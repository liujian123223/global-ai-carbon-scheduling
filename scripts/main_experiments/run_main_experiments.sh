#!/usr/bin/env bash
set -Eeuo pipefail

# Full manuscript grid: 5 years x 4 demand pathways x 3 grid pathways x
# 3 governance cases. Each case runs LI, TM, SM and JSM in one Python process.
# Scientific settings match the recorded manuscript commands and defaults.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/main_experiments/run_main_experiments.sh"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/main_experiments/run_scheduling.py}"
WORKLOAD_PATH="${WORKLOAD_PATH:-dataset/cluster-trace-gpu-v2026}"
WORKLOAD_CACHE_DIR="${WORKLOAD_CACHE_DIR:-results/workload_profile_cache}"
WORKLOAD_CACHE_WORKERS="${WORKLOAD_CACHE_WORKERS:-8}"
# All pathways default to the same project-relative carbon directory.
# A shared explicit directory overrides any pathway-specific overrides.
if [[ -n "${HOURLY_CARBON_DIR:-}" ]]; then
  CP_HOURLY_CARBON_DIR="$HOURLY_CARBON_DIR"
  IEA_HOURLY_CARBON_DIR="$HOURLY_CARBON_DIR"
else
  CP_HOURLY_CARBON_DIR="${CP_HOURLY_CARBON_DIR:-dataset/EM-CPNDCNZ}"
  IEA_HOURLY_CARBON_DIR="${IEA_HOURLY_CARBON_DIR:-dataset/EM-CPNDCNZ}"
fi
AZURE_TRANSMISSION_DIR="${AZURE_TRANSMISSION_DIR:-dataset/Azure}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/main_experiments}"
WORKERS="${WORKERS:-1}"
PREPARE_CACHE="${PREPARE_CACHE:-1}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

YEARS=(2026 2027 2028 2029 2030)
SCENARIOS=("Base" "Lift-Off" "High Efficiency" "Headwinds")
POLICIES=(CP NDC NZ)
GOVERNANCE=(global_transfer_with_safeguards eea_adequacy_trusted_zone eea_only_regional_residency)
declare -A SCENARIO_SLUG=([Base]=base [Lift-Off]=lift_off ["High Efficiency"]=high_efficiency [Headwinds]=headwinds)
declare -A GOVERNANCE_SLUG=([global_transfer_with_safeguards]=global [eea_adequacy_trusted_zone]=trusted [eea_only_regional_residency]=eea_only)
BASELINE_METHODS=(
  local_immediate_training_overflow_constrained
  local_training_time_shift_greedy_constrained
  annual_average_immediate_overflow_greedy_constrained
)
CASE_YEARS=() CASE_SCENARIOS=() CASE_POLICIES=() CASE_GOVERNANCE=()
COMMAND=() METHODS=() PIDS=()
RUN_ID="" CARBON_DIR=""

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
die() { log "ERROR: $*"; exit 1; }
usage() {
  cat <<'EOF'
Usage: bash scripts/main_experiments/run_main_experiments.sh

Runs all 180 configurations (four manuscript methods each).
  WORKERS=1                  Serial by default; set WORKERS=N for parallel cases.
  DRY_RUN=1                  Write run_plan.tsv and print commands; no solving.
  OUTPUT_ROOT=...            Results (default: results/main_experiments).
  WORKLOAD_PATH=...          Alibaba GPU v2026 directory.
  HOURLY_CARBON_DIR=...       Shared carbon directory for CP/NDC/NZ; overrides both below.
  CP_HOURLY_CARBON_DIR=...    Optional CP override (default: dataset/EM-CPNDCNZ).
  IEA_HOURLY_CARBON_DIR=...   Optional NDC/NZ override (default: dataset/EM-CPNDCNZ).
  PYTHON_BIN=...             Python executable in the experiment environment.
  PREPARE_CACHE=0            Use existing caches without a preparation pass.
  FORCE=1                   Recompute and replace results in OUTPUT_ROOT.
  STOP_ON_ERROR=1            Stop the assigned worker after a failed case.

Training/offline delays 24/6 h; commit window 24 h; 8760 h;
matched_window baselines; SciPy solver; one numeric thread per process.
Missing data is allowed only for DRY_RUN. CP, NDC and NZ all default to
dataset/EM-CPNDCNZ. Set input overrides before generating run_plan.tsv.
EOF
}

build_design() {
  local year scenario policy governance key
  local -A seen=()
  for year in "${YEARS[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
      for policy in "${POLICIES[@]}"; do
        for governance in "${GOVERNANCE[@]}"; do
          key="$year|$scenario|$policy|$governance"
          [[ ! -v 'seen[$key]' ]] || die "Duplicate case: $key"
          seen["$key"]=1
          CASE_YEARS+=("$year"); CASE_SCENARIOS+=("$scenario")
          CASE_POLICIES+=("$policy"); CASE_GOVERNANCE+=("$governance")
        done
      done
    done
  done
  [[ ${#seen[@]} -eq 180 ]] || die "Expected 180 unique cases."
}

validate_environment() {
  [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] && (( WORKERS <= 180 )) || die "WORKERS must be between 1 and 180."
  [[ "$WORKLOAD_CACHE_WORKERS" =~ ^[1-9][0-9]*$ ]] || die "WORKLOAD_CACHE_WORKERS must be positive."
  local name
  for name in DRY_RUN PREPARE_CACHE FORCE STOP_ON_ERROR; do
    [[ "${!name}" == 0 || "${!name}" == 1 ]] || die "$name must be 0 or 1."
  done
  PYTHON_BIN="$(command -v "$PYTHON_BIN" 2>/dev/null)" || die "Python executable not found."
  [[ -f "$RUN_SCRIPT" ]] || die "Runner not found: $RUN_SCRIPT"
  if [[ "$DRY_RUN" == 1 ]]; then
    log "DRY_RUN: input availability is not required; this checks the run plan only."
    return 0
  fi
  [[ -d "$WORKLOAD_PATH" ]] || die "Workload directory missing: $WORKLOAD_PATH"
  [[ -f dataset/Factors.py ]] || die "Missing model parameters: dataset/Factors.py"
  local policy directory year count governance
  for policy in "${POLICIES[@]}"; do
    directory="$IEA_HOURLY_CARBON_DIR"
    [[ "$policy" != CP ]] || directory="$CP_HOURLY_CARBON_DIR"
    [[ -d "$directory" ]] || die "Missing hourly carbon directory: $directory"
    for year in "${YEARS[@]}"; do
      count="$(find "$directory" -type f -name "*-${policy}-${year}-hourly.csv" | wc -l)"
      [[ "$count" -eq 24 ]] || die "Expected 24 hourly files for $policy/$year in $directory; found $count."
    done
  done
  for governance in "${GOVERNANCE[@]}"; do
    [[ -f "$AZURE_TRANSMISSION_DIR/azure_${governance}.csv" ]] || die "Missing governance matrix: $governance"
  done
}

build_command() {
  local index="$1"
  local year="${CASE_YEARS[$index]}" scenario="${CASE_SCENARIOS[$index]}"
  local policy="${CASE_POLICIES[$index]}" governance="${CASE_GOVERNANCE[$index]}"
  RUN_ID="${year}_${SCENARIO_SLUG[$scenario]}_${policy,,}_${GOVERNANCE_SLUG[$governance]}"
  CARBON_DIR="$IEA_HOURLY_CARBON_DIR"
  METHODS=("${BASELINE_METHODS[@]}" linear_programming)
  if [[ "$policy" == CP ]]; then
    CARBON_DIR="$CP_HOURLY_CARBON_DIR"
    # CP's historical full-method commands placed LP first.
    METHODS=(linear_programming "${BASELINE_METHODS[@]}")
  fi
  COMMAND=(
    "$PYTHON_BIN" "$RUN_SCRIPT"
    --policy "$policy" --scenarios "$scenario" --years 1 --year-start "$year"
    --constraint global --delay-hours 24 --offline-inference-delay-hours 6
    --solve-mode windowed --commit-hours 24 --baseline-scheduling-mode matched_window
    --solver scipy --max-intervals 8760
    --execution-policy capacity --inference-origin-fraction 0.75 --other-origin-fraction 0.50
    --capacity-quantile 0.96 --max-resource-utilization 1.0
    --workload-path "$WORKLOAD_PATH" --workload-cache-dir "$WORKLOAD_CACHE_DIR"
    --workload-cache-workers "$WORKLOAD_CACHE_WORKERS"
    --hourly-carbon-dir "$CARBON_DIR" --hourly-carbon-scope direct --strict-hourly-carbon
    --azure-transmission-policy "$governance" --azure-transmission-dir "$AZURE_TRANSMISSION_DIR"
    --methods "${METHODS[@]}" --ramping-trace-cycle-hours 4440
    --output-dir "$OUTPUT_ROOT/$RUN_ID"
  )
}

write_plan() {
  mkdir -p "$OUTPUT_ROOT"
  local index temporary="$OUTPUT_ROOT/run_plan.tsv.tmp.$$"
  printf 'run_id\tyear\tdemand_pathway\tgrid_pathway\tgovernance_case\thourly_carbon_dir\tcommand\n' >"$temporary"
  for index in "${!CASE_YEARS[@]}"; do
    build_command "$index"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t' "$RUN_ID" "${CASE_YEARS[$index]}" \
      "${CASE_SCENARIOS[$index]}" "${CASE_POLICIES[$index]}" "${CASE_GOVERNANCE[$index]}" "$CARBON_DIR" >>"$temporary"
    printf '%q ' "${COMMAND[@]}" >>"$temporary"
    printf '\n' >>"$temporary"
  done
  mv "$temporary" "$OUTPUT_ROOT/run_plan.tsv"
  log "PLAN CHECK: 180 unique configurations; 60 CP, 60 NDC, 60 NZ; 720 method runs."
}

prepare_cache() {
  [[ "$PREPARE_CACHE" == 1 && "$DRY_RUN" != 1 ]] || return 0
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$PYTHON_BIN" "$RUN_SCRIPT" --workload-path "$WORKLOAD_PATH" \
    --workload-cache-dir "$WORKLOAD_CACHE_DIR" --workload-cache-workers "$WORKLOAD_CACHE_WORKERS" \
    --capacity-quantile 0.96 --max-intervals 8760 --prepare-workload-cache-only
}

has_complete_outputs() {
  "$PYTHON_BIN" - "$1" "${METHODS[@]}" <<'PY'
import csv
import sys
from pathlib import Path

directory = Path(sys.argv[1])
expected = set(sys.argv[2:])
for pattern in (
    "Training_Migration_Execution_*.csv",
    "Offline_Inference_Migration_Execution_*.csv",
    "Country_Resource_Capacity_*.csv",
    "Country_Component_Energy_*.csv",
    "Migration_Flow_Summary_*.csv",
    "Flexible_Migration_Flow_Summary_*.csv",
    "Transmission_Policy_Summary_*.csv",
    "Capacity_Binding_Summary_*.csv",
    "Training_Migration_Workload_Profile_Summary.csv",
):
    if len(list(directory.glob(pattern))) != 1:
        sys.exit(1)
patterns = (
    "Training_Migration_Annual_Summary_*.csv",
    "Training_Migration_Country_Summary_*.csv",
    "Training_Migration_System_Ramping_Summary_*.csv",
    "Training_Migration_Country_Ramping_Summary_*.csv",
)
for index, pattern in enumerate(patterns):
    files = list(directory.glob(pattern))
    if len(files) != 1:
        sys.exit(1)
    try:
        with files[0].open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if {row.get("method") for row in rows} != expected:
            sys.exit(1)
        if index == 0 and (
            len(rows) != len(expected)
            or any(float(row.get("solver_status", "nan")) != 0 for row in rows)
        ):
            sys.exit(1)
    except (OSError, ValueError):
        sys.exit(1)
PY
}

run_case() {
  build_command "$1"
  local output_dir="$OUTPUT_ROOT/$RUN_ID" command_text
  printf -v command_text '%q ' "${COMMAND[@]}"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[DRY-RUN] %s\n' "$command_text"
    return 0
  fi
  if [[ "$FORCE" != 1 && -f "$output_dir/command.sh" ]]; then
    if [[ "$(cat "$output_dir/command.sh")" != "$command_text" ]]; then
      log "ERROR: existing command differs for $RUN_ID; choose a new OUTPUT_ROOT or explicitly set FORCE=1."
      return 1
    fi
    if has_complete_outputs "$output_dir"; then
      log "SKIP complete: $RUN_ID"
      return 0
    fi
  elif [[ "$FORCE" != 1 ]] && compgen -G "$output_dir/*.csv" >/dev/null; then
    log "ERROR: existing CSVs lack a matching command record for $RUN_ID; choose a new OUTPUT_ROOT or explicitly set FORCE=1."
    return 1
  fi
  mkdir -p "$output_dir"
  printf '%s\n' "$command_text" >"$output_dir/command.sh"
  log "RUN $RUN_ID"
  if ! env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "${COMMAND[@]}" >"$output_dir/run.log" 2>&1; then
    log "FAILED $RUN_ID; see $output_dir/run.log"
    return 1
  fi
  if ! has_complete_outputs "$output_dir"; then
    log "FAILED output validation: $RUN_ID"
    return 1
  fi
  log "DONE $RUN_ID"
}

run_worker() {
  local worker="$1" index failures=0 assigned=0
  [[ "$worker" =~ ^[0-9]+$ ]] && (( worker < WORKERS )) || die "Invalid worker index."
  for index in "${!CASE_YEARS[@]}"; do
    (( index % WORKERS == worker )) || continue
    assigned=$((assigned + 1))
    if ! run_case "$index"; then
      failures=$((failures + 1))
      [[ "$STOP_ON_ERROR" != 1 ]] || return 1
    fi
  done
  log "Worker $((worker + 1))/$WORKERS: assigned=$assigned failed=$failures dry_run=$DRY_RUN"
  (( failures == 0 ))
}

stop_workers() {
  local pid
  for pid in "${PIDS[@]}"; do
    [[ -n "$pid" ]] || continue
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}

main() {
  validate_environment
  build_design
  write_plan
  prepare_cache
  if [[ "$WORKERS" == 1 ]]; then
    run_worker 0
    return
  fi
  export PYTHON_BIN RUN_SCRIPT WORKLOAD_PATH WORKLOAD_CACHE_DIR WORKLOAD_CACHE_WORKERS
  export CP_HOURLY_CARBON_DIR IEA_HOURLY_CARBON_DIR AZURE_TRANSMISSION_DIR OUTPUT_ROOT
  export WORKERS DRY_RUN FORCE STOP_ON_ERROR
  mkdir -p "$OUTPUT_ROOT/_worker_logs"
  local worker index failures=0 log_path
  trap 'stop_workers; exit 130' INT TERM
  for ((worker=0; worker<WORKERS; worker++)); do
    printf -v log_path '%s/_worker_logs/worker%03d.log' "$OUTPUT_ROOT" "$((worker + 1))"
    PREPARE_CACHE=0 bash "$SCRIPT_PATH" --worker "$worker" >"$log_path" 2>&1 &
    PIDS+=("$!")
  done
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$index]}"; then failures=$((failures + 1)); fi
    PIDS[$index]=""
  done
  trap - INT TERM
  log "Workers=$WORKERS failed=$failures dry_run=$DRY_RUN; logs: $OUTPUT_ROOT/_worker_logs"
  (( failures == 0 ))
}

case "${1:-}" in
  --help|-h) usage ;;
  --worker)
    [[ $# == 2 ]] || die "Internal --worker requires one index."
    validate_environment
    build_design
    run_worker "$2"
    ;;
  "") main ;;
  *) usage >&2; exit 2 ;;
esac
