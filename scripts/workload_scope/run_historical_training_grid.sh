#!/usr/bin/env bash
set -Eeuo pipefail

# Run the 8182 carbon-only workload training migration optimization grid.
#
# Default grid:
#   years:      2026, 2027, 2028, 2029, 2030
#   policies:   CP, NDC, NZ
#   scenarios:  Base, Lift-Off, High Efficiency, Headwinds
#   migration:  global, EEA-only, EEA adequacy/trusted zone
#
# Output directory pattern:
#   <OUTPUT_ROOT>/<year>_<scenario>_<policy>_<migration>
# Example:
#   results/8182_grid_2026_2030_carbon_only/2026_base_cp_eea_trust
#
# Useful overrides:
#   DRY_RUN=1 bash scripts/workload_scope/run_historical_training_grid.sh
#   FORCE=1 bash scripts/workload_scope/run_historical_training_grid.sh
#   WORKLOAD_PATH=/path/to/cluster-trace-gpu-v2026 bash scripts/workload_scope/run_historical_training_grid.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/workload_scope/run_historical_training_grid.sh"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/workload_scope/run_training_only.py}"
WORKLOAD_PATH="${WORKLOAD_PATH:-dataset/cluster-trace-gpu-v2026}"
WORKLOAD_CACHE_DIR="${WORKLOAD_CACHE_DIR:-results/workload_profile_cache}"
HOURLY_CARBON_DIR="${HOURLY_CARBON_DIR:-dataset/EM-CPNDCNZ}"
AZURE_TRANSMISSION_DIR="${AZURE_TRANSMISSION_DIR:-dataset/Azure}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/8182_grid_2026_2030_carbon_only}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/_worker_logs}"
WORKERS="${WORKERS:-1}"
PREPARE_CACHE="${PREPARE_CACHE:-1}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
PIDS=()

CONSTRAINT="${CONSTRAINT:-global}"
DELAY_HOURS="${DELAY_HOURS:-24}"
SOLVE_MODE="${SOLVE_MODE:-windowed}"
COMMIT_HOURS="${COMMIT_HOURS:-24}"
SOLVER="${SOLVER:-scipy}"
MAX_INTERVALS="${MAX_INTERVALS:-8760}"
METHODS="${METHODS:-all}"
STRICT_HOURLY_CARBON="${STRICT_HOURLY_CARBON:-1}"

DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
RUN_COUNT=0
SKIP_COUNT=0

YEARS=(2026 2027 2028 2029 2030)
POLICIES=(CP NDC NZ)
SCENARIOS=("Base" "Lift-Off" "High Efficiency" "Headwinds")
MIGRATION_POLICIES=(
  "global_transfer_with_safeguards"
  "eea_only_regional_residency"
  "eea_adequacy_trusted_zone"
)

declare -A SCENARIO_SLUG=(
  ["Base"]="base"
  ["Lift-Off"]="lift_off"
  ["High Efficiency"]="high_efficiency"
  ["Headwinds"]="headwinds"
)

declare -A MIGRATION_SLUG=(
  ["global_transfer_with_safeguards"]="global"
  ["eea_only_regional_residency"]="eea_only"
  ["eea_adequacy_trusted_zone"]="eea_trust"
)

declare -A MIGRATION_FILE=(
  ["global_transfer_with_safeguards"]="azure_global_transfer_with_safeguards.csv"
  ["eea_only_regional_residency"]="azure_eea_only_regional_residency.csv"
  ["eea_adequacy_trusted_zone"]="azure_eea_adequacy_trusted_zone.csv"
)

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

has_annual_summary() {
  local out_dir="$1"
  local summaries=()
  shopt -s nullglob
  summaries=("$out_dir"/Training_Migration_Annual_Summary_*.csv)
  shopt -u nullglob
  [[ ${#summaries[@]} -gt 0 ]]
}

validate_inputs() {
  [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] && (( WORKERS <= 180 )) || die "WORKERS must be between 1 and 180."
  [[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || die "SHARD_COUNT must be positive."
  [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] && (( SHARD_INDEX < SHARD_COUNT )) || die "Invalid SHARD_INDEX."
  [[ -f "$RUN_SCRIPT" ]] || die "Run script not found: $RUN_SCRIPT"
  [[ -d "$HOURLY_CARBON_DIR" ]] || die "Hourly carbon directory not found: $HOURLY_CARBON_DIR"
  [[ -d "$AZURE_TRANSMISSION_DIR" ]] || die "Azure transmission directory not found: $AZURE_TRANSMISSION_DIR"

  if [[ ! -d "$WORKLOAD_PATH" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      log "WARN: workload path not found, allowed because DRY_RUN=1: $WORKLOAD_PATH"
    else
      die "Workload path not found: $WORKLOAD_PATH"
    fi
  fi

  local expected_hourly_count=$(( ${#YEARS[@]} * ${#POLICIES[@]} * 24 ))
  local hourly_count
  hourly_count="$(find "$HOURLY_CARBON_DIR" -type f -name '*-hourly.csv' | wc -l | tr -d ' ')"
  [[ "$hourly_count" -ge "$expected_hourly_count" ]] || die \
    "Hourly carbon CSV count is $hourly_count, expected at least $expected_hourly_count"

  local year policy count matrix
  for year in "${YEARS[@]}"; do
    for policy in "${POLICIES[@]}"; do
      count="$(find "$HOURLY_CARBON_DIR" -type f -name "*-${policy}-${year}-hourly.csv" | wc -l | tr -d ' ')"
      [[ "$count" -eq 24 ]] || die \
        "Hourly carbon data for ${policy}-${year} has $count country files, expected 24"
    done
  done

  local migration_policy
  for migration_policy in "${MIGRATION_POLICIES[@]}"; do
    matrix="$AZURE_TRANSMISSION_DIR/${MIGRATION_FILE[$migration_policy]}"
    [[ -f "$matrix" ]] || die "Azure migration matrix not found: $matrix"
  done
}

run_one() {
  local year="$1"
  local policy="$2"
  local scenario="$3"
  local migration_policy="$4"

  local scenario_slug="${SCENARIO_SLUG[$scenario]}"
  local policy_slug
  policy_slug="$(printf '%s' "$policy" | tr '[:upper:]' '[:lower:]')"
  local migration_slug="${MIGRATION_SLUG[$migration_policy]}"
  local run_id="${year}_${scenario_slug}_${policy_slug}_${migration_slug}"
  local out_dir="$OUTPUT_ROOT/$run_id"

  if [[ "$FORCE" != "1" ]] && has_annual_summary "$out_dir"; then
    log "SKIP existing annual summary: $run_id"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return 0
  fi

  mkdir -p "$out_dir"

  local methods_args=()
  read -r -a methods_args <<< "$METHODS"

  local cmd=(
    "$PYTHON_BIN" "$RUN_SCRIPT"
    --policy "$policy"
    --scenarios "$scenario"
    --years 1
    --year-start "$year"
    --constraint "$CONSTRAINT"
    --delay-hours "$DELAY_HOURS"
    --solve-mode "$SOLVE_MODE"
    --commit-hours "$COMMIT_HOURS"
    --solver "$SOLVER"
    --max-intervals "$MAX_INTERVALS"
    --workload-path "$WORKLOAD_PATH"
    --hourly-carbon-dir "$HOURLY_CARBON_DIR"
    --azure-transmission-policy "$migration_policy"
    --azure-transmission-dir "$AZURE_TRANSMISSION_DIR"
    --methods "${methods_args[@]}"
    --workload-cache-dir "$WORKLOAD_CACHE_DIR"
    --output-dir "$out_dir"
  )

  if [[ "$STRICT_HOURLY_CARBON" == "1" ]]; then
    cmd+=(--strict-hourly-carbon)
  fi

  log "RUN $run_id"
  RUN_COUNT=$((RUN_COUNT + 1))
  printf '%q ' "${cmd[@]}" > "$out_dir/command.sh"
  printf '\n' >> "$out_dir/command.sh"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY-RUN] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}" 2>&1 | tee "$out_dir/run.log"
}

main() {
  validate_inputs
  mkdir -p "$OUTPUT_ROOT"

  local total=0
  local started_at=$SECONDS

  local year policy scenario migration_policy
  for year in "${YEARS[@]}"; do
    for policy in "${POLICIES[@]}"; do
      for scenario in "${SCENARIOS[@]}"; do
        for migration_policy in "${MIGRATION_POLICIES[@]}"; do
          total=$((total + 1))
          (( (total - 1) % SHARD_COUNT == SHARD_INDEX )) || continue
          run_one "$year" "$policy" "$scenario" "$migration_policy"
        done
      done
    done
  done

  log "DONE grid traversal: $total combinations visited, $RUN_COUNT run/planned, $SKIP_COUNT skipped, elapsed $((SECONDS - started_at))s"
}

prepare_cache_once() {
  [[ "$PREPARE_CACHE" == 1 && "$DRY_RUN" != 1 ]] || return 0
  log "Preparing the training-only workload cache before parallel execution."
  "$PYTHON_BIN" - "$RUN_SCRIPT" "$WORKLOAD_PATH" "$WORKLOAD_CACHE_DIR" "$MAX_INTERVALS" <<'PY'
import runpy
import sys
from pathlib import Path

# Load the same engine and defaults as the existing training-only entry point.
runner = runpy.run_path(sys.argv[1])
runner["opt"].build_workload_profile(
    workload_profile_path=Path(sys.argv[2]),
    workload_cache_dir=Path(sys.argv[3]),
    capacity_quantile=0.96,
    max_intervals=int(sys.argv[4]),
    verbose=True,
)
PY
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

dispatch() {
  if [[ "$WORKERS" == 1 ]]; then main; return; fi
  [[ "$SHARD_COUNT" == 1 && "$SHARD_INDEX" == 0 ]] || die \
    "Use WORKERS for automatic parallel execution; manual shards require WORKERS=1."
  validate_inputs
  prepare_cache_once
  export PYTHON_BIN RUN_SCRIPT WORKLOAD_PATH WORKLOAD_CACHE_DIR
  export HOURLY_CARBON_DIR AZURE_TRANSMISSION_DIR OUTPUT_ROOT LOG_DIR
  export CONSTRAINT DELAY_HOURS SOLVE_MODE COMMIT_HOURS SOLVER MAX_INTERVALS METHODS
  export STRICT_HOURLY_CARBON DRY_RUN FORCE
  local worker index failures=0 log_path worker_count="$WORKERS"
  mkdir -p "$LOG_DIR"
  trap 'stop_workers; exit 130' INT TERM
  for ((worker=0; worker<worker_count; worker++)); do
    printf -v log_path '%s/worker%02d.log' "$LOG_DIR" "$((worker + 1))"
    WORKERS=1 SHARD_COUNT="$worker_count" SHARD_INDEX="$worker" \
      PREPARE_CACHE=0 bash "$SCRIPT_PATH" --worker >"$log_path" 2>&1 &
    PIDS+=("$!")
  done
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$index]}"; then failures=$((failures + 1)); fi
    PIDS[$index]=""
  done
  trap - INT TERM
  log "Workers=$WORKERS failed=$failures dry_run=$DRY_RUN; 180 configurations assigned; logs: $LOG_DIR"
  (( failures == 0 ))
}

case "${1:-}" in
  --help|-h)
    printf '%s\n' 'Usage: bash scripts/workload_scope/run_historical_training_grid.sh' \
      'WORKERS=1 by default (serial); set WORKERS=N for parallel execution.' \
      'The historical grid includes all 180 configurations, including 2026 CP Base.' \
      'Existing result directories are skipped unless FORCE=1. DRY_RUN=1 prints commands only.' \
      'PREPARE_CACHE=0 skips cache preparation; use only with a prepared cache.'
    ;;
  --worker) [[ $# == 1 ]] || die "Internal --worker takes no arguments."; main ;;
  "") dispatch ;;
  *) die "Unknown argument: $1" ;;
esac
