#!/usr/bin/env bash
set -Eeuo pipefail

# Result-828 one-factor-at-a-time sensitivity experiments.
#
# Fixed setting: CP-Base-global, years 2026-2030.
#   A. commit window:             12, 24, 48, 96 h
#   B. training migration delay:   0, 12, 24, 48 h
#   C. offline-inference delay:     0,  2,  6, 12 h
#
# The existing main point (commit=24, training=24, offline=6) is omitted.
# This leaves 3 experiments x 5 years x 3 new levels = 45 parameter points.
# Each point runs LP and all four matched-window fair baselines.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/time_sensitivity/run_time_sensitivity_45_cases.sh"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/main_experiments/run_scheduling.py}"
WORKLOAD_PATH="${WORKLOAD_PATH:-dataset/cluster-trace-gpu-v2026}"
WORKLOAD_CACHE_DIR="${WORKLOAD_CACHE_DIR:-results/workload_profile_cache}"
WORKLOAD_CACHE_WORKERS="${WORKLOAD_CACHE_WORKERS:-8}"
HOURLY_CARBON_DIR="${HOURLY_CARBON_DIR:-dataset/EM-CPNDCNZ}"
AZURE_TRANSMISSION_DIR="${AZURE_TRANSMISSION_DIR:-dataset/Azure}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/Result-V3/Result-V3-parameter-sensitivity}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/_worker_logs}"
WORKERS="${WORKERS:-1}"
PREPARE_CACHE="${PREPARE_CACHE:-1}"

BASELINE_SCHEDULING_MODE="${BASELINE_SCHEDULING_MODE:-matched_window}"
SOLVER="${SOLVER:-scipy}"
MAX_INTERVALS="${MAX_INTERVALS:-8760}"
RAMPING_TRACE_CYCLE_HOURS="${RAMPING_TRACE_CYCLE_HOURS:-4440}"

DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

YEARS=(2026 2027 2028 2029 2030)
COMMIT_WINDOW_LEVELS=(12 48 96)
TRAINING_DELAY_LEVELS=(0 12 48)
OFFLINE_DELAY_LEVELS=(0 2 12)

POLICY="CP"
SCENARIO="Base"
MIGRATION_POLICY="global_transfer_with_safeguards"
MIGRATION_MATRIX="azure_global_transfer_with_safeguards.csv"

ALL_METHODS=(
  linear_programming
  local_immediate_training_overflow_constrained
  local_training_time_shift_greedy_constrained
  annual_average_immediate_overflow_greedy_constrained
  hourly_arrival_greedy_constrained_fallback
)

DESIGN_COUNT=0
ASSIGNED_COUNT=0
SHARD_FILTERED_COUNT=0
RUN_COUNT=0
SKIP_COUNT=0
METHOD_SOLVE_COUNT=0
FAILED_COUNT=0
FAILED_RUN_IDS=()
PIDS=()
declare -A SEEN_PARAMETERS=()

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

has_complete_outputs() {
  local output_dir="$1"
  local annual_files=()
  local offline_execution_files=()
  local country_ramping_files=()
  local system_ramping_files=()

  shopt -s nullglob
  annual_files=("$output_dir"/Training_Migration_Annual_Summary_*.csv)
  offline_execution_files=("$output_dir"/Offline_Inference_Migration_Execution_*.csv)
  country_ramping_files=("$output_dir"/Training_Migration_Country_Ramping_Summary_*.csv)
  system_ramping_files=("$output_dir"/Training_Migration_System_Ramping_Summary_*.csv)
  shopt -u nullglob

  [[ ${#annual_files[@]} -gt 0 && \
     ${#offline_execution_files[@]} -gt 0 && \
     ${#country_ramping_files[@]} -gt 0 && \
     ${#system_ramping_files[@]} -gt 0 ]] || return 1

  local method
  for method in "${ALL_METHODS[@]}"; do
    awk -F, -v target="$method" '
      NR == 1 {
        for (i = 1; i <= NF; i++) {
          if ($i == "method") method_col = i
          if ($i == "solver_status") status_col = i
        }
        next
      }
      method_col && status_col && $method_col == target &&
        $status_col != "" && ($status_col + 0) == 0 { found = 1 }
      END { exit(found ? 0 : 1) }
    ' "${annual_files[0]}" || return 1
  done
}

validate_inputs() {
  [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] && (( WORKERS <= 45 )) || die "WORKERS must be between 1 and 45."
  [[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || die \
    "SHARD_COUNT must be a positive integer: $SHARD_COUNT"
  [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] || die \
    "SHARD_INDEX must be a non-negative integer: $SHARD_INDEX"
  (( SHARD_INDEX < SHARD_COUNT )) || die \
    "SHARD_INDEX must be smaller than SHARD_COUNT: ${SHARD_INDEX} >= ${SHARD_COUNT}"
  [[ "$BASELINE_SCHEDULING_MODE" == "matched_window" ]] || die \
    "Fair baselines require BASELINE_SCHEDULING_MODE=matched_window"
  [[ "$RUN_SCRIPT" == *"run_scheduling.py" ]] || die \
    "Sensitivity experiments must use the unchanged 828 runner: $RUN_SCRIPT"
  [[ -f "$RUN_SCRIPT" ]] || die "Run script not found: $RUN_SCRIPT"
  [[ -d "$HOURLY_CARBON_DIR" ]] || die \
    "Hourly carbon directory not found: $HOURLY_CARBON_DIR"
  [[ -d "$AZURE_TRANSMISSION_DIR" ]] || die \
    "Azure transmission directory not found: $AZURE_TRANSMISSION_DIR"
  [[ -f "$AZURE_TRANSMISSION_DIR/$MIGRATION_MATRIX" ]] || die \
    "Migration matrix not found: $AZURE_TRANSMISSION_DIR/$MIGRATION_MATRIX"

  if [[ ! -d "$WORKLOAD_PATH" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      log "WARN: workload path not found, allowed for DRY_RUN: $WORKLOAD_PATH"
    else
      die "Workload path not found: $WORKLOAD_PATH (set WORKLOAD_PATH explicitly)"
    fi
  fi

  local year hourly_count
  for year in "${YEARS[@]}"; do
    hourly_count="$(find "$HOURLY_CARBON_DIR" -type f \
      -name "*-${POLICY}-${year}-hourly.csv" | wc -l | tr -d ' ')"
    [[ "$hourly_count" -eq 24 ]] || die \
      "Hourly carbon data for ${POLICY}-${year} has ${hourly_count} files; expected 24"
  done
}

execute_point() {
  local experiment="$1"
  local year="$2"
  local commit_hours="$3"
  local training_delay_hours="$4"
  local offline_delay_hours="$5"
  local parameter_key="${year}|${commit_hours}|${training_delay_hours}|${offline_delay_hours}"
  local run_id="${experiment}_${year}_cp_base_global_commit${commit_hours}_train${training_delay_hours}_offline${offline_delay_hours}"
  local output_dir="$OUTPUT_ROOT/$run_id"

  [[ -z "${SEEN_PARAMETERS[$parameter_key]:-}" ]] || die \
    "Duplicate parameter point generated by ${SEEN_PARAMETERS[$parameter_key]} and ${experiment}: $parameter_key"
  SEEN_PARAMETERS[$parameter_key]="$experiment"

  local design_position="$DESIGN_COUNT"
  DESIGN_COUNT=$((DESIGN_COUNT + 1))
  if [[ $((design_position % SHARD_COUNT)) -ne "$SHARD_INDEX" ]]; then
    SHARD_FILTERED_COUNT=$((SHARD_FILTERED_COUNT + 1))
    return 0
  fi
  ASSIGNED_COUNT=$((ASSIGNED_COUNT + 1))

  if [[ "$FORCE" != "1" ]] && has_complete_outputs "$output_dir"; then
    log "SKIP complete: $run_id"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return 0
  fi

  local command=(
    "$PYTHON_BIN" "$RUN_SCRIPT"
    --policy "$POLICY"
    --scenarios "$SCENARIO"
    --years 1
    --year-start "$year"
    --constraint global
    --delay-hours "$training_delay_hours"
    --offline-inference-delay-hours "$offline_delay_hours"
    --solve-mode windowed
    --commit-hours "$commit_hours"
    --baseline-scheduling-mode "$BASELINE_SCHEDULING_MODE"
    --solver "$SOLVER"
    --max-intervals "$MAX_INTERVALS"
    --workload-path "$WORKLOAD_PATH"
    --workload-cache-dir "$WORKLOAD_CACHE_DIR"
    --workload-cache-workers "$WORKLOAD_CACHE_WORKERS"
    --hourly-carbon-dir "$HOURLY_CARBON_DIR"
    --strict-hourly-carbon
    --azure-transmission-policy "$MIGRATION_POLICY"
    --azure-transmission-dir "$AZURE_TRANSMISSION_DIR"
    --methods "${ALL_METHODS[@]}"
    --ramping-trace-cycle-hours "$RAMPING_TRACE_CYCLE_HOURS"
    --output-dir "$output_dir"
  )

  RUN_COUNT=$((RUN_COUNT + 1))
  METHOD_SOLVE_COUNT=$((METHOD_SOLVE_COUNT + ${#ALL_METHODS[@]}))
  log "PLAN experiment=${experiment} run=${run_id} methods=${#ALL_METHODS[@]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY-RUN] '
    printf '%q ' "${command[@]}"
    printf '\n'
    return 0
  fi

  mkdir -p "$output_dir"
  printf '%q ' "${command[@]}" > "$output_dir/command.sh"
  printf '\n' >> "$output_dir/command.sh"

  if "${command[@]}" 2>&1 | tee "$output_dir/run.log"; then
    log "COMPLETE: $run_id"
  else
    local exit_code=$?
    FAILED_COUNT=$((FAILED_COUNT + 1))
    FAILED_RUN_IDS+=("$run_id")
    log "FAILED (${exit_code}): $run_id"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      return "$exit_code"
    fi
  fi
}

run_experiment_a() {
  local year commit_hours
  for year in "${YEARS[@]}"; do
    for commit_hours in "${COMMIT_WINDOW_LEVELS[@]}"; do
      execute_point "A" "$year" "$commit_hours" 24 6
    done
  done
}

run_experiment_b() {
  local year training_delay_hours
  for year in "${YEARS[@]}"; do
    for training_delay_hours in "${TRAINING_DELAY_LEVELS[@]}"; do
      execute_point "B" "$year" 24 "$training_delay_hours" 6
    done
  done
}

run_experiment_c() {
  local year offline_delay_hours
  for year in "${YEARS[@]}"; do
    for offline_delay_hours in "${OFFLINE_DELAY_LEVELS[@]}"; do
      execute_point "C" "$year" 24 24 "$offline_delay_hours"
    done
  done
}

check_plan_counts() {
  [[ "$DESIGN_COUNT" -eq 45 ]] || die \
    "Unexpected sensitivity design size: ${DESIGN_COUNT}; expected 45"
  [[ "${#SEEN_PARAMETERS[@]}" -eq 45 ]] || die \
    "Unexpected unique parameter count: ${#SEEN_PARAMETERS[@]}; expected 45"

  local expected_assigned=$((45 / SHARD_COUNT))
  if (( SHARD_INDEX < 45 % SHARD_COUNT )); then
    expected_assigned=$((expected_assigned + 1))
  fi
  [[ "$ASSIGNED_COUNT" -eq "$expected_assigned" ]] || die \
    "Unexpected shard assignment: ${ASSIGNED_COUNT}; expected ${expected_assigned}"

  log "PLAN CHECK: 45-point sensitivity design confirmed; shard $((SHARD_INDEX + 1))/${SHARD_COUNT} owns ${expected_assigned} points."
}

main() {
  validate_inputs

  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$OUTPUT_ROOT"
  fi

  local started_at=$SECONDS
  log "Starting Result-828 CP-Base-global sensitivity design; shard=$((SHARD_INDEX + 1))/${SHARD_COUNT}"
  log "All runs use LP plus four matched-window fair baselines."
  run_experiment_a
  run_experiment_b
  run_experiment_c
  check_plan_counts

  log "DONE: ${ASSIGNED_COUNT} assigned, ${RUN_COUNT} runs, ${METHOD_SOLVE_COUNT} method solves, ${SKIP_COUNT} skips, ${SHARD_FILTERED_COUNT} assigned to other shards, ${FAILED_COUNT} failed, elapsed $((SECONDS - started_at))s"

  if [[ "$FAILED_COUNT" -gt 0 ]]; then
    log "FAILED RUNS: ${FAILED_RUN_IDS[*]}"
    return 1
  fi
}

prepare_cache_once() {
  [[ "$PREPARE_CACHE" == 1 && "$DRY_RUN" != 1 ]] || return 0
  log "Preparing shared workload caches before parallel execution."
  "$PYTHON_BIN" "$RUN_SCRIPT" \
    --workload-path "$WORKLOAD_PATH" \
    --workload-cache-dir "$WORKLOAD_CACHE_DIR" \
    --workload-cache-workers "$WORKLOAD_CACHE_WORKERS" \
    --capacity-quantile 0.96 --max-intervals "$MAX_INTERVALS" \
    --prepare-workload-cache-only
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
  export PYTHON_BIN RUN_SCRIPT WORKLOAD_PATH WORKLOAD_CACHE_DIR WORKLOAD_CACHE_WORKERS
  export HOURLY_CARBON_DIR AZURE_TRANSMISSION_DIR OUTPUT_ROOT LOG_DIR
  export BASELINE_SCHEDULING_MODE SOLVER MAX_INTERVALS RAMPING_TRACE_CYCLE_HOURS
  export DRY_RUN FORCE STOP_ON_ERROR
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
  log "Workers=$WORKERS failed=$failures dry_run=$DRY_RUN; 45 settings assigned; logs: $LOG_DIR"
  (( failures == 0 ))
}

case "${1:-}" in
  --help|-h)
    printf '%s\n' 'Usage: bash scripts/time_sensitivity/run_time_sensitivity_45_cases.sh' \
      'WORKERS=1 by default (serial); set WORKERS=N for parallel execution.' \
      'DRY_RUN=1 prints the 45-case plan; carbon inputs are still checked.' \
      'PREPARE_CACHE=0 skips cache preparation; use only with prepared caches.'
    ;;
  --worker) [[ $# == 1 ]] || die "Internal --worker takes no arguments."; main ;;
  "") dispatch ;;
  *) die "Unknown argument: $1" ;;
esac
