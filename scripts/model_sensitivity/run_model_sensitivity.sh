#!/usr/bin/env bash
set -Eeuo pipefail

# One-factor-at-a-time sensitivity design for the manuscript's representative
# 2030--CP--Base--GT setting. Each perturbed case runs LI, TM, SM, and JSM.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/model_sensitivity/run_model_sensitivity.sh"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/model_sensitivity/run_model_sensitivity.py}"
VALIDATOR="${VALIDATOR:-scripts/model_sensitivity/validate_results.py}"
WORKLOAD_PATH="${WORKLOAD_PATH:-dataset/cluster-trace-gpu-v2026}"
WORKLOAD_CACHE_DIR="${WORKLOAD_CACHE_DIR:-results/workload_profile_cache}"
WORKLOAD_CACHE_WORKERS="${WORKLOAD_CACHE_WORKERS:-8}"
HOURLY_CARBON_DIR="${HOURLY_CARBON_DIR:-dataset/EM-CPNDCNZ}"
AZURE_TRANSMISSION_DIR="${AZURE_TRANSMISSION_DIR:-dataset/Azure}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/Result-V3/Result-V3-nine-parameter-sensitivity-831}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/_worker_logs}"
WORKERS="${WORKERS:-1}"

YEAR=2030
SCENARIO="Base"
GRID_POLICY="CP"
MIGRATION_CONSTRAINT="global"
TRANSMISSION_POLICY="global_transfer_with_safeguards"
TRAINING_DELAY_HOURS=24
OFFLINE_INFERENCE_DELAY_HOURS=6
COMMIT_HOURS=24
CAPACITY_QUANTILE=0.96
MAX_RESOURCE_UTILIZATION=1.0
MAX_INTERVALS="${MAX_INTERVALS:-8760}"
RAMPING_TRACE_CYCLE_HOURS="${RAMPING_TRACE_CYCLE_HOURS:-4440}"
SOLVER="${SOLVER:-scipy}"

METHODS=(
  local_immediate_training_overflow_constrained
  local_training_time_shift_greedy_constrained
  annual_average_immediate_overflow_greedy_constrained
  linear_programming
)

# case | parameter | CLI option | value | level | neutral | interpretation
SENSITIVITY_CASES=(
  "gpu_idle_0p05|gpu_idle_power_fraction|--gpu-idle-power-fraction|0.05|low|0.10|GPU idle/full-load power ratio"
  "gpu_idle_0p15|gpu_idle_power_fraction|--gpu-idle-power-fraction|0.15|high|0.10|GPU idle/full-load power ratio"
  "training_gpu_0p8|training_gpu_load_scale|--training-gpu-load-scale|0.8|low|1.0|Training GPU load scale"
  "training_gpu_1p2|training_gpu_load_scale|--training-gpu-load-scale|1.2|high|1.0|Training GPU load scale"
  "offline_inference_gpu_0p8|offline_inference_gpu_load_scale|--offline-inference-gpu-load-scale|0.8|low|1.0|Direct offline-inference GPU load scale"
  "offline_inference_gpu_1p2|offline_inference_gpu_load_scale|--offline-inference-gpu-load-scale|1.2|high|1.0|Direct offline-inference GPU load scale"
  "flexible_gpu_share_0p8|flexible_gpu_share_scale|--flexible-gpu-share-scale|0.8|low|1.0|Training-plus-offline-inference share of GPU load"
  "flexible_gpu_share_1p2|flexible_gpu_share_scale|--flexible-gpu-share-scale|1.2|high|1.0|Training-plus-offline-inference share of GPU load"
  "gpu_power_share_0p40|gpu_power_share|--gpu-power-share|0.40|low|0.50|GPU share of full-load IT power"
  "gpu_power_share_0p60|gpu_power_share|--gpu-power-share|0.60|high|0.50|GPU share of full-load IT power"
  "hourly_carbon_variability_0p5|hourly_carbon_variability_scale|--hourly-carbon-variability-scale|0.5|low|1.0|Within-country hourly carbon-intensity variability"
  "hourly_carbon_variability_1p5|hourly_carbon_variability_scale|--hourly-carbon-variability-scale|1.5|high|1.0|Within-country hourly carbon-intensity variability"
  "gpu_power_curve_alpha_0|gpu_power_curve_alpha|--gpu-power-curve-alpha|0|linear|1.0|GPU utilization-power curve shape"
  "gpu_power_curve_alpha_3|gpu_power_curve_alpha|--gpu-power-curve-alpha|3|more_concave|1.0|GPU utilization-power curve shape"
  "spatial_carbon_contrast_0p5|spatial_carbon_contrast_scale|--spatial-carbon-contrast-scale|0.5|low|1.0|Between-country annual carbon-intensity contrast"
  "spatial_carbon_contrast_1p5|spatial_carbon_contrast_scale|--spatial-carbon-contrast-scale|1.5|high|1.0|Between-country annual carbon-intensity contrast"
  "usable_residual_capacity_0p50|usable_residual_capacity_fraction|--usable-residual-capacity-fraction|0.50|strong_restriction|1.0|Usable migration headroom above local-immediate load"
  "usable_residual_capacity_0p75|usable_residual_capacity_fraction|--usable-residual-capacity-fraction|0.75|moderate_restriction|1.0|Usable migration headroom above local-immediate load"
)

PREPARE_CACHE="${PREPARE_CACHE:-1}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
MANIFEST_ONLY="${MANIFEST_ONLY:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
WRITE_MANIFEST="${WRITE_MANIFEST:-1}"

DESIGN_COUNT=0
ASSIGNED_COUNT=0
RUN_COUNT=0
SKIP_COUNT=0
FAILED_COUNT=0
FAILED_RUN_IDS=()
PIDS=()

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

validate_environment() {
  [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] && (( WORKERS <= 18 )) || die "WORKERS must be between 1 and 18."
  [[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || die "SHARD_COUNT must be a positive integer."
  [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] || die "SHARD_INDEX must be a non-negative integer."
  (( SHARD_INDEX < SHARD_COUNT )) || die "SHARD_INDEX must be smaller than SHARD_COUNT."
  [[ -f "$RUN_SCRIPT" ]] || die "831 runner not found: $RUN_SCRIPT"
  [[ -f "$VALIDATOR" ]] || die "831 validator not found: $VALIDATOR"
  [[ "$RUN_SCRIPT" == *"run_model_sensitivity.py" ]] || die \
    "This design must use the independent 831 runner: $RUN_SCRIPT"
  [[ -d "$HOURLY_CARBON_DIR" ]] || die "Hourly carbon directory not found: $HOURLY_CARBON_DIR"
  [[ -d "$AZURE_TRANSMISSION_DIR" ]] || die "Azure policy directory not found: $AZURE_TRANSMISSION_DIR"

  local python_path
  python_path="$(command -v "$PYTHON_BIN" 2>/dev/null || true)"
  [[ -n "$python_path" ]] || die "Python executable not found: $PYTHON_BIN"
  PYTHON_BIN="$python_path"

  if [[ "$DRY_RUN" != "1" && "$MANIFEST_ONLY" != "1" ]]; then
    [[ -d "$WORKLOAD_PATH" ]] || die \
      "Alibaba GPU v2026 dataset not found: $WORKLOAD_PATH. Set WORKLOAD_PATH explicitly."
  fi
}

prepare_cache() {
  [[ "$PREPARE_CACHE" == "1" && "$DRY_RUN" != "1" && "$MANIFEST_ONLY" != "1" ]] || return 0
  log "Preparing the shared 828-compatible workload caches once."
  env \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "$PYTHON_BIN" "$RUN_SCRIPT" \
    --workload-path "$WORKLOAD_PATH" \
    --workload-cache-dir "$WORKLOAD_CACHE_DIR" \
    --workload-cache-workers "$WORKLOAD_CACHE_WORKERS" \
    --capacity-quantile "$CAPACITY_QUANTILE" \
    --max-intervals "$MAX_INTERVALS" \
    --prepare-workload-cache-only
  log "Shared workload caches are ready."
}

write_manifest() {
  [[ "$WRITE_MANIFEST" == "1" ]] || return 0
  if [[ "$MANIFEST_ONLY" != "1" && "$SHARD_INDEX" != "0" ]]; then
    return 0
  fi
  mkdir -p "$OUTPUT_ROOT"
  local manifest="$OUTPUT_ROOT/Nine_Parameter_Sensitivity_Design_18cases.tsv"
  local temporary="$manifest.tmp.$$"
  printf 'run_id\tyear\tscenario\tgrid_policy\tmigration_constraint\ttransmission_policy\tparameter\tlevel\tsetting\tneutral\tinterpretation\tneutral_source\n' >"$temporary"
  local definition case_key parameter_key option setting level neutral interpretation run_id
  for definition in "${SENSITIVITY_CASES[@]}"; do
    IFS='|' read -r case_key parameter_key option setting level neutral interpretation <<<"$definition"
    run_id="${YEAR}_base_cp_global__${case_key}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$run_id" "$YEAR" "$SCENARIO" "$GRID_POLICY" "$MIGRATION_CONSTRAINT" \
      "$TRANSMISSION_POLICY" "$parameter_key" "$level" "$setting" "$neutral" \
      "$interpretation" "828_main_experiment__2030_CP_Base_GT" >>"$temporary"
  done
  mv "$temporary" "$manifest"
}

has_complete_output() {
  local output_dir="$1"
  local parameter_key="$2"
  local setting="$3"
  [[ -d "$output_dir" ]] || return 1
  "$PYTHON_BIN" "$VALIDATOR" "$output_dir" \
    --expect "${parameter_key}=${setting}" --quiet >/dev/null 2>&1
}

run_one() {
  local design_index="$1"
  local definition="$2"
  local case_key parameter_key sensitivity_option setting level neutral interpretation
  IFS='|' read -r case_key parameter_key sensitivity_option setting level neutral interpretation <<<"$definition"

  DESIGN_COUNT=$((DESIGN_COUNT + 1))
  (( design_index % SHARD_COUNT == SHARD_INDEX )) || return 0
  ASSIGNED_COUNT=$((ASSIGNED_COUNT + 1))

  local run_id="${YEAR}_base_cp_global__${case_key}"
  local output_dir="$OUTPUT_ROOT/$run_id"
  if [[ "$FORCE" != "1" && "$DRY_RUN" != "1" ]] \
    && has_complete_output "$output_dir" "$parameter_key" "$setting"; then
    log "SKIP complete: $run_id"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return 0
  fi

  local -a command=(
    "$PYTHON_BIN" "$RUN_SCRIPT"
    --policy "$GRID_POLICY"
    --scenarios "$SCENARIO"
    --years 1
    --year-start "$YEAR"
    --constraint "$MIGRATION_CONSTRAINT"
    --delay-hours "$TRAINING_DELAY_HOURS"
    --offline-inference-delay-hours "$OFFLINE_INFERENCE_DELAY_HOURS"
    --solve-mode windowed
    --commit-hours "$COMMIT_HOURS"
    --baseline-scheduling-mode matched_window
    --solver "$SOLVER"
    --capacity-quantile "$CAPACITY_QUANTILE"
    --max-resource-utilization "$MAX_RESOURCE_UTILIZATION"
    --max-intervals "$MAX_INTERVALS"
    --workload-path "$WORKLOAD_PATH"
    --workload-cache-dir "$WORKLOAD_CACHE_DIR"
    --workload-cache-workers "$WORKLOAD_CACHE_WORKERS"
    --hourly-carbon-dir "$HOURLY_CARBON_DIR"
    --hourly-carbon-scope direct
    --strict-hourly-carbon
    --azure-transmission-policy "$TRANSMISSION_POLICY"
    --azure-transmission-dir "$AZURE_TRANSMISSION_DIR"
    --methods "${METHODS[@]}"
    --ramping-trace-cycle-hours "$RAMPING_TRACE_CYCLE_HOURS"
    "$sensitivity_option" "$setting"
    --output-dir "$output_dir"
  )

  RUN_COUNT=$((RUN_COUNT + 1))
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY-RUN] '
    printf '%q ' "${command[@]}"
    printf '\n'
    return 0
  fi

  mkdir -p "$output_dir"
  {
    printf 'run_id\tparameter\tlevel\tsetting\tneutral\tinterpretation\n'
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$run_id" "$parameter_key" "$level" "$setting" "$neutral" "$interpretation"
  } >"$output_dir/sensitivity_case.tsv"
  {
    printf '#!/usr/bin/env bash\n'
    printf '%q ' "${command[@]}"
    printf '\n'
  } >"$output_dir/command.sh"

  log "RUN $run_id parameter=$parameter_key level=$level setting=$setting"
  if env \
    OMP_NUM_THREADS="${NUMERIC_THREADS:-1}" \
    OPENBLAS_NUM_THREADS="${NUMERIC_THREADS:-1}" \
    MKL_NUM_THREADS="${NUMERIC_THREADS:-1}" \
    NUMEXPR_NUM_THREADS="${NUMERIC_THREADS:-1}" \
    "${command[@]}" 2>&1 | tee "$output_dir/run.log"; then
    if has_complete_output "$output_dir" "$parameter_key" "$setting"; then
      log "DONE $run_id"
      return 0
    fi
    log "FAILED output validation: $run_id"
  else
    log "FAILED command: $run_id"
  fi
  FAILED_COUNT=$((FAILED_COUNT + 1))
  FAILED_RUN_IDS+=("$run_id")
  if [[ "$STOP_ON_ERROR" == "1" ]]; then
    return 1
  fi
  return 0
}

main() {
  validate_environment
  write_manifest
  if [[ "$MANIFEST_ONLY" == "1" ]]; then
    log "Design manifest written; no experiments launched."
    return 0
  fi
  prepare_cache

  local design_index=0
  local definition
  for definition in "${SENSITIVITY_CASES[@]}"; do
    run_one "$design_index" "$definition"
    design_index=$((design_index + 1))
  done

  log "Design=$DESIGN_COUNT assigned=$ASSIGNED_COUNT run=$RUN_COUNT skipped=$SKIP_COUNT failed=$FAILED_COUNT"
  if (( FAILED_COUNT > 0 )); then
    printf 'Failed runs: %s\n' "${FAILED_RUN_IDS[*]}" >&2
    return 1
  fi
}

usage() {
  cat <<'EOF'
Usage: bash scripts/model_sensitivity/run_model_sensitivity.sh

Runs the original 18 parameter settings at 2030--CP--Base--GT.
Current manuscript: 16 settings; the two GPU curve settings remain for parity.
  WORKERS=1         Serial by default; set WORKERS=N for parallel cases.
  DRY_RUN=1         Print commands without solving (carbon directory required).
  PREPARE_CACHE=0   Skip shared-cache preparation.
  OUTPUT_ROOT=...   Override the existing default result directory.
  WORKLOAD_PATH=... Alibaba GPU v2026 input directory.
  HOURLY_CARBON_DIR=...  CP carbon inputs (default: dataset/EM-CPNDCNZ).
  MANIFEST_ONLY=1   Write the design table without running experiments.
  FORCE=1          Recompute existing cases.
  STOP_ON_ERROR=1  Stop a worker on its first failed case.

The default-parameter reference remains a separate run, as in the old batch.
EOF
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
  if [[ "$WORKERS" == 1 || "$MANIFEST_ONLY" == 1 ]]; then
    main
    return
  fi
  [[ "$SHARD_COUNT" == 1 && "$SHARD_INDEX" == 0 ]] || die \
    "Use WORKERS for automatic parallel execution, or SHARD_COUNT/SHARD_INDEX with WORKERS=1."
  validate_environment
  write_manifest
  prepare_cache
  export PYTHON_BIN RUN_SCRIPT VALIDATOR WORKLOAD_PATH WORKLOAD_CACHE_DIR WORKLOAD_CACHE_WORKERS
  export HOURLY_CARBON_DIR AZURE_TRANSMISSION_DIR OUTPUT_ROOT LOG_DIR
  export MAX_INTERVALS RAMPING_TRACE_CYCLE_HOURS SOLVER DRY_RUN FORCE STOP_ON_ERROR
  local worker index failures=0 log_path worker_count="$WORKERS"
  mkdir -p "$LOG_DIR"
  trap 'stop_workers; exit 130' INT TERM
  for ((worker=0; worker<WORKERS; worker++)); do
    printf -v log_path '%s/worker%02d.log' "$LOG_DIR" "$((worker + 1))"
    WORKERS=1 SHARD_COUNT="$worker_count" SHARD_INDEX="$worker" \
      PREPARE_CACHE=0 WRITE_MANIFEST=0 bash "$SCRIPT_PATH" --worker >"$log_path" 2>&1 &
    PIDS+=("$!")
  done
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$index]}"; then failures=$((failures + 1)); fi
    PIDS[$index]=""
  done
  trap - INT TERM
  log "Workers=$WORKERS failed=$failures dry_run=$DRY_RUN; 18 settings assigned; logs: $LOG_DIR"
  (( failures == 0 ))
}

case "${1:-}" in
  --help|-h) usage ;;
  --worker) [[ $# == 1 ]] || die "Internal --worker takes no arguments."; main ;;
  "") dispatch ;;
  *) usage >&2; exit 2 ;;
esac
