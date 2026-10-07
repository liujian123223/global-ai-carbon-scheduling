import argparse
import importlib.util
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core import workload_training_migration_methods as migration_methods


def _load_optimization_module():
    module_path = ROOT_DIR / "core" / "workload_scheduling_engine.py"
    spec = importlib.util.spec_from_file_location(
        "workload_training_migration_optimization_8182",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load optimization module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


opt = _load_optimization_module()


DEFAULT_RAMPING_TRACE_CYCLE_HOURS = 4440
DEFAULT_OFFLINE_INFERENCE_DELAY_HOURS = 6
DEFAULT_WORKLOAD_CACHE_WORKERS = 8
DIRECT_OFFLINE_CACHE_SUBDIR = "827_direct_offline_labels_v1"
FLEXIBLE_TASK_TYPES = ("training", "offline_inference")
BASELINE_SCHEDULING_MODES = ("matched_window", "legacy_sequential")
DEFAULT_BASELINE_SCHEDULING_MODE = "matched_window"
RAMPING_CHANGE_METRICS = (
    "mean_power_mw",
    "peak_power_mw",
    "p99_power_mw",
    "peak_to_mean_ratio",
    "max_abs_ramp_mw_per_h",
    "mean_abs_ramp_mw_per_h",
    "p95_abs_ramp_mw_per_h",
    "p99_abs_ramp_mw_per_h",
    "total_variation_mw",
    "max_abs_ramp_pct_of_mean_per_h",
    "p99_abs_ramp_pct_of_mean_per_h",
    "max_abs_ramp_pct_of_capacity_per_h",
    "p99_abs_ramp_pct_of_capacity_per_h",
    "max_abs_ramp_excluding_trace_seams_mw_per_h",
    "p95_abs_ramp_excluding_trace_seams_mw_per_h",
    "p99_abs_ramp_excluding_trace_seams_mw_per_h",
    "p99_abs_ramp_excluding_trace_seams_pct_of_capacity_per_h",
    "p99_abs_ramp_at_commit_boundaries_mw_per_h",
    "p99_abs_ramp_outside_commit_boundaries_mw_per_h",
)


_827_CACHE_WORKER_CONFIG = None


def _initialize_827_cache_worker(config: dict) -> None:
    global _827_CACHE_WORKER_CONFIG
    _827_CACHE_WORKER_CONFIG = config


def _normalize_827_task_label(value, aliases: dict) -> str:
    if pd.isna(value):
        return "unclassified"
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return aliases.get(label, "unclassified")


def _process_827_pod_partition(partition):
    """Read and fully aggregate one pod-hour partition in a worker process."""
    config = _827_CACHE_WORKER_CONFIG
    if config is None:
        raise RuntimeError("827 cache worker was not initialized.")
    day, hour, path = partition
    n_types = len(config["task_types"])
    n_resources = len(config["resources"])
    interval_load = np.zeros((n_types, n_resources), dtype=np.float64)
    pod_hour_counts = np.zeros(n_types, dtype=np.int64)
    source_used_gpu_hours = np.zeros(n_types, dtype=np.float64)
    source_gpu_memory_gib_hours = np.zeros(n_types, dtype=np.float64)
    paper_validation_used_gpu_hours = np.zeros(n_types, dtype=np.float64)

    df = pd.read_parquet(
        path,
        columns=list(config["pod_columns"]),
        use_threads=False,
    )
    if df.empty:
        return (
            day * 24 + hour,
            interval_load,
            pod_hour_counts,
            source_used_gpu_hours,
            source_gpu_memory_gib_hours,
            paper_validation_used_gpu_hours,
        )

    aliases = config["task_aliases"]
    id_by_type = config["id_by_type"]
    mapped_task_type_ids = (
        df["job_type_public"]
        .map(lambda value: _normalize_827_task_label(value, aliases))
        .map(id_by_type)
        .to_numpy(dtype=np.int64)
    )
    state = df["state_public"].fillna("").astype(str)

    def numeric(column: str) -> np.ndarray:
        return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)

    gpu_request = numeric("gpu_request")
    gpu_mem_request = numeric("gpu_mem_request")
    used_gpu_hours = numeric("used_gpu_hours")
    gpu_sm = numeric("avg_gpu_sm_util")
    gpu_mem = numeric("avg_gpu_mem_gib")
    cpu_request = numeric("cpu_request_cores")
    cpu_request_util = numeric("avg_cpu_request_util")
    memory_util = numeric("avg_memory_util")

    if config["paper_start_day"] <= day <= config["paper_end_day"]:
        paper_valid = (
            np.isfinite(gpu_mem_request)
            & (gpu_mem_request > 0)
            & np.isfinite(used_gpu_hours)
            & (mapped_task_type_ids != config["unclassified_id"])
        )
        if np.any(paper_valid):
            paper_validation_used_gpu_hours += np.bincount(
                mapped_task_type_ids[paper_valid],
                weights=np.maximum(used_gpu_hours[paper_valid], 0.0),
                minlength=n_types,
            )

    gpu_request = np.where(
        np.isfinite(gpu_request), np.maximum(gpu_request, 0.0), 0.0
    )
    cpu_request = np.where(
        np.isfinite(cpu_request), np.maximum(cpu_request, 0.0), 0.0
    )
    gpu_sm = np.where(np.isfinite(gpu_sm), np.maximum(gpu_sm, 0.0), 0.0)
    gpu_mem = np.where(np.isfinite(gpu_mem), np.maximum(gpu_mem, 0.0), 0.0)
    cpu_request_util = np.where(
        np.isfinite(cpu_request_util), np.maximum(cpu_request_util, 0.0), 0.0
    )
    memory_util = np.where(
        np.isfinite(memory_util), np.maximum(memory_util, 0.0), 0.0
    )
    used_gpu_hours = np.where(
        np.isfinite(used_gpu_hours), np.maximum(used_gpu_hours, 0.0), 0.0
    )

    active = state.isin(config["active_states"]).to_numpy()
    active |= used_gpu_hours > 0
    active |= gpu_sm > 0
    active |= cpu_request_util > 0
    active &= ~state.eq("Standby").to_numpy()
    if not config["include_zero_gpu_pods"]:
        active &= gpu_request > 0
    if not np.any(active):
        return (
            day * 24 + hour,
            interval_load,
            pod_hour_counts,
            source_used_gpu_hours,
            source_gpu_memory_gib_hours,
            paper_validation_used_gpu_hours,
        )

    mapped_task_type_ids = mapped_task_type_ids[active]
    gpu_request = gpu_request[active]
    used_gpu_hours = used_gpu_hours[active]
    gpu_sm = gpu_sm[active]
    gpu_mem = gpu_mem[active]
    cpu_request = cpu_request[active]
    cpu_request_util = cpu_request_util[active]
    memory_util = memory_util[active]

    cpu_used_cores = cpu_request * cpu_request_util
    gpu_sm_equivalent = gpu_sm / 100.0
    if config["cap_gpu_sm_by_request"]:
        gpu_sm_equivalent = np.minimum(
            gpu_sm_equivalent,
            np.where(gpu_request > 0, gpu_request, gpu_sm_equivalent),
        )
    resource_values = (
        cpu_used_cores,
        gpu_sm_equivalent,
        cpu_request * memory_util,
        np.zeros_like(cpu_used_cores),
    )

    pod_hour_counts += np.bincount(mapped_task_type_ids, minlength=n_types)
    source_used_gpu_hours += np.bincount(
        mapped_task_type_ids,
        weights=used_gpu_hours,
        minlength=n_types,
    )
    source_gpu_memory_gib_hours += np.bincount(
        mapped_task_type_ids,
        weights=gpu_mem,
        minlength=n_types,
    )
    for resource_id, values in enumerate(resource_values):
        interval_load[:, resource_id] += np.bincount(
            mapped_task_type_ids,
            weights=values,
            minlength=n_types,
        )
    return (
        day * 24 + hour,
        interval_load,
        pod_hour_counts,
        source_used_gpu_hours,
        source_gpu_memory_gib_hours,
        paper_validation_used_gpu_hours,
    )


def _process_827_server_partition(partition):
    """Read and aggregate one server-hour partition in a worker process."""
    config = _827_CACHE_WORKER_CONFIG
    if config is None:
        raise RuntimeError("827 cache worker was not initialized.")
    day, hour, path = partition
    df = pd.read_parquet(
        path,
        columns=list(config["server_columns"]),
        use_threads=False,
    )
    gpu_count = (
        pd.to_numeric(df["gpu_count"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    cpu_cores = (
        pd.to_numeric(df["cpu_capacity_cores"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    return day * 24 + hour, float(cpu_cores.sum()), float(gpu_count.sum())


def _parallel_cache_progress_interval(total: int) -> int:
    return max(1, (total + 99) // 100)


def _build_workload_profile_827(
    workload_profile_path: Path,
    server_profile_path,
    network_profile_path,
    capacity_quantile: float,
    max_intervals,
    workload_cache_dir,
    refresh_workload_cache: bool,
    workload_cache_workers: int,
    verbose: bool,
):
    """827-only multiprocessing equivalent of the unchanged 8.182 builder."""
    if workload_cache_workers <= 1:
        return opt.build_workload_profile(
            workload_profile_path=workload_profile_path,
            server_profile_path=server_profile_path,
            network_profile_path=network_profile_path,
            capacity_quantile=capacity_quantile,
            trace_config=None,
            max_intervals=max_intervals,
            workload_cache_dir=workload_cache_dir,
            refresh_workload_cache=refresh_workload_cache,
            verbose=verbose,
        )
    if not (0 < capacity_quantile <= 1):
        raise ValueError("capacity_quantile must be in (0, 1].")
    if max_intervals is not None and max_intervals <= 0:
        raise ValueError("max_intervals must be positive when provided.")

    trace_config = opt.Alibaba2026TraceConfig()
    if network_profile_path is not None:
        opt._print_progress(
            verbose,
            "Ignoring network_profile_path because this carbon-only variant does "
            "not read Alibaba 2026 traffic data.",
        )
    pod_root = opt._resolve_alibaba_2026_table_root(
        workload_profile_path,
        opt.ALIBABA_2026_POD_TABLE,
    )
    server_root = (
        Path(server_profile_path)
        if server_profile_path is not None
        else opt._resolve_alibaba_2026_table_root(
            workload_profile_path,
            opt.ALIBABA_2026_SERVER_TABLE,
        )
    )
    profile_cache_key = {
        "version": opt.WORKLOAD_PROFILE_CACHE_VERSION,
        "kind": "workload_profile",
        "workload_root": opt._path_cache_value(workload_profile_path),
        "pod_root": opt._path_cache_value(pod_root),
        "server_root": opt._path_cache_value(server_root),
        "capacity_quantile": float(capacity_quantile),
        "max_intervals": None if max_intervals is None else int(max_intervals),
        "trace_config": opt._trace_config_cache_dict(trace_config),
        "task_types": list(opt.TASK_TYPES),
        "resources": list(opt.RESOURCES),
    }
    profile_cache_path = opt._cache_file_path(
        workload_cache_dir,
        "workload_profile",
        profile_cache_key,
    )
    if not refresh_workload_cache:
        cached = opt._load_pickle_cache(
            profile_cache_path,
            profile_cache_key,
            verbose=verbose,
            label="workload profile",
        )
        if cached is not None:
            return opt._profile_from_cache_payload(cached)

    pod_files = opt._alibaba_2026_partition_files(pod_root)
    if not pod_files:
        raise FileNotFoundError(
            f"No Alibaba GPU v2026 pod-hour parquet partitions found under {pod_root}. "
            "Expected day=<day>/hour=<hour>/*.parquet."
        )
    max_trace_pos = max(day * 24 + hour for day, hour, _ in pod_files) + 1
    n_intervals = (
        min(max_trace_pos, max_intervals)
        if max_intervals is not None
        else max_trace_pos
    )
    pod_files_to_process = [
        (day, hour, path)
        for day, hour, path in pod_files
        if day * 24 + hour < n_intervals
    ]
    opt._print_progress(
        verbose,
        f"Found {len(pod_files_to_process)} pod partitions under {pod_root}; "
        f"building {n_intervals} hourly intervals with "
        f"{workload_cache_workers} worker processes.",
    )

    interval_hours = 1.0
    interval_index = pd.date_range(
        start=f"{trace_config.trace_anchor_year}-01-01",
        periods=n_intervals,
        freq="h",
        tz="UTC",
    )
    load = np.zeros(
        (len(opt.TASK_TYPES), len(opt.RESOURCES), n_intervals),
        dtype=np.float64,
    )
    pod_hour_counts = np.zeros(len(opt.TASK_TYPES), dtype=np.int64)
    source_used_gpu_hours = np.zeros(len(opt.TASK_TYPES), dtype=np.float64)
    source_gpu_memory_gib_hours = np.zeros(len(opt.TASK_TYPES), dtype=np.float64)
    paper_validation_used_gpu_hours = np.zeros(
        len(opt.TASK_TYPES),
        dtype=np.float64,
    )
    worker_config = {
        "task_types": tuple(opt.TASK_TYPES),
        "resources": tuple(opt.RESOURCES),
        "task_aliases": dict(opt._PUBLIC_TASK_TYPE_ALIASES),
        "id_by_type": {
            task_type: index for index, task_type in enumerate(opt.TASK_TYPES)
        },
        "unclassified_id": opt.TASK_TYPES.index("unclassified"),
        "active_states": tuple(trace_config.active_states),
        "include_zero_gpu_pods": bool(trace_config.include_zero_gpu_pods),
        "cap_gpu_sm_by_request": bool(trace_config.cap_gpu_sm_by_request),
        "paper_start_day": int(opt.ALIBABA_2026_JOB_TYPE_START_DAY),
        "paper_end_day": int(opt.ALIBABA_2026_JOB_TYPE_END_DAY),
        "pod_columns": tuple(opt.POD_PROFILE_COLUMNS),
        "server_columns": tuple(opt.SERVER_PROFILE_COLUMNS),
    }
    progress_interval = _parallel_cache_progress_interval(len(pod_files_to_process))
    process_context = multiprocessing.get_context("spawn")
    server_capacity = None
    with ProcessPoolExecutor(
        max_workers=workload_cache_workers,
        mp_context=process_context,
        initializer=_initialize_827_cache_worker,
        initargs=(worker_config,),
    ) as executor:
        pod_results = executor.map(
            _process_827_pod_partition,
            pod_files_to_process,
            chunksize=1,
        )
        for file_index, result in enumerate(pod_results, start=1):
            if (
                file_index == 1
                or file_index == len(pod_files_to_process)
                or file_index % progress_interval == 0
            ):
                opt._print_progress(
                    verbose,
                    f"Processed pod partitions: {file_index}/"
                    f"{len(pod_files_to_process)} "
                    f"({file_index / len(pod_files_to_process):.0%})",
                )
            (
                pos,
                interval_load,
                interval_counts,
                interval_used_gpu,
                interval_gpu_memory,
                interval_paper_validation,
            ) = result
            load[:, :, pos] += interval_load
            pod_hour_counts += interval_counts
            source_used_gpu_hours += interval_used_gpu
            source_gpu_memory_gib_hours += interval_gpu_memory
            paper_validation_used_gpu_hours += interval_paper_validation

        if server_root.exists():
            opt._print_progress(verbose, f"Reading server inventory from {server_root}.")
            server_files = [
                (day, hour, path)
                for day, hour, path in opt._alibaba_2026_partition_files(server_root)
                if day * 24 + hour < n_intervals
            ]
            if server_files:
                capacity = np.full((2, n_intervals), np.nan, dtype=np.float64)
                server_progress = _parallel_cache_progress_interval(len(server_files))
                server_results = executor.map(
                    _process_827_server_partition,
                    server_files,
                    chunksize=1,
                )
                for file_index, (pos, cpu_cores, gpu_count) in enumerate(
                    server_results,
                    start=1,
                ):
                    if (
                        file_index == 1
                        or file_index == len(server_files)
                        or file_index % server_progress == 0
                    ):
                        opt._print_progress(
                            verbose,
                            f"Processed server partitions: {file_index}/"
                            f"{len(server_files)} "
                            f"({file_index / len(server_files):.0%})",
                        )
                    capacity[0, pos] = cpu_cores
                    capacity[1, pos] = gpu_count
                candidate = np.full(2, np.nan, dtype=np.float64)
                for resource_id in range(2):
                    valid = capacity[
                        resource_id,
                        np.isfinite(capacity[resource_id])
                        & (capacity[resource_id] > 0),
                    ]
                    if valid.size:
                        candidate[resource_id] = float(
                            np.quantile(valid, capacity_quantile)
                        )
                if np.all(np.isfinite(candidate)):
                    server_capacity = candidate
        else:
            opt._print_progress(
                verbose,
                f"Server inventory not found under {server_root}; "
                "using pod-load capacity estimates.",
            )

    total_load = load.sum(axis=0)
    trace_capacity = np.ones(len(opt.RESOURCES), dtype=np.float64)
    capacity_source = ["pod_load_quantile"] * len(opt.RESOURCES)
    if server_capacity is not None:
        trace_capacity[opt.RESOURCES.index("cpu")] = server_capacity[0]
        trace_capacity[opt.RESOURCES.index("gpu")] = server_capacity[1]
        capacity_source[opt.RESOURCES.index("cpu")] = "server_inventory_quantile"
        capacity_source[opt.RESOURCES.index("gpu")] = "server_inventory_quantile"
    for resource in ("cpu", "gpu", "memory"):
        resource_id = opt.RESOURCES.index(resource)
        if capacity_source[resource_id] == "server_inventory_quantile":
            continue
        values = total_load[resource_id]
        positive = values[np.isfinite(values) & (values > 0)]
        trace_capacity[resource_id] = (
            float(np.quantile(positive, capacity_quantile))
            if positive.size
            else 1.0
        )
    trace_capacity[opt.RESOURCES.index("storage")] = 1.0
    capacity_source[opt.RESOURCES.index("storage")] = "not_observed_idle_only"
    capacity_source[opt.RESOURCES.index("memory")] = (
        "cpu_weighted_memory_util_proxy"
    )
    trace_capacity = np.maximum(trace_capacity, 1e-12)

    resource_hours = load.sum(axis=2) * interval_hours
    annualization = 8760.0 / (n_intervals * interval_hours)
    total_pod_hour_rows = int(pod_hour_counts.sum())
    total_source_used_gpu_hours = float(source_used_gpu_hours.sum())
    classified_source_used_gpu_hours = float(
        source_used_gpu_hours[: opt.TASK_TYPES.index("unclassified")].sum()
    )
    paper_validation_total = float(paper_validation_used_gpu_hours.sum())
    summary_records = []
    for task_type_id, task_type in enumerate(opt.TASK_TYPES):
        summary_records.append(
            {
                "task_type": task_type,
                "is_classified": task_type != "unclassified",
                "pod_hour_rows": int(pod_hour_counts[task_type_id]),
                "pod_hour_rows_share_all": (
                    float(pod_hour_counts[task_type_id]) / total_pod_hour_rows
                    if total_pod_hour_rows > 0
                    else 0.0
                ),
                "source_used_gpu_hours": float(
                    source_used_gpu_hours[task_type_id]
                ),
                "source_used_gpu_hours_share_all": (
                    float(source_used_gpu_hours[task_type_id])
                    / total_source_used_gpu_hours
                    if total_source_used_gpu_hours > 0
                    else 0.0
                ),
                "source_used_gpu_hours_share_classified": (
                    float(source_used_gpu_hours[task_type_id])
                    / classified_source_used_gpu_hours
                    if task_type != "unclassified"
                    and classified_source_used_gpu_hours > 0
                    else np.nan
                ),
                "paper_validation_used_gpu_hours": float(
                    paper_validation_used_gpu_hours[task_type_id]
                ),
                "paper_validation_used_gpu_hours_share_classified": (
                    float(paper_validation_used_gpu_hours[task_type_id])
                    / paper_validation_total
                    if task_type != "unclassified" and paper_validation_total > 0
                    else np.nan
                ),
                "source_gpu_memory_gib_hours": float(
                    source_gpu_memory_gib_hours[task_type_id]
                ),
                "cpu_core_hours": float(
                    resource_hours[task_type_id, opt.RESOURCES.index("cpu")]
                ),
                "gpu_sm_equivalent_hours": float(
                    resource_hours[task_type_id, opt.RESOURCES.index("gpu")]
                ),
                "memory_proxy_hours": float(
                    resource_hours[task_type_id, opt.RESOURCES.index("memory")]
                ),
                "annualized_cpu_core_hours": float(
                    resource_hours[task_type_id, opt.RESOURCES.index("cpu")]
                    * annualization
                ),
                "annualized_gpu_sm_equivalent_hours": float(
                    resource_hours[task_type_id, opt.RESOURCES.index("gpu")]
                    * annualization
                ),
            }
        )
    profile = opt.WorkloadProfile(
        interval_index=interval_index,
        interval_hours=interval_hours,
        load=load,
        trace_capacity=trace_capacity,
        task_counts=pod_hour_counts,
        task_type_summary=pd.DataFrame(summary_records),
        relative_time=True,
        trace_capacity_source=tuple(capacity_source),
    )
    opt._write_pickle_cache(
        profile_cache_path,
        profile_cache_key,
        opt._profile_cache_payload(profile),
        verbose=verbose,
        label="workload profile",
    )
    return profile


def _build_direct_offline_inference_profile(
    workload_profile_path: Path,
    server_profile_path,
    network_profile_path,
    capacity_quantile: float,
    max_intervals,
    workload_cache_dir,
    refresh_workload_cache: bool,
    workload_cache_workers: int,
    verbose: bool,
):
    """Read only raw ``offline_inference`` rows into the inference slice."""
    original_aliases = opt._PUBLIC_TASK_TYPE_ALIASES
    direct_offline_aliases = dict(original_aliases)
    direct_offline_aliases.update(
        {
            "inference": "unclassified",
            "online_inference": "unclassified",
            "offline_inference": "inference",
        }
    )
    direct_cache_dir = (
        None
        if workload_cache_dir is None
        else Path(workload_cache_dir) / DIRECT_OFFLINE_CACHE_SUBDIR
    )
    try:
        opt._PUBLIC_TASK_TYPE_ALIASES = direct_offline_aliases
        return _build_workload_profile_827(
            workload_profile_path=workload_profile_path,
            server_profile_path=server_profile_path,
            network_profile_path=network_profile_path,
            capacity_quantile=capacity_quantile,
            max_intervals=max_intervals,
            workload_cache_dir=direct_cache_dir,
            refresh_workload_cache=refresh_workload_cache,
            workload_cache_workers=workload_cache_workers,
            verbose=verbose,
        )
    finally:
        opt._PUBLIC_TASK_TYPE_ALIASES = original_aliases


def _validate_direct_offline_profile(profile, direct_offline_profile) -> None:
    if (
        profile.n_intervals != direct_offline_profile.n_intervals
        or profile.interval_hours != direct_offline_profile.interval_hours
        or profile.relative_time != direct_offline_profile.relative_time
        or not profile.interval_index.equals(direct_offline_profile.interval_index)
    ):
        raise ValueError(
            "Direct offline-inference profile is not aligned with the main profile."
        )
    if not np.allclose(
        profile.trace_capacity,
        direct_offline_profile.trace_capacity,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Direct offline-inference profile has inconsistent trace capacity.")
    if not np.allclose(
        profile.load.sum(axis=0),
        direct_offline_profile.load.sum(axis=0),
        rtol=1e-10,
        atol=1e-8,
    ):
        raise ValueError(
            "Direct offline-inference classification changed total resource load."
        )


def _prepare_workload_profiles(
    workload_profile_path: Path,
    server_profile_path,
    network_profile_path,
    capacity_quantile: float,
    max_intervals,
    workload_cache_dir,
    refresh_workload_cache: bool,
    workload_cache_workers: int,
    verbose: bool,
):
    opt._print_progress(verbose, "Building workload profile from trace.")
    profile = _build_workload_profile_827(
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        network_profile_path=network_profile_path,
        capacity_quantile=capacity_quantile,
        max_intervals=max_intervals,
        workload_cache_dir=workload_cache_dir,
        refresh_workload_cache=refresh_workload_cache,
        workload_cache_workers=workload_cache_workers,
        verbose=verbose,
    )
    # Keep the original four-class profile for backward-compatible summaries.
    opt._print_progress(
        verbose,
        f"Workload profile ready: {profile.n_intervals} intervals, "
        f"{profile.interval_hours:.2f} hours per interval, "
        f"annualization factor {profile.annualization_factor_8760:.3f}.",
    )
    opt._print_progress(
        verbose,
        "Building direct offline-inference profile from job_type_public labels.",
    )
    direct_offline_profile = _build_direct_offline_inference_profile(
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        network_profile_path=network_profile_path,
        capacity_quantile=capacity_quantile,
        max_intervals=max_intervals,
        workload_cache_dir=workload_cache_dir,
        refresh_workload_cache=refresh_workload_cache,
        workload_cache_workers=workload_cache_workers,
        verbose=verbose,
    )
    _validate_direct_offline_profile(profile, direct_offline_profile)
    opt._print_progress(
        verbose,
        "Direct offline-inference profile ready; online inference remains fixed.",
    )
    return profile, direct_offline_profile


def _percentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan
    return float(np.percentile(values, percentile))


def _relative_change_pct(optimized: float, baseline: float) -> float:
    if not np.isfinite(baseline) or not np.isfinite(optimized) or abs(baseline) <= 1e-12:
        return np.nan
    return float((optimized - baseline) / baseline * 100.0)


def _ramping_metrics(
    power_mw: np.ndarray,
    facility_power_capacity_mw: float,
    trace_cycle_hours: int,
    commit_hours: int,
) -> dict:
    power = np.asarray(power_mw, dtype=np.float64).reshape(-1)
    if power.size == 0 or not np.all(np.isfinite(power)):
        raise ValueError("Hourly facility power must be finite and non-empty.")

    mean_power = float(power.mean())
    peak_power = float(power.max())
    p99_power = _percentile(power, 99.0)
    ramp = np.diff(power)
    abs_ramp = np.abs(ramp)
    transition_hours = np.arange(1, power.size, dtype=np.int64)

    trace_seam_mask = np.zeros(ramp.size, dtype=bool)
    if trace_cycle_hours > 0:
        trace_seam_mask = transition_hours % int(trace_cycle_hours) == 0

    commit_boundary_mask = np.zeros(ramp.size, dtype=bool)
    if commit_hours > 0:
        commit_boundary_mask = transition_hours % int(commit_hours) == 0

    non_seam_abs_ramp = abs_ramp[~trace_seam_mask]
    commit_abs_ramp = abs_ramp[commit_boundary_mask]
    non_commit_abs_ramp = abs_ramp[~commit_boundary_mask]

    max_abs_ramp = float(abs_ramp.max()) if abs_ramp.size else np.nan
    p99_abs_ramp = _percentile(abs_ramp, 99.0)
    capacity = float(facility_power_capacity_mw)

    return {
        "mean_power_mw": mean_power,
        "peak_power_mw": peak_power,
        "p99_power_mw": p99_power,
        "peak_to_mean_ratio": peak_power / mean_power if mean_power > 0 else np.nan,
        "max_ramp_up_mw_per_h": max(float(ramp.max()), 0.0) if ramp.size else np.nan,
        "max_ramp_down_mw_per_h": max(float(-ramp.min()), 0.0) if ramp.size else np.nan,
        "max_abs_ramp_mw_per_h": max_abs_ramp,
        "mean_abs_ramp_mw_per_h": float(abs_ramp.mean()) if abs_ramp.size else np.nan,
        "p95_abs_ramp_mw_per_h": _percentile(abs_ramp, 95.0),
        "p99_abs_ramp_mw_per_h": p99_abs_ramp,
        "total_variation_mw": float(abs_ramp.sum()),
        "max_abs_ramp_pct_of_mean_per_h": (
            max_abs_ramp / mean_power * 100.0 if mean_power > 0 else np.nan
        ),
        "p99_abs_ramp_pct_of_mean_per_h": (
            p99_abs_ramp / mean_power * 100.0 if mean_power > 0 else np.nan
        ),
        "max_abs_ramp_pct_of_capacity_per_h": (
            max_abs_ramp / capacity * 100.0 if capacity > 0 else np.nan
        ),
        "p99_abs_ramp_pct_of_capacity_per_h": (
            p99_abs_ramp / capacity * 100.0 if capacity > 0 else np.nan
        ),
        "max_abs_ramp_excluding_trace_seams_mw_per_h": (
            float(non_seam_abs_ramp.max()) if non_seam_abs_ramp.size else np.nan
        ),
        "p95_abs_ramp_excluding_trace_seams_mw_per_h": _percentile(
            non_seam_abs_ramp, 95.0
        ),
        "p99_abs_ramp_excluding_trace_seams_mw_per_h": _percentile(
            non_seam_abs_ramp, 99.0
        ),
        "p99_abs_ramp_excluding_trace_seams_pct_of_capacity_per_h": (
            _percentile(non_seam_abs_ramp, 99.0) / capacity * 100.0
            if capacity > 0 and non_seam_abs_ramp.size
            else np.nan
        ),
        "p99_abs_ramp_at_commit_boundaries_mw_per_h": _percentile(
            commit_abs_ramp, 99.0
        ),
        "p99_abs_ramp_outside_commit_boundaries_mw_per_h": _percentile(
            non_commit_abs_ramp, 99.0
        ),
        "hour_count": int(power.size),
        "transition_count": int(ramp.size),
        "trace_seam_transition_count": int(trace_seam_mask.sum()),
        "commit_boundary_transition_count": int(commit_boundary_mask.sum()),
    }


def _paired_ramping_record(
    scenario: str,
    year: int,
    method: str,
    baseline_power_mw: np.ndarray,
    optimized_power_mw: np.ndarray,
    facility_power_capacity_mw: float,
    trace_cycle_hours: int,
    commit_hours: int,
    country: str | None = None,
) -> dict:
    baseline_metrics = _ramping_metrics(
        power_mw=baseline_power_mw,
        facility_power_capacity_mw=facility_power_capacity_mw,
        trace_cycle_hours=trace_cycle_hours,
        commit_hours=commit_hours,
    )
    optimized_metrics = _ramping_metrics(
        power_mw=optimized_power_mw,
        facility_power_capacity_mw=facility_power_capacity_mw,
        trace_cycle_hours=trace_cycle_hours,
        commit_hours=commit_hours,
    )
    record = {
        "scenario": scenario,
        "year": int(year),
        "method": method,
        "facility_power_capacity_mw": float(facility_power_capacity_mw),
        "trace_cycle_hours": int(trace_cycle_hours),
        "commit_hours": int(commit_hours),
        "hour_count": baseline_metrics["hour_count"],
        "transition_count": baseline_metrics["transition_count"],
        "trace_seam_transition_count": baseline_metrics["trace_seam_transition_count"],
        "commit_boundary_transition_count": baseline_metrics[
            "commit_boundary_transition_count"
        ],
    }
    if country is not None:
        record["country"] = country

    metric_names = [
        key
        for key in baseline_metrics
        if key
        not in {
            "hour_count",
            "transition_count",
            "trace_seam_transition_count",
            "commit_boundary_transition_count",
        }
    ]
    for metric in metric_names:
        baseline_value = baseline_metrics[metric]
        optimized_value = optimized_metrics[metric]
        record[f"baseline_{metric}"] = baseline_value
        record[f"optimized_{metric}"] = optimized_value
        if metric in RAMPING_CHANGE_METRICS:
            record[f"optimized_minus_baseline_{metric}"] = optimized_value - baseline_value
            record[f"optimized_vs_baseline_{metric}_change_pct"] = _relative_change_pct(
                optimized=optimized_value,
                baseline=baseline_value,
            )
    return record


def _slug(value: str) -> str:
    return "_".join(str(value).strip().replace("-", "_").split()).lower()


class _RampingCollector:
    def __init__(
        self,
        trace_cycle_hours: int,
        commit_hours: int,
        hourly_output_dir: Path | None,
        run_label: str,
    ):
        self.trace_cycle_hours = int(trace_cycle_hours)
        self.commit_hours = int(commit_hours)
        self.hourly_output_dir = hourly_output_dir
        self.run_label = run_label
        self.country_records = []
        self.system_records = []

    def capture(
        self,
        scenario: str,
        year: int,
        method: str,
        countries,
        baseline: dict,
        optimized: dict,
    ) -> None:
        baseline_power = np.asarray(baseline["facility_energy_mwh"], dtype=np.float64)
        optimized_power = np.asarray(optimized["facility_energy_mwh"], dtype=np.float64)
        if baseline_power.shape != optimized_power.shape:
            raise ValueError("Baseline and optimized hourly facility power shapes do not match.")
        if baseline_power.shape[0] != len(countries):
            raise ValueError("Country count does not match hourly facility power rows.")

        capacity = np.asarray(
            baseline["_ramping_facility_power_capacity_mw"], dtype=np.float64
        )
        if capacity.shape != (len(countries),):
            raise ValueError("Facility power capacity does not match the selected countries.")

        for country_id, country in enumerate(countries):
            self.country_records.append(
                _paired_ramping_record(
                    scenario=scenario,
                    year=year,
                    method=method,
                    country=str(country),
                    baseline_power_mw=baseline_power[country_id],
                    optimized_power_mw=optimized_power[country_id],
                    facility_power_capacity_mw=capacity[country_id],
                    trace_cycle_hours=self.trace_cycle_hours,
                    commit_hours=self.commit_hours,
                )
            )

        self.system_records.append(
            _paired_ramping_record(
                scenario=scenario,
                year=year,
                method=method,
                baseline_power_mw=baseline_power.sum(axis=0),
                optimized_power_mw=optimized_power.sum(axis=0),
                facility_power_capacity_mw=float(capacity.sum()),
                trace_cycle_hours=self.trace_cycle_hours,
                commit_hours=self.commit_hours,
            )
        )

        if self.hourly_output_dir is not None:
            self._write_hourly_power(
                scenario=scenario,
                year=year,
                method=method,
                countries=countries,
                baseline_power=baseline_power,
                optimized_power=optimized_power,
                capacity=capacity,
            )

    def _write_hourly_power(
        self,
        scenario: str,
        year: int,
        method: str,
        countries,
        baseline_power: np.ndarray,
        optimized_power: np.ndarray,
        capacity: np.ndarray,
    ) -> None:
        self.hourly_output_dir.mkdir(parents=True, exist_ok=True)
        n_countries, n_hours = baseline_power.shape
        hours = np.arange(n_hours, dtype=np.int64)
        baseline_ramp = np.full_like(baseline_power, np.nan, dtype=np.float64)
        optimized_ramp = np.full_like(optimized_power, np.nan, dtype=np.float64)
        baseline_ramp[:, 1:] = np.diff(baseline_power, axis=1)
        optimized_ramp[:, 1:] = np.diff(optimized_power, axis=1)

        is_trace_seam = np.zeros(n_hours, dtype=bool)
        if self.trace_cycle_hours > 0:
            is_trace_seam = (hours > 0) & (hours % self.trace_cycle_hours == 0)
        is_commit_boundary = np.zeros(n_hours, dtype=bool)
        if self.commit_hours > 0:
            is_commit_boundary = (hours > 0) & (hours % self.commit_hours == 0)

        country_frame = pd.DataFrame(
            {
                "scenario": scenario,
                "year": int(year),
                "method": method,
                "scope": "country",
                "country": np.repeat(np.asarray(countries, dtype=object), n_hours),
                "hour": np.tile(hours, n_countries),
                "facility_power_capacity_mw": np.repeat(capacity, n_hours),
                "baseline_facility_power_mw": baseline_power.reshape(-1),
                "optimized_facility_power_mw": optimized_power.reshape(-1),
                "baseline_ramp_mw_per_h": baseline_ramp.reshape(-1),
                "optimized_ramp_mw_per_h": optimized_ramp.reshape(-1),
                "is_trace_seam_transition": np.tile(is_trace_seam, n_countries),
                "is_commit_boundary_transition": np.tile(
                    is_commit_boundary, n_countries
                ),
            }
        )

        system_baseline = baseline_power.sum(axis=0)
        system_optimized = optimized_power.sum(axis=0)
        system_baseline_ramp = np.full(n_hours, np.nan, dtype=np.float64)
        system_optimized_ramp = np.full(n_hours, np.nan, dtype=np.float64)
        system_baseline_ramp[1:] = np.diff(system_baseline)
        system_optimized_ramp[1:] = np.diff(system_optimized)
        system_frame = pd.DataFrame(
            {
                "scenario": scenario,
                "year": int(year),
                "method": method,
                "scope": "covered_system",
                "country": "__SYSTEM__",
                "hour": hours,
                "facility_power_capacity_mw": float(capacity.sum()),
                "baseline_facility_power_mw": system_baseline,
                "optimized_facility_power_mw": system_optimized,
                "baseline_ramp_mw_per_h": system_baseline_ramp,
                "optimized_ramp_mw_per_h": system_optimized_ramp,
                "is_trace_seam_transition": is_trace_seam,
                "is_commit_boundary_transition": is_commit_boundary,
            }
        )
        output = pd.concat([country_frame, system_frame], ignore_index=True)
        filename = (
            f"Hourly_Facility_Power_{self.run_label}_{_slug(scenario)}_{year}_"
            f"{_slug(method)}.csv.gz"
        )
        output.to_csv(self.hourly_output_dir / filename, index=False, compression="gzip")


def _empty_flexible_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "movable_type_id",
            "movable_task_type",
            "source_country_id",
            "source_hour",
            "execution_hour",
            "execution_country_id",
            "delay_hours",
            "share",
        ]
    )


def _active_flexible_groups(source_type_country_load: np.ndarray):
    if source_type_country_load.ndim != 4:
        raise ValueError(
            "source_type_country_load must have shape "
            "(movable_type, country, resource, hour)."
        )
    active = np.any(source_type_country_load > 1e-12, axis=2)
    type_ids, source_country_ids, source_hour_ids = np.nonzero(active)
    return (
        type_ids.astype(np.int32),
        source_country_ids.astype(np.int32),
        source_hour_ids.astype(np.int32),
    )


def _build_flexible_variable_index(
    active_type_ids: np.ndarray,
    active_source_country_ids: np.ndarray,
    active_source_hour_ids: np.ndarray,
    allowed_destination_mask: np.ndarray,
    n_hours: int,
    delay_hours_by_type: np.ndarray,
):
    type_parts = []
    source_country_parts = []
    source_hour_parts = []
    destination_parts = []
    execution_parts = []
    group_parts = []

    for group_id, (type_id, source_country, source_hour) in enumerate(
        zip(active_type_ids, active_source_country_ids, active_source_hour_ids)
    ):
        destinations = np.flatnonzero(
            allowed_destination_mask[int(source_country)]
        ).astype(np.int32)
        if destinations.size == 0:
            raise ValueError(
                f"No allowed destinations for source country index {int(source_country)}."
            )
        latest = min(
            n_hours - 1,
            int(source_hour) + int(delay_hours_by_type[int(type_id)]),
        )
        execution_hours = np.arange(int(source_hour), latest + 1, dtype=np.int32)
        variable_count = destinations.size * execution_hours.size
        type_parts.append(np.full(variable_count, int(type_id), dtype=np.int32))
        source_country_parts.append(
            np.full(variable_count, int(source_country), dtype=np.int32)
        )
        source_hour_parts.append(
            np.full(variable_count, int(source_hour), dtype=np.int32)
        )
        destination_parts.append(np.repeat(destinations, execution_hours.size))
        execution_parts.append(np.tile(execution_hours, destinations.size))
        group_parts.append(np.full(variable_count, group_id, dtype=np.int32))

    if not type_parts:
        empty = np.array([], dtype=np.int32)
        return empty, empty, empty, empty, empty, empty
    return (
        np.concatenate(type_parts),
        np.concatenate(source_country_parts),
        np.concatenate(source_hour_parts),
        np.concatenate(destination_parts),
        np.concatenate(execution_parts),
        np.concatenate(group_parts),
    )


def _estimate_flexible_variable_count(
    source_type_country_load: np.ndarray,
    allowed_destination_mask: np.ndarray,
    delay_hours_by_type: np.ndarray,
) -> int:
    type_ids, source_country_ids, source_hour_ids = _active_flexible_groups(
        source_type_country_load
    )
    if source_hour_ids.size == 0:
        return 0
    n_hours = source_type_country_load.shape[-1]
    execution_counts = (
        np.minimum(
            n_hours - 1,
            source_hour_ids + delay_hours_by_type[type_ids],
        )
        - source_hour_ids
        + 1
    )
    destination_counts = allowed_destination_mask[source_country_ids].sum(axis=1)
    return int(np.sum(execution_counts * destination_counts))


def _solve_flexible_migration_lp(
    source_type_country_load: np.ndarray,
    allowed_destination_mask: np.ndarray,
    residual_capacity: np.ndarray,
    marginal_carbon_cost: np.ndarray,
    delay_hours_by_type: np.ndarray,
    task_type_names=FLEXIBLE_TASK_TYPES,
    return_schedule: bool = False,
    verbose: bool = False,
    progress_prefix: str = "",
    solver: str = "scipy",
    linprog_options=None,
    gurobi_options=None,
):
    source = np.asarray(source_type_country_load, dtype=np.float64)
    delays = np.asarray(delay_hours_by_type, dtype=np.int32)
    n_types, n_source_countries, n_resources, n_hours = source.shape
    n_countries = residual_capacity.shape[0]
    if n_source_countries != n_countries:
        raise ValueError("Flexible source and residual-capacity country dimensions differ.")
    if residual_capacity.shape != (n_countries, n_resources, n_hours):
        raise ValueError("Unexpected residual-capacity shape for flexible migration.")
    if marginal_carbon_cost.shape != residual_capacity.shape:
        raise ValueError("Marginal carbon cost and residual capacity shapes differ.")
    if delays.shape != (n_types,) or np.any(delays < 0):
        raise ValueError("Each movable task type must have one non-negative delay window.")
    if len(task_type_names) != n_types:
        raise ValueError("task_type_names must match the movable type dimension.")
    if allowed_destination_mask.shape != (n_countries, n_countries):
        raise ValueError("allowed_destination_mask must be country-by-country.")

    type_ids, source_country_ids, source_hour_ids = _active_flexible_groups(source)
    optimized_type_load = np.zeros_like(source)
    migrated_gpu_hours = np.zeros((n_types, n_countries), dtype=np.float64)
    if source_hour_ids.size == 0:
        return optimized_type_load, _empty_flexible_schedule(), {
            "status": 0,
            "message": "No movable training or offline-inference workload.",
            "objective": 0.0,
            "carbon_objective_tco2": 0.0,
            "source_migrated_gpu_hours_by_type": migrated_gpu_hours,
            "variables": 0,
        }

    prefix = f"{progress_prefix}: " if progress_prefix else ""
    if verbose:
        print(
            f"[flexible-migration] {prefix}building joint LP for "
            f"{source_hour_ids.size} type-country-hour groups.",
            flush=True,
        )
    (
        variable_type_ids,
        variable_source_country_ids,
        variable_source_hour_ids,
        destination_ids,
        execution_hour_ids,
        group_ids,
    ) = _build_flexible_variable_index(
        active_type_ids=type_ids,
        active_source_country_ids=source_country_ids,
        active_source_hour_ids=source_hour_ids,
        allowed_destination_mask=allowed_destination_mask,
        n_hours=n_hours,
        delay_hours_by_type=delays,
    )
    n_variables = variable_type_ids.size
    variable_ids = np.arange(n_variables, dtype=np.int32)

    carbon_cost = np.zeros(n_variables, dtype=np.float64)
    for resource_id in range(n_resources):
        carbon_cost += (
            source[
                variable_type_ids,
                variable_source_country_ids,
                resource_id,
                variable_source_hour_ids,
            ]
            * marginal_carbon_cost[
                destination_ids,
                resource_id,
                execution_hour_ids,
            ]
        )

    equality_matrix = opt.sparse.coo_matrix(
        (
            np.ones(n_variables, dtype=np.float64),
            (group_ids, variable_ids),
        ),
        shape=(source_hour_ids.size, n_variables),
    ).tocsr()
    equality_rhs = np.ones(source_hour_ids.size, dtype=np.float64)

    capacity_rows = []
    capacity_cols = []
    capacity_data = []
    for resource_id in range(n_resources):
        values = source[
            variable_type_ids,
            variable_source_country_ids,
            resource_id,
            variable_source_hour_ids,
        ]
        positive = values > 1e-12
        if not np.any(positive):
            continue
        rows = (
            (destination_ids[positive] * n_resources + resource_id) * n_hours
            + execution_hour_ids[positive]
        )
        capacity_rows.append(rows.astype(np.int64, copy=False))
        capacity_cols.append(variable_ids[positive].astype(np.int64, copy=False))
        capacity_data.append(values[positive])

    if capacity_rows:
        flat_rows = np.concatenate(capacity_rows)
        unique_rows, compact_rows = np.unique(flat_rows, return_inverse=True)
        inequality_matrix = opt.sparse.coo_matrix(
            (
                np.concatenate(capacity_data),
                (compact_rows, np.concatenate(capacity_cols)),
            ),
            shape=(unique_rows.size, n_variables),
        ).tocsr()
        inequality_rhs = residual_capacity.reshape(-1)[unique_rows]
    else:
        inequality_matrix = opt.sparse.csr_matrix((0, n_variables))
        inequality_rhs = np.zeros(0, dtype=np.float64)

    if not np.all(np.isfinite(carbon_cost)):
        raise ValueError("Flexible LP objective contains NaN or infinite values.")
    if inequality_rhs.size and inequality_rhs.min() < -opt.CAPACITY_TOLERANCE:
        raise ValueError(
            f"Flexible LP has negative residual capacity: {inequality_rhs.min():.6g}."
        )
    inequality_rhs = np.maximum(inequality_rhs, 0.0)

    if solver == "scipy":
        solution, objective, solver_status, solver_message = opt._solve_sparse_lp_scipy(
            cost=carbon_cost,
            equality_matrix=equality_matrix,
            equality_rhs=equality_rhs,
            inequality_matrix=inequality_matrix,
            inequality_rhs=inequality_rhs,
            linprog_options=linprog_options,
        )
    elif solver == "gurobi":
        solution, objective, solver_status, solver_message = opt._solve_sparse_lp_gurobi(
            cost=carbon_cost,
            equality_matrix=equality_matrix,
            equality_rhs=equality_rhs,
            inequality_matrix=inequality_matrix,
            inequality_rhs=inequality_rhs,
            verbose=verbose,
            gurobi_options=gurobi_options,
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    solution = np.where(solution < 1e-12, 0.0, solution)
    for resource_id in range(n_resources):
        contribution = solution * source[
            variable_type_ids,
            variable_source_country_ids,
            resource_id,
            variable_source_hour_ids,
        ]
        np.add.at(
            optimized_type_load[:, :, resource_id, :],
            (variable_type_ids, destination_ids, execution_hour_ids),
            contribution,
        )

    gpu_id = opt.RESOURCES.index("gpu")
    cross_country = destination_ids != variable_source_country_ids
    migrated_contribution = solution[cross_country] * source[
        variable_type_ids[cross_country],
        variable_source_country_ids[cross_country],
        gpu_id,
        variable_source_hour_ids[cross_country],
    ]
    np.add.at(
        migrated_gpu_hours,
        (
            variable_type_ids[cross_country],
            variable_source_country_ids[cross_country],
        ),
        migrated_contribution,
    )

    if return_schedule:
        names = np.asarray(task_type_names, dtype=object)
        schedule = pd.DataFrame(
            {
                "movable_type_id": variable_type_ids,
                "movable_task_type": names[variable_type_ids],
                "source_country_id": variable_source_country_ids,
                "source_hour": variable_source_hour_ids,
                "execution_hour": execution_hour_ids,
                "execution_country_id": destination_ids,
                "delay_hours": execution_hour_ids - variable_source_hour_ids,
                "share": solution,
            }
        )
        schedule = schedule[schedule["share"] > 1e-10].reset_index(drop=True)
    else:
        schedule = _empty_flexible_schedule()

    return optimized_type_load, schedule, {
        "status": int(solver_status),
        "message": str(solver_message),
        "objective": float(objective),
        "carbon_objective_tco2": float(np.dot(solution, carbon_cost)),
        "source_migrated_gpu_hours_by_type": migrated_gpu_hours,
        "variables": int(n_variables),
    }


def _immediate_flexible_solution(
    source_type_country_load: np.ndarray,
    residual_capacity: np.ndarray,
    marginal_carbon_cost: np.ndarray,
    task_type_names=FLEXIBLE_TASK_TYPES,
    return_schedule: bool = False,
    message: str = "Immediate local fallback.",
):
    source = np.asarray(source_type_country_load, dtype=np.float64)
    combined = source.sum(axis=0)
    excess = combined - residual_capacity
    if excess.size and float(excess.max()) > opt.CAPACITY_TOLERANCE:
        idx = np.unravel_index(np.argmax(excess), excess.shape)
        raise RuntimeError(
            "Immediate flexible fallback would exceed residual capacity: "
            f"country/resource/hour index={idx}, excess={float(excess.max()):.6g}."
        )
    type_ids, source_country_ids, source_hour_ids = _active_flexible_groups(source)
    if return_schedule and source_hour_ids.size:
        names = np.asarray(task_type_names, dtype=object)
        schedule = pd.DataFrame(
            {
                "movable_type_id": type_ids,
                "movable_task_type": names[type_ids],
                "source_country_id": source_country_ids,
                "source_hour": source_hour_ids,
                "execution_hour": source_hour_ids,
                "execution_country_id": source_country_ids,
                "delay_hours": np.zeros(source_hour_ids.size, dtype=np.int32),
                "share": np.ones(source_hour_ids.size, dtype=np.float64),
            }
        )
    else:
        schedule = _empty_flexible_schedule()
    objective = float(np.sum(combined * marginal_carbon_cost))
    return source.copy(), schedule, {
        "status": 0,
        "message": message,
        "objective": objective,
        "carbon_objective_tco2": objective,
        "source_migrated_gpu_hours_by_type": np.zeros(
            (source.shape[0], source.shape[1]), dtype=np.float64
        ),
        "variables": int(source_hour_ids.size),
    }


def _solve_flexible_migration_windowed(
    source_type_country_load: np.ndarray,
    allowed_destination_mask: np.ndarray,
    capacity_limit: np.ndarray,
    fixed_for_optimization: np.ndarray,
    baseline_movable_type_load: np.ndarray,
    country_it_mw: np.ndarray,
    resource_capacities: np.ndarray,
    pue: np.ndarray,
    hourly_emission_kg_per_mwh: np.ndarray,
    hardware_config,
    delay_hours_by_type: np.ndarray,
    commit_hours: int,
    task_type_names=FLEXIBLE_TASK_TYPES,
    return_schedule: bool = False,
    verbose: bool = False,
    progress_prefix: str = "",
    solver: str = "scipy",
    linprog_options=None,
    gurobi_options=None,
):
    if commit_hours <= 0:
        raise ValueError("commit_hours must be positive.")
    source = np.asarray(source_type_country_load, dtype=np.float64)
    n_types, n_countries, _, n_hours = source.shape
    windows = [
        (start, min(n_hours, start + commit_hours))
        for start in range(0, n_hours, commit_hours)
    ]
    scheduled = np.zeros_like(source)
    reservation_remaining = np.asarray(baseline_movable_type_load, dtype=np.float64).sum(axis=0)
    schedule_frames = []
    total_objective = 0.0
    total_carbon_objective = 0.0
    total_variables = 0
    migrated = np.zeros((n_types, n_countries), dtype=np.float64)
    fallback_messages = []

    for window_id, (source_start, source_end) in enumerate(windows, start=1):
        reservation_remaining[:, :, source_start:source_end] = 0.0
        local_source = np.zeros_like(source)
        local_source[:, :, :, source_start:source_end] = source[
            :, :, :, source_start:source_end
        ]
        if not np.any(local_source > 1e-12):
            continue

        committed_load = fixed_for_optimization + scheduled.sum(axis=0)
        committed_residual = capacity_limit - committed_load
        if committed_residual.size and float(committed_residual.min()) < -opt.CAPACITY_TOLERANCE:
            idx = np.unravel_index(np.argmin(committed_residual), committed_residual.shape)
            raise ValueError(
                "Committed flexible workload exceeds capacity before LP solve: "
                f"country/resource/hour index={idx}, "
                f"residual={float(committed_residual.min()):.6g}."
            )
        available = np.maximum(committed_residual, 0.0)
        effective_reservation = np.minimum(reservation_remaining, available)
        base_load = committed_load + effective_reservation
        residual_capacity = np.maximum(capacity_limit - base_load, 0.0)
        marginal_it = opt._marginal_it_mwh_per_resource_hour(
            country_it_mw=country_it_mw,
            fixed_resource_load=base_load,
            resource_capacities=resource_capacities,
            config=hardware_config,
        )
        marginal_carbon_cost = (
            marginal_it
            * pue[:, None, None]
            * hourly_emission_kg_per_mwh[:, None, :]
            / 1000.0
        )
        estimated_variables = _estimate_flexible_variable_count(
            source_type_country_load=local_source,
            allowed_destination_mask=allowed_destination_mask,
            delay_hours_by_type=np.asarray(delay_hours_by_type, dtype=np.int32),
        )
        if verbose:
            print(
                f"[flexible-migration] {progress_prefix}: window {window_id}/{len(windows)} "
                f"source hours {source_start}-{source_end - 1}, "
                f"about {estimated_variables} variables.",
                flush=True,
            )
        terminal_short_window = source_end == n_hours and (source_end - source_start) <= 24
        try:
            window_load, window_schedule, info = _solve_flexible_migration_lp(
                source_type_country_load=local_source,
                allowed_destination_mask=allowed_destination_mask,
                residual_capacity=residual_capacity,
                marginal_carbon_cost=marginal_carbon_cost,
                delay_hours_by_type=np.asarray(delay_hours_by_type, dtype=np.int32),
                task_type_names=task_type_names,
                return_schedule=return_schedule,
                verbose=False,
                progress_prefix=f"{progress_prefix} window {window_id}/{len(windows)}",
                solver=solver,
                linprog_options=linprog_options,
                gurobi_options=gurobi_options,
            )
        except RuntimeError as exc:
            if not terminal_short_window:
                raise
            message = (
                "Terminal LP failed; used immediate local execution fallback. "
                f"Original error: {exc}"
            )
            window_load, window_schedule, info = _immediate_flexible_solution(
                source_type_country_load=local_source,
                residual_capacity=residual_capacity,
                marginal_carbon_cost=marginal_carbon_cost,
                task_type_names=task_type_names,
                return_schedule=return_schedule,
                message=message,
            )
            info["variables"] = int(estimated_variables)
            fallback_messages.append(message)

        scheduled += window_load
        total_objective += info["objective"]
        total_carbon_objective += info["carbon_objective_tco2"]
        total_variables += info["variables"]
        migrated += info["source_migrated_gpu_hours_by_type"]
        if return_schedule and not window_schedule.empty:
            schedule_frames.append(window_schedule)

    schedule = (
        pd.concat(schedule_frames, ignore_index=True)
        if return_schedule and schedule_frames
        else _empty_flexible_schedule()
    )
    return scheduled, schedule, {
        "status": 0,
        "message": (
            "Windowed joint flexible-workload LP completed."
            if not fallback_messages
            else "Windowed joint flexible-workload LP completed with terminal fallback. "
            + " ".join(fallback_messages)
        ),
        "objective": float(total_objective),
        "carbon_objective_tco2": float(total_carbon_objective),
        "source_migrated_gpu_hours_by_type": migrated,
        "variables": int(total_variables),
        "windows": int(len(windows)),
    }


def _matched_window_group_order(
    method_name: str,
    type_ids: np.ndarray,
    source_country_ids: np.ndarray,
    source_hour_ids: np.ndarray,
    allowed_destination_mask: np.ndarray,
    n_hours: int,
    delay_hours_by_type: np.ndarray,
    order_mode: str,
) -> np.ndarray:
    latest = np.minimum(
        n_hours - 1,
        source_hour_ids + delay_hours_by_type[type_ids],
    )
    if order_mode == "arrival":
        return np.lexsort(
            (source_country_ids, type_ids, latest, source_hour_ids)
        )
    if order_mode != "constrained":
        raise ValueError(f"Unknown matched-window greedy order mode: {order_mode}.")

    execution_counts = latest - source_hour_ids + 1
    if method_name in (
        migration_methods.LOCAL_IMMEDIATE_TRAINING_OVERFLOW,
        migration_methods.LOCAL_TRAINING_TIME_SHIFT_GREEDY,
    ):
        destination_counts = allowed_destination_mask[
            source_country_ids, source_country_ids
        ].astype(np.int64)
    else:
        destination_counts = allowed_destination_mask[source_country_ids].sum(axis=1)
    option_counts = execution_counts * destination_counts
    return np.lexsort(
        (
            source_country_ids,
            type_ids,
            source_hour_ids,
            latest,
            option_counts,
        )
    )


def _matched_window_candidate_arrays(
    method_name: str,
    source_country: int,
    source_hour: int,
    allowed_destination_mask: np.ndarray,
    n_hours: int,
    delay_hours: int,
    resources: np.ndarray,
    marginal_carbon_cost: np.ndarray,
    annual_marginal_carbon_cost: np.ndarray,
):
    if method_name == migration_methods.ANNUAL_AVERAGE_IMMEDIATE_OVERFLOW_GREEDY:
        delay_hours = 0
    last_hour = min(n_hours - 1, int(source_hour) + int(delay_hours))
    execution_base = np.arange(int(source_hour), last_hour + 1, dtype=np.int32)
    if method_name in (
        migration_methods.LOCAL_IMMEDIATE_TRAINING_OVERFLOW,
        migration_methods.LOCAL_TRAINING_TIME_SHIFT_GREEDY,
    ):
        if not allowed_destination_mask[int(source_country), int(source_country)]:
            empty = np.array([], dtype=np.int32)
            costs = np.array([], dtype=np.float64)
            return empty, empty, costs, costs
        destination_ids = np.full(
            execution_base.size,
            int(source_country),
            dtype=np.int32,
        )
        execution_hours = execution_base
    else:
        destination_base = np.flatnonzero(
            allowed_destination_mask[int(source_country)]
        ).astype(np.int32)
        destination_ids = np.repeat(destination_base, execution_base.size)
        execution_hours = np.tile(execution_base, destination_base.size)

    hourly_carbon = np.sum(
        marginal_carbon_cost[destination_ids, :, execution_hours]
        * resources[None, :],
        axis=1,
    )
    if method_name == migration_methods.ANNUAL_AVERAGE_IMMEDIATE_OVERFLOW_GREEDY:
        sort_carbon = np.sum(
            annual_marginal_carbon_cost[destination_ids] * resources[None, :],
            axis=1,
        )
    else:
        sort_carbon = hourly_carbon

    if method_name == migration_methods.LOCAL_IMMEDIATE_TRAINING_OVERFLOW:
        order = np.argsort(execution_hours, kind="mergesort")
    elif method_name in (
        migration_methods.LOCAL_TRAINING_TIME_SHIFT_GREEDY,
        migration_methods.HOURLY_ARRIVAL_GREEDY_FALLBACK,
    ):
        order = np.lexsort((execution_hours, sort_carbon))
    else:
        immediate = execution_hours == int(source_hour)
        immediate_ids = np.flatnonzero(immediate)
        delayed_ids = np.flatnonzero(~immediate)
        immediate_order = immediate_ids[
            np.lexsort(
                (
                    destination_ids[immediate_ids],
                    sort_carbon[immediate_ids],
                )
            )
        ]
        delayed_order = delayed_ids[
            np.lexsort(
                (
                    sort_carbon[delayed_ids],
                    execution_hours[delayed_ids],
                )
            )
        ]
        order = np.concatenate((immediate_order, delayed_order))
    return (
        destination_ids[order],
        execution_hours[order],
        hourly_carbon[order],
        sort_carbon[order],
    )


def _solve_flexible_greedy_window_matched(
    method_name: str,
    source_type_country_load: np.ndarray,
    allowed_destination_mask: np.ndarray,
    residual_capacity: np.ndarray,
    marginal_carbon_cost: np.ndarray,
    delay_hours_by_type: np.ndarray,
    task_type_names=FLEXIBLE_TASK_TYPES,
    return_schedule: bool = False,
    objective_normalization: str = "benchmark",
    order_mode: str = "constrained",
):
    source = np.asarray(source_type_country_load, dtype=np.float64)
    delays = np.asarray(delay_hours_by_type, dtype=np.int32)
    n_types, n_countries, n_resources, n_hours = source.shape
    if method_name not in migration_methods._GREEDY_METHODS:
        raise ValueError(f"Matched-window greedy does not support method: {method_name}.")
    if delays.shape != (n_types,):
        raise ValueError("Each movable task type must have one delay window.")
    if len(task_type_names) != n_types:
        raise ValueError("task_type_names must match the movable type dimension.")
    if residual_capacity.shape != (n_countries, n_resources, n_hours):
        raise ValueError("Unexpected matched-window residual-capacity shape.")
    if marginal_carbon_cost.shape != residual_capacity.shape:
        raise ValueError("Matched-window marginal cost and capacity shapes differ.")

    type_ids, source_country_ids, source_hour_ids = _active_flexible_groups(source)
    optimized = np.zeros_like(source)
    migrated = np.zeros((n_types, n_countries), dtype=np.float64)
    if source_hour_ids.size == 0:
        return optimized, _empty_flexible_schedule(), {
            "status": 0,
            "message": f"{method_name} matched-window greedy had no movable workload.",
            "objective": 0.0,
            "carbon_objective_tco2": 0.0,
            "source_migrated_gpu_hours_by_type": migrated,
            "variables": 0,
            "solver": "greedy",
            "solve_mode": "matched_window",
        }

    if objective_normalization == "none":
        carbon_scale = 1.0
    elif objective_normalization == "benchmark":
        carbon_scale = 0.0
        for type_id, source_country, source_hour in zip(
            type_ids, source_country_ids, source_hour_ids
        ):
            resources = source[int(type_id), int(source_country), :, int(source_hour)]
            carbon_scale += float(
                np.dot(
                    marginal_carbon_cost[
                        int(source_country), :, int(source_hour)
                    ],
                    resources,
                )
            )
        if carbon_scale <= 1e-12:
            carbon_scale = 1.0
    else:
        raise ValueError("objective_normalization must be 'benchmark' or 'none'.")

    group_order = _matched_window_group_order(
        method_name=method_name,
        type_ids=type_ids,
        source_country_ids=source_country_ids,
        source_hour_ids=source_hour_ids,
        allowed_destination_mask=allowed_destination_mask,
        n_hours=n_hours,
        delay_hours_by_type=delays,
        order_mode=order_mode,
    )
    annual_marginal_carbon_cost = marginal_carbon_cost.mean(axis=2)
    usage = np.zeros_like(residual_capacity)
    reservation_remaining = source.sum(axis=0)
    rows = []
    total_objective = 0.0
    total_carbon_objective = 0.0
    total_variables = 0
    gpu_id = opt.RESOURCES.index("gpu")

    for group_id in group_order:
        type_id = int(type_ids[group_id])
        source_country = int(source_country_ids[group_id])
        source_hour = int(source_hour_ids[group_id])
        resources = source[type_id, source_country, :, source_hour]
        reservation_remaining[source_country, :, source_hour] = np.maximum(
            reservation_remaining[source_country, :, source_hour] - resources,
            0.0,
        )
        (
            destination_ids,
            execution_hours,
            hourly_carbon,
            sort_carbon,
        ) = _matched_window_candidate_arrays(
            method_name=method_name,
            source_country=source_country,
            source_hour=source_hour,
            allowed_destination_mask=allowed_destination_mask,
            n_hours=n_hours,
            delay_hours=int(delays[type_id]),
            resources=resources,
            marginal_carbon_cost=marginal_carbon_cost,
            annual_marginal_carbon_cost=annual_marginal_carbon_cost,
        )
        total_variables += int(destination_ids.size)
        if destination_ids.size == 0:
            raise ValueError(
                f"{method_name} has no matched-window candidate for "
                f"type={type_id}, source_country={source_country}, "
                f"source_hour={source_hour}."
            )

        remaining = 1.0
        positive_resources = resources > 1e-12
        for destination, execution, candidate_carbon, candidate_sort_carbon in zip(
            destination_ids,
            execution_hours,
            hourly_carbon,
            sort_carbon,
        ):
            destination = int(destination)
            execution = int(execution)
            if np.any(positive_resources):
                slack = (
                    residual_capacity[destination, positive_resources, execution]
                    - usage[destination, positive_resources, execution]
                    - reservation_remaining[
                        destination, positive_resources, execution
                    ]
                )
                max_fraction = float(
                    np.min(slack / resources[positive_resources])
                )
            else:
                max_fraction = 1.0
            take = min(remaining, max(0.0, max_fraction))
            if take <= 1e-10:
                continue

            contribution = resources * take
            usage[destination, :, execution] += contribution
            optimized[type_id, destination, :, execution] += contribution
            if destination != source_country:
                migrated[type_id, source_country] += resources[gpu_id] * take
            total_objective += float(candidate_sort_carbon) / carbon_scale * take
            total_carbon_objective += float(candidate_carbon) * take
            if return_schedule:
                rows.append(
                    {
                        "movable_type_id": type_id,
                        "movable_task_type": task_type_names[type_id],
                        "source_country_id": source_country,
                        "source_hour": source_hour,
                        "execution_hour": execution,
                        "execution_country_id": destination,
                        "delay_hours": execution - source_hour,
                        "share": take,
                    }
                )
            remaining -= take
            if remaining <= 1e-9:
                break

        if remaining > 1e-8:
            raise ValueError(
                f"{method_name} could not place matched-window group "
                f"type={type_id}, source_country={source_country}, "
                f"source_hour={source_hour}, remaining={remaining:.6g}."
            )

    schedule = (
        pd.DataFrame(rows, columns=_empty_flexible_schedule().columns)
        if return_schedule and rows
        else _empty_flexible_schedule()
    )
    return optimized, schedule, {
        "status": 0,
        "message": f"{method_name} matched-window {order_mode} greedy completed.",
        "objective": float(total_objective),
        "carbon_objective_tco2": float(total_carbon_objective),
        "source_migrated_gpu_hours_by_type": migrated,
        "variables": int(total_variables),
        "solver": "greedy",
        "solve_mode": "matched_window",
    }


def _solve_flexible_baseline_method_matched_window(
    method_name: str,
    source_type_country_load: np.ndarray,
    allowed_destination_mask: np.ndarray,
    capacity_limit: np.ndarray,
    fixed_for_optimization: np.ndarray,
    country_it_mw: np.ndarray,
    resource_capacities: np.ndarray,
    pue: np.ndarray,
    hourly_emission_kg_per_mwh: np.ndarray,
    hardware_config,
    delay_hours_by_type: np.ndarray,
    commit_hours: int,
    task_type_names=FLEXIBLE_TASK_TYPES,
    return_schedule: bool = False,
    objective_normalization: str = "benchmark",
    linprog_options=None,
):
    """Run baselines with the LP's source-window and reservation protocol."""
    if commit_hours <= 0:
        raise ValueError("commit_hours must be positive.")
    source = np.asarray(source_type_country_load, dtype=np.float64)
    delays = np.asarray(delay_hours_by_type, dtype=np.int32)
    n_types, n_countries, _, n_hours = source.shape
    if delays.shape != (n_types,):
        raise ValueError("Each movable task type must have one delay window.")

    windows = [
        (start, min(n_hours, start + commit_hours))
        for start in range(0, n_hours, commit_hours)
    ]
    scheduled = np.zeros_like(source)
    reservation_remaining = source.sum(axis=0).copy()
    schedule_frames = []
    migrated = np.zeros((n_types, n_countries), dtype=np.float64)
    total_objective = 0.0
    total_carbon_objective = 0.0
    total_variables = 0
    constrained_fallback_windows = 0
    method_solver = method_name

    for window_id, (source_start, source_end) in enumerate(windows, start=1):
        reservation_remaining[:, :, source_start:source_end] = 0.0
        local_source = np.zeros_like(source)
        local_source[:, :, :, source_start:source_end] = source[
            :, :, :, source_start:source_end
        ]
        if not np.any(local_source > 1e-12):
            continue

        committed_load = fixed_for_optimization + scheduled.sum(axis=0)
        committed_residual = capacity_limit - committed_load
        if committed_residual.size and float(
            committed_residual.min()
        ) < -opt.CAPACITY_TOLERANCE:
            idx = np.unravel_index(
                np.argmin(committed_residual), committed_residual.shape
            )
            raise ValueError(
                "Committed flexible workload exceeds capacity before matched "
                f"baseline solve: country/resource/hour index={idx}, "
                f"residual={float(committed_residual.min()):.6g}."
            )
        available = np.maximum(committed_residual, 0.0)
        effective_reservation = np.minimum(reservation_remaining, available)
        base_load = committed_load + effective_reservation
        residual_capacity = np.maximum(capacity_limit - base_load, 0.0)
        marginal_it = opt._marginal_it_mwh_per_resource_hour(
            country_it_mw=country_it_mw,
            fixed_resource_load=base_load,
            resource_capacities=resource_capacities,
            config=hardware_config,
        )
        marginal_carbon_cost = (
            marginal_it
            * pue[:, None, None]
            * hourly_emission_kg_per_mwh[:, None, :]
            / 1000.0
        )

        if method_name in migration_methods._GREEDY_METHODS:
            order_mode = (
                "arrival"
                if method_name
                == migration_methods.HOURLY_ARRIVAL_GREEDY_FALLBACK
                else "constrained"
            )
            try:
                window_load, window_schedule, info = (
                    _solve_flexible_greedy_window_matched(
                        method_name=method_name,
                        source_type_country_load=local_source,
                        allowed_destination_mask=allowed_destination_mask,
                        residual_capacity=residual_capacity,
                        marginal_carbon_cost=marginal_carbon_cost,
                        delay_hours_by_type=delays,
                        task_type_names=task_type_names,
                        return_schedule=return_schedule,
                        objective_normalization=objective_normalization,
                        order_mode=order_mode,
                    )
                )
            except ValueError:
                if order_mode != "arrival":
                    raise
                window_load, window_schedule, info = (
                    _solve_flexible_greedy_window_matched(
                        method_name=method_name,
                        source_type_country_load=local_source,
                        allowed_destination_mask=allowed_destination_mask,
                        residual_capacity=residual_capacity,
                        marginal_carbon_cost=marginal_carbon_cost,
                        delay_hours_by_type=delays,
                        task_type_names=task_type_names,
                        return_schedule=return_schedule,
                        objective_normalization=objective_normalization,
                        order_mode="constrained",
                    )
                )
                info["message"] = (
                    "Matched-window arrival order was infeasible; constrained "
                    "group-order fallback completed."
                )
                constrained_fallback_windows += 1
        else:
            window_load, window_schedule, info = _solve_flexible_baseline_method(
                method_name=method_name,
                source_type_country_load=local_source,
                allowed_destination_mask=allowed_destination_mask,
                capacity_limit=capacity_limit,
                fixed_for_optimization=base_load,
                country_it_mw=country_it_mw,
                resource_capacities=resource_capacities,
                pue=pue,
                hourly_emission_kg_per_mwh=hourly_emission_kg_per_mwh,
                hardware_config=hardware_config,
                delay_hours_by_type=delays,
                commit_hours=n_hours,
                task_type_names=task_type_names,
                return_schedule=return_schedule,
                objective_normalization=objective_normalization,
                linprog_options=linprog_options,
                marginal_carbon_cost_override=marginal_carbon_cost,
            )

        scheduled += window_load
        total_objective += float(info.get("objective", 0.0))
        total_carbon_objective += float(info.get("carbon_objective_tco2", 0.0))
        total_variables += int(info.get("variables", 0))
        migrated += info["source_migrated_gpu_hours_by_type"]
        method_solver = info.get("solver", method_solver)
        if return_schedule and not window_schedule.empty:
            schedule_frames.append(window_schedule)

    schedule = (
        pd.concat(schedule_frames, ignore_index=True)
        if return_schedule and schedule_frames
        else _empty_flexible_schedule()
    )
    return scheduled, schedule, {
        "status": 0,
        "message": (
            f"Matched-window flexible baseline completed across {len(windows)} "
            "source windows."
            + (
                f" Constrained group-order fallback was used in "
                f"{constrained_fallback_windows} windows."
                if constrained_fallback_windows
                else ""
            )
        ),
        "objective": float(total_objective),
        "carbon_objective_tco2": float(total_carbon_objective),
        "source_migrated_gpu_hours_by_type": migrated,
        "variables": int(total_variables),
        "windows": int(len(windows)),
        "commit_hours": int(commit_hours),
        "solver": method_solver,
        "solve_mode": "matched_window",
    }


def _solve_flexible_baseline_method(
    method_name: str,
    source_type_country_load: np.ndarray,
    allowed_destination_mask: np.ndarray,
    capacity_limit: np.ndarray,
    fixed_for_optimization: np.ndarray,
    country_it_mw: np.ndarray,
    resource_capacities: np.ndarray,
    pue: np.ndarray,
    hourly_emission_kg_per_mwh: np.ndarray,
    hardware_config,
    delay_hours_by_type: np.ndarray,
    commit_hours: int,
    task_type_names=FLEXIBLE_TASK_TYPES,
    return_schedule: bool = False,
    objective_normalization: str = "benchmark",
    linprog_options=None,
    marginal_carbon_cost_override=None,
):
    """Run the unchanged 8.182 baseline solver for each flexible task class.

    Baseline heuristics are intrinsically sequential.  Classes are therefore
    placed by earliest deadline (offline inference before training), while
    unscheduled classes retain their local baseline reservations.  This keeps
    the original method definitions and guarantees a shared capacity budget.
    """
    source = np.asarray(source_type_country_load, dtype=np.float64)
    delays = np.asarray(delay_hours_by_type, dtype=np.int32)
    n_types, n_countries, _, n_hours = source.shape
    if delays.shape != (n_types,):
        raise ValueError("Each movable task type must have one delay window.")

    scheduled = np.zeros_like(source)
    completed = np.zeros_like(fixed_for_optimization)
    schedule_frames = []
    migrated_by_type = np.zeros((n_types, n_countries), dtype=np.float64)
    total_objective = 0.0
    total_carbon_objective = 0.0
    total_variables = 0
    total_windows = 0
    messages = []
    method_solver = method_name
    method_solve_mode = method_name
    pending = set(range(n_types))

    for type_id in np.argsort(delays, kind="stable"):
        type_id = int(type_id)
        pending.remove(type_id)
        pending_reservation = (
            source[list(sorted(pending))].sum(axis=0)
            if pending
            else np.zeros_like(fixed_for_optimization)
        )
        method_fixed = fixed_for_optimization + completed + pending_reservation
        residual_capacity = capacity_limit - method_fixed
        min_residual = float(residual_capacity.min()) if residual_capacity.size else 0.0
        if min_residual < -opt.CAPACITY_TOLERANCE:
            idx = np.unravel_index(np.argmin(residual_capacity), residual_capacity.shape)
            raise ValueError(
                "Flexible baseline reservations exceed capacity before scheduling: "
                f"country/resource/hour index={idx}, residual={min_residual:.6g}."
            )
        residual_capacity = np.maximum(residual_capacity, 0.0)
        if marginal_carbon_cost_override is None:
            marginal_it = opt._marginal_it_mwh_per_resource_hour(
                country_it_mw=country_it_mw,
                fixed_resource_load=method_fixed,
                resource_capacities=resource_capacities,
                config=hardware_config,
            )
            marginal_carbon_cost = (
                marginal_it
                * pue[:, None, None]
                * hourly_emission_kg_per_mwh[:, None, :]
                / 1000.0
            )
        else:
            marginal_carbon_cost = np.asarray(
                marginal_carbon_cost_override,
                dtype=np.float64,
            )
            if marginal_carbon_cost.shape != residual_capacity.shape:
                raise ValueError(
                    "Overridden marginal carbon cost has an unexpected shape."
                )
        transfer_gb = np.zeros((n_countries, n_hours), dtype=np.float64)
        type_load, type_schedule, info = opt.solve_training_migration_baseline_method(
            method_name=method_name,
            source_country_training_load=source[type_id],
            source_country_transfer_gb=transfer_gb,
            allowed_destination_mask=allowed_destination_mask,
            residual_capacity=residual_capacity,
            capacity_limit=capacity_limit,
            fixed_for_optimization=method_fixed,
            baseline_movable_load=source[type_id],
            marginal_carbon_cost=marginal_carbon_cost,
            annual_marginal_carbon_cost=marginal_carbon_cost.mean(axis=2),
            tx_cost_per_gb=np.zeros((n_countries, n_countries), dtype=np.float64),
            delay_hours=int(delays[type_id]),
            commit_hours=commit_hours,
            return_schedule=return_schedule,
            carbon_weight=1.0,
            tx_weight=0.0,
            objective_normalization=objective_normalization,
            linprog_options=linprog_options,
        )
        scheduled[type_id] = type_load
        completed += type_load
        migrated_by_type[type_id] = info["source_migrated_training_load"]
        total_objective += float(info.get("objective", 0.0))
        total_carbon_objective += float(info.get("carbon_objective_tco2", 0.0))
        total_variables += int(info.get("variables", 0))
        total_windows += int(info.get("windows", 1))
        messages.append(f"{task_type_names[type_id]}: {info.get('message', '')}")
        method_solver = info.get("solver", method_solver)
        method_solve_mode = info.get("solve_mode", method_solve_mode)
        if return_schedule and not type_schedule.empty:
            type_schedule = type_schedule.copy()
            type_schedule.insert(0, "movable_task_type", task_type_names[type_id])
            type_schedule.insert(0, "movable_type_id", type_id)
            schedule_frames.append(type_schedule)

    schedule = (
        pd.concat(schedule_frames, ignore_index=True)
        if return_schedule and schedule_frames
        else _empty_flexible_schedule()
    )
    return scheduled, schedule, {
        "status": 0,
        "message": (
            "Deadline-priority flexible baseline completed; " + " ".join(messages)
        ),
        "objective": float(total_objective),
        "carbon_objective_tco2": float(total_carbon_objective),
        "source_migrated_gpu_hours_by_type": migrated_by_type,
        "variables": int(total_variables),
        "windows": int(total_windows),
        "commit_hours": int(commit_hours),
        "solver": method_solver,
        "solve_mode": method_solve_mode,
    }


def _offline_inference_execution_records(
    scenario: str,
    year: int,
    method: str,
    countries,
    source_offline_load: np.ndarray,
    baseline_offline_load: np.ndarray,
    optimized_offline_load: np.ndarray,
    source_migrated_offline_load: np.ndarray,
):
    records = []
    global_resource_hours = source_offline_load.sum(axis=1)
    baseline_resource_hours = baseline_offline_load.sum(axis=2)
    optimized_resource_hours = optimized_offline_load.sum(axis=2)
    total_by_resource = baseline_resource_hours.sum(axis=0)
    gpu_id = opt.RESOURCES.index("gpu")
    for country_id, country in enumerate(countries):
        source_gpu_hours = baseline_resource_hours[country_id, gpu_id]
        record = {
            "scenario": scenario,
            "year": year,
            "method": method,
            "country": country,
            "origin_offline_inference_share": (
                source_gpu_hours / total_by_resource[gpu_id]
                if total_by_resource[gpu_id] > 0
                else 0.0
            ),
            "migrated_offline_inference_share_pct": (
                source_migrated_offline_load[country_id] / source_gpu_hours * 100.0
                if source_gpu_hours > 0
                else 0.0
            ),
        }
        for resource_id, resource in enumerate(opt.RESOURCES):
            baseline_hours = baseline_resource_hours[country_id, resource_id]
            optimized_hours = optimized_resource_hours[country_id, resource_id]
            record[f"global_offline_inference_{resource}_hours"] = global_resource_hours[
                resource_id
            ]
            record[f"baseline_execution_{resource}_hours"] = baseline_hours
            record[f"optimized_execution_{resource}_hours"] = optimized_hours
            record[f"net_execution_change_{resource}_hours"] = (
                optimized_hours - baseline_hours
            )
        records.append(record)
    return records


def _schedule_for_type(schedule: pd.DataFrame, type_id: int) -> pd.DataFrame:
    if schedule.empty or "movable_type_id" not in schedule.columns:
        return _empty_flexible_schedule()
    return schedule.loc[schedule["movable_type_id"] == int(type_id)].copy()


def _flexible_migration_flow_records(
    scenario: str,
    year: int,
    method: str,
    countries,
    schedule: pd.DataFrame,
    source_type_country_load: np.ndarray,
):
    records = []
    for type_id, task_type in enumerate(FLEXIBLE_TASK_TYPES):
        type_schedule = _schedule_for_type(schedule, type_id)
        type_records = opt._migration_flow_summary_records_carbon_only(
            scenario=scenario,
            year=year,
            method=method,
            countries=countries,
            schedule=type_schedule,
            source_country_training_load=source_type_country_load[type_id],
        )
        for record in type_records:
            record["movable_task_type"] = task_type
            for resource in opt.RESOURCES:
                old_key = f"training_{resource}_hours"
                if old_key in record:
                    record[f"movable_{resource}_hours"] = record.pop(old_key)
            records.append(record)
    return records


def _run_flexible_workload_migration_optimization(
    renewable_energy_policy: str,
    scenarios,
    years: int,
    countries,
    workload_profile_path: Path,
    server_profile_path,
    network_profile_path,
    year_start: int,
    output_dir: str,
    save_outputs: bool,
    execution_policy: str,
    inference_origin_fraction: float,
    other_origin_fraction: float,
    capacity_quantile: float,
    max_resource_utilization: float,
    hourly_carbon_factors_dir: Path,
    hourly_carbon_scope: str,
    hourly_carbon_fallback_to_annual: bool,
    training_delay_hours: int,
    offline_inference_delay_hours: int,
    migration_constraint: str,
    save_schedule: bool,
    save_analysis_outputs: bool,
    max_intervals,
    workload_cache_dir,
    refresh_workload_cache: bool,
    workload_cache_workers: int,
    solve_mode: str,
    commit_hours: int,
    monolithic_variable_limit: int,
    solver: str,
    linprog_options,
    policy_transmission_matrix,
    azure_transmission_policy,
    azure_transmission_dir: Path,
    policy_block_cost: float,
    objective_normalization: str,
    scheduling_methods,
    baseline_scheduling_mode: str = DEFAULT_BASELINE_SCHEDULING_MODE,
    verbose: bool = True,
):
    if years <= 0:
        raise ValueError("years must be positive.")
    if training_delay_hours < 0 or offline_inference_delay_hours < 0:
        raise ValueError("Flexible-workload delay windows must be non-negative.")
    if migration_constraint not in opt.MIGRATION_CONSTRAINTS:
        raise ValueError(f"Unknown migration constraint: {migration_constraint}.")
    if solve_mode not in opt.SOLVE_MODES:
        raise ValueError(f"Unknown solve mode: {solve_mode}.")
    if solver not in opt.SOLVERS:
        raise ValueError(f"Unknown solver: {solver}.")
    if baseline_scheduling_mode not in BASELINE_SCHEDULING_MODES:
        raise ValueError(
            f"Unknown baseline scheduling mode: {baseline_scheduling_mode}."
        )
    if commit_hours <= 0 or monolithic_variable_limit <= 0:
        raise ValueError("commit-hours and monolithic-variable-limit must be positive.")
    if not 0 < max_resource_utilization <= 1:
        raise ValueError("max-resource-utilization must be in (0, 1].")

    selected_methods = []
    for method_name in scheduling_methods or [opt.LINEAR_PROGRAMMING_METHOD]:
        if method_name == "all":
            selected_methods.extend(opt.DEFAULT_SCHEDULING_METHODS)
        else:
            selected_methods.append(str(method_name))
    selected_methods = list(dict.fromkeys(selected_methods))
    unknown = [name for name in selected_methods if name not in opt.SCHEDULING_METHODS]
    if unknown:
        raise ValueError(f"Unknown scheduling methods: {unknown}.")

    data_year_end = opt.DATA_YEAR_START + opt.it_capacity.shape[0] - 1
    if year_start < opt.DATA_YEAR_START or year_start + years - 1 > data_year_end:
        raise ValueError(
            f"Requested years must be within {opt.DATA_YEAR_START}-{data_year_end}."
        )
    ai_capacity_factors = opt._resolve_ai_capacity_factors(
        year_start=year_start,
        years=years,
        overrides=None,
    )
    hardware_config = opt.HardwarePowerConfig()
    hardware_config.validate()
    countries = list(countries or opt.DEFAULT_COUNTRIES)
    for scenario in scenarios:
        if scenario not in opt.SCENARIO_COL_MAP:
            raise ValueError(f"Unknown scenario: {scenario}.")
    unknown_countries = [country for country in countries if country not in opt.it_ratio]
    if unknown_countries:
        raise ValueError(f"Unknown countries: {unknown_countries}.")

    origin_weights = opt._as_task_weight_table(countries, None)
    training_origin_weights = origin_weights[opt.TASK_TYPES.index("training")]
    country_share = np.array([float(opt.it_ratio[country]) for country in countries])
    europe_set = set(opt.EUROPEAN_COUNTRIES)
    europe_ids = np.array(
        [idx for idx, country in enumerate(countries) if country in europe_set],
        dtype=np.int32,
    )
    if migration_constraint == "europe_only" and europe_ids.size == 0:
        raise ValueError("No selected country belongs to the configured European set.")
    (
        transmission_matrix_values,
        policy_allowed_mask,
        transmission_matrix_path,
        transmission_policy_name,
    ) = opt._load_transmission_policy_matrix(
        countries=countries,
        policy_transmission_matrix=policy_transmission_matrix,
        azure_transmission_policy=azure_transmission_policy,
        azure_transmission_dir=azure_transmission_dir,
        policy_block_cost=policy_block_cost,
    )
    transmission_policy_records = opt._transmission_policy_restriction_records(
        policy_name=transmission_policy_name,
        matrix_path=transmission_matrix_path,
        countries=countries,
        allowed_mask=policy_allowed_mask,
        policy_block_cost=policy_block_cost,
    )
    allowed_destination_mask = opt._apply_migration_constraint_to_allowed_mask(
        allowed_mask=policy_allowed_mask,
        migration_constraint=migration_constraint,
        europe_ids=europe_ids,
    )

    annual_records = []
    country_records = []
    training_execution_records = []
    offline_execution_records = []
    country_capacity_records = []
    component_energy_records = []
    migration_flow_records = []
    flexible_flow_records = []
    capacity_binding_records = []
    schedule_frames = []
    return_schedule_for_analysis = save_schedule or save_analysis_outputs

    profile, direct_offline_profile = _prepare_workload_profiles(
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        network_profile_path=network_profile_path,
        capacity_quantile=capacity_quantile,
        max_intervals=max_intervals,
        workload_cache_dir=workload_cache_dir,
        refresh_workload_cache=refresh_workload_cache,
        workload_cache_workers=workload_cache_workers,
        verbose=verbose,
    )

    training_type_id = opt.TASK_TYPES.index("training")
    inference_type_id = opt.TASK_TYPES.index("inference")
    gpu_id = opt.RESOURCES.index("gpu")
    delay_hours_by_type = np.array(
        [training_delay_hours, offline_inference_delay_hours], dtype=np.int32
    )

    for scenario in scenarios:
        opt._print_progress(verbose, f"Starting scenario '{scenario}'.")
        scenario_col = opt.SCENARIO_COL_MAP[scenario]
        for output_year_idx in range(years):
            year = year_start + output_year_idx
            data_year_idx = year - opt.DATA_YEAR_START
            progress_label = f"{scenario} {year}"
            hourly_timestamps, hourly_emission, used_hourly_factors = opt._hourly_emission_factors(
                countries=countries,
                renewable_energy_policy=renewable_energy_policy,
                year=year,
                year_idx=data_year_idx,
                hourly_carbon_factors_dir=hourly_carbon_factors_dir,
                hourly_carbon_scope=hourly_carbon_scope,
                hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
            )
            if max_intervals is not None:
                horizon_hours = max(1, int(np.ceil(max_intervals * profile.interval_hours)))
                hourly_timestamps = hourly_timestamps[:horizon_hours]
                hourly_emission = hourly_emission[:, :horizon_hours]

            total_data_center_global_it_mw = float(
                opt.it_capacity[data_year_idx, scenario_col]
            ) * 1e3
            ai_capacity_factor = ai_capacity_factors[year]
            global_it_mw = total_data_center_global_it_mw * ai_capacity_factor
            country_it_mw = global_it_mw * country_share
            resource_capacities = opt._resource_capacity(country_it_mw, hardware_config)
            global_resource_capacity = resource_capacities.sum(axis=0)
            pue = np.array(
                [opt.PUE[country][data_year_idx, scenario_col] for country in countries],
                dtype=float,
            )
            if save_analysis_outputs:
                country_capacity_records.extend(
                    opt.country_resource_capacity_records(
                        scenario=scenario,
                        year=year,
                        countries=countries,
                        country_share=country_share,
                        country_it_mw=country_it_mw,
                        resource_capacities=resource_capacities,
                        pue=pue,
                        ai_capacity_factor=ai_capacity_factor,
                        total_data_center_global_it_mw=total_data_center_global_it_mw,
                        global_ai_it_mw=global_it_mw,
                        max_resource_utilization=max_resource_utilization,
                    )
                )

            interval_global_load = opt._scale_workload_to_capacity(
                profile=profile,
                global_resource_capacity=global_resource_capacity,
                max_resource_utilization=max_resource_utilization,
            )
            hourly_global_type_load = opt._hourly_load_from_intervals(
                interval_load=interval_global_load,
                interval_index=profile.interval_index,
                interval_hours=profile.interval_hours,
                target_timestamps=hourly_timestamps,
                relative_trace=profile.relative_time,
            )
            direct_offline_interval_load = opt._scale_workload_to_capacity(
                profile=direct_offline_profile,
                global_resource_capacity=global_resource_capacity,
                max_resource_utilization=max_resource_utilization,
            )
            direct_offline_hourly_type_load = opt._hourly_load_from_intervals(
                interval_load=direct_offline_interval_load,
                interval_index=direct_offline_profile.interval_index,
                interval_hours=direct_offline_profile.interval_hours,
                target_timestamps=hourly_timestamps,
                relative_trace=direct_offline_profile.relative_time,
            )
            source_training_load = hourly_global_type_load[training_type_id]
            # This slice contains only raw job_type_public=offline_inference rows.
            source_offline_load = direct_offline_hourly_type_load[inference_type_id]
            execution_weights = opt._build_execution_weights(
                countries=countries,
                country_it_mw=country_it_mw,
                origin_weights=origin_weights,
                execution_policy=execution_policy,
                inference_origin_fraction=inference_origin_fraction,
                other_origin_fraction=other_origin_fraction,
                task_execution_weights=None,
            )
            country_type_load = (
                hourly_global_type_load[:, None, :, :]
                * execution_weights[:, :, None, None]
            )
            baseline_training_load = (
                source_training_load[None, :, :] * training_origin_weights[:, None, None]
            )
            baseline_inference_load = country_type_load[inference_type_id]
            baseline_offline_load = (
                source_offline_load[None, :, :]
                * execution_weights[inference_type_id, :, None, None]
            )
            fixed_online_inference_load = baseline_inference_load - baseline_offline_load
            min_online_load = float(fixed_online_inference_load.min())
            if min_online_load < -opt.CAPACITY_TOLERANCE:
                raise ValueError(
                    "Direct offline-inference load exceeds combined inference load: "
                    f"minimum online remainder={min_online_load:.6g}."
                )
            fixed_online_inference_load = np.maximum(fixed_online_inference_load, 0.0)
            fixed_non_flexible_load = fixed_online_inference_load.copy()
            for task_id, task_type in enumerate(opt.TASK_TYPES):
                if task_type not in {"training", "inference"}:
                    fixed_non_flexible_load += country_type_load[task_id]
            baseline_total_load = (
                fixed_non_flexible_load + baseline_training_load + baseline_offline_load
            )
            baseline_eval = opt._evaluate_country_hourly_carbon(
                country_it_mw=country_it_mw,
                country_resource_load=baseline_total_load,
                resource_capacities=resource_capacities,
                pue=pue,
                hourly_emission_kg_per_mwh=hourly_emission,
                hardware_config=hardware_config,
                max_resource_utilization=max_resource_utilization,
            )
            baseline_task_metrics = opt._allocate_task_energy_carbon(
                country_it_mw=country_it_mw,
                task_country_resource_load=opt._task_country_resource_load(
                    country_type_load=country_type_load,
                    training_load=baseline_training_load,
                ),
                resource_capacities=resource_capacities,
                pue=pue,
                hourly_emission_kg_per_mwh=hourly_emission,
                hardware_config=hardware_config,
                max_resource_utilization=max_resource_utilization,
            )

            if migration_constraint == "global":
                fixed_training_load = np.zeros_like(baseline_training_load)
                fixed_offline_load = np.zeros_like(baseline_offline_load)
                movable_training_share = float(training_origin_weights.sum())
                movable_offline_share = 1.0
            else:
                fixed_training_load = baseline_training_load.copy()
                fixed_offline_load = baseline_offline_load.copy()
                fixed_training_load[europe_ids] = 0.0
                fixed_offline_load[europe_ids] = 0.0
                movable_training_share = float(training_origin_weights[europe_ids].sum())
                total_offline_gpu = float(baseline_offline_load[:, gpu_id, :].sum())
                movable_offline_share = (
                    float(baseline_offline_load[europe_ids, gpu_id, :].sum())
                    / total_offline_gpu
                    if total_offline_gpu > 0
                    else 0.0
                )
            baseline_movable_training = np.maximum(
                baseline_training_load - fixed_training_load, 0.0
            )
            baseline_movable_offline = np.maximum(
                baseline_offline_load - fixed_offline_load, 0.0
            )
            baseline_movable_type_load = np.stack(
                [baseline_movable_training, baseline_movable_offline], axis=0
            )
            fixed_for_optimization = (
                fixed_non_flexible_load + fixed_training_load + fixed_offline_load
            )
            capacity_limit = (
                resource_capacities[:, :, None] * max_resource_utilization
            )
            fixed_residual_capacity = capacity_limit - fixed_for_optimization
            min_residual = float(fixed_residual_capacity.min())
            if min_residual < -opt.CAPACITY_TOLERANCE:
                idx = np.unravel_index(
                    np.argmin(fixed_residual_capacity), fixed_residual_capacity.shape
                )
                raise ValueError(
                    "Fixed workload exceeds capacity before flexible migration: "
                    f"country={countries[idx[0]]}, resource={opt.RESOURCES[idx[1]]}, "
                    f"hour={idx[2]}, residual={min_residual:.6g}."
                )
            fixed_residual_capacity = np.maximum(fixed_residual_capacity, 0.0)
            base_marginal_it = opt._marginal_it_mwh_per_resource_hour(
                country_it_mw=country_it_mw,
                fixed_resource_load=fixed_for_optimization,
                resource_capacities=resource_capacities,
                config=hardware_config,
            )
            base_marginal_carbon_cost = (
                base_marginal_it
                * pue[:, None, None]
                * hourly_emission[:, None, :]
                / 1000.0
            )
            estimated_variables = _estimate_flexible_variable_count(
                source_type_country_load=baseline_movable_type_load,
                allowed_destination_mask=allowed_destination_mask,
                delay_hours_by_type=delay_hours_by_type,
            )
            baseline_carbon = float(baseline_eval["carbon_tco2"].sum())
            baseline_facility_energy = float(
                baseline_eval["facility_energy_mwh"].sum()
            )
            total_training_gpu_hours = float(
                baseline_training_load[:, gpu_id, :].sum()
            )
            total_offline_gpu_hours = float(
                baseline_offline_load[:, gpu_id, :].sum()
            )
            total_inference_gpu_hours = float(
                baseline_inference_load[:, gpu_id, :].sum()
            )
            direct_offline_share_of_inference = (
                total_offline_gpu_hours / total_inference_gpu_hours
                if total_inference_gpu_hours > 0
                else 0.0
            )
            total_flexible_gpu_hours = total_training_gpu_hours + total_offline_gpu_hours

            for scheduling_method in selected_methods:
                method_label = f"{progress_label} {scheduling_method}"
                opt._print_progress(verbose, f"{method_label}: starting scheduling method.")
                selected_solve_mode = (
                    solve_mode
                    if scheduling_method == opt.LINEAR_PROGRAMMING_METHOD
                    else scheduling_method
                )
                method_solver = (
                    solver
                    if scheduling_method == opt.LINEAR_PROGRAMMING_METHOD
                    else scheduling_method
                )
                try:
                    if scheduling_method == opt.LINEAR_PROGRAMMING_METHOD:
                        if solve_mode == "auto":
                            selected_solve_mode = (
                                "windowed"
                                if estimated_variables > monolithic_variable_limit
                                else "monolithic"
                            )
                        if selected_solve_mode == "monolithic":
                            movable_type_load, schedule, solver_info = (
                                _solve_flexible_migration_lp(
                                    source_type_country_load=baseline_movable_type_load,
                                    allowed_destination_mask=allowed_destination_mask,
                                    residual_capacity=fixed_residual_capacity,
                                    marginal_carbon_cost=base_marginal_carbon_cost,
                                    delay_hours_by_type=delay_hours_by_type,
                                    return_schedule=return_schedule_for_analysis,
                                    verbose=verbose,
                                    progress_prefix=method_label,
                                    solver=solver,
                                    linprog_options=linprog_options,
                                )
                            )
                            solver_info["windows"] = 1
                        else:
                            movable_type_load, schedule, solver_info = (
                                _solve_flexible_migration_windowed(
                                    source_type_country_load=baseline_movable_type_load,
                                    allowed_destination_mask=allowed_destination_mask,
                                    capacity_limit=capacity_limit,
                                    fixed_for_optimization=fixed_for_optimization,
                                    baseline_movable_type_load=baseline_movable_type_load,
                                    country_it_mw=country_it_mw,
                                    resource_capacities=resource_capacities,
                                    pue=pue,
                                    hourly_emission_kg_per_mwh=hourly_emission,
                                    hardware_config=hardware_config,
                                    delay_hours_by_type=delay_hours_by_type,
                                    commit_hours=commit_hours,
                                    return_schedule=return_schedule_for_analysis,
                                    verbose=verbose,
                                    progress_prefix=method_label,
                                    solver=solver,
                                    linprog_options=linprog_options,
                                )
                            )
                    else:
                        baseline_solver = (
                            _solve_flexible_baseline_method_matched_window
                            if baseline_scheduling_mode == "matched_window"
                            else _solve_flexible_baseline_method
                        )
                        movable_type_load, schedule, solver_info = (
                            baseline_solver(
                                method_name=scheduling_method,
                                source_type_country_load=baseline_movable_type_load,
                                allowed_destination_mask=allowed_destination_mask,
                                capacity_limit=capacity_limit,
                                fixed_for_optimization=fixed_for_optimization,
                                country_it_mw=country_it_mw,
                                resource_capacities=resource_capacities,
                                pue=pue,
                                hourly_emission_kg_per_mwh=hourly_emission,
                                hardware_config=hardware_config,
                                delay_hours_by_type=delay_hours_by_type,
                                commit_hours=commit_hours,
                                return_schedule=return_schedule_for_analysis,
                                objective_normalization=objective_normalization,
                                linprog_options=linprog_options,
                            )
                        )
                        selected_solve_mode = solver_info.get(
                            "solve_mode", scheduling_method
                        )
                        method_solver = solver_info.get("solver", scheduling_method)
                except Exception as exc:
                    opt._print_progress(
                        verbose,
                        f"{method_label}: failed; continuing with remaining methods. {exc}",
                    )
                    failed = opt._failed_method_annual_record(
                        scenario=scenario,
                        year=year,
                        method=scheduling_method,
                        migration_constraint=migration_constraint,
                        solver=method_solver,
                        delay_hours=training_delay_hours,
                        solve_mode=selected_solve_mode,
                        commit_hours=commit_hours,
                        transmission_policy_name=transmission_policy_name,
                        transmission_matrix_path=transmission_matrix_path,
                        ai_capacity_factor=ai_capacity_factor,
                        total_data_center_global_it_mw=total_data_center_global_it_mw,
                        global_it_mw=global_it_mw,
                        movable_share=movable_training_share,
                        baseline_eval=baseline_eval,
                        baseline_carbon=baseline_carbon,
                        used_hourly_factors=used_hourly_factors,
                        error=exc,
                    )
                    failed.update(
                        {
                            "offline_inference_share_of_inference": direct_offline_share_of_inference,
                            "offline_inference_delay_hours": offline_inference_delay_hours,
                            "movable_offline_inference_share": movable_offline_share,
                            "migrated_offline_inference_share_pct": np.nan,
                            "migrated_flexible_workload_share_pct": np.nan,
                        }
                    )
                    annual_records.append(failed)
                    continue

                optimized_training_load = fixed_training_load + movable_type_load[0]
                optimized_offline_load = fixed_offline_load + movable_type_load[1]
                optimized_total_load = (
                    fixed_non_flexible_load
                    + optimized_training_load
                    + optimized_offline_load
                )
                optimized_eval = opt._evaluate_country_hourly_carbon(
                    country_it_mw=country_it_mw,
                    country_resource_load=optimized_total_load,
                    resource_capacities=resource_capacities,
                    pue=pue,
                    hourly_emission_kg_per_mwh=hourly_emission,
                    hardware_config=hardware_config,
                    max_resource_utilization=max_resource_utilization,
                )
                optimized_task_load = opt._task_country_resource_load(
                    country_type_load=country_type_load,
                    training_load=optimized_training_load,
                )
                optimized_task_load[inference_type_id] = (
                    fixed_online_inference_load + optimized_offline_load
                )
                optimized_task_metrics = opt._allocate_task_energy_carbon(
                    country_it_mw=country_it_mw,
                    task_country_resource_load=optimized_task_load,
                    resource_capacities=resource_capacities,
                    pue=pue,
                    hourly_emission_kg_per_mwh=hourly_emission,
                    hardware_config=hardware_config,
                    max_resource_utilization=max_resource_utilization,
                )

                optimized_carbon = float(optimized_eval["carbon_tco2"].sum())
                optimized_facility_energy = float(
                    optimized_eval["facility_energy_mwh"].sum()
                )
                carbon_reduction = baseline_carbon - optimized_carbon
                facility_energy_reduction = (
                    baseline_facility_energy - optimized_facility_energy
                )
                migrated_by_type = solver_info[
                    "source_migrated_gpu_hours_by_type"
                ]
                migrated_training_gpu_hours = float(migrated_by_type[0].sum())
                migrated_offline_gpu_hours = float(migrated_by_type[1].sum())
                migrated_training_share_pct = (
                    migrated_training_gpu_hours / total_training_gpu_hours * 100.0
                    if total_training_gpu_hours > 0
                    else 0.0
                )
                migrated_offline_share_pct = (
                    migrated_offline_gpu_hours / total_offline_gpu_hours * 100.0
                    if total_offline_gpu_hours > 0
                    else 0.0
                )
                migrated_flexible_share_pct = (
                    (migrated_training_gpu_hours + migrated_offline_gpu_hours)
                    / total_flexible_gpu_hours
                    * 100.0
                    if total_flexible_gpu_hours > 0
                    else 0.0
                )
                linearized_training_objective = float(
                    np.sum(movable_type_load[0] * base_marginal_carbon_cost)
                )
                annual_record = {
                    "scenario": scenario,
                    "year": year,
                    "method": scheduling_method,
                    "migration_constraint": migration_constraint,
                    "solver": method_solver,
                    "delay_hours": training_delay_hours,
                    "training_delay_hours": training_delay_hours,
                    "offline_inference_delay_hours": offline_inference_delay_hours,
                    "offline_inference_share_of_inference": direct_offline_share_of_inference,
                    "solve_mode": selected_solve_mode,
                    "commit_hours": solver_info.get(
                        "commit_hours",
                        commit_hours if selected_solve_mode == "windowed" else 0,
                    ),
                    "transmission_policy": transmission_policy_name,
                    "transmission_matrix_path": (
                        str(transmission_matrix_path)
                        if transmission_matrix_path is not None
                        else ""
                    ),
                    "ai_capacity_factor": float(ai_capacity_factor),
                    "total_data_center_global_it_mw": float(
                        total_data_center_global_it_mw
                    ),
                    "global_ai_it_mw": float(global_it_mw),
                    "movable_training_share": movable_training_share,
                    "movable_offline_inference_share": movable_offline_share,
                    "baseline_facility_energy_mwh": baseline_facility_energy,
                    "optimized_facility_energy_mwh": optimized_facility_energy,
                    "facility_energy_reduction_mwh": facility_energy_reduction,
                    "facility_energy_reduction_pct": opt._fraction(
                        facility_energy_reduction, baseline_facility_energy
                    ),
                    "baseline_carbon_tco2": baseline_carbon,
                    "optimized_carbon_tco2": optimized_carbon,
                    "carbon_reduction_tco2": carbon_reduction,
                    "carbon_reduction_pct": opt._fraction(
                        carbon_reduction, baseline_carbon
                    ),
                    "linearized_training_objective_tco2": linearized_training_objective,
                    "linearized_flexible_objective_tco2": solver_info[
                        "carbon_objective_tco2"
                    ],
                    "baseline_training_gpu_hours": total_training_gpu_hours,
                    "baseline_offline_inference_gpu_hours": total_offline_gpu_hours,
                    "baseline_flexible_gpu_hours": total_flexible_gpu_hours,
                    "migrated_training_gpu_hours": migrated_training_gpu_hours,
                    "migrated_offline_inference_gpu_hours": migrated_offline_gpu_hours,
                    "migrated_training_share_pct": migrated_training_share_pct,
                    "migrated_offline_inference_share_pct": migrated_offline_share_pct,
                    "migrated_flexible_workload_share_pct": migrated_flexible_share_pct,
                    "optimization_variables": solver_info["variables"],
                    "optimization_windows": solver_info["windows"],
                    "hourly_factor_countries": int(used_hourly_factors.sum()),
                    "annual_fallback_countries": int((~used_hourly_factors).sum()),
                    "solver_status": solver_info["status"],
                    "solver_message": solver_info["message"],
                }
                annual_record.update(
                    opt._task_metric_output_fields(
                        baseline_task_metrics=baseline_task_metrics,
                        optimized_task_metrics=optimized_task_metrics,
                    )
                )
                annual_records.append(annual_record)
                opt._print_progress(
                    verbose,
                    f"{method_label}: completed with {carbon_reduction:.6f} tCO2 "
                    f"reduction ({opt._fraction(carbon_reduction, baseline_carbon):.6%}).",
                )
                country_records.extend(
                    opt._country_records(
                        scenario=scenario,
                        year=year,
                        method=scheduling_method,
                        countries=countries,
                        baseline=baseline_eval,
                        optimized=optimized_eval,
                        baseline_task_metrics=baseline_task_metrics,
                        optimized_task_metrics=optimized_task_metrics,
                    )
                )
                training_execution_records.extend(
                    opt._training_execution_records(
                        scenario=scenario,
                        year=year,
                        method=scheduling_method,
                        countries=countries,
                        training_origin_weights=training_origin_weights,
                        source_training_load=source_training_load,
                        baseline_training_load=baseline_training_load,
                        optimized_training_load=optimized_training_load,
                        source_migrated_training_load=migrated_by_type[0],
                    )
                )
                offline_execution_records.extend(
                    _offline_inference_execution_records(
                        scenario=scenario,
                        year=year,
                        method=scheduling_method,
                        countries=countries,
                        source_offline_load=source_offline_load,
                        baseline_offline_load=baseline_offline_load,
                        optimized_offline_load=optimized_offline_load,
                        source_migrated_offline_load=migrated_by_type[1],
                    )
                )
                if save_analysis_outputs:
                    component_energy_records.extend(
                        opt.country_component_energy_records(
                            scenario=scenario,
                            year=year,
                            method=scheduling_method,
                            countries=countries,
                            country_it_mw=country_it_mw,
                            baseline_resource_load=baseline_total_load,
                            optimized_resource_load=optimized_total_load,
                            resource_capacities=resource_capacities,
                            pue=pue,
                            hardware_config=hardware_config,
                            max_resource_utilization=max_resource_utilization,
                        )
                    )
                    capacity_binding_records.extend(
                        opt.capacity_binding_summary_records(
                            scenario=scenario,
                            year=year,
                            method=scheduling_method,
                            countries=countries,
                            baseline_eval=baseline_eval,
                            optimized_eval=optimized_eval,
                        )
                    )
                    training_schedule = _schedule_for_type(schedule, 0)
                    migration_flow_records.extend(
                        opt._migration_flow_summary_records_carbon_only(
                            scenario=scenario,
                            year=year,
                            method=scheduling_method,
                            countries=countries,
                            schedule=training_schedule,
                            source_country_training_load=baseline_movable_training,
                        )
                    )
                    flexible_flow_records.extend(
                        _flexible_migration_flow_records(
                            scenario=scenario,
                            year=year,
                            method=scheduling_method,
                            countries=countries,
                            schedule=schedule,
                            source_type_country_load=baseline_movable_type_load,
                        )
                    )
                if save_schedule:
                    schedule_frames.append(
                        opt._schedule_records(
                            schedule=schedule,
                            scenario=scenario,
                            year=year,
                            method=scheduling_method,
                            countries=countries,
                            timestamps=hourly_timestamps,
                        )
                    )

    tag = "-".join([scenario.replace(" ", "") for scenario in scenarios]) or "None"
    results = {
        "annual_summary": pd.DataFrame(annual_records),
        "country_summary": pd.DataFrame(country_records),
        "training_execution": pd.DataFrame(training_execution_records),
        "offline_inference_execution": pd.DataFrame(offline_execution_records),
        "country_resource_capacity": pd.DataFrame(country_capacity_records),
        "country_component_energy": pd.DataFrame(component_energy_records),
        "migration_flow_summary": pd.DataFrame(migration_flow_records),
        "flexible_migration_flow_summary": pd.DataFrame(flexible_flow_records),
        "transmission_policy_summary": pd.DataFrame(transmission_policy_records),
        "capacity_binding_summary": pd.DataFrame(capacity_binding_records),
        "migration_schedule": (
            pd.concat(schedule_frames, ignore_index=True)
            if schedule_frames
            else pd.DataFrame(
                columns=[
                    "scenario",
                    "year",
                    "method",
                    "movable_type_id",
                    "movable_task_type",
                    "source_country",
                    "source_hour",
                    "execution_hour",
                    "delay_hours",
                    "share",
                    "source_timestamp_utc",
                    "execution_timestamp_utc",
                    "execution_country",
                ]
            )
        ),
        "workload_profile_summary": profile.task_type_summary,
    }
    if save_outputs:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        suffix = f"{renewable_energy_policy}_{migration_constraint}_{tag}"
        if transmission_policy_name != "none":
            suffix = f"{suffix}_{transmission_policy_name}"
        output_specs = {
            "annual_summary": f"Training_Migration_Annual_Summary_{suffix}.csv",
            "country_summary": f"Training_Migration_Country_Summary_{suffix}.csv",
            "training_execution": f"Training_Migration_Execution_{suffix}.csv",
            "offline_inference_execution": (
                f"Offline_Inference_Migration_Execution_{suffix}.csv"
            ),
            "country_resource_capacity": f"Country_Resource_Capacity_{suffix}.csv",
            "country_component_energy": f"Country_Component_Energy_{suffix}.csv",
            "migration_flow_summary": f"Migration_Flow_Summary_{suffix}.csv",
            "flexible_migration_flow_summary": (
                f"Flexible_Migration_Flow_Summary_{suffix}.csv"
            ),
            "transmission_policy_summary": f"Transmission_Policy_Summary_{suffix}.csv",
            "capacity_binding_summary": f"Capacity_Binding_Summary_{suffix}.csv",
        }
        for result_key, filename in output_specs.items():
            results[result_key].to_csv(output_path / filename, index=False)
        results["workload_profile_summary"].to_csv(
            output_path / "Training_Migration_Workload_Profile_Summary.csv", index=False
        )
        if save_schedule:
            results["migration_schedule"].to_csv(
                output_path / f"Training_Migration_Schedule_{suffix}.csv", index=False
            )
        opt._print_progress(verbose, f"Output files saved to {output_path.resolve()}.")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Alibaba GPU v2026 training and offline-inference migration "
            "optimization with "
            "EM-CPNDCNZ carbon factors. Azure matrices are used only as "
            "country-pair transmission restrictions, not as cost terms."
        )
    )
    parser.add_argument("--policy", choices=["CP", "NDC", "NZ"], default="CP")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["Base"],
        choices=["Base", "Lift-Off", "High Efficiency", "Headwinds"],
    )
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--year-start", type=int, default=2026)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument(
        "--constraint",
        choices=opt.MIGRATION_CONSTRAINTS,
        default="global",
        help=(
            "global allows all selected countries; europe_only moves European-origin "
            "training and offline inference only within Europe."
        ),
    )
    parser.add_argument(
        "--delay-hours",
        type=int,
        default=24,
        help="Maximum delay for movable training workload. Default: 24 hours.",
    )
    parser.add_argument(
        "--offline-inference-delay-hours",
        type=int,
        default=DEFAULT_OFFLINE_INFERENCE_DELAY_HOURS,
        help="Maximum delay for movable offline-inference workload. Default: 6 hours.",
    )
    parser.add_argument(
        "--disable-offline-inference-migration",
        action="store_true",
        help=(
            "Use the original training-only 8.182 workflow exactly; useful for strict "
            "backward-comparison runs."
        ),
    )
    parser.add_argument("--solve-mode", choices=opt.SOLVE_MODES, default="windowed")
    parser.add_argument("--solver", choices=opt.SOLVERS, default="scipy")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[opt.LINEAR_PROGRAMMING_METHOD],
        choices=["all", *opt.SCHEDULING_METHODS],
    )
    parser.add_argument(
        "--baseline-scheduling-mode",
        choices=BASELINE_SCHEDULING_MODES,
        default=DEFAULT_BASELINE_SCHEDULING_MODE,
        help=(
            "matched_window gives baselines the LP rolling-window, reservation, "
            "and marginal-cost protocol; legacy_sequential preserves the 827 "
            "full-year deadline-priority baseline behavior. Default: matched_window."
        ),
    )
    parser.add_argument(
        "--commit-hours",
        type=int,
        default=24,
        help="Source-hour commit window used by the rolling LP and matched baselines. Default: 24.",
    )
    parser.add_argument("--monolithic-variable-limit", type=int, default=1_000_000)
    parser.add_argument(
        "--workload-path",
        default=str(ROOT_DIR / "dataset" / "cluster-trace-gpu-v2026"),
        help="Alibaba GPU v2026 release directory containing data/asi_opensource_* tables.",
    )
    parser.add_argument("--server-profile-path", default=None)
    parser.add_argument(
        "--network-profile-path",
        default=None,
        help="Accepted for CLI compatibility but ignored by the carbon-only 8.182 core.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "results" / "workload_training_migration_optimization"),
    )
    parser.add_argument(
        "--execution-policy",
        choices=["capacity", "origin", "hybrid"],
        default="capacity",
        help="Execution policy for fixed non-training workloads.",
    )
    parser.add_argument("--inference-origin-fraction", type=float, default=0.75)
    parser.add_argument("--other-origin-fraction", type=float, default=0.50)
    parser.add_argument("--capacity-quantile", type=float, default=0.96)
    parser.add_argument("--max-resource-utilization", type=float, default=1.0)
    parser.add_argument(
        "--hourly-carbon-dir",
        default=str(opt.EM_CPNDCNZ_DIR),
        help="Must resolve to dataset/EM-CPNDCNZ; other paths are redirected to the EM-CPNDCNZ dataset.",
    )
    parser.add_argument("--hourly-carbon-scope", choices=["direct", "life_cycle"], default="direct")
    parser.add_argument(
        "--strict-hourly-carbon",
        action="store_true",
        help="Fail when an EM-CPNDCNZ hourly carbon CSV is missing instead of using EM-CPNDCNZ annual fallback.",
    )
    parser.add_argument("--save-schedule", action="store_true")
    parser.add_argument(
        "--no-analysis-outputs",
        action="store_true",
        help="Do not build extra analysis tables such as capacity, component energy, and migration flows.",
    )
    parser.add_argument("--max-intervals", type=int, default=None)
    parser.add_argument(
        "--workload-cache-dir",
        default=str(ROOT_DIR / "results" / "workload_profile_cache"),
        help="Directory for cached Alibaba v2026 carbon-only workload profiles.",
    )
    parser.add_argument(
        "--no-workload-cache",
        action="store_true",
        help="Disable workload/profile caching and read Alibaba parquet inputs directly.",
    )
    parser.add_argument(
        "--refresh-workload-cache",
        action="store_true",
        help="Ignore existing workload caches and rebuild them from Alibaba parquet inputs.",
    )
    parser.add_argument(
        "--workload-cache-workers",
        type=int,
        default=DEFAULT_WORKLOAD_CACHE_WORKERS,
        help=(
            "Worker processes used to read, classify, and aggregate Parquet partitions "
            "while building workload caches. "
            "Default: 8; use 1 for the original serial behavior."
        ),
    )
    parser.add_argument(
        "--prepare-workload-cache-only",
        action="store_true",
        help="Build and validate both 827 workload caches, then exit before optimization.",
    )
    parser.add_argument("--linprog-time-limit", type=float, default=None)
    parser.add_argument(
        "--policy-transmission-matrix",
        default=None,
        help="Optional country-by-country policy matrix. Values >= --policy-block-cost block migration.",
    )
    parser.add_argument(
        "--azure-transmission-policy",
        choices=opt.AZURE_TRANSMISSION_POLICIES,
        default=None,
        help="Read one of the three Azure policy matrices from dataset/Azure as migration restrictions.",
    )
    parser.add_argument("--azure-transmission-dir", default=str(opt.AZURE_TRANSMISSION_DIR))
    parser.add_argument("--policy-block-cost", type=float, default=opt.DEFAULT_POLICY_BLOCK_COST)
    parser.add_argument(
        "--carbon-weight",
        type=float,
        default=1.0,
        help="Accepted for compatibility; the 8.182 core always optimizes carbon only.",
    )
    parser.add_argument(
        "--tx-weight",
        type=float,
        default=1.0,
        help="Accepted for compatibility but ignored by the carbon-only 8.182 core.",
    )
    parser.add_argument(
        "--include-transmission-cost",
        dest="include_transmission_cost",
        action="store_true",
        default=False,
        help="Accepted for compatibility but ignored by the carbon-only 8.182 core.",
    )
    parser.add_argument(
        "--no-transmission-cost",
        dest="include_transmission_cost",
        action="store_false",
        help="Accepted for compatibility; this is the effective behavior of the 8.182 core.",
    )
    parser.add_argument(
        "--objective-normalization",
        choices=opt.OBJECTIVE_NORMALIZATIONS,
        default="benchmark",
        help="Used only by non-LP baseline helpers; transmission terms are zero.",
    )
    parser.add_argument(
        "--ramping-only",
        action="store_true",
        help=(
            "Save only the new ramping outputs. The optimization still runs normally, "
            "but the pre-existing annual/country output files are not written."
        ),
    )
    parser.add_argument(
        "--save-ramping-hourly-power",
        action="store_true",
        help=(
            "Also save per-method country and covered-system hourly facility power as "
            "compressed CSV files. Ramping summary CSVs are always saved unless --no-save."
        ),
    )
    parser.add_argument(
        "--ramping-trace-cycle-hours",
        type=int,
        default=DEFAULT_RAMPING_TRACE_CYCLE_HOURS,
        help=(
            "Repeated workload-trace cycle used to flag artificial seam transitions; "
            "set to 0 to disable seam flags. Default: 4440."
        ),
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def _transmission_label(args) -> str:
    if args.azure_transmission_policy:
        return str(args.azure_transmission_policy)
    if args.policy_transmission_matrix:
        return Path(args.policy_transmission_matrix).stem
    return "none"


def _ramping_output_suffix(args, results: dict) -> str:
    tag = "-".join([scenario.replace(" ", "") for scenario in args.scenarios]) or "None"
    transmission_policy = _transmission_label(args)
    annual = results.get("annual_summary", pd.DataFrame())
    if not annual.empty and "transmission_policy" in annual:
        values = annual["transmission_policy"].dropna().astype(str)
        if not values.empty:
            transmission_policy = values.iloc[0]
    suffix = f"{args.policy}_{args.constraint}_{tag}"
    if transmission_policy != "none":
        suffix = f"{suffix}_{transmission_policy}"
    return suffix


def _save_ramping_outputs(args, results: dict, collector: _RampingCollector) -> None:
    if args.no_save:
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _ramping_output_suffix(args=args, results=results)
    transmission_policy = _transmission_label(args)

    country = pd.DataFrame(collector.country_records)
    system = pd.DataFrame(collector.system_records)

    annual = results.get("annual_summary", pd.DataFrame())
    context_columns = [
        "scenario",
        "year",
        "method",
        "solver",
        "solver_status",
        "solver_message",
        "solve_mode",
        "optimization_windows",
        "migrated_training_share_pct",
        "migrated_offline_inference_share_pct",
        "migrated_flexible_workload_share_pct",
        "offline_inference_share_of_inference",
        "training_delay_hours",
        "offline_inference_delay_hours",
        "carbon_reduction_tco2",
        "facility_energy_reduction_mwh",
    ]
    available_context = [column for column in context_columns if column in annual.columns]
    if {"scenario", "year", "method"}.issubset(available_context):
        context = annual[available_context].drop_duplicates(
            subset=["scenario", "year", "method"], keep="last"
        )
        if "carbon_reduction_pct" in annual.columns:
            context["carbon_reduction_percent"] = (
                pd.to_numeric(annual.loc[context.index, "carbon_reduction_pct"], errors="coerce")
                * 100.0
            )
        if "facility_energy_reduction_pct" in annual.columns:
            context["facility_energy_reduction_percent"] = (
                pd.to_numeric(
                    annual.loc[context.index, "facility_energy_reduction_pct"],
                    errors="coerce",
                )
                * 100.0
            )
        if "solver_message" in context.columns:
            context["terminal_fallback_used"] = context["solver_message"].fillna("").str.contains(
                r"terminal.*fallback", case=False, regex=True
            )
        if not country.empty:
            country = country.merge(context, on=["scenario", "year", "method"], how="left")
        if not system.empty:
            system = system.merge(context, on=["scenario", "year", "method"], how="left")

    for frame in (country, system):
        if frame.empty:
            continue
        frame.insert(0, "renewable_energy_policy", args.policy)
        frame.insert(1, "migration_constraint", args.constraint)
        frame.insert(2, "transmission_policy", transmission_policy)

    country_path = output_dir / f"Training_Migration_Country_Ramping_Summary_{suffix}.csv"
    system_path = output_dir / f"Training_Migration_System_Ramping_Summary_{suffix}.csv"
    country.to_csv(country_path, index=False)
    system.to_csv(system_path, index=False)
    print(f"Country ramping summary saved to: {country_path.resolve()}")
    print(f"System ramping summary saved to: {system_path.resolve()}")


def main():
    args = parse_args()
    if args.ramping_trace_cycle_hours < 0:
        raise ValueError("--ramping-trace-cycle-hours must be non-negative.")
    if args.delay_hours < 0 or args.offline_inference_delay_hours < 0:
        raise ValueError("Task-specific delay windows must be non-negative.")
    if args.workload_cache_workers <= 0:
        raise ValueError("--workload-cache-workers must be positive.")
    if args.prepare_workload_cache_only:
        if args.no_workload_cache:
            raise ValueError(
                "--prepare-workload-cache-only cannot be combined with --no-workload-cache."
            )
        _prepare_workload_profiles(
            workload_profile_path=Path(args.workload_path),
            server_profile_path=(
                Path(args.server_profile_path) if args.server_profile_path else None
            ),
            network_profile_path=(
                Path(args.network_profile_path) if args.network_profile_path else None
            ),
            capacity_quantile=args.capacity_quantile,
            max_intervals=args.max_intervals,
            workload_cache_dir=Path(args.workload_cache_dir),
            refresh_workload_cache=args.refresh_workload_cache,
            workload_cache_workers=args.workload_cache_workers,
            verbose=True,
        )
        print("827 workload caches are ready for parallel runs.")
        return
    linprog_options = {}
    if args.linprog_time_limit is not None:
        linprog_options["time_limit"] = args.linprog_time_limit

    run_label = "_".join(
        [
            _slug(args.policy),
            _slug(args.constraint),
            _slug(_transmission_label(args)),
        ]
    )
    hourly_output_dir = (
        Path(args.output_dir) / "Training_Migration_Ramping_Hourly_Power"
        if args.save_ramping_hourly_power and not args.no_save
        else None
    )
    collector = _RampingCollector(
        trace_cycle_hours=args.ramping_trace_cycle_hours,
        commit_hours=args.commit_hours,
        hourly_output_dir=hourly_output_dir,
        run_label=run_label,
    )

    original_evaluate = opt._evaluate_country_hourly_carbon
    original_country_records = opt._country_records

    def evaluate_with_ramping_capacity(
        country_it_mw,
        country_resource_load,
        resource_capacities,
        pue,
        hourly_emission_kg_per_mwh,
        hardware_config,
        max_resource_utilization,
    ):
        evaluation = original_evaluate(
            country_it_mw=country_it_mw,
            country_resource_load=country_resource_load,
            resource_capacities=resource_capacities,
            pue=pue,
            hourly_emission_kg_per_mwh=hourly_emission_kg_per_mwh,
            hardware_config=hardware_config,
            max_resource_utilization=max_resource_utilization,
        )
        evaluation["_ramping_facility_power_capacity_mw"] = (
            np.asarray(country_it_mw, dtype=np.float64)
            * np.asarray(pue, dtype=np.float64)
        )
        return evaluation

    def country_records_with_ramping(
        scenario,
        year,
        method,
        countries,
        baseline,
        optimized,
        baseline_task_metrics=None,
        optimized_task_metrics=None,
    ):
        records = original_country_records(
            scenario=scenario,
            year=year,
            method=method,
            countries=countries,
            baseline=baseline,
            optimized=optimized,
            baseline_task_metrics=baseline_task_metrics,
            optimized_task_metrics=optimized_task_metrics,
        )
        collector.capture(
            scenario=scenario,
            year=year,
            method=method,
            countries=countries,
            baseline=baseline,
            optimized=optimized,
        )
        return records

    opt._evaluate_country_hourly_carbon = evaluate_with_ramping_capacity
    opt._country_records = country_records_with_ramping
    try:
        if args.disable_offline_inference_migration:
            print(
                "Offline-inference migration disabled: running the original "
                "8.182 training-only workflow."
            )
            results = opt.run_training_migration_optimization(
                renewable_energy_policy=args.policy,
                scenarios=args.scenarios,
                years=args.years,
                countries=args.countries,
                workload_profile_path=Path(args.workload_path),
                server_profile_path=(
                    Path(args.server_profile_path) if args.server_profile_path else None
                ),
                network_profile_path=(
                    Path(args.network_profile_path) if args.network_profile_path else None
                ),
                year_start=args.year_start,
                output_dir=args.output_dir,
                save_outputs=not args.no_save and not args.ramping_only,
                execution_policy=args.execution_policy,
                inference_origin_fraction=args.inference_origin_fraction,
                other_origin_fraction=args.other_origin_fraction,
                capacity_quantile=args.capacity_quantile,
                max_resource_utilization=args.max_resource_utilization,
                hourly_carbon_factors_dir=Path(args.hourly_carbon_dir),
                hourly_carbon_scope=args.hourly_carbon_scope,
                hourly_carbon_fallback_to_annual=not args.strict_hourly_carbon,
                delay_hours=args.delay_hours,
                migration_constraint=args.constraint,
                save_schedule=args.save_schedule,
                save_analysis_outputs=not args.no_analysis_outputs,
                max_intervals=args.max_intervals,
                workload_cache_dir=(
                    None if args.no_workload_cache else Path(args.workload_cache_dir)
                ),
                refresh_workload_cache=args.refresh_workload_cache,
                solve_mode=args.solve_mode,
                commit_hours=args.commit_hours,
                monolithic_variable_limit=args.monolithic_variable_limit,
                solver=args.solver,
                linprog_options=linprog_options or None,
                policy_transmission_matrix=(
                    Path(args.policy_transmission_matrix)
                    if args.policy_transmission_matrix
                    else None
                ),
                azure_transmission_policy=args.azure_transmission_policy,
                azure_transmission_dir=Path(args.azure_transmission_dir),
                policy_block_cost=args.policy_block_cost,
                carbon_weight=args.carbon_weight,
                tx_weight=args.tx_weight,
                include_transmission_cost=args.include_transmission_cost,
                objective_normalization=args.objective_normalization,
                scheduling_methods=args.methods,
                verbose=True,
            )
        else:
            print(
                "Flexible workload enabled: training delay "
                f"{args.delay_hours} h; offline-inference delay "
                f"{args.offline_inference_delay_hours} h; offline inference is read "
                "directly from job_type_public labels."
            )
            results = _run_flexible_workload_migration_optimization(
                renewable_energy_policy=args.policy,
                scenarios=args.scenarios,
                years=args.years,
                countries=args.countries,
                workload_profile_path=Path(args.workload_path),
                server_profile_path=(
                    Path(args.server_profile_path) if args.server_profile_path else None
                ),
                network_profile_path=(
                    Path(args.network_profile_path) if args.network_profile_path else None
                ),
                year_start=args.year_start,
                output_dir=args.output_dir,
                save_outputs=not args.no_save and not args.ramping_only,
                execution_policy=args.execution_policy,
                inference_origin_fraction=args.inference_origin_fraction,
                other_origin_fraction=args.other_origin_fraction,
                capacity_quantile=args.capacity_quantile,
                max_resource_utilization=args.max_resource_utilization,
                hourly_carbon_factors_dir=Path(args.hourly_carbon_dir),
                hourly_carbon_scope=args.hourly_carbon_scope,
                hourly_carbon_fallback_to_annual=not args.strict_hourly_carbon,
                training_delay_hours=args.delay_hours,
                offline_inference_delay_hours=args.offline_inference_delay_hours,
                migration_constraint=args.constraint,
                save_schedule=args.save_schedule,
                save_analysis_outputs=not args.no_analysis_outputs,
                max_intervals=args.max_intervals,
                workload_cache_dir=(
                    None if args.no_workload_cache else Path(args.workload_cache_dir)
                ),
                refresh_workload_cache=args.refresh_workload_cache,
                workload_cache_workers=args.workload_cache_workers,
                solve_mode=args.solve_mode,
                commit_hours=args.commit_hours,
                monolithic_variable_limit=args.monolithic_variable_limit,
                solver=args.solver,
                linprog_options=linprog_options or None,
                policy_transmission_matrix=(
                    Path(args.policy_transmission_matrix)
                    if args.policy_transmission_matrix
                    else None
                ),
                azure_transmission_policy=args.azure_transmission_policy,
                azure_transmission_dir=Path(args.azure_transmission_dir),
                policy_block_cost=args.policy_block_cost,
                objective_normalization=args.objective_normalization,
                scheduling_methods=args.methods,
                baseline_scheduling_mode=args.baseline_scheduling_mode,
                verbose=True,
            )
    finally:
        opt._evaluate_country_hourly_carbon = original_evaluate
        opt._country_records = original_country_records

    _save_ramping_outputs(args=args, results=results, collector=collector)

    print()
    print("Aggregate optimization summary:")
    print(
        results["annual_summary"].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


if __name__ == "__main__":
    main()
