import os
import re
import sys
import json
import pickle
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.workload_component_model import (  # noqa: E402
    COMPONENTS,
    DATA_YEAR_START,
    RESOURCES,
    SCENARIO_COL_MAP,
    HardwarePowerConfig,
    _align_interval_energy_to_target_hours,
    _component_full_power,
    _component_power_timeseries,
    _load_hourly_carbon_factors,
    _normalize_weights,
    _policy_factors,
    _resource_capacity,
    _scale_workload_to_capacity,
    _standard_hourly_index,
)
from core.workload_training_migration_analysis_outputs import (  # noqa: E402
    capacity_binding_summary_records,
    country_component_energy_records,
    country_resource_capacity_records,
)
from core.workload_training_migration_methods import (  # noqa: E402
    DEFAULT_SCHEDULING_METHODS,
    LINEAR_PROGRAMMING_METHOD,
    SCHEDULING_METHODS,
    solve_training_migration_baseline_method,
)
from dataset import Factors as factor_data  # noqa: E402
from dataset.Factors import PUE  # noqa: E402
from dataset.Installed_capacity_data import (  # noqa: E402
    countries as DEFAULT_COUNTRIES,
    it_capacity,
    it_ratio,
    total_ratio,
)


TASK_TYPES = ("training", "inference", "other", "unclassified")
CLASSIFIED_TASK_TYPES = tuple(task_type for task_type in TASK_TYPES if task_type != "unclassified")
ALIBABA_2026_POD_TABLE = "asi_opensource_pod_hourly"
ALIBABA_2026_SERVER_TABLE = "asi_opensource_server_hourly"
ALIBABA_2026_PARTITION_RE = re.compile(r"day=(?P<day>\d+)/hour=(?P<hour>\d+)")
ALIBABA_2026_JOB_TYPE_START_DAY = 109
ALIBABA_2026_JOB_TYPE_END_DAY = 184

POD_PROFILE_COLUMNS = (
    "state_public",
    "job_type_public",
    "gpu_request",
    "gpu_mem_request",
    "used_gpu_hours",
    "avg_gpu_sm_util",
    "avg_gpu_mem_gib",
    "cpu_request_cores",
    "avg_cpu_request_util",
    "avg_memory_util",
)
SERVER_PROFILE_COLUMNS = ("gpu_count", "cpu_capacity_cores")

# Calibrated in the referenced GitHub repository from IEA (2026), "Key
# Questions on Energy and AI", Figure 1.5. The factors convert the all-data-
# centre IT-capacity series to AI-focused data-centre capacity.
DEFAULT_AI_CAPACITY_FACTORS = {
    2025: 0.388349425,
    2026: 0.434657544,
    2027: 0.480929326,
    2028: 0.526529347,
    2029: 0.565874480,
    2030: 0.597655819,
}

EM_CPNDCNZ_DIR = ROOT_DIR / "dataset" / "EM-CPNDCNZ"
AZURE_TRANSMISSION_DIR = ROOT_DIR / "dataset" / "Azure"
AZURE_TRANSMISSION_POLICY_FILES = {
    "global_transfer_with_safeguards": "azure_global_transfer_with_safeguards.csv",
    "eea_only_regional_residency": "azure_eea_only_regional_residency.csv",
    "eea_adequacy_trusted_zone": "azure_eea_adequacy_trusted_zone.csv",
}
AZURE_TRANSMISSION_POLICIES = tuple(AZURE_TRANSMISSION_POLICY_FILES)

_PUBLIC_TASK_TYPE_ALIASES = {
    "training": "training",
    "train": "training",
    "inference": "inference",
    "online_inference": "inference",
    "offline_inference": "inference",
    "other": "other",
    "dev": "other",
    "development": "other",
    "unclassified": "unclassified",
    "unknown": "unclassified",
}


@dataclass(frozen=True)
class Alibaba2026TraceConfig:
    """Controls conversion of the public Alibaba GPU v2026 trace."""

    trace_anchor_year: int = 2001
    include_zero_gpu_pods: bool = True
    active_states: Tuple[str, ...] = ("Running",)
    cap_gpu_sm_by_request: bool = True


@dataclass
class WorkloadProfile:
    interval_index: pd.DatetimeIndex
    interval_hours: float
    load: np.ndarray
    trace_capacity: np.ndarray
    task_counts: np.ndarray
    task_type_summary: pd.DataFrame
    relative_time: bool = True
    trace_capacity_source: Tuple[str, ...] = ()

    @property
    def n_intervals(self) -> int:
        return len(self.interval_index)

    @property
    def trace_hours(self) -> float:
        return self.n_intervals * self.interval_hours

    @property
    def annualization_factor_8760(self) -> float:
        if self.trace_hours <= 0:
            raise ValueError("Workload trace must contain at least one positive-duration interval.")
        return 8760.0 / self.trace_hours


EUROPEAN_COUNTRIES = (
    "France",
    "Germany",
    "United_Kingdom",
    "Italy",
    "Ireland",
    "Netherlands",
    "Spain",
    "Sweden",
    "Belgium",
    "Norway",
    "Poland",
    "Switzerland",
)

MIGRATION_CONSTRAINTS = ("global", "europe_only")
SOLVE_MODES = ("auto", "monolithic", "windowed")
SOLVERS = ("scipy", "gurobi")
OBJECTIVE_NORMALIZATIONS = ("benchmark", "none")
CAPACITY_TOLERANCE = 1e-3
DEFAULT_POLICY_BLOCK_COST = 1e9
WORKLOAD_PROFILE_CACHE_VERSION = "alibaba-gpu-v2026-workload-profile-carbon-only-v1"


def _print_progress(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[training-migration] {message}", flush=True)


def _progress_interval(total: int) -> int:
    return max(1, (total + 9) // 10)


def _path_cache_value(path: Optional[Union[str, Path]]) -> str:
    if path is None:
        return ""
    return str(Path(path).expanduser().resolve())


def _trace_config_cache_dict(trace_config: Alibaba2026TraceConfig) -> Dict[str, object]:
    return {
        "trace_anchor_year": int(trace_config.trace_anchor_year),
        "include_zero_gpu_pods": bool(trace_config.include_zero_gpu_pods),
        "active_states": list(trace_config.active_states),
        "cap_gpu_sm_by_request": bool(trace_config.cap_gpu_sm_by_request),
    }


def _cache_digest(cache_key: Mapping[str, object]) -> str:
    encoded = json.dumps(
        cache_key,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_file_path(
    cache_dir: Optional[Union[str, Path]],
    prefix: str,
    cache_key: Mapping[str, object],
) -> Optional[Path]:
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"{prefix}_{_cache_digest(cache_key)[:20]}.pkl"


def _load_pickle_cache(
    cache_path: Optional[Path],
    expected_key: Mapping[str, object],
    verbose: bool,
    label: str,
):
    if cache_path is None or not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        _print_progress(verbose, f"Ignoring unreadable {label} cache {cache_path}: {exc}.")
        return None

    if not isinstance(payload, dict) or payload.get("cache_key") != expected_key:
        _print_progress(verbose, f"Ignoring stale {label} cache {cache_path}.")
        return None
    _print_progress(verbose, f"Loaded {label} cache from {cache_path}.")
    return payload


def _write_pickle_cache(
    cache_path: Optional[Path],
    cache_key: Mapping[str, object],
    payload: Mapping[str, object],
    verbose: bool,
    label: str,
) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    data = dict(payload)
    data["cache_key"] = dict(cache_key)
    with tmp_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)
    _print_progress(verbose, f"Saved {label} cache to {cache_path}.")


def _profile_cache_payload(profile: WorkloadProfile) -> Dict[str, object]:
    return {
        "interval_index": profile.interval_index,
        "interval_hours": float(profile.interval_hours),
        "load": profile.load,
        "trace_capacity": profile.trace_capacity,
        "task_counts": profile.task_counts,
        "task_type_summary": profile.task_type_summary,
        "relative_time": bool(profile.relative_time),
        "trace_capacity_source": tuple(profile.trace_capacity_source),
    }


def _profile_from_cache_payload(payload: Mapping[str, object]) -> WorkloadProfile:
    return WorkloadProfile(
        interval_index=payload["interval_index"],
        interval_hours=float(payload["interval_hours"]),
        load=payload["load"],
        trace_capacity=payload["trace_capacity"],
        task_counts=payload["task_counts"],
        task_type_summary=payload["task_type_summary"],
        relative_time=bool(payload.get("relative_time", True)),
        trace_capacity_source=tuple(payload.get("trace_capacity_source", ())),
    )


def _resolve_ai_capacity_factors(
    year_start: int,
    years: int,
    overrides: Optional[Mapping[int, float]],
) -> Dict[int, float]:
    source = DEFAULT_AI_CAPACITY_FACTORS if overrides is None else overrides
    requested_years = range(year_start, year_start + years)
    missing_years = [year for year in requested_years if year not in source]
    if missing_years:
        raise ValueError(f"Missing AI capacity factors for years: {missing_years}")

    factors = {year: float(source[year]) for year in requested_years}
    invalid = {
        year: factor
        for year, factor in factors.items()
        if not np.isfinite(factor) or not (0 < factor <= 1)
    }
    if invalid:
        raise ValueError(f"AI capacity factors must be finite and in (0, 1], got: {invalid}")
    return factors


def _normalize_task_label(value: object) -> str:
    if pd.isna(value):
        return "unclassified"
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _PUBLIC_TASK_TYPE_ALIASES.get(label, "unclassified")


def _task_type_ids(values: pd.Series) -> np.ndarray:
    id_by_type = {task_type: index for index, task_type in enumerate(TASK_TYPES)}
    return values.map(_normalize_task_label).map(id_by_type).to_numpy(dtype=np.int64)


def _as_task_weight_table(
    countries: Sequence[str],
    task_weights: Optional[Mapping[str, Mapping[str, float]]],
) -> np.ndarray:
    if task_weights is None:
        defaults = {
            "training": it_ratio,
            "inference": total_ratio,
            "other": total_ratio,
            "unclassified": total_ratio,
        }
        return np.stack([_normalize_weights(countries, defaults[task_type]) for task_type in TASK_TYPES])

    table = []
    for task_type in TASK_TYPES:
        if task_type not in task_weights:
            if task_type == "unclassified":
                table.append(_normalize_weights(countries, total_ratio))
                continue
            raise ValueError(f"Missing weights for task type '{task_type}'.")
        table.append(_normalize_weights(countries, task_weights[task_type]))
    return np.stack(table)


def _build_execution_weights(
    countries: Sequence[str],
    country_it_mw: np.ndarray,
    origin_weights: np.ndarray,
    execution_policy: str,
    inference_origin_fraction: float,
    other_origin_fraction: float,
    task_execution_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> np.ndarray:
    if task_execution_weights is not None:
        return _as_task_weight_table(countries, task_execution_weights)

    capacity_weight = country_it_mw / country_it_mw.sum()
    if execution_policy == "capacity":
        return np.repeat(capacity_weight[None, :], len(TASK_TYPES), axis=0)
    if execution_policy == "origin":
        return origin_weights.copy()
    if execution_policy != "hybrid":
        raise ValueError("execution_policy must be one of: capacity, origin, hybrid.")

    if not (0 <= inference_origin_fraction <= 1):
        raise ValueError("inference_origin_fraction must be in [0, 1].")
    if not (0 <= other_origin_fraction <= 1):
        raise ValueError("other_origin_fraction must be in [0, 1].")

    weights = np.zeros_like(origin_weights)
    weights[TASK_TYPES.index("training")] = capacity_weight
    weights[TASK_TYPES.index("inference")] = (
        inference_origin_fraction * origin_weights[TASK_TYPES.index("inference")]
        + (1 - inference_origin_fraction) * capacity_weight
    )
    weights[TASK_TYPES.index("other")] = (
        other_origin_fraction * origin_weights[TASK_TYPES.index("other")]
        + (1 - other_origin_fraction) * capacity_weight
    )
    weights[TASK_TYPES.index("unclassified")] = capacity_weight
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights


def _resolve_alibaba_2026_table_root(base_path: Union[str, Path], table_name: str) -> Path:
    base = Path(base_path)
    table_names = {
        ALIBABA_2026_POD_TABLE,
        ALIBABA_2026_SERVER_TABLE,
    }
    candidates = []
    if base.name in table_names:
        candidates.append(base.parent / table_name)
        if base.name == table_name:
            candidates.insert(0, base)
    candidates.extend(
        [
            base / table_name,
            base / "data" / table_name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else base / table_name


def _alibaba_2026_partition_files(table_root: Path) -> list[tuple[int, int, Path]]:
    rows = []
    for path in sorted(table_root.glob("day=*/hour=*/*.parquet")):
        match = ALIBABA_2026_PARTITION_RE.search(path.as_posix())
        if match:
            rows.append((int(match.group("day")), int(match.group("hour")), path))
    return rows


def _read_parquet_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=list(columns))
    except ImportError as exc:
        raise ImportError(
            "Reading Alibaba GPU v2026 parquet files requires a parquet engine. "
            "Install `pyarrow` before running the 2026 trace reader."
        ) from exc


def _to_numeric_array(df: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)


def _server_trace_capacity(
    server_root: Path,
    n_intervals: int,
    capacity_quantile: float,
    verbose: bool = False,
) -> Optional[np.ndarray]:
    files = _alibaba_2026_partition_files(server_root)
    if not files:
        return None

    files_to_process = [
        (day, hour, path) for day, hour, path in files if day * 24 + hour < n_intervals
    ]
    progress_interval = _progress_interval(len(files_to_process))
    capacity = np.full((2, n_intervals), np.nan, dtype=np.float64)
    for file_index, (day, hour, path) in enumerate(files_to_process, start=1):
        if (
            file_index == 1
            or file_index == len(files_to_process)
            or file_index % progress_interval == 0
        ):
            _print_progress(
                verbose,
                f"Reading server partitions: {file_index}/{len(files_to_process)} "
                f"({file_index / len(files_to_process):.0%})",
            )
        pos = day * 24 + hour
        df = _read_parquet_columns(path, SERVER_PROFILE_COLUMNS)
        gpu_count = pd.to_numeric(df["gpu_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
        cpu_cores = pd.to_numeric(df["cpu_capacity_cores"], errors="coerce").fillna(0.0).clip(lower=0.0)
        capacity[0, pos] = float(cpu_cores.sum())
        capacity[1, pos] = float(gpu_count.sum())

    result = np.full((2,), np.nan, dtype=np.float64)
    for idx in range(2):
        valid = capacity[idx, np.isfinite(capacity[idx]) & (capacity[idx] > 0)]
        if valid.size:
            result[idx] = float(np.quantile(valid, capacity_quantile))
    if not np.all(np.isfinite(result)):
        return None
    return result


def build_workload_profile(
    workload_profile_path: Union[str, Path],
    server_profile_path: Optional[Union[str, Path]] = None,
    network_profile_path: Optional[Union[str, Path]] = None,
    capacity_quantile: float = 0.96,
    trace_config: Optional[Alibaba2026TraceConfig] = None,
    max_intervals: Optional[int] = None,
    workload_cache_dir: Optional[Union[str, Path]] = None,
    refresh_workload_cache: bool = False,
    verbose: bool = False,
) -> WorkloadProfile:
    """
    Convert Alibaba Cluster Trace GPU v2026 pod-hour parquet partitions into
    hourly resource-load series for cross-country scheduling.

    This carbon-only variant uses Alibaba v2026 pod/server resource data only.
    The public server-level network sample is intentionally not read because
    Azure policy matrices are used only to block disallowed country pairs, not
    to price transmission volume.
    """
    if not (0 < capacity_quantile <= 1):
        raise ValueError("capacity_quantile must be in (0, 1].")
    if max_intervals is not None and max_intervals <= 0:
        raise ValueError("max_intervals must be positive when provided.")

    trace_config = trace_config or Alibaba2026TraceConfig()
    if network_profile_path is not None:
        _print_progress(
            verbose,
            "Ignoring network_profile_path because this carbon-only variant does not read Alibaba 2026 traffic data.",
        )
    pod_root = _resolve_alibaba_2026_table_root(workload_profile_path, ALIBABA_2026_POD_TABLE)
    server_root = (
        Path(server_profile_path)
        if server_profile_path is not None
        else _resolve_alibaba_2026_table_root(workload_profile_path, ALIBABA_2026_SERVER_TABLE)
    )
    profile_cache_key = {
        "version": WORKLOAD_PROFILE_CACHE_VERSION,
        "kind": "workload_profile",
        "workload_root": _path_cache_value(workload_profile_path),
        "pod_root": _path_cache_value(pod_root),
        "server_root": _path_cache_value(server_root),
        "capacity_quantile": float(capacity_quantile),
        "max_intervals": None if max_intervals is None else int(max_intervals),
        "trace_config": _trace_config_cache_dict(trace_config),
        "task_types": list(TASK_TYPES),
        "resources": list(RESOURCES),
    }
    profile_cache_path = _cache_file_path(
        workload_cache_dir,
        "workload_profile",
        profile_cache_key,
    )
    if not refresh_workload_cache:
        cached = _load_pickle_cache(
            profile_cache_path,
            profile_cache_key,
            verbose=verbose,
            label="workload profile",
        )
        if cached is not None:
            return _profile_from_cache_payload(cached)

    pod_files = _alibaba_2026_partition_files(pod_root)
    if not pod_files:
        raise FileNotFoundError(
            f"No Alibaba GPU v2026 pod-hour parquet partitions found under {pod_root}. "
            "Expected day=<day>/hour=<hour>/*.parquet."
        )

    max_trace_pos = max(day * 24 + hour for day, hour, _ in pod_files) + 1
    n_intervals = min(max_trace_pos, max_intervals) if max_intervals is not None else max_trace_pos
    pod_files_to_process = [
        (day, hour, path) for day, hour, path in pod_files if day * 24 + hour < n_intervals
    ]
    _print_progress(
        verbose,
        f"Found {len(pod_files_to_process)} pod partitions under {pod_root}; "
        f"building {n_intervals} hourly intervals.",
    )

    interval_hours = 1.0

    interval_index = pd.date_range(
        start=f"{trace_config.trace_anchor_year}-01-01",
        periods=n_intervals,
        freq="h",
        tz="UTC",
    )

    load = np.zeros((len(TASK_TYPES), len(RESOURCES), n_intervals), dtype=np.float64)
    pod_hour_counts = np.zeros((len(TASK_TYPES),), dtype=np.int64)
    source_used_gpu_hours = np.zeros((len(TASK_TYPES),), dtype=np.float64)
    source_gpu_memory_gib_hours = np.zeros((len(TASK_TYPES),), dtype=np.float64)
    paper_validation_used_gpu_hours = np.zeros((len(TASK_TYPES),), dtype=np.float64)

    progress_interval = _progress_interval(len(pod_files_to_process))
    for file_index, (day, hour, path) in enumerate(pod_files_to_process, start=1):
        if (
            file_index == 1
            or file_index == len(pod_files_to_process)
            or file_index % progress_interval == 0
        ):
            _print_progress(
                verbose,
                f"Reading pod partitions: {file_index}/{len(pod_files_to_process)} "
                f"({file_index / len(pod_files_to_process):.0%})",
            )
        pos = day * 24 + hour
        df = _read_parquet_columns(path, POD_PROFILE_COLUMNS)
        if df.empty:
            continue

        mapped_task_type_ids = _task_type_ids(df["job_type_public"])
        state = df["state_public"].fillna("").astype(str)
        gpu_request = _to_numeric_array(df, "gpu_request")
        gpu_mem_request = _to_numeric_array(df, "gpu_mem_request")
        used_gpu_hours = _to_numeric_array(df, "used_gpu_hours")
        gpu_sm = _to_numeric_array(df, "avg_gpu_sm_util")
        gpu_mem = _to_numeric_array(df, "avg_gpu_mem_gib")
        cpu_request = _to_numeric_array(df, "cpu_request_cores")
        cpu_request_util = _to_numeric_array(df, "avg_cpu_request_util")
        memory_util = _to_numeric_array(df, "avg_memory_util")

        if ALIBABA_2026_JOB_TYPE_START_DAY <= day <= ALIBABA_2026_JOB_TYPE_END_DAY:
            paper_valid = (
                np.isfinite(gpu_mem_request)
                & (gpu_mem_request > 0)
                & np.isfinite(used_gpu_hours)
                & (mapped_task_type_ids != TASK_TYPES.index("unclassified"))
            )
            if np.any(paper_valid):
                paper_validation_used_gpu_hours += np.bincount(
                    mapped_task_type_ids[paper_valid],
                    weights=np.maximum(used_gpu_hours[paper_valid], 0.0),
                    minlength=len(TASK_TYPES),
                )

        gpu_request = np.where(np.isfinite(gpu_request), np.maximum(gpu_request, 0.0), 0.0)
        cpu_request = np.where(np.isfinite(cpu_request), np.maximum(cpu_request, 0.0), 0.0)
        gpu_sm = np.where(np.isfinite(gpu_sm), np.maximum(gpu_sm, 0.0), 0.0)
        gpu_mem = np.where(np.isfinite(gpu_mem), np.maximum(gpu_mem, 0.0), 0.0)
        cpu_request_util = np.where(
            np.isfinite(cpu_request_util), np.maximum(cpu_request_util, 0.0), 0.0
        )
        memory_util = np.where(np.isfinite(memory_util), np.maximum(memory_util, 0.0), 0.0)
        used_gpu_hours = np.where(
            np.isfinite(used_gpu_hours), np.maximum(used_gpu_hours, 0.0), 0.0
        )

        active = state.isin(trace_config.active_states).to_numpy()
        active |= used_gpu_hours > 0
        active |= gpu_sm > 0
        active |= cpu_request_util > 0
        active &= ~state.eq("Standby").to_numpy()
        if not trace_config.include_zero_gpu_pods:
            active &= gpu_request > 0
        if not np.any(active):
            continue

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
        if trace_config.cap_gpu_sm_by_request:
            gpu_sm_equivalent = np.minimum(
                gpu_sm_equivalent,
                np.where(gpu_request > 0, gpu_request, gpu_sm_equivalent),
            )
        memory_activity_proxy = cpu_request * memory_util
        storage_activity = np.zeros_like(cpu_used_cores)
        resource_values = (
            cpu_used_cores,
            gpu_sm_equivalent,
            memory_activity_proxy,
            storage_activity,
        )

        pod_hour_counts += np.bincount(mapped_task_type_ids, minlength=len(TASK_TYPES))
        source_used_gpu_hours += np.bincount(
            mapped_task_type_ids, weights=used_gpu_hours, minlength=len(TASK_TYPES)
        )
        source_gpu_memory_gib_hours += np.bincount(
            mapped_task_type_ids, weights=gpu_mem, minlength=len(TASK_TYPES)
        )
        for resource_id, values in enumerate(resource_values):
            load[:, resource_id, pos] += np.bincount(
                mapped_task_type_ids, weights=values, minlength=len(TASK_TYPES)
            )

    total_load = load.sum(axis=0)
    trace_capacity = np.ones((len(RESOURCES),), dtype=np.float64)
    capacity_source = ["pod_load_quantile"] * len(RESOURCES)

    if server_root.exists():
        _print_progress(verbose, f"Reading server inventory from {server_root}.")
        server_capacity = _server_trace_capacity(
            server_root=server_root,
            n_intervals=n_intervals,
            capacity_quantile=capacity_quantile,
            verbose=verbose,
        )
    else:
        _print_progress(
            verbose,
            f"Server inventory not found under {server_root}; using pod-load capacity estimates.",
        )
        server_capacity = None
    if server_capacity is not None:
        trace_capacity[RESOURCES.index("cpu")] = server_capacity[0]
        trace_capacity[RESOURCES.index("gpu")] = server_capacity[1]
        capacity_source[RESOURCES.index("cpu")] = "server_inventory_quantile"
        capacity_source[RESOURCES.index("gpu")] = "server_inventory_quantile"

    for resource in ("cpu", "gpu", "memory"):
        resource_id = RESOURCES.index(resource)
        if capacity_source[resource_id] == "server_inventory_quantile":
            continue
        values = total_load[resource_id]
        positive = values[np.isfinite(values) & (values > 0)]
        trace_capacity[resource_id] = float(np.quantile(positive, capacity_quantile)) if positive.size else 1.0

    trace_capacity[RESOURCES.index("storage")] = 1.0
    capacity_source[RESOURCES.index("storage")] = "not_observed_idle_only"
    capacity_source[RESOURCES.index("memory")] = "cpu_weighted_memory_util_proxy"
    trace_capacity = np.maximum(trace_capacity, 1e-12)

    resource_hours = load.sum(axis=2) * interval_hours
    annualization = 8760.0 / (n_intervals * interval_hours)
    total_pod_hour_rows = int(pod_hour_counts.sum())
    total_source_used_gpu_hours = float(source_used_gpu_hours.sum())
    classified_source_used_gpu_hours = float(
        source_used_gpu_hours[: TASK_TYPES.index("unclassified")].sum()
    )
    paper_validation_total = float(paper_validation_used_gpu_hours.sum())

    summary_records = []
    for task_type_id, task_type in enumerate(TASK_TYPES):
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
                "source_used_gpu_hours": float(source_used_gpu_hours[task_type_id]),
                "source_used_gpu_hours_share_all": (
                    float(source_used_gpu_hours[task_type_id]) / total_source_used_gpu_hours
                    if total_source_used_gpu_hours > 0
                    else 0.0
                ),
                "source_used_gpu_hours_share_classified": (
                    float(source_used_gpu_hours[task_type_id]) / classified_source_used_gpu_hours
                    if task_type != "unclassified" and classified_source_used_gpu_hours > 0
                    else np.nan
                ),
                "paper_validation_used_gpu_hours": float(
                    paper_validation_used_gpu_hours[task_type_id]
                ),
                "paper_validation_used_gpu_hours_share_classified": (
                    float(paper_validation_used_gpu_hours[task_type_id]) / paper_validation_total
                    if task_type != "unclassified" and paper_validation_total > 0
                    else np.nan
                ),
                "source_gpu_memory_gib_hours": float(source_gpu_memory_gib_hours[task_type_id]),
                "cpu_core_hours": float(resource_hours[task_type_id, RESOURCES.index("cpu")]),
                "gpu_sm_equivalent_hours": float(resource_hours[task_type_id, RESOURCES.index("gpu")]),
                "memory_proxy_hours": float(resource_hours[task_type_id, RESOURCES.index("memory")]),
                "annualized_cpu_core_hours": float(
                    resource_hours[task_type_id, RESOURCES.index("cpu")] * annualization
                ),
                "annualized_gpu_sm_equivalent_hours": float(
                    resource_hours[task_type_id, RESOURCES.index("gpu")] * annualization
                ),
            }
        )

    profile = WorkloadProfile(
        interval_index=interval_index,
        interval_hours=interval_hours,
        load=load,
        trace_capacity=trace_capacity,
        task_counts=pod_hour_counts,
        task_type_summary=pd.DataFrame(summary_records),
        relative_time=True,
        trace_capacity_source=tuple(capacity_source),
    )
    _write_pickle_cache(
        profile_cache_path,
        profile_cache_key,
        _profile_cache_payload(profile),
        verbose=verbose,
        label="workload profile",
    )
    return profile


def _tile_relative_interval_energy_to_target_hours(
    interval_energy: np.ndarray,
    interval_index: pd.DatetimeIndex,
    target_timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    original_shape = interval_energy.shape
    flat = interval_energy.reshape(-1, original_shape[-1])
    source_hours = pd.DatetimeIndex(interval_index).floor("h")
    hourly = pd.DataFrame(flat.T, index=source_hours).groupby(level=0, sort=True).sum()
    hourly_values = hourly.to_numpy(dtype=np.float64).T
    if hourly_values.shape[1] == 0:
        raise ValueError("Relative workload trace must contain at least one hourly interval.")

    target_hours = len(target_timestamps)
    repeat_count = int(np.ceil(target_hours / hourly_values.shape[1]))
    aligned = np.tile(hourly_values, repeat_count)[:, :target_hours]
    return aligned.reshape(original_shape[:-1] + (target_hours,))


def _hourly_load_from_intervals(
    interval_load: np.ndarray,
    interval_index: pd.DatetimeIndex,
    interval_hours: float,
    target_timestamps: pd.DatetimeIndex,
    relative_trace: bool = False,
) -> np.ndarray:
    resource_hours = interval_load * interval_hours
    if relative_trace:
        return _tile_relative_interval_energy_to_target_hours(
            interval_energy=resource_hours,
            interval_index=interval_index,
            target_timestamps=target_timestamps,
        )
    return _align_interval_energy_to_target_hours(
        interval_energy_mwh=resource_hours,
        interval_index=interval_index,
        target_timestamps=target_timestamps,
    )


def _load_em_cpndcnz_annual_emission_factors(
    hourly_carbon_factors_dir: Optional[Union[str, Path]],
    renewable_energy_policy: str,
):
    """Load the shared annual factors; the directory argument selects hourly CSVs only."""
    attr = f"CF_{renewable_energy_policy}_electricity_map"
    if not hasattr(factor_data, attr):
        raise FileNotFoundError(f"Missing annual carbon factors in dataset/Factors.py: {attr}")
    # Preserve the fresh dictionaries/lists returned by the former file loader.
    factors = getattr(factor_data, attr)
    return {country: list(values) for country, values in factors.items()}, 2017


def _resolve_em_cpndcnz_dir(hourly_carbon_factors_dir: Optional[Union[str, Path]]) -> Path:
    candidates = []
    if hourly_carbon_factors_dir is not None:
        requested = Path(hourly_carbon_factors_dir)
        if requested.name == EM_CPNDCNZ_DIR.name:
            candidates.append(requested)
        candidates.append(requested / EM_CPNDCNZ_DIR.name)
    candidates.append(EM_CPNDCNZ_DIR)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Alibaba 2026 carbon-factor runs must read dataset/EM-CPNDCNZ. "
        f"Searched: {searched}"
    )


def _hourly_emission_factors(
    countries: Sequence[str],
    renewable_energy_policy: str,
    year: int,
    year_idx: int,
    hourly_carbon_factors_dir: Optional[Union[str, Path]],
    hourly_carbon_scope: str,
    hourly_carbon_fallback_to_annual: bool,
):
    em_cpndcnz_dir = _resolve_em_cpndcnz_dir(hourly_carbon_factors_dir)
    emission_factors, emission_factor_year_start = _load_em_cpndcnz_annual_emission_factors(
        hourly_carbon_factors_dir=em_cpndcnz_dir,
        renewable_energy_policy=renewable_energy_policy,
    )
    annual_year_idx = year - emission_factor_year_start
    annual = np.array(
        [emission_factors[country][annual_year_idx] for country in countries],
        dtype=np.float64,
    )

    return _load_hourly_carbon_factors(
        countries=countries,
        renewable_energy_policy=renewable_energy_policy,
        year=year,
        year_idx=annual_year_idx,
        annual_emission_factors=emission_factors,
        hourly_carbon_factors_dir=em_cpndcnz_dir,
        hourly_carbon_scope=hourly_carbon_scope,
        hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
    )


def _load_transmission_policy_matrix(
    countries: Sequence[str],
    policy_transmission_matrix: Optional[Union[str, Path]],
    azure_transmission_policy: Optional[str],
    azure_transmission_dir: Union[str, Path],
    policy_block_cost: float,
):
    n_countries = len(countries)
    transmission_matrix_path = None
    transmission_policy_name = None

    if policy_transmission_matrix is not None:
        transmission_matrix_path = Path(policy_transmission_matrix)
        transmission_policy_name = transmission_matrix_path.stem
    elif azure_transmission_policy is not None:
        normalized_policy = str(azure_transmission_policy).strip().lower().replace("-", "_")
        if normalized_policy.startswith("azure_"):
            normalized_policy = normalized_policy.removeprefix("azure_")
        if normalized_policy not in AZURE_TRANSMISSION_POLICY_FILES:
            raise ValueError(
                f"Unknown azure_transmission_policy '{azure_transmission_policy}'. "
                f"Allowed: {list(AZURE_TRANSMISSION_POLICIES)}."
            )
        transmission_matrix_path = (
            Path(azure_transmission_dir) / AZURE_TRANSMISSION_POLICY_FILES[normalized_policy]
        )
        transmission_policy_name = normalized_policy

    if transmission_matrix_path is None:
        transmission_policy_name = "none"
        return (
            np.zeros((n_countries, n_countries), dtype=np.float64),
            np.ones((n_countries, n_countries), dtype=bool),
            None,
            transmission_policy_name,
        )

    matrix_path = transmission_matrix_path
    if not matrix_path.exists():
        raise FileNotFoundError(f"Transmission policy matrix not found: {matrix_path}")
    raw = pd.read_csv(matrix_path, index_col=0)
    missing_rows = [country for country in countries if country not in raw.index]
    missing_cols = [country for country in countries if country not in raw.columns]
    if missing_rows or missing_cols:
        raise ValueError(
            "Policy transmission matrix must contain all selected countries. "
            f"Missing rows={missing_rows}, missing columns={missing_cols}."
        )

    values = raw.loc[list(countries), list(countries)].apply(pd.to_numeric, errors="coerce")
    policy_matrix = values.to_numpy(dtype=np.float64)
    if np.any(np.isnan(policy_matrix)):
        raise ValueError("Policy transmission matrix contains NaN or non-numeric values.")
    if np.any(policy_matrix < 0):
        raise ValueError("Policy transmission matrix values must be non-negative.")

    allowed_mask = np.isfinite(policy_matrix) & (policy_matrix < policy_block_cost)
    np.fill_diagonal(policy_matrix, 0.0)
    np.fill_diagonal(allowed_mask, True)
    policy_matrix = np.where(allowed_mask, policy_matrix, 0.0)
    return policy_matrix, allowed_mask, matrix_path, transmission_policy_name


def _apply_migration_constraint_to_allowed_mask(
    allowed_mask: np.ndarray,
    migration_constraint: str,
    europe_ids: np.ndarray,
) -> np.ndarray:
    constrained = np.asarray(allowed_mask, dtype=bool).copy()
    if migration_constraint == "europe_only":
        europe_destination_mask = np.zeros((constrained.shape[0],), dtype=bool)
        europe_destination_mask[europe_ids] = True
        constrained &= europe_destination_mask[None, :]
        np.fill_diagonal(constrained, True)
    return constrained


def _active_source_country_hours(source_country_training_load: np.ndarray):
    active = np.any(source_country_training_load > 1e-12, axis=1)
    source_country_ids, source_hour_ids = np.nonzero(active)
    return source_country_ids.astype(np.int32), source_hour_ids.astype(np.int32)


def _marginal_it_mwh_per_resource_hour(
    country_it_mw: np.ndarray,
    fixed_resource_load: np.ndarray,
    resource_capacities: np.ndarray,
    config: HardwarePowerConfig,
) -> np.ndarray:
    n_countries, _, n_hours = fixed_resource_load.shape
    coeff = np.zeros((n_countries, len(RESOURCES), n_hours), dtype=np.float64)
    component_full_mw = _component_full_power(country_it_mw, config)
    utilization = np.clip(fixed_resource_load / resource_capacities[:, :, None], 0.0, 1.0)

    cpu_id = RESOURCES.index("cpu")
    gpu_id = RESOURCES.index("gpu")
    memory_id = RESOURCES.index("memory")
    storage_id = RESOURCES.index("storage")

    cpu_idle = config.cpu_idle_power_w_per_core / config.cpu_full_power_w_per_core
    gpu_idle = config.gpu_idle_power_w / config.gpu_full_power_w

    coeff[:, cpu_id, :] += (
        component_full_mw[:, 0, None]
        * (1 - cpu_idle)
        / resource_capacities[:, cpu_id, None]
    )
    coeff[:, gpu_id, :] += (
        component_full_mw[:, 1, None]
        * (1 - gpu_idle)
        / ((1 + utilization[:, gpu_id, :]) * np.log(2.0))
        / resource_capacities[:, gpu_id, None]
    )
    coeff[:, memory_id, :] += (
        component_full_mw[:, 2, None]
        * (1 - config.memory_idle_fraction)
        / resource_capacities[:, memory_id, None]
    )
    coeff[:, storage_id, :] += (
        component_full_mw[:, 3, None]
        * (1 - config.storage_idle_fraction)
        / resource_capacities[:, storage_id, None]
    )

    fan_weight_sum = config.fan_cpu_weight + config.fan_gpu_weight + config.fan_memory_weight
    heat_load = (
        config.fan_cpu_weight * utilization[:, cpu_id, :]
        + config.fan_gpu_weight * utilization[:, gpu_id, :]
        + config.fan_memory_weight * utilization[:, memory_id, :]
    ) / fan_weight_sum
    fan_slope = component_full_mw[:, 4, None] * (1 - config.it_fan_idle_fraction) * 3 * heat_load**2
    coeff[:, cpu_id, :] += fan_slope * (
        config.fan_cpu_weight / fan_weight_sum / resource_capacities[:, cpu_id, None]
    )
    coeff[:, gpu_id, :] += fan_slope * (
        config.fan_gpu_weight / fan_weight_sum / resource_capacities[:, gpu_id, None]
    )
    coeff[:, memory_id, :] += fan_slope * (
        config.fan_memory_weight / fan_weight_sum / resource_capacities[:, memory_id, None]
    )
    return coeff


def _evaluate_country_hourly_carbon(
    country_it_mw: np.ndarray,
    country_resource_load: np.ndarray,
    resource_capacities: np.ndarray,
    pue: np.ndarray,
    hourly_emission_kg_per_mwh: np.ndarray,
    hardware_config: HardwarePowerConfig,
    max_resource_utilization: float,
) -> Dict[str, np.ndarray]:
    utilization = np.clip(
        country_resource_load / resource_capacities[:, :, None],
        0.0,
        max_resource_utilization,
    )
    component_power_mw = _component_power_timeseries(
        country_it_mw=country_it_mw,
        resource_utilization=utilization,
        config=hardware_config,
    )
    it_energy_mwh = component_power_mw.sum(axis=0)
    facility_energy_mwh = it_energy_mwh * pue[:, None]
    carbon_tco2 = facility_energy_mwh * hourly_emission_kg_per_mwh / 1000.0
    capacity_limit = resource_capacities[:, :, None] * max_resource_utilization
    overflow = np.maximum(country_resource_load - capacity_limit, 0.0)
    return {
        "it_energy_mwh": it_energy_mwh,
        "facility_energy_mwh": facility_energy_mwh,
        "carbon_tco2": carbon_tco2,
        "utilization": utilization,
        "overflow": overflow,
    }


def _task_country_resource_load(
    country_type_load: np.ndarray,
    training_load: np.ndarray,
) -> np.ndarray:
    task_load = country_type_load.copy()
    task_load[TASK_TYPES.index("training")] = training_load
    return task_load


def _allocate_task_energy_carbon(
    country_it_mw: np.ndarray,
    task_country_resource_load: np.ndarray,
    resource_capacities: np.ndarray,
    pue: np.ndarray,
    hourly_emission_kg_per_mwh: np.ndarray,
    hardware_config: HardwarePowerConfig,
    max_resource_utilization: float,
) -> Dict[str, np.ndarray]:
    country_resource_load = task_country_resource_load.sum(axis=0)
    utilization = np.clip(
        country_resource_load / resource_capacities[:, :, None],
        0.0,
        max_resource_utilization,
    )
    component_power_mw = _component_power_timeseries(
        country_it_mw=country_it_mw,
        resource_utilization=utilization,
        config=hardware_config,
    )

    n_tasks, n_countries, _, n_hours = task_country_resource_load.shape
    task_facility_energy_mwh = np.zeros((n_tasks, n_countries), dtype=np.float64)
    task_carbon_tco2 = np.zeros((n_tasks, n_countries), dtype=np.float64)
    component_driver_resource = {
        "cpu": "cpu",
        "gpu": "gpu",
        "memory": "memory",
        "storage": "storage",
    }

    for component_id, component in enumerate(COMPONENTS):
        if component == "it_fan":
            cpu = task_country_resource_load[:, :, RESOURCES.index("cpu"), :] / resource_capacities[
                None, :, RESOURCES.index("cpu"), None
            ]
            gpu = task_country_resource_load[:, :, RESOURCES.index("gpu"), :] / resource_capacities[
                None, :, RESOURCES.index("gpu"), None
            ]
            memory = task_country_resource_load[:, :, RESOURCES.index("memory"), :] / resource_capacities[
                None, :, RESOURCES.index("memory"), None
            ]
            driver = (
                hardware_config.fan_cpu_weight * cpu
                + hardware_config.fan_gpu_weight * gpu
                + hardware_config.fan_memory_weight * memory
            )
        else:
            resource_id = RESOURCES.index(component_driver_resource[component])
            driver = task_country_resource_load[:, :, resource_id, :]

        for country_id in range(n_countries):
            local_driver = driver[:, country_id, :]
            total_driver = local_driver.sum(axis=0)
            annual_driver = local_driver.sum(axis=1)
            if annual_driver.sum() > 0:
                fallback = annual_driver / annual_driver.sum()
            else:
                fallback = np.full((n_tasks,), 1.0 / n_tasks, dtype=np.float64)

            shares = np.repeat(fallback[:, None], n_hours, axis=1)
            active = total_driver > 0
            shares[:, active] = local_driver[:, active] / total_driver[active][None, :]

            component_facility_energy = (
                component_power_mw[component_id, country_id, :] * pue[country_id]
            )
            component_carbon = (
                component_facility_energy
                * hourly_emission_kg_per_mwh[country_id, :]
                / 1000.0
            )
            task_facility_energy_mwh[:, country_id] += (
                shares * component_facility_energy[None, :]
            ).sum(axis=1)
            task_carbon_tco2[:, country_id] += (
                shares * component_carbon[None, :]
            ).sum(axis=1)

    return {
        "facility_energy_mwh": task_facility_energy_mwh,
        "carbon_tco2": task_carbon_tco2,
    }


def _fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _task_metric_output_fields(
    baseline_task_metrics: Dict[str, np.ndarray],
    optimized_task_metrics: Dict[str, np.ndarray],
    country_id: Optional[int] = None,
) -> Dict[str, float]:
    if country_id is None:
        baseline_energy = baseline_task_metrics["facility_energy_mwh"].sum(axis=1)
        optimized_energy = optimized_task_metrics["facility_energy_mwh"].sum(axis=1)
        baseline_carbon = baseline_task_metrics["carbon_tco2"].sum(axis=1)
        optimized_carbon = optimized_task_metrics["carbon_tco2"].sum(axis=1)
    else:
        baseline_energy = baseline_task_metrics["facility_energy_mwh"][:, country_id]
        optimized_energy = optimized_task_metrics["facility_energy_mwh"][:, country_id]
        baseline_carbon = baseline_task_metrics["carbon_tco2"][:, country_id]
        optimized_carbon = optimized_task_metrics["carbon_tco2"][:, country_id]

    baseline_total_energy = float(baseline_energy.sum())
    optimized_total_energy = float(optimized_energy.sum())
    baseline_total_carbon = float(baseline_carbon.sum())
    optimized_total_carbon = float(optimized_carbon.sum())
    classified_ids = [TASK_TYPES.index(task_type) for task_type in CLASSIFIED_TASK_TYPES]
    baseline_classified_energy = float(baseline_energy[classified_ids].sum())
    optimized_classified_energy = float(optimized_energy[classified_ids].sum())
    baseline_classified_carbon = float(baseline_carbon[classified_ids].sum())
    optimized_classified_carbon = float(optimized_carbon[classified_ids].sum())

    fields: Dict[str, float] = {
        "baseline_classified_only_facility_energy_mwh": baseline_classified_energy,
        "optimized_classified_only_facility_energy_mwh": optimized_classified_energy,
        "classified_only_facility_energy_reduction_mwh": (
            baseline_classified_energy - optimized_classified_energy
        ),
        "classified_only_facility_energy_reduction_pct": _fraction(
            baseline_classified_energy - optimized_classified_energy,
            baseline_classified_energy,
        ),
        "baseline_classified_only_carbon_tco2": baseline_classified_carbon,
        "optimized_classified_only_carbon_tco2": optimized_classified_carbon,
        "classified_only_carbon_reduction_tco2": (
            baseline_classified_carbon - optimized_classified_carbon
        ),
        "classified_only_carbon_reduction_pct": _fraction(
            baseline_classified_carbon - optimized_classified_carbon,
            baseline_classified_carbon,
        ),
    }

    for task_id, task_type in enumerate(TASK_TYPES):
        task_baseline_energy = float(baseline_energy[task_id])
        task_optimized_energy = float(optimized_energy[task_id])
        task_baseline_carbon = float(baseline_carbon[task_id])
        task_optimized_carbon = float(optimized_carbon[task_id])

        fields[f"baseline_{task_type}_facility_energy_mwh"] = task_baseline_energy
        fields[f"optimized_{task_type}_facility_energy_mwh"] = task_optimized_energy
        fields[f"{task_type}_facility_energy_reduction_mwh"] = (
            task_baseline_energy - task_optimized_energy
        )
        fields[f"{task_type}_facility_energy_reduction_pct"] = _fraction(
            task_baseline_energy - task_optimized_energy,
            task_baseline_energy,
        )
        fields[f"baseline_{task_type}_facility_energy_share_all"] = _fraction(
            task_baseline_energy,
            baseline_total_energy,
        )
        fields[f"optimized_{task_type}_facility_energy_share_all"] = _fraction(
            task_optimized_energy,
            optimized_total_energy,
        )
        fields[f"baseline_{task_type}_carbon_tco2"] = task_baseline_carbon
        fields[f"optimized_{task_type}_carbon_tco2"] = task_optimized_carbon
        fields[f"{task_type}_carbon_reduction_tco2"] = (
            task_baseline_carbon - task_optimized_carbon
        )
        fields[f"{task_type}_carbon_reduction_pct"] = _fraction(
            task_baseline_carbon - task_optimized_carbon,
            task_baseline_carbon,
        )
        fields[f"baseline_{task_type}_carbon_share_all"] = _fraction(
            task_baseline_carbon,
            baseline_total_carbon,
        )
        fields[f"optimized_{task_type}_carbon_share_all"] = _fraction(
            task_optimized_carbon,
            optimized_total_carbon,
        )

        if task_type == "unclassified":
            fields[f"baseline_{task_type}_facility_energy_share_classified_only"] = np.nan
            fields[f"optimized_{task_type}_facility_energy_share_classified_only"] = np.nan
            fields[f"baseline_{task_type}_carbon_share_classified_only"] = np.nan
            fields[f"optimized_{task_type}_carbon_share_classified_only"] = np.nan
        else:
            fields[f"baseline_{task_type}_facility_energy_share_classified_only"] = _fraction(
                task_baseline_energy,
                baseline_classified_energy,
            )
            fields[f"optimized_{task_type}_facility_energy_share_classified_only"] = _fraction(
                task_optimized_energy,
                optimized_classified_energy,
            )
            fields[f"baseline_{task_type}_carbon_share_classified_only"] = _fraction(
                task_baseline_carbon,
                baseline_classified_carbon,
            )
            fields[f"optimized_{task_type}_carbon_share_classified_only"] = _fraction(
                task_optimized_carbon,
                optimized_classified_carbon,
            )

    return fields


def _build_variable_index(
    active_source_country_ids: np.ndarray,
    active_source_hour_ids: np.ndarray,
    allowed_destination_mask: np.ndarray,
    n_hours: int,
    delay_hours: int,
):
    source_country_parts = []
    source_hour_parts = []
    destination_parts = []
    execution_parts = []
    group_parts = []

    for group_id, (source_country, source_hour) in enumerate(
        zip(active_source_country_ids, active_source_hour_ids)
    ):
        destinations = np.flatnonzero(allowed_destination_mask[int(source_country)]).astype(np.int32)
        if len(destinations) == 0:
            raise ValueError(f"No allowed destinations for source country index {int(source_country)}.")
        last_hour = min(n_hours - 1, int(source_hour) + delay_hours)
        execution_hours = np.arange(int(source_hour), last_hour + 1, dtype=np.int32)
        n_local_variables = len(destinations) * len(execution_hours)
        local_source_countries = np.full(
            n_local_variables,
            int(source_country),
            dtype=np.int32,
        )
        local_source_hours = np.full(
            n_local_variables,
            int(source_hour),
            dtype=np.int32,
        )
        local_destinations = np.repeat(destinations, len(execution_hours))
        local_executions = np.tile(execution_hours, len(destinations))
        local_groups = np.full(n_local_variables, group_id, dtype=np.int32)
        source_country_parts.append(local_source_countries)
        source_hour_parts.append(local_source_hours)
        destination_parts.append(local_destinations)
        execution_parts.append(local_executions)
        group_parts.append(local_groups)

    if not source_country_parts:
        empty = np.array([], dtype=np.int32)
        return empty, empty, empty, empty, empty

    return (
        np.concatenate(source_country_parts),
        np.concatenate(source_hour_parts),
        np.concatenate(destination_parts),
        np.concatenate(execution_parts),
        np.concatenate(group_parts),
    )


def _estimate_variable_count(
    active_source_country_ids: np.ndarray,
    active_source_hour_ids: np.ndarray,
    allowed_destination_mask: np.ndarray,
    n_hours: int,
    delay_hours: int,
) -> int:
    if len(active_source_hour_ids) == 0:
        return 0
    execution_counts = (
        np.minimum(n_hours - 1, active_source_hour_ids + delay_hours) - active_source_hour_ids + 1
    )
    destination_counts = allowed_destination_mask[active_source_country_ids].sum(axis=1)
    return int(np.sum(execution_counts * destination_counts))


def _immediate_training_migration_solution(
    source_country_training_load: np.ndarray,
    residual_capacity: np.ndarray,
    marginal_carbon_cost: np.ndarray,
    return_schedule: bool,
    message: str,
):
    excess = source_country_training_load - residual_capacity
    max_excess = float(excess.max()) if excess.size else 0.0
    if max_excess > CAPACITY_TOLERANCE:
        idx = np.unravel_index(np.argmax(excess), excess.shape)
        raise RuntimeError(
            "Immediate terminal fallback would exceed residual capacity: "
            f"country/resource/hour index={idx}, excess={max_excess:.6g}."
        )

    n_countries = source_country_training_load.shape[0]
    movable_load = source_country_training_load.copy()
    source_country_ids, source_hour_ids = _active_source_country_hours(source_country_training_load)
    if return_schedule and len(source_hour_ids):
        schedule = pd.DataFrame(
            {
                "source_country_id": source_country_ids,
                "source_hour": source_hour_ids,
                "execution_hour": source_hour_ids,
                "execution_country_id": source_country_ids,
                "delay_hours": np.zeros_like(source_hour_ids),
                "share": np.ones_like(source_hour_ids, dtype=np.float64),
            }
        )
    else:
        schedule = pd.DataFrame(
            columns=[
                "source_country_id",
                "source_hour",
                "execution_hour",
                "execution_country_id",
                "delay_hours",
                "share",
            ]
        )

    carbon_objective = float(np.sum(source_country_training_load * marginal_carbon_cost))
    return movable_load, schedule, {
        "status": 0,
        "message": message,
        "objective": carbon_objective,
        "carbon_objective_tco2": carbon_objective,
        "source_migrated_training_load": np.zeros((n_countries,), dtype=np.float64),
        "variables": int(len(source_hour_ids)),
    }


def _solve_sparse_lp_scipy(
    cost: np.ndarray,
    equality_matrix: sparse.csr_matrix,
    equality_rhs: np.ndarray,
    inequality_matrix: sparse.csr_matrix,
    inequality_rhs: np.ndarray,
    linprog_options: Optional[Mapping[str, Union[bool, float, int]]] = None,
):
    options = {"presolve": True}
    if linprog_options:
        options.update(dict(linprog_options))

    result = linprog(
        c=cost,
        A_ub=inequality_matrix,
        b_ub=inequality_rhs,
        A_eq=equality_matrix,
        b_eq=equality_rhs,
        bounds=(0, None),
        method="highs",
        options=options,
    )
    if not result.success:
        raise RuntimeError(f"Training migration optimization failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64), float(result.fun), int(result.status), str(result.message)


def _solve_sparse_lp_gurobi(
    cost: np.ndarray,
    equality_matrix: sparse.csr_matrix,
    equality_rhs: np.ndarray,
    inequality_matrix: sparse.csr_matrix,
    inequality_rhs: np.ndarray,
    verbose: bool = False,
    gurobi_options: Optional[Mapping[str, Union[bool, float, int, str]]] = None,
):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError(
            "solver='gurobi' requires the gurobipy package and a valid Gurobi license."
        ) from exc

    try:
        model = gp.Model("training_migration_lp")
        params = {
            "OutputFlag": 0,
            "NumericFocus": 3,
            "FeasibilityTol": 1e-9,
            "OptimalityTol": 1e-9,
            "ScaleFlag": 2,
        }
        if gurobi_options:
            params.update(dict(gurobi_options))
        for name, value in params.items():
            model.setParam(name, value)

        variables = model.addMVar(shape=cost.shape[0], lb=0.0, name="x")
        model.setObjective(cost @ variables, GRB.MINIMIZE)
        model.addMConstr(equality_matrix, variables, GRB.EQUAL, equality_rhs, name="source_balance")
        if inequality_matrix.shape[0] > 0:
            model.addMConstr(inequality_matrix, variables, GRB.LESS_EQUAL, inequality_rhs, name="capacity")

        if verbose:
            print("[training-migration] Gurobi model built; optimizing.", flush=True)
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            status_name = _gurobi_status_name(model.Status)
            raise RuntimeError(
                "Training migration optimization failed with Gurobi: "
                f"{status_name} (status {model.Status})."
            )

        solution = np.asarray(variables.X, dtype=np.float64)
        return solution, float(model.ObjVal), int(model.Status), _gurobi_status_name(model.Status)
    except gp.GurobiError as exc:
        raise RuntimeError(f"Training migration optimization failed with Gurobi: {exc}") from exc


def _gurobi_status_name(status: int) -> str:
    status_names = {
        1: "LOADED",
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        6: "CUTOFF",
        7: "ITERATION_LIMIT",
        8: "NODE_LIMIT",
        9: "TIME_LIMIT",
        10: "SOLUTION_LIMIT",
        11: "INTERRUPTED",
        12: "NUMERIC",
        13: "SUBOPTIMAL",
        14: "INPROGRESS",
        15: "USER_OBJ_LIMIT",
        16: "WORK_LIMIT",
        17: "MEM_LIMIT",
    }
    return status_names.get(status, f"UNKNOWN_STATUS_{status}")


def _solve_training_migration_lp(
    source_country_training_load: np.ndarray,
    allowed_destination_mask: np.ndarray,
    residual_capacity: np.ndarray,
    marginal_carbon_cost: np.ndarray,
    delay_hours: int,
    return_schedule: bool,
    verbose: bool = False,
    progress_prefix: str = "",
    solver: str = "scipy",
    linprog_options: Optional[Mapping[str, Union[bool, float, int]]] = None,
    gurobi_options: Optional[Mapping[str, Union[bool, float, int, str]]] = None,
):
    prefix = f"{progress_prefix}: " if progress_prefix else ""
    _print_progress(verbose, f"{prefix}building LP source-country-hour set.")
    n_countries = residual_capacity.shape[0]
    n_source_countries, n_resources, n_hours = source_country_training_load.shape
    if n_source_countries != n_countries:
        raise ValueError("source_country_training_load and residual_capacity country dimensions differ.")
    active_source_country_ids, active_source_hour_ids = _active_source_country_hours(
        source_country_training_load
    )

    movable_load = np.zeros((n_countries, n_resources, n_hours), dtype=np.float64)
    source_migrated_training_load = np.zeros((n_countries,), dtype=np.float64)
    empty_schedule = pd.DataFrame(
        columns=[
            "source_country_id",
            "source_hour",
            "execution_hour",
            "execution_country_id",
            "delay_hours",
            "share",
        ]
    )
    if len(active_source_hour_ids) == 0:
        return movable_load, empty_schedule, {
            "status": 0,
            "message": "No movable training workload.",
            "objective": 0.0,
            "carbon_objective_tco2": 0.0,
            "source_migrated_training_load": source_migrated_training_load,
            "variables": 0,
        }

    if delay_hours < 0:
        raise ValueError("delay_hours must be non-negative.")
    if allowed_destination_mask.shape != (n_countries, n_countries):
        raise ValueError("allowed_destination_mask must be a square country-by-country matrix.")
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of: {SOLVERS}.")

    _print_progress(
        verbose,
        f"{prefix}building LP variables for {len(active_source_hour_ids)} source-country-hour groups "
        f"and a {delay_hours}-hour delay window.",
    )
    source_country_ids, source_hour_ids, country_ids, execution_ids, group_ids = _build_variable_index(
        active_source_country_ids=active_source_country_ids,
        active_source_hour_ids=active_source_hour_ids,
        allowed_destination_mask=allowed_destination_mask,
        n_hours=n_hours,
        delay_hours=delay_hours,
    )
    n_variables = len(source_country_ids)
    variable_ids = np.arange(n_variables, dtype=np.int32)

    _print_progress(verbose, f"{prefix}building LP objective vector with {n_variables} variables.")
    carbon_cost = np.zeros((n_variables,), dtype=np.float64)
    for resource_id in range(n_resources):
        carbon_cost += (
            source_country_training_load[source_country_ids, resource_id, source_hour_ids]
            * marginal_carbon_cost[country_ids, resource_id, execution_ids]
        )
    cost = carbon_cost.copy()

    _print_progress(verbose, f"{prefix}building LP equality constraints.")
    equality_matrix = sparse.coo_matrix(
        (
            np.ones((n_variables,), dtype=np.float64),
            (group_ids, variable_ids),
        ),
        shape=(len(active_source_hour_ids), n_variables),
    ).tocsr()
    equality_rhs = np.ones((len(active_source_hour_ids),), dtype=np.float64)

    _print_progress(verbose, f"{prefix}building LP capacity constraints.")
    capacity_rows = []
    capacity_cols = []
    capacity_data = []
    for resource_id in range(n_resources):
        values = source_country_training_load[source_country_ids, resource_id, source_hour_ids]
        positive = values > 1e-12
        if not np.any(positive):
            continue
        rows = (
            (country_ids[positive] * n_resources + resource_id) * n_hours
            + execution_ids[positive]
        )
        capacity_rows.append(rows.astype(np.int64, copy=False))
        capacity_cols.append(variable_ids[positive].astype(np.int64, copy=False))
        capacity_data.append(values[positive])

    if capacity_rows:
        flat_capacity_rows = np.concatenate(capacity_rows)
        unique_capacity_rows, compact_rows = np.unique(flat_capacity_rows, return_inverse=True)
        inequality_matrix = sparse.coo_matrix(
            (
                np.concatenate(capacity_data),
                (compact_rows, np.concatenate(capacity_cols)),
            ),
            shape=(len(unique_capacity_rows), n_variables),
        ).tocsr()
        inequality_rhs = residual_capacity.reshape(-1)[unique_capacity_rows]
    else:
        inequality_matrix = sparse.csr_matrix((0, n_variables))
        inequality_rhs = np.zeros((0,), dtype=np.float64)
    if not np.all(np.isfinite(cost)):
        raise ValueError("LP objective contains NaN or infinite values.")
    if not np.all(np.isfinite(equality_rhs)) or not np.all(np.isfinite(inequality_rhs)):
        raise ValueError("LP constraint right-hand side contains NaN or infinite values.")
    if inequality_rhs.size and inequality_rhs.min() < -CAPACITY_TOLERANCE:
        raise ValueError(f"LP has negative residual capacity: {inequality_rhs.min():.6g}.")
    inequality_rhs = np.maximum(inequality_rhs, 0.0)

    _print_progress(verbose, f"{prefix}solving LP with {solver}.")
    if solver == "scipy":
        solution, objective, solver_status, solver_message = _solve_sparse_lp_scipy(
            cost=cost,
            equality_matrix=equality_matrix,
            equality_rhs=equality_rhs,
            inequality_matrix=inequality_matrix,
            inequality_rhs=inequality_rhs,
            linprog_options=linprog_options,
        )
    else:
        solution, objective, solver_status, solver_message = _solve_sparse_lp_gurobi(
            cost=cost,
            equality_matrix=equality_matrix,
            equality_rhs=equality_rhs,
            inequality_matrix=inequality_matrix,
            inequality_rhs=inequality_rhs,
            verbose=verbose,
            gurobi_options=gurobi_options,
        )

    solution = np.where(solution < 1e-12, 0.0, solution)
    _print_progress(verbose, f"{prefix}LP solved: {solver_message}")
    _print_progress(verbose, f"{prefix}decoding optimized training allocation.")
    for resource_id in range(n_resources):
        contribution = solution * source_country_training_load[
            source_country_ids, resource_id, source_hour_ids
        ]
        np.add.at(movable_load[:, resource_id, :], (country_ids, execution_ids), contribution)
    gpu_id = RESOURCES.index("gpu")
    migrated = country_ids != source_country_ids
    migrated_training_contribution = (
        solution[migrated]
        * source_country_training_load[source_country_ids[migrated], gpu_id, source_hour_ids[migrated]]
    )
    np.add.at(
        source_migrated_training_load,
        source_country_ids[migrated],
        migrated_training_contribution,
    )

    if return_schedule:
        _print_progress(verbose, f"{prefix}building non-zero migration schedule table.")
        schedule = pd.DataFrame(
            {
                "source_country_id": source_country_ids,
                "source_hour": source_hour_ids,
                "execution_hour": execution_ids,
                "execution_country_id": country_ids,
                "delay_hours": execution_ids - source_hour_ids,
                "share": solution,
            }
        )
        schedule = schedule[schedule["share"] > 1e-10].reset_index(drop=True)
    else:
        schedule = empty_schedule

    return movable_load, schedule, {
        "status": int(solver_status),
        "message": str(solver_message),
        "objective": float(objective),
        "carbon_objective_tco2": float(np.dot(solution, carbon_cost)),
        "source_migrated_training_load": source_migrated_training_load,
        "variables": int(n_variables),
    }


def _solve_training_migration_windowed(
    source_country_training_load: np.ndarray,
    allowed_destination_mask: np.ndarray,
    capacity_limit: np.ndarray,
    fixed_for_optimization: np.ndarray,
    baseline_movable_load: np.ndarray,
    country_it_mw: np.ndarray,
    resource_capacities: np.ndarray,
    pue: np.ndarray,
    hourly_emission_kg_per_mwh: np.ndarray,
    hardware_config: HardwarePowerConfig,
    delay_hours: int,
    commit_hours: int,
    return_schedule: bool,
    verbose: bool = False,
    progress_prefix: str = "",
    solver: str = "scipy",
    linprog_options: Optional[Mapping[str, Union[bool, float, int]]] = None,
    gurobi_options: Optional[Mapping[str, Union[bool, float, int, str]]] = None,
):
    if commit_hours <= 0:
        raise ValueError("commit_hours must be positive.")
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of: {SOLVERS}.")

    n_countries, n_resources, n_hours = source_country_training_load.shape
    window_ranges = [
        (source_start, min(n_hours, source_start + commit_hours))
        for source_start in range(0, n_hours, commit_hours)
    ]
    n_windows = len(window_ranges)
    scheduled_movable_load = np.zeros_like(baseline_movable_load)
    reservation_remaining = baseline_movable_load.copy()
    schedule_frames = []
    total_objective = 0.0
    total_carbon_objective = 0.0
    total_variables = 0
    source_migrated_training_load = np.zeros((n_countries,), dtype=np.float64)
    terminal_fallback_used = False
    terminal_fallback_messages = []

    prefix = f"{progress_prefix}: " if progress_prefix else ""
    _print_progress(
        verbose,
        f"{prefix}using windowed LP solve with {n_windows} windows "
        f"({commit_hours} committed hours per window).",
    )

    for window_id, (source_start, source_end) in enumerate(window_ranges, start=1):
        reservation_remaining[:, :, source_start:source_end] = 0.0

        local_source_load = np.zeros_like(source_country_training_load)
        local_source_load[:, :, source_start:source_end] = source_country_training_load[
            :, :, source_start:source_end
        ]
        active_source_country_ids, active_source_hour_ids = _active_source_country_hours(local_source_load)
        if len(active_source_hour_ids) == 0:
            _print_progress(
                verbose,
                f"{prefix}window {window_id}/{n_windows}: source hours "
                f"{source_start}-{source_end - 1} have no movable training load.",
            )
            continue

        committed_load = fixed_for_optimization + scheduled_movable_load
        committed_residual = capacity_limit - committed_load
        min_committed_residual = committed_residual.min()
        if min_committed_residual < -CAPACITY_TOLERANCE:
            idx = np.unravel_index(np.argmin(committed_residual), committed_residual.shape)
            raise ValueError(
                "Windowed committed workload exceeds capacity before LP solve: "
                f"country/resource/hour index={idx}, residual={min_committed_residual:.6g}."
            )

        available_capacity = np.maximum(committed_residual, 0.0)
        effective_reservation = np.minimum(reservation_remaining, available_capacity)
        reservation_excess = reservation_remaining - effective_reservation
        max_reservation_excess = reservation_excess.max()
        if max_reservation_excess > CAPACITY_TOLERANCE:
            idx = np.unravel_index(np.argmax(reservation_excess), reservation_excess.shape)
            _print_progress(
                verbose,
                f"{prefix}window {window_id}/{n_windows}: clipped future reservation "
                f"at country/resource/hour index={idx} by up to {max_reservation_excess:.6g}.",
            )

        base_load_for_window = committed_load + effective_reservation
        residual_capacity = capacity_limit - base_load_for_window
        residual_capacity = np.maximum(residual_capacity, 0.0)

        marginal_it = _marginal_it_mwh_per_resource_hour(
            country_it_mw=country_it_mw,
            fixed_resource_load=base_load_for_window,
            resource_capacities=resource_capacities,
            config=hardware_config,
        )
        marginal_carbon_cost = (
            marginal_it * pue[:, None, None] * hourly_emission_kg_per_mwh[:, None, :] / 1000.0
        )

        estimated_variables = _estimate_variable_count(
            active_source_country_ids=active_source_country_ids,
            active_source_hour_ids=active_source_hour_ids,
            allowed_destination_mask=allowed_destination_mask,
            n_hours=n_hours,
            delay_hours=delay_hours,
        )
        _print_progress(
            verbose,
            f"{prefix}window {window_id}/{n_windows}: solving source hours "
            f"{source_start}-{source_end - 1} with {estimated_variables} variables using {solver}.",
        )
        terminal_24h_window = source_end == n_hours and (source_end - source_start) <= 24
        try:
            window_load, window_schedule, solver_info = _solve_training_migration_lp(
                source_country_training_load=local_source_load,
                allowed_destination_mask=allowed_destination_mask,
                residual_capacity=residual_capacity,
                marginal_carbon_cost=marginal_carbon_cost,
                delay_hours=delay_hours,
                return_schedule=return_schedule,
                verbose=False,
                progress_prefix=f"{progress_prefix} window {window_id}/{n_windows}",
                solver=solver,
                linprog_options=linprog_options,
                gurobi_options=gurobi_options,
            )
        except RuntimeError as exc:
            if not terminal_24h_window:
                raise
            fallback_message = (
                "Terminal 24h LP failed; used immediate local execution fallback. "
                f"Original error: {exc}"
            )
            _print_progress(verbose, f"{prefix}window {window_id}/{n_windows}: {fallback_message}")
            window_load, window_schedule, solver_info = _immediate_training_migration_solution(
                source_country_training_load=local_source_load,
                residual_capacity=residual_capacity,
                marginal_carbon_cost=marginal_carbon_cost,
                return_schedule=return_schedule,
                message=fallback_message,
            )
            solver_info["variables"] = int(estimated_variables)
            terminal_fallback_used = True
            terminal_fallback_messages.append(fallback_message)
        scheduled_movable_load += window_load
        total_objective += solver_info["objective"]
        total_carbon_objective += solver_info["carbon_objective_tco2"]
        total_variables += solver_info["variables"]
        source_migrated_training_load += solver_info["source_migrated_training_load"]
        if return_schedule and not window_schedule.empty:
            schedule_frames.append(window_schedule)
        _print_progress(
            verbose,
            f"{prefix}window {window_id}/{n_windows}: solved "
            f"({solver_info['variables']} variables).",
        )

    if return_schedule and schedule_frames:
        schedule = pd.concat(schedule_frames, ignore_index=True)
    else:
        schedule = pd.DataFrame(
            columns=[
                "source_country_id",
                "source_hour",
                "execution_hour",
                "execution_country_id",
                "delay_hours",
                "share",
            ]
        )

    return scheduled_movable_load, schedule, {
        "status": 0,
        "message": (
            "Windowed LP solve completed with terminal 24h immediate fallback. "
            + " ".join(terminal_fallback_messages)
            if terminal_fallback_used
            else "Windowed LP solve completed."
        ),
        "objective": float(total_objective),
        "carbon_objective_tco2": float(total_carbon_objective),
        "source_migrated_training_load": source_migrated_training_load,
        "variables": int(total_variables),
        "windows": int(n_windows),
    }


def _country_records(
    scenario: str,
    year: int,
    method: str,
    countries: Sequence[str],
    baseline: Dict[str, np.ndarray],
    optimized: Dict[str, np.ndarray],
    baseline_task_metrics: Optional[Dict[str, np.ndarray]] = None,
    optimized_task_metrics: Optional[Dict[str, np.ndarray]] = None,
):
    records = []
    baseline_country_carbon = baseline["carbon_tco2"].sum(axis=1)
    optimized_country_carbon = optimized["carbon_tco2"].sum(axis=1)
    baseline_facility = baseline["facility_energy_mwh"].sum(axis=1)
    optimized_facility = optimized["facility_energy_mwh"].sum(axis=1)

    for country_id, country in enumerate(countries):
        reduction = baseline_country_carbon[country_id] - optimized_country_carbon[country_id]
        facility_reduction = baseline_facility[country_id] - optimized_facility[country_id]
        record = {
            "scenario": scenario,
            "year": year,
            "method": method,
            "country": country,
            "baseline_facility_energy_mwh": baseline_facility[country_id],
            "optimized_facility_energy_mwh": optimized_facility[country_id],
            "facility_energy_reduction_mwh": facility_reduction,
            "facility_energy_reduction_pct": _fraction(
                facility_reduction,
                baseline_facility[country_id],
            ),
            "baseline_carbon_tco2": baseline_country_carbon[country_id],
            "optimized_carbon_tco2": optimized_country_carbon[country_id],
            "carbon_reduction_tco2": reduction,
            "carbon_reduction_pct": _fraction(
                reduction,
                baseline_country_carbon[country_id],
            ),
            "baseline_peak_gpu_utilization": baseline["utilization"][
                country_id, RESOURCES.index("gpu")
            ].max(),
            "optimized_peak_gpu_utilization": optimized["utilization"][
                country_id, RESOURCES.index("gpu")
            ].max(),
            "optimized_overflow_resource_hours": optimized["overflow"][country_id].sum(),
        }
        if baseline_task_metrics is not None and optimized_task_metrics is not None:
            record.update(
                _task_metric_output_fields(
                    baseline_task_metrics=baseline_task_metrics,
                    optimized_task_metrics=optimized_task_metrics,
                    country_id=country_id,
                )
            )
        records.append(record)
    return records


def _training_execution_records(
    scenario: str,
    year: int,
    method: str,
    countries: Sequence[str],
    training_origin_weights: np.ndarray,
    source_training_load: np.ndarray,
    baseline_training_load: np.ndarray,
    optimized_training_load: np.ndarray,
    source_migrated_training_load: np.ndarray,
):
    records = []
    global_training_resource_hours = source_training_load.sum(axis=1)
    baseline_resource_hours = baseline_training_load.sum(axis=2)
    optimized_resource_hours = optimized_training_load.sum(axis=2)
    gpu_id = RESOURCES.index("gpu")

    for country_id, country in enumerate(countries):
        source_training_gpu_hours = baseline_resource_hours[country_id, gpu_id]
        record = {
            "scenario": scenario,
            "year": year,
            "method": method,
            "country": country,
            "origin_training_share": training_origin_weights[country_id],
            "migrated_training_share_pct": (
                source_migrated_training_load[country_id] / source_training_gpu_hours * 100.0
                if source_training_gpu_hours > 0
                else 0.0
            ),
        }
        for resource_id, resource in enumerate(RESOURCES):
            origin_hours = training_origin_weights[country_id] * global_training_resource_hours[resource_id]
            baseline_hours = baseline_resource_hours[country_id, resource_id]
            optimized_hours = optimized_resource_hours[country_id, resource_id]
            record[f"origin_{resource}_hours"] = origin_hours
            record[f"baseline_execution_{resource}_hours"] = baseline_hours
            record[f"optimized_execution_{resource}_hours"] = optimized_hours
            record[f"net_execution_change_{resource}_hours"] = optimized_hours - baseline_hours
        records.append(record)
    return records


def _schedule_records(
    schedule: pd.DataFrame,
    scenario: str,
    year: int,
    method: str,
    countries: Sequence[str],
    timestamps: pd.DatetimeIndex,
):
    if schedule.empty:
        return schedule

    result = schedule.copy()
    result.insert(0, "year", year)
    result.insert(0, "scenario", scenario)
    result.insert(2, "method", method)
    result["source_timestamp_utc"] = timestamps[result["source_hour"].to_numpy()].strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    result["execution_timestamp_utc"] = timestamps[result["execution_hour"].to_numpy()].strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    country_array = np.array(countries, dtype=object)
    result["source_country"] = country_array[result["source_country_id"].to_numpy()]
    result["execution_country"] = country_array[result["execution_country_id"].to_numpy()]
    if "tx_cost_usd" in result.columns:
        result = result.drop(columns=["tx_cost_usd"])
    return result.drop(columns=["source_country_id", "execution_country_id"])


def _migration_flow_summary_records_carbon_only(
    scenario: str,
    year: int,
    method: str,
    countries: Sequence[str],
    schedule: pd.DataFrame,
    source_country_training_load: np.ndarray,
):
    if schedule.empty:
        return []

    required = {
        "source_country_id",
        "execution_country_id",
        "source_hour",
        "execution_hour",
        "delay_hours",
        "share",
    }
    if not required.issubset(schedule.columns):
        return []

    df = schedule.loc[schedule["share"] > 1e-10, list(required)].copy()
    if df.empty:
        return []

    source_country_ids = df["source_country_id"].to_numpy(dtype=np.int64)
    execution_country_ids = df["execution_country_id"].to_numpy(dtype=np.int64)
    source_hour_ids = df["source_hour"].to_numpy(dtype=np.int64)
    shares = df["share"].to_numpy(dtype=np.float64)
    country_array = np.asarray(countries, dtype=object)

    df["scenario"] = scenario
    df["year"] = year
    df["method"] = method
    df["source_country"] = country_array[source_country_ids]
    df["execution_country"] = country_array[execution_country_ids]
    df["is_cross_country"] = source_country_ids != execution_country_ids
    for resource_id, resource in enumerate(RESOURCES):
        df[f"training_{resource}_hours"] = (
            shares * source_country_training_load[source_country_ids, resource_id, source_hour_ids]
        )

    gpu_hours = df[f"training_{RESOURCES[RESOURCES.index('gpu')]}_hours"].to_numpy(dtype=np.float64)
    delay_weight = np.where(gpu_hours > 1e-12, gpu_hours, shares)
    df["delay_weight"] = delay_weight
    df["weighted_delay_hours"] = df["delay_hours"].to_numpy(dtype=np.float64) * delay_weight

    agg_spec: Dict[str, tuple] = {
        "decision_count": ("share", "size"),
        "decision_share_sum": ("share", "sum"),
        "source_hour_count": ("source_hour", "nunique"),
        "execution_hour_count": ("execution_hour", "nunique"),
        "delay_weight": ("delay_weight", "sum"),
        "weighted_delay_hours": ("weighted_delay_hours", "sum"),
    }
    for resource in RESOURCES:
        agg_spec[f"training_{resource}_hours"] = (f"training_{resource}_hours", "sum")

    grouped = (
        df.groupby(
            ["scenario", "year", "method", "source_country", "execution_country", "is_cross_country"],
            as_index=False,
        )
        .agg(**agg_spec)
    )
    grouped["average_delay_hours"] = np.where(
        grouped["delay_weight"] > 1e-12,
        grouped["weighted_delay_hours"] / grouped["delay_weight"],
        0.0,
    )
    grouped = grouped.drop(columns=["delay_weight", "weighted_delay_hours"])
    return grouped.to_dict("records")


def _transmission_policy_restriction_records(
    policy_name: str,
    matrix_path,
    countries: Sequence[str],
    allowed_mask: np.ndarray,
    policy_block_cost: float,
):
    n_countries = len(countries)
    off_diagonal = ~np.eye(n_countries, dtype=bool)
    allowed_off_diagonal = allowed_mask & off_diagonal
    blocked_off_diagonal = (~allowed_mask) & off_diagonal
    return [
        {
            "transmission_policy": policy_name,
            "transmission_matrix_path": "" if matrix_path is None else str(matrix_path),
            "country_count": n_countries,
            "intercountry_path_count": int(off_diagonal.sum()),
            "allowed_intercountry_path_count": int(allowed_off_diagonal.sum()),
            "blocked_intercountry_path_count": int(blocked_off_diagonal.sum()),
            "allowed_intercountry_path_share": float(
                allowed_off_diagonal.sum() / off_diagonal.sum() if off_diagonal.sum() else 0.0
            ),
            "policy_block_threshold": float(policy_block_cost),
        }
    ]


def _failed_method_annual_record(
    scenario: str,
    year: int,
    method: str,
    migration_constraint: str,
    solver: str,
    delay_hours: int,
    solve_mode: str,
    commit_hours: int,
    transmission_policy_name: str,
    transmission_matrix_path,
    ai_capacity_factor: float,
    total_data_center_global_it_mw: float,
    global_it_mw: float,
    movable_share: float,
    baseline_eval: Dict[str, np.ndarray],
    baseline_carbon: float,
    used_hourly_factors: np.ndarray,
    error: Exception,
) -> Dict[str, object]:
    return {
        "scenario": scenario,
        "year": year,
        "method": method,
        "migration_constraint": migration_constraint,
        "solver": solver,
        "delay_hours": delay_hours,
        "solve_mode": solve_mode,
        "commit_hours": commit_hours if solve_mode == "windowed" else 0,
        "transmission_policy": transmission_policy_name,
        "transmission_matrix_path": (
            str(transmission_matrix_path) if transmission_matrix_path is not None else ""
        ),
        "ai_capacity_factor": float(ai_capacity_factor),
        "total_data_center_global_it_mw": float(total_data_center_global_it_mw),
        "global_ai_it_mw": float(global_it_mw),
        "movable_training_share": movable_share,
        "baseline_facility_energy_mwh": float(baseline_eval["facility_energy_mwh"].sum()),
        "optimized_facility_energy_mwh": np.nan,
        "baseline_carbon_tco2": baseline_carbon,
        "optimized_carbon_tco2": np.nan,
        "carbon_reduction_tco2": np.nan,
        "carbon_reduction_pct": np.nan,
        "linearized_training_objective_tco2": np.nan,
        "migrated_training_share_pct": np.nan,
        "optimization_variables": 0,
        "optimization_windows": 0,
        "hourly_factor_countries": int(used_hourly_factors.sum()),
        "annual_fallback_countries": int((~used_hourly_factors).sum()),
        "solver_status": -1,
        "solver_message": f"FAILED: {error}",
    }


def run_training_migration_optimization(
    renewable_energy_policy: str,
    scenarios: Sequence[str],
    years: int = 5,
    countries: Optional[Sequence[str]] = None,
    workload_profile_path: Union[str, Path] = ROOT_DIR / "dataset" / "cluster-trace-gpu-v2026",
    server_profile_path: Optional[Union[str, Path]] = None,
    network_profile_path: Optional[Union[str, Path]] = None,
    workload_year: Optional[int] = None,
    year_start: int = 2026,
    output_dir: Union[str, Path] = ROOT_DIR / "results" / "workload_training_migration_optimization",
    save_outputs: bool = True,
    verbose: bool = True,
    hardware_config: Optional[HardwarePowerConfig] = None,
    trace_config: Optional[Alibaba2026TraceConfig] = None,
    classification_config: Optional[object] = None,
    task_origin_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
    task_execution_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
    execution_policy: str = "capacity",
    inference_origin_fraction: float = 0.75,
    other_origin_fraction: float = 0.50,
    cpu_data_origin_fraction: Optional[float] = None,
    capacity_quantile: float = 0.96,
    max_resource_utilization: float = 1.0,
    pue_scale: float = 1.0,
    ai_capacity_factors: Optional[Mapping[int, float]] = None,
    hourly_carbon_factors_dir: Optional[Union[str, Path]] = EM_CPNDCNZ_DIR,
    hourly_carbon_scope: str = "direct",
    hourly_carbon_fallback_to_annual: bool = True,
    delay_hours: int = 24,
    migration_constraint: str = "global",
    europe_countries: Sequence[str] = EUROPEAN_COUNTRIES,
    save_schedule: bool = False,
    save_analysis_outputs: bool = True,
    max_intervals: Optional[int] = None,
    workload_cache_dir: Optional[Union[str, Path]] = None,
    refresh_workload_cache: bool = False,
    solve_mode: str = "auto",
    commit_hours: int = 24,
    monolithic_variable_limit: int = 1_000_000,
    solver: str = "scipy",
    linprog_options: Optional[Mapping[str, Union[bool, float, int]]] = None,
    gurobi_options: Optional[Mapping[str, Union[bool, float, int, str]]] = None,
    policy_transmission_matrix: Optional[Union[str, Path]] = None,
    azure_transmission_policy: Optional[str] = None,
    azure_transmission_dir: Union[str, Path] = AZURE_TRANSMISSION_DIR,
    policy_block_cost: float = DEFAULT_POLICY_BLOCK_COST,
    carbon_weight: float = 1.0,
    tx_weight: float = 1.0,
    include_transmission_cost: bool = False,
    objective_normalization: str = "benchmark",
    scheduling_methods: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Optimize training workload migration over carbon emissions using Alibaba
    GPU v2026 workload traces.

    Training workload is aggregated to hourly resource demand. For each source
    country and source hour, the optimizer can execute the movable training
    workload in the same hour or delay it by up to ``delay_hours``.
    ``migration_constraint="global"`` allows all selected countries to receive
    movable training workload. ``"europe_only"`` allows only European-origin
    training workload to move within Europe, while non-European training remains
    fixed in its origin country and source hour. ``solve_mode="auto"`` uses a
    single LP for small problems and a rolling window LP for full-year runs that
    would otherwise exceed solver limits.

    Azure transmission policies are read from ``dataset/Azure`` when
    ``azure_transmission_policy`` is one of ``AZURE_TRANSMISSION_POLICIES``.
    Manual ``policy_transmission_matrix`` paths remain supported and take
    precedence when both interfaces are provided. These matrices are used only
    to block disallowed source-destination country pairs; USD/GB values are
    not part of the optimization objective.
    """
    if years <= 0:
        raise ValueError("years must be positive.")
    if delay_hours < 0:
        raise ValueError("delay_hours must be non-negative.")
    if migration_constraint not in MIGRATION_CONSTRAINTS:
        raise ValueError(f"migration_constraint must be one of: {MIGRATION_CONSTRAINTS}.")
    if solve_mode not in SOLVE_MODES:
        raise ValueError(f"solve_mode must be one of: {SOLVE_MODES}.")
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of: {SOLVERS}.")
    if commit_hours <= 0:
        raise ValueError("commit_hours must be positive.")
    if monolithic_variable_limit <= 0:
        raise ValueError("monolithic_variable_limit must be positive.")
    if not (0 < max_resource_utilization <= 1):
        raise ValueError("max_resource_utilization must be in (0, 1].")
    if policy_block_cost <= 0:
        raise ValueError("policy_block_cost must be positive.")
    if include_transmission_cost or tx_weight != 1.0 or carbon_weight != 1.0:
        _print_progress(
            verbose,
            "Ignoring transmission/objective weight arguments; this variant optimizes carbon emissions only.",
        )
    if scheduling_methods is None:
        selected_scheduling_methods = list(DEFAULT_SCHEDULING_METHODS)
    else:
        selected_scheduling_methods = []
        for method_name in scheduling_methods:
            if method_name == "all":
                selected_scheduling_methods.extend(DEFAULT_SCHEDULING_METHODS)
            else:
                selected_scheduling_methods.append(str(method_name))
    selected_scheduling_methods = list(dict.fromkeys(selected_scheduling_methods))
    unknown_methods = [
        method_name for method_name in selected_scheduling_methods if method_name not in SCHEDULING_METHODS
    ]
    if unknown_methods:
        raise ValueError(f"Unknown scheduling methods: {unknown_methods}. Allowed: {SCHEDULING_METHODS}.")

    data_year_end = DATA_YEAR_START + it_capacity.shape[0] - 1
    if year_start < DATA_YEAR_START or year_start + years - 1 > data_year_end:
        raise ValueError(f"Requested years must be within {DATA_YEAR_START}-{data_year_end}.")
    resolved_ai_capacity_factors = _resolve_ai_capacity_factors(
        year_start=year_start,
        years=years,
        overrides=ai_capacity_factors,
    )

    hardware_config = hardware_config or HardwarePowerConfig()
    hardware_config.validate()
    if cpu_data_origin_fraction is not None:
        other_origin_fraction = float(cpu_data_origin_fraction)
    countries = list(countries or DEFAULT_COUNTRIES)
    for scenario in scenarios:
        if scenario not in SCENARIO_COL_MAP:
            raise ValueError(f"Unknown scenario '{scenario}'. Allowed: {list(SCENARIO_COL_MAP.keys())}")
    unknown_countries = [country for country in countries if country not in it_ratio]
    if unknown_countries:
        raise ValueError(f"Unknown countries: {unknown_countries}")

    origin_weights = _as_task_weight_table(countries, task_origin_weights)
    training_origin_weights = origin_weights[TASK_TYPES.index("training")]
    country_share = np.array([float(it_ratio[country]) for country in countries], dtype=float)
    europe_set = set(europe_countries)
    europe_ids = np.array(
        [country_id for country_id, country in enumerate(countries) if country in europe_set],
        dtype=np.int32,
    )
    if migration_constraint == "europe_only" and len(europe_ids) == 0:
        raise ValueError("No selected countries are included in europe_countries.")
    transmission_matrix_values, policy_allowed_mask, transmission_matrix_path, transmission_policy_name = (
        _load_transmission_policy_matrix(
            countries=countries,
            policy_transmission_matrix=policy_transmission_matrix,
            azure_transmission_policy=azure_transmission_policy,
            azure_transmission_dir=azure_transmission_dir,
            policy_block_cost=policy_block_cost,
        )
    )
    transmission_policy_records = _transmission_policy_restriction_records(
        policy_name=transmission_policy_name,
        matrix_path=transmission_matrix_path,
        countries=countries,
        allowed_mask=policy_allowed_mask,
        policy_block_cost=policy_block_cost,
    )
    allowed_destination_mask = _apply_migration_constraint_to_allowed_mask(
        allowed_mask=policy_allowed_mask,
        migration_constraint=migration_constraint,
        europe_ids=europe_ids,
    )

    annual_records = []
    country_records = []
    execution_records = []
    country_capacity_records = []
    component_energy_records = []
    migration_flow_records = []
    capacity_binding_records = []
    schedule_frames = []
    return_schedule_for_analysis = save_schedule or save_analysis_outputs

    _print_progress(verbose, "Building workload profile from trace.")
    profile = build_workload_profile(
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        network_profile_path=network_profile_path,
        capacity_quantile=capacity_quantile,
        trace_config=trace_config,
        max_intervals=max_intervals,
        workload_cache_dir=workload_cache_dir,
        refresh_workload_cache=refresh_workload_cache,
        verbose=verbose,
    )
    _print_progress(
        verbose,
        f"Workload profile ready: {profile.n_intervals} intervals, "
        f"{profile.interval_hours:.2f} hours per interval, "
        f"annualization factor {profile.annualization_factor_8760:.3f}.",
    )

    for scenario in scenarios:
        _print_progress(verbose, f"Starting scenario '{scenario}'.")
        scenario_col = SCENARIO_COL_MAP[scenario]
        for output_year_idx in range(years):
            year = year_start + output_year_idx
            data_year_idx = year - DATA_YEAR_START
            progress_label = f"{scenario} {year}"
            _print_progress(verbose, f"{progress_label}: loading carbon factors.")
            hourly_timestamps, hourly_emission, used_hourly_factors = _hourly_emission_factors(
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
            _print_progress(
                verbose,
                f"{progress_label}: carbon factors ready for {len(hourly_timestamps)} hours "
                f"({int(used_hourly_factors.sum())} hourly, {int((~used_hourly_factors).sum())} annual fallback).",
            )

            _print_progress(verbose, f"{progress_label}: building country capacities and PUE.")
            total_data_center_global_it_mw = float(it_capacity[data_year_idx, scenario_col]) * 1e3
            ai_capacity_factor = resolved_ai_capacity_factors[year]
            global_it_mw = total_data_center_global_it_mw * ai_capacity_factor
            country_it_mw = global_it_mw * country_share
            resource_capacities = _resource_capacity(country_it_mw, hardware_config)
            global_resource_capacity = resource_capacities.sum(axis=0)
            pue = np.array([PUE[country][data_year_idx, scenario_col] for country in countries], dtype=float)
            pue *= pue_scale
            if save_analysis_outputs:
                country_capacity_records.extend(
                    country_resource_capacity_records(
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

            _print_progress(verbose, f"{progress_label}: scaling workload to global capacity.")
            interval_global_load = _scale_workload_to_capacity(
                profile=profile,
                global_resource_capacity=global_resource_capacity,
                max_resource_utilization=max_resource_utilization,
            )
            _print_progress(verbose, f"{progress_label}: aggregating 15-minute workload to hourly demand.")
            hourly_global_type_load = _hourly_load_from_intervals(
                interval_load=interval_global_load,
                interval_index=profile.interval_index,
                interval_hours=profile.interval_hours,
                target_timestamps=hourly_timestamps,
                relative_trace=profile.relative_time,
            )
            source_training_load = hourly_global_type_load[TASK_TYPES.index("training")]

            _print_progress(verbose, f"{progress_label}: building fixed workload execution weights.")
            execution_weights = _build_execution_weights(
                countries=countries,
                country_it_mw=country_it_mw,
                origin_weights=origin_weights,
                execution_policy=execution_policy,
                inference_origin_fraction=inference_origin_fraction,
                other_origin_fraction=other_origin_fraction,
                task_execution_weights=task_execution_weights,
            )
            country_type_load = (
                hourly_global_type_load[:, None, :, :] * execution_weights[:, :, None, None]
            )
            non_training_ids = [
                task_id for task_id, task_type in enumerate(TASK_TYPES) if task_type != "training"
            ]
            fixed_non_training_load = country_type_load[non_training_ids].sum(axis=0)
            baseline_training_load = (
                source_training_load[None, :, :] * training_origin_weights[:, None, None]
            )
            baseline_total_load = fixed_non_training_load + baseline_training_load
            _print_progress(verbose, f"{progress_label}: evaluating baseline emissions.")
            baseline_eval = _evaluate_country_hourly_carbon(
                country_it_mw=country_it_mw,
                country_resource_load=baseline_total_load,
                resource_capacities=resource_capacities,
                pue=pue,
                hourly_emission_kg_per_mwh=hourly_emission,
                hardware_config=hardware_config,
                max_resource_utilization=max_resource_utilization,
            )
            baseline_task_metrics = _allocate_task_energy_carbon(
                country_it_mw=country_it_mw,
                task_country_resource_load=_task_country_resource_load(
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
                movable_share = float(training_origin_weights.sum())
                fixed_training_load = np.zeros_like(baseline_training_load)
            else:
                movable_share = float(training_origin_weights[europe_ids].sum())
                fixed_training_load = baseline_training_load.copy()
                fixed_training_load[europe_ids] = 0.0
            baseline_movable_load = np.maximum(baseline_training_load - fixed_training_load, 0.0)
            zero_movable_transfer_gb = np.zeros(
                (len(countries), source_training_load.shape[1]),
                dtype=np.float64,
            )

            _print_progress(verbose, f"{progress_label}: building migration capacity constraints.")
            fixed_for_optimization = fixed_non_training_load + fixed_training_load
            capacity_limit = resource_capacities[:, :, None] * max_resource_utilization
            fixed_residual_capacity = capacity_limit - fixed_for_optimization
            min_residual = fixed_residual_capacity.min()
            if min_residual < -CAPACITY_TOLERANCE:
                idx = np.unravel_index(np.argmin(fixed_residual_capacity), fixed_residual_capacity.shape)
                raise ValueError(
                    "Fixed workload exceeds capacity before training migration: "
                    f"country={countries[idx[0]]}, resource={RESOURCES[idx[1]]}, "
                    f"hour={idx[2]}, residual={min_residual:.6g}."
                )
            fixed_residual_capacity = np.maximum(fixed_residual_capacity, 0.0)

            _print_progress(verbose, f"{progress_label}: estimating baseline marginal carbon costs.")
            base_marginal_it = _marginal_it_mwh_per_resource_hour(
                country_it_mw=country_it_mw,
                fixed_resource_load=fixed_for_optimization,
                resource_capacities=resource_capacities,
                config=hardware_config,
            )
            base_marginal_carbon_cost = (
                base_marginal_it * pue[:, None, None] * hourly_emission[:, None, :] / 1000.0
            )
            annual_marginal_carbon_cost = base_marginal_carbon_cost.mean(axis=2)

            active_source_country_ids, active_source_hour_ids = _active_source_country_hours(
                baseline_movable_load
            )
            estimated_variables = _estimate_variable_count(
                active_source_country_ids=active_source_country_ids,
                active_source_hour_ids=active_source_hour_ids,
                allowed_destination_mask=allowed_destination_mask,
                n_hours=source_training_load.shape[1],
                delay_hours=delay_hours,
            )
            baseline_carbon = float(baseline_eval["carbon_tco2"].sum())
            training_gpu_id = RESOURCES.index("gpu")
            total_training_gpu_hours = float(baseline_training_load[:, training_gpu_id, :].sum())

            for scheduling_method in selected_scheduling_methods:
                method_label = f"{progress_label} {scheduling_method}"
                _print_progress(verbose, f"{method_label}: starting scheduling method.")
                selected_solve_mode = solve_mode if scheduling_method == LINEAR_PROGRAMMING_METHOD else scheduling_method
                method_solver = solver if scheduling_method == LINEAR_PROGRAMMING_METHOD else scheduling_method
                try:
                    if scheduling_method == LINEAR_PROGRAMMING_METHOD:
                        if solve_mode == "auto":
                            selected_solve_mode = (
                                "windowed" if estimated_variables > monolithic_variable_limit else "monolithic"
                            )
                        _print_progress(
                            verbose,
                            f"{method_label}: selected {selected_solve_mode} solve mode "
                            f"for {estimated_variables} estimated variables.",
                        )
                        if selected_solve_mode == "monolithic":
                            movable_training_load, schedule, solver_info = _solve_training_migration_lp(
                                source_country_training_load=baseline_movable_load,
                                allowed_destination_mask=allowed_destination_mask,
                                residual_capacity=fixed_residual_capacity,
                                marginal_carbon_cost=base_marginal_carbon_cost,
                                delay_hours=delay_hours,
                                return_schedule=return_schedule_for_analysis,
                                verbose=verbose,
                                progress_prefix=method_label,
                                solver=solver,
                                linprog_options=linprog_options,
                                gurobi_options=gurobi_options,
                            )
                            solver_info["windows"] = 1
                        else:
                            movable_training_load, schedule, solver_info = _solve_training_migration_windowed(
                                source_country_training_load=baseline_movable_load,
                                allowed_destination_mask=allowed_destination_mask,
                                capacity_limit=capacity_limit,
                                fixed_for_optimization=fixed_for_optimization,
                                baseline_movable_load=baseline_movable_load,
                                country_it_mw=country_it_mw,
                                resource_capacities=resource_capacities,
                                pue=pue,
                                hourly_emission_kg_per_mwh=hourly_emission,
                                hardware_config=hardware_config,
                                delay_hours=delay_hours,
                                commit_hours=commit_hours,
                                return_schedule=return_schedule_for_analysis,
                                verbose=verbose,
                                progress_prefix=method_label,
                                solver=solver,
                                linprog_options=linprog_options,
                                gurobi_options=gurobi_options,
                            )
                        method_solver = solver
                    else:
                        movable_training_load, schedule, solver_info = solve_training_migration_baseline_method(
                            method_name=scheduling_method,
                            source_country_training_load=baseline_movable_load,
                            source_country_transfer_gb=zero_movable_transfer_gb,
                            allowed_destination_mask=allowed_destination_mask,
                            residual_capacity=fixed_residual_capacity,
                            capacity_limit=capacity_limit,
                            fixed_for_optimization=fixed_for_optimization,
                            baseline_movable_load=baseline_movable_load,
                            marginal_carbon_cost=base_marginal_carbon_cost,
                            annual_marginal_carbon_cost=annual_marginal_carbon_cost,
                            tx_cost_per_gb=np.zeros_like(transmission_matrix_values),
                            delay_hours=delay_hours,
                            commit_hours=commit_hours,
                            return_schedule=return_schedule_for_analysis,
                            carbon_weight=1.0,
                            tx_weight=0.0,
                            objective_normalization=objective_normalization,
                            linprog_options=linprog_options,
                        )
                        selected_solve_mode = solver_info.get("solve_mode", scheduling_method)
                        method_solver = solver_info.get("solver", scheduling_method)
                except Exception as exc:
                    _print_progress(verbose, f"{method_label}: failed; continuing with remaining methods. {exc}")
                    annual_records.append(
                        _failed_method_annual_record(
                            scenario=scenario,
                            year=year,
                            method=scheduling_method,
                            migration_constraint=migration_constraint,
                            solver=method_solver,
                            delay_hours=delay_hours,
                            solve_mode=selected_solve_mode,
                            commit_hours=commit_hours,
                            transmission_policy_name=transmission_policy_name,
                            transmission_matrix_path=transmission_matrix_path,
                            ai_capacity_factor=ai_capacity_factor,
                            total_data_center_global_it_mw=total_data_center_global_it_mw,
                            global_it_mw=global_it_mw,
                            movable_share=movable_share,
                            baseline_eval=baseline_eval,
                            baseline_carbon=baseline_carbon,
                            used_hourly_factors=used_hourly_factors,
                            error=exc,
                        )
                    )
                    continue

                optimized_training_load = fixed_training_load + movable_training_load
                optimized_total_load = fixed_non_training_load + optimized_training_load
                _print_progress(verbose, f"{method_label}: evaluating optimized emissions.")
                optimized_eval = _evaluate_country_hourly_carbon(
                    country_it_mw=country_it_mw,
                    country_resource_load=optimized_total_load,
                    resource_capacities=resource_capacities,
                    pue=pue,
                    hourly_emission_kg_per_mwh=hourly_emission,
                    hardware_config=hardware_config,
                    max_resource_utilization=max_resource_utilization,
                )
                optimized_task_metrics = _allocate_task_energy_carbon(
                    country_it_mw=country_it_mw,
                    task_country_resource_load=_task_country_resource_load(
                        country_type_load=country_type_load,
                        training_load=optimized_training_load,
                    ),
                    resource_capacities=resource_capacities,
                    pue=pue,
                    hourly_emission_kg_per_mwh=hourly_emission,
                    hardware_config=hardware_config,
                    max_resource_utilization=max_resource_utilization,
                )

                optimized_carbon = float(optimized_eval["carbon_tco2"].sum())
                carbon_reduction = baseline_carbon - optimized_carbon
                baseline_facility_energy = float(baseline_eval["facility_energy_mwh"].sum())
                optimized_facility_energy = float(optimized_eval["facility_energy_mwh"].sum())
                facility_energy_reduction = baseline_facility_energy - optimized_facility_energy
                migrated_training_share_pct = (
                    float(solver_info["source_migrated_training_load"].sum())
                    / total_training_gpu_hours
                    * 100.0
                    if total_training_gpu_hours > 0
                    else 0.0
                )
                _print_progress(
                    verbose,
                    f"{method_label}: completed with {carbon_reduction:.6f} tCO2 reduction "
                    f"({carbon_reduction / baseline_carbon:.6%}).",
                )
                annual_records.append(
                    {
                        "scenario": scenario,
                        "year": year,
                        "method": scheduling_method,
                        "migration_constraint": migration_constraint,
                        "solver": method_solver,
                        "delay_hours": delay_hours,
                        "solve_mode": selected_solve_mode,
                        "commit_hours": solver_info.get(
                            "commit_hours",
                            commit_hours if selected_solve_mode == "windowed" else 0,
                        ),
                        "transmission_policy": transmission_policy_name,
                        "transmission_matrix_path": (
                            str(transmission_matrix_path) if transmission_matrix_path is not None else ""
                        ),
                        "ai_capacity_factor": float(ai_capacity_factor),
                        "total_data_center_global_it_mw": float(total_data_center_global_it_mw),
                        "global_ai_it_mw": float(global_it_mw),
                        "movable_training_share": movable_share,
                        "baseline_facility_energy_mwh": baseline_facility_energy,
                        "optimized_facility_energy_mwh": optimized_facility_energy,
                        "facility_energy_reduction_mwh": facility_energy_reduction,
                        "facility_energy_reduction_pct": _fraction(
                            facility_energy_reduction,
                            baseline_facility_energy,
                        ),
                        "baseline_carbon_tco2": baseline_carbon,
                        "optimized_carbon_tco2": optimized_carbon,
                        "carbon_reduction_tco2": carbon_reduction,
                        "carbon_reduction_pct": _fraction(carbon_reduction, baseline_carbon),
                        "linearized_training_objective_tco2": solver_info["carbon_objective_tco2"],
                        "migrated_training_share_pct": migrated_training_share_pct,
                        "optimization_variables": solver_info["variables"],
                        "optimization_windows": solver_info["windows"],
                        "hourly_factor_countries": int(used_hourly_factors.sum()),
                        "annual_fallback_countries": int((~used_hourly_factors).sum()),
                        "solver_status": solver_info["status"],
                        "solver_message": solver_info["message"],
                    }
                )
                annual_records[-1].update(
                    _task_metric_output_fields(
                        baseline_task_metrics=baseline_task_metrics,
                        optimized_task_metrics=optimized_task_metrics,
                    )
                )
                country_records.extend(
                    _country_records(
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
                execution_records.extend(
                    _training_execution_records(
                        scenario=scenario,
                        year=year,
                        method=scheduling_method,
                        countries=countries,
                        training_origin_weights=training_origin_weights,
                        source_training_load=source_training_load,
                        baseline_training_load=baseline_training_load,
                        optimized_training_load=optimized_training_load,
                        source_migrated_training_load=solver_info["source_migrated_training_load"],
                    )
                )
                if save_analysis_outputs:
                    component_energy_records.extend(
                        country_component_energy_records(
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
                        capacity_binding_summary_records(
                            scenario=scenario,
                            year=year,
                            method=scheduling_method,
                            countries=countries,
                            baseline_eval=baseline_eval,
                            optimized_eval=optimized_eval,
                        )
                    )
                    migration_flow_records.extend(
                        _migration_flow_summary_records_carbon_only(
                            scenario=scenario,
                            year=year,
                            method=scheduling_method,
                            countries=countries,
                            schedule=schedule,
                            source_country_training_load=baseline_movable_load,
                        )
                    )
                if save_schedule:
                    schedule_frames.append(
                        _schedule_records(
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
        "training_execution": pd.DataFrame(execution_records),
        "country_resource_capacity": pd.DataFrame(country_capacity_records),
        "country_component_energy": pd.DataFrame(component_energy_records),
        "migration_flow_summary": pd.DataFrame(migration_flow_records),
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
        _print_progress(verbose, "Saving optimization outputs.")
        output_path = Path(output_dir)
        os.makedirs(output_path, exist_ok=True)
        suffix = f"{renewable_energy_policy}_{migration_constraint}_{tag}"
        if transmission_policy_name != "none":
            suffix = f"{suffix}_{transmission_policy_name}"
        results["annual_summary"].to_csv(
            output_path / f"Training_Migration_Annual_Summary_{suffix}.csv",
            index=False,
        )
        results["country_summary"].to_csv(
            output_path / f"Training_Migration_Country_Summary_{suffix}.csv",
            index=False,
        )
        results["training_execution"].to_csv(
            output_path / f"Training_Migration_Execution_{suffix}.csv",
            index=False,
        )
        results["country_resource_capacity"].to_csv(
            output_path / f"Country_Resource_Capacity_{suffix}.csv",
            index=False,
        )
        results["country_component_energy"].to_csv(
            output_path / f"Country_Component_Energy_{suffix}.csv",
            index=False,
        )
        results["migration_flow_summary"].to_csv(
            output_path / f"Migration_Flow_Summary_{suffix}.csv",
            index=False,
        )
        results["transmission_policy_summary"].to_csv(
            output_path / f"Transmission_Policy_Summary_{suffix}.csv",
            index=False,
        )
        results["capacity_binding_summary"].to_csv(
            output_path / f"Capacity_Binding_Summary_{suffix}.csv",
            index=False,
        )
        results["workload_profile_summary"].to_csv(
            output_path / "Training_Migration_Workload_Profile_Summary.csv",
            index=False,
        )
        if save_schedule:
            results["migration_schedule"].to_csv(
                output_path / f"Training_Migration_Schedule_{suffix}.csv",
                index=False,
            )
        _print_progress(verbose, f"Output files saved to {os.path.abspath(output_dir)}.")

    if verbose:
        summary = results["annual_summary"][
            [
                "scenario",
                "year",
                "method",
                "migration_constraint",
                "baseline_carbon_tco2",
                "optimized_carbon_tco2",
                "carbon_reduction_pct",
                "migrated_training_share_pct",
            ]
        ]
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        if save_outputs:
            print("Saved training migration optimization results to:", os.path.abspath(output_dir))

    return results


if __name__ == "__main__":
    run_training_migration_optimization(
        renewable_energy_policy="CP",
        scenarios=["Base"],
        years=5,
        year_start=2026,
        migration_constraint="europe_only",  #  global  europe_only
        delay_hours=72,
        commit_hours=168,
        solver='gurobi',  # scipy  gurobi
        include_transmission_cost=False,  # Kept for compatibility; this variant is carbon-only.
        azure_transmission_policy="global_transfer_with_safeguards",
    )
