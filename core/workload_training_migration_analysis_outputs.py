from typing import Dict, Sequence

import numpy as np
import pandas as pd

from core.workload_component_model import (
    COMPONENTS,
    RESOURCES,
    _component_power_timeseries,
)


def country_resource_capacity_records(
    scenario: str,
    year: int,
    countries: Sequence[str],
    country_share: np.ndarray,
    country_it_mw: np.ndarray,
    resource_capacities: np.ndarray,
    pue: np.ndarray,
    ai_capacity_factor: float,
    total_data_center_global_it_mw: float,
    global_ai_it_mw: float,
    max_resource_utilization: float,
):
    records = []
    effective_capacities = resource_capacities * float(max_resource_utilization)
    for country_id, country in enumerate(countries):
        record = {
            "scenario": scenario,
            "year": year,
            "country": country,
            "country_capacity_share": float(country_share[country_id]),
            "country_ai_it_mw": float(country_it_mw[country_id]),
            "ai_capacity_factor": float(ai_capacity_factor),
            "total_data_center_global_it_mw": float(total_data_center_global_it_mw),
            "global_ai_it_mw": float(global_ai_it_mw),
            "pue": float(pue[country_id]),
            "max_resource_utilization": float(max_resource_utilization),
        }
        for resource_id, resource in enumerate(RESOURCES):
            record[f"{resource}_capacity"] = float(resource_capacities[country_id, resource_id])
            record[f"{resource}_capacity_limit"] = float(effective_capacities[country_id, resource_id])
        records.append(record)
    return records


def country_component_energy_records(
    scenario: str,
    year: int,
    method: str,
    countries: Sequence[str],
    country_it_mw: np.ndarray,
    baseline_resource_load: np.ndarray,
    optimized_resource_load: np.ndarray,
    resource_capacities: np.ndarray,
    pue: np.ndarray,
    hardware_config,
    max_resource_utilization: float,
):
    baseline_it = _component_it_energy_by_country(
        country_it_mw=country_it_mw,
        country_resource_load=baseline_resource_load,
        resource_capacities=resource_capacities,
        hardware_config=hardware_config,
        max_resource_utilization=max_resource_utilization,
    )
    optimized_it = _component_it_energy_by_country(
        country_it_mw=country_it_mw,
        country_resource_load=optimized_resource_load,
        resource_capacities=resource_capacities,
        hardware_config=hardware_config,
        max_resource_utilization=max_resource_utilization,
    )

    records = []
    for country_id, country in enumerate(countries):
        for component_id, component in enumerate(COMPONENTS):
            baseline_it_mwh = float(baseline_it[country_id, component_id])
            optimized_it_mwh = float(optimized_it[country_id, component_id])
            baseline_facility_mwh = baseline_it_mwh * float(pue[country_id])
            optimized_facility_mwh = optimized_it_mwh * float(pue[country_id])
            records.append(
                {
                    "scenario": scenario,
                    "year": year,
                    "method": method,
                    "country": country,
                    "component": component,
                    "baseline_it_energy_mwh": baseline_it_mwh,
                    "optimized_it_energy_mwh": optimized_it_mwh,
                    "it_energy_change_mwh": optimized_it_mwh - baseline_it_mwh,
                    "baseline_facility_energy_mwh": baseline_facility_mwh,
                    "optimized_facility_energy_mwh": optimized_facility_mwh,
                    "facility_energy_change_mwh": optimized_facility_mwh - baseline_facility_mwh,
                }
            )
    return records


def _component_it_energy_by_country(
    country_it_mw: np.ndarray,
    country_resource_load: np.ndarray,
    resource_capacities: np.ndarray,
    hardware_config,
    max_resource_utilization: float,
) -> np.ndarray:
    utilization = np.clip(
        country_resource_load / resource_capacities[:, :, None],
        0.0,
        float(max_resource_utilization),
    )
    component_power_mw = _component_power_timeseries(
        country_it_mw=country_it_mw,
        resource_utilization=utilization,
        config=hardware_config,
    )
    return component_power_mw.sum(axis=2).T


def migration_flow_summary_records(
    scenario: str,
    year: int,
    method: str,
    countries: Sequence[str],
    schedule: pd.DataFrame,
    source_country_training_load: np.ndarray,
    source_country_transfer_gb: np.ndarray,
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
        "tx_cost_usd",
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
    df["transfer_gb"] = shares * source_country_transfer_gb[source_country_ids, source_hour_ids]
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
        "transfer_gb": ("transfer_gb", "sum"),
        "tx_cost_usd": ("tx_cost_usd", "sum"),
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


def transmission_policy_summary_records(
    policy_name: str,
    matrix_path,
    countries: Sequence[str],
    tx_cost_per_gb: np.ndarray,
    allowed_mask: np.ndarray,
    policy_block_cost: float,
):
    n_countries = len(countries)
    off_diagonal = ~np.eye(n_countries, dtype=bool)
    allowed_off_diagonal = allowed_mask & off_diagonal
    blocked_off_diagonal = (~allowed_mask) & off_diagonal
    allowed_costs = tx_cost_per_gb[allowed_off_diagonal]
    positive_allowed_costs = allowed_costs[allowed_costs > 0]

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
            "policy_block_cost": float(policy_block_cost),
            "mean_allowed_cost_usd_per_gb": _safe_stat(positive_allowed_costs, np.mean),
            "median_allowed_cost_usd_per_gb": _safe_stat(positive_allowed_costs, np.median),
            "min_allowed_cost_usd_per_gb": _safe_stat(positive_allowed_costs, np.min),
            "max_allowed_cost_usd_per_gb": _safe_stat(positive_allowed_costs, np.max),
        }
    ]


def _safe_stat(values: np.ndarray, fn) -> float:
    if values.size == 0:
        return float("nan")
    return float(fn(values))


def capacity_binding_summary_records(
    scenario: str,
    year: int,
    method: str,
    countries: Sequence[str],
    baseline_eval: Dict[str, np.ndarray],
    optimized_eval: Dict[str, np.ndarray],
    thresholds=(0.90, 0.95, 0.99),
):
    records = []
    baseline_util = baseline_eval["utilization"]
    optimized_util = optimized_eval["utilization"]
    optimized_overflow = optimized_eval["overflow"]

    for country_id, country in enumerate(countries):
        for resource_id, resource in enumerate(RESOURCES):
            baseline_series = baseline_util[country_id, resource_id]
            optimized_series = optimized_util[country_id, resource_id]
            record = {
                "scenario": scenario,
                "year": year,
                "method": method,
                "country": country,
                "resource": resource,
                "baseline_mean_utilization": float(np.mean(baseline_series)),
                "optimized_mean_utilization": float(np.mean(optimized_series)),
                "baseline_peak_utilization": float(np.max(baseline_series)),
                "optimized_peak_utilization": float(np.max(optimized_series)),
                "optimized_overflow_resource_hours": float(
                    optimized_overflow[country_id, resource_id].sum()
                ),
            }
            for threshold in thresholds:
                pct = int(round(threshold * 100))
                record[f"baseline_hours_ge_{pct}pct_utilization"] = int(
                    np.count_nonzero(baseline_series >= threshold)
                )
                record[f"optimized_hours_ge_{pct}pct_utilization"] = int(
                    np.count_nonzero(optimized_series >= threshold)
                )
            records.append(record)
    return records
