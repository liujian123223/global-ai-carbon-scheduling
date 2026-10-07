from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from core.workload_component_model import RESOURCES


LINEAR_PROGRAMMING_METHOD = "linear_programming"
LOCAL_IMMEDIATE_TRAINING_OVERFLOW = "local_immediate_training_overflow_constrained"
LOCAL_TRAINING_TIME_SHIFT_GREEDY = "local_training_time_shift_greedy_constrained"
ANNUAL_AVERAGE_IMMEDIATE_OVERFLOW_GREEDY = "annual_average_immediate_overflow_greedy_constrained"
HOURLY_IMMEDIATE_OVERFLOW_GREEDY = "hourly_immediate_overflow_greedy_constrained"
HOURLY_ARRIVAL_GREEDY_FALLBACK = "hourly_arrival_greedy_constrained_fallback"
SINGLE_CAPACITY_MIN_COST_FLOW = "single_capacity_min_cost_flow"

BASELINE_SCHEDULING_METHODS = (
    LOCAL_IMMEDIATE_TRAINING_OVERFLOW,
    LOCAL_TRAINING_TIME_SHIFT_GREEDY,
    ANNUAL_AVERAGE_IMMEDIATE_OVERFLOW_GREEDY,
    HOURLY_IMMEDIATE_OVERFLOW_GREEDY,
    HOURLY_ARRIVAL_GREEDY_FALLBACK,
    SINGLE_CAPACITY_MIN_COST_FLOW,
)
DEFAULT_SCHEDULING_METHODS = (LINEAR_PROGRAMMING_METHOD,) + BASELINE_SCHEDULING_METHODS
SCHEDULING_METHODS = DEFAULT_SCHEDULING_METHODS

_GREEDY_METHODS = {
    LOCAL_IMMEDIATE_TRAINING_OVERFLOW,
    LOCAL_TRAINING_TIME_SHIFT_GREEDY,
    ANNUAL_AVERAGE_IMMEDIATE_OVERFLOW_GREEDY,
    HOURLY_IMMEDIATE_OVERFLOW_GREEDY,
    HOURLY_ARRIVAL_GREEDY_FALLBACK,
}


def _empty_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_country_id",
            "source_hour",
            "execution_hour",
            "execution_country_id",
            "delay_hours",
            "share",
            "tx_cost_usd",
        ]
    )


def _active_source_country_hours(source_country_training_load: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    active = np.any(source_country_training_load > 1e-12, axis=1)
    source_country_ids, source_hour_ids = np.nonzero(active)
    return source_country_ids.astype(np.int32), source_hour_ids.astype(np.int32)


def _candidate_count(
    method_name: str,
    source_country: int,
    source_hour: int,
    allowed_destination_mask: np.ndarray,
    n_hours: int,
    delay_hours: int,
) -> int:
    last_hour = min(n_hours - 1, int(source_hour) + int(delay_hours))
    n_starts = max(0, last_hour - int(source_hour) + 1)
    if method_name in (LOCAL_IMMEDIATE_TRAINING_OVERFLOW, LOCAL_TRAINING_TIME_SHIFT_GREEDY):
        return n_starts if allowed_destination_mask[int(source_country), int(source_country)] else 0
    return int(allowed_destination_mask[int(source_country)].sum()) * n_starts


def _normalization_scale(
    benchmark_value: float,
    fallback_value: float,
) -> float:
    if benchmark_value > 1e-12:
        return float(benchmark_value)
    if fallback_value > 1e-12:
        return float(fallback_value)
    return 1.0


def _option_cost_parts(
    source_country: int,
    source_hour: int,
    destination_ids: np.ndarray,
    execution_hours: np.ndarray,
    resources: np.ndarray,
    transfer_gb: float,
    marginal_carbon_cost: np.ndarray,
    annual_marginal_carbon_cost: Optional[np.ndarray],
    tx_cost_per_gb: np.ndarray,
    carbon_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hourly_carbon = np.sum(
        marginal_carbon_cost[destination_ids, :, execution_hours] * resources[None, :],
        axis=1,
    )
    if carbon_mode == "annual":
        if annual_marginal_carbon_cost is None:
            raise ValueError("annual_marginal_carbon_cost is required for annual objective scores.")
        sort_carbon = np.sum(
            annual_marginal_carbon_cost[destination_ids] * resources[None, :],
            axis=1,
        )
    else:
        sort_carbon = hourly_carbon
    tx_cost = float(transfer_gb) * tx_cost_per_gb[int(source_country), destination_ids]
    return hourly_carbon, sort_carbon, tx_cost


def _objective_scales_for_options(
    source_country_training_load: np.ndarray,
    source_country_transfer_gb: np.ndarray,
    allowed_destination_mask: np.ndarray,
    marginal_carbon_cost: np.ndarray,
    tx_cost_per_gb: np.ndarray,
    delay_hours: int,
    objective_normalization: str,
) -> Tuple[float, float]:
    if objective_normalization == "none":
        return 1.0, 1.0
    if objective_normalization != "benchmark":
        raise ValueError("objective_normalization must be 'benchmark' or 'none'.")

    source_country_ids, source_hour_ids = _active_source_country_hours(source_country_training_load)
    n_hours = source_country_training_load.shape[2]
    carbon_benchmark = 0.0
    tx_benchmark = 0.0
    carbon_fallback = 0.0
    tx_fallback = 0.0

    for source_country, source_hour in zip(source_country_ids, source_hour_ids):
        source_country = int(source_country)
        source_hour = int(source_hour)
        resources = source_country_training_load[source_country, :, source_hour]
        transfer_gb = float(source_country_transfer_gb[source_country, source_hour])
        last_hour = min(n_hours - 1, source_hour + int(delay_hours))
        destination_base = np.flatnonzero(allowed_destination_mask[source_country]).astype(np.int32)
        if len(destination_base) == 0:
            continue
        hours = np.arange(source_hour, last_hour + 1, dtype=np.int32)
        destination_ids = np.repeat(destination_base, len(hours))
        execution_hours = np.tile(hours, len(destination_base))
        hourly_carbon, _, tx_cost = _option_cost_parts(
            source_country=source_country,
            source_hour=source_hour,
            destination_ids=destination_ids,
            execution_hours=execution_hours,
            resources=resources,
            transfer_gb=transfer_gb,
            marginal_carbon_cost=marginal_carbon_cost,
            annual_marginal_carbon_cost=None,
            tx_cost_per_gb=tx_cost_per_gb,
            carbon_mode="hourly",
        )
        local_immediate = (destination_ids == source_country) & (execution_hours == source_hour)
        if np.any(local_immediate):
            carbon_benchmark += float(hourly_carbon[local_immediate][0])
            tx_benchmark += float(tx_cost[local_immediate][0])
        else:
            carbon_fallback += float(np.max(hourly_carbon)) if len(hourly_carbon) else 0.0
            tx_fallback += float(np.max(tx_cost)) if len(tx_cost) else 0.0
        carbon_fallback += 0.0 if np.any(local_immediate) else 0.0
        if len(tx_cost):
            tx_fallback += float(np.max(tx_cost))

    return (
        _normalization_scale(carbon_benchmark, carbon_fallback),
        _normalization_scale(tx_benchmark, tx_fallback),
    )


def _combined_objective(
    carbon_cost: np.ndarray,
    tx_cost: np.ndarray,
    carbon_scale: float,
    tx_cost_scale: float,
    carbon_weight: float,
    tx_weight: float,
) -> np.ndarray:
    return (
        float(carbon_weight) * carbon_cost / float(carbon_scale)
        + float(tx_weight) * tx_cost / float(tx_cost_scale)
    )


def _candidate_arrays(
    method_name: str,
    source_country: int,
    source_hour: int,
    allowed_destination_mask: np.ndarray,
    n_hours: int,
    delay_hours: int,
    resources: np.ndarray,
    transfer_gb: float,
    marginal_carbon_cost: np.ndarray,
    annual_marginal_carbon_cost: Optional[np.ndarray],
    tx_cost_per_gb: np.ndarray,
    carbon_scale: float,
    tx_cost_scale: float,
    carbon_weight: float,
    tx_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_country = int(source_country)
    source_hour = int(source_hour)
    last_hour = min(n_hours - 1, source_hour + int(delay_hours))
    if last_hour < source_hour:
        empty = np.array([], dtype=np.int32)
        costs = np.array([], dtype=np.float64)
        return empty, empty, costs, costs, costs

    if method_name in (LOCAL_IMMEDIATE_TRAINING_OVERFLOW, LOCAL_TRAINING_TIME_SHIFT_GREEDY):
        if not allowed_destination_mask[source_country, source_country]:
            empty = np.array([], dtype=np.int32)
            costs = np.array([], dtype=np.float64)
            return empty, empty, costs, costs, costs
        execution_hours = np.arange(source_hour, last_hour + 1, dtype=np.int32)
        destination_ids = np.full(len(execution_hours), source_country, dtype=np.int32)
    else:
        destination_base = np.flatnonzero(allowed_destination_mask[source_country]).astype(np.int32)
        execution_base = np.arange(source_hour, last_hour + 1, dtype=np.int32)
        destination_ids = np.repeat(destination_base, len(execution_base))
        execution_hours = np.tile(execution_base, len(destination_base))

    hourly_carbon, sort_carbon, tx_cost = _option_cost_parts(
        source_country=source_country,
        source_hour=source_hour,
        destination_ids=destination_ids,
        execution_hours=execution_hours,
        resources=resources,
        transfer_gb=transfer_gb,
        marginal_carbon_cost=marginal_carbon_cost,
        annual_marginal_carbon_cost=annual_marginal_carbon_cost,
        tx_cost_per_gb=tx_cost_per_gb,
        carbon_mode="annual"
        if method_name == ANNUAL_AVERAGE_IMMEDIATE_OVERFLOW_GREEDY
        else "hourly",
    )
    objective = _combined_objective(
        carbon_cost=sort_carbon,
        tx_cost=tx_cost,
        carbon_scale=carbon_scale,
        tx_cost_scale=tx_cost_scale,
        carbon_weight=carbon_weight,
        tx_weight=tx_weight,
    )

    if method_name == LOCAL_IMMEDIATE_TRAINING_OVERFLOW:
        order = np.argsort(execution_hours, kind="mergesort")
    elif method_name in (LOCAL_TRAINING_TIME_SHIFT_GREEDY, HOURLY_ARRIVAL_GREEDY_FALLBACK):
        order = np.lexsort((execution_hours, objective))
    else:
        immediate = execution_hours == source_hour
        immediate_ids = np.flatnonzero(immediate)
        delayed_ids = np.flatnonzero(~immediate)
        immediate_order = immediate_ids[
            np.lexsort((destination_ids[immediate_ids], objective[immediate_ids]))
        ]
        delayed_order = delayed_ids[
            np.lexsort((objective[delayed_ids], execution_hours[delayed_ids]))
        ]
        order = np.concatenate((immediate_order, delayed_order))

    return (
        destination_ids[order],
        execution_hours[order],
        hourly_carbon[order],
        tx_cost[order],
        objective[order],
    )


def _ordered_groups(
    method_name: str,
    source_country_ids: np.ndarray,
    source_hour_ids: np.ndarray,
    allowed_destination_mask: np.ndarray,
    n_hours: int,
    delay_hours: int,
    order_mode: str,
) -> np.ndarray:
    if order_mode == "arrival":
        return np.lexsort((source_country_ids, source_hour_ids))
    if order_mode != "constrained":
        raise ValueError(f"Unknown greedy order mode: {order_mode}.")
    option_counts = np.array(
        [
            _candidate_count(
                method_name=method_name,
                source_country=int(source_country),
                source_hour=int(source_hour),
                allowed_destination_mask=allowed_destination_mask,
                n_hours=n_hours,
                delay_hours=delay_hours,
            )
            for source_country, source_hour in zip(source_country_ids, source_hour_ids)
        ],
        dtype=np.int64,
    )
    latest = np.minimum(n_hours - 1, source_hour_ids + int(delay_hours))
    return np.lexsort((source_country_ids, source_hour_ids, latest, option_counts))


def _decode_choice(
    solution: np.ndarray,
    source_country_ids: np.ndarray,
    source_hour_ids: np.ndarray,
    destination_ids: np.ndarray,
    execution_hours: np.ndarray,
    hourly_carbon_cost: np.ndarray,
    tx_cost: np.ndarray,
    source_country_training_load: np.ndarray,
    transferred_gb: np.ndarray,
    return_schedule: bool,
):
    n_countries, n_resources, n_hours = source_country_training_load.shape
    movable_load = np.zeros((n_countries, n_resources, n_hours), dtype=np.float64)
    source_tx_cost_usd = np.zeros((n_countries,), dtype=np.float64)
    source_migrated_training_load = np.zeros((n_countries,), dtype=np.float64)
    source_migrated_transfer_gb = np.zeros((n_countries,), dtype=np.float64)

    for resource_id in range(n_resources):
        contribution = solution * source_country_training_load[
            source_country_ids, resource_id, source_hour_ids
        ]
        np.add.at(movable_load[:, resource_id, :], (destination_ids, execution_hours), contribution)

    source_tx_contribution = solution * tx_cost
    np.add.at(source_tx_cost_usd, source_country_ids, source_tx_contribution)
    gpu_id = RESOURCES.index("gpu")
    migrated = destination_ids != source_country_ids
    migrated_training_contribution = (
        solution[migrated]
        * source_country_training_load[source_country_ids[migrated], gpu_id, source_hour_ids[migrated]]
    )
    migrated_transfer_contribution = solution[migrated] * transferred_gb[migrated]
    np.add.at(
        source_migrated_training_load,
        source_country_ids[migrated],
        migrated_training_contribution,
    )
    np.add.at(
        source_migrated_transfer_gb,
        source_country_ids[migrated],
        migrated_transfer_contribution,
    )

    if return_schedule:
        schedule = pd.DataFrame(
            {
                "source_country_id": source_country_ids,
                "source_hour": source_hour_ids,
                "execution_hour": execution_hours,
                "execution_country_id": destination_ids,
                "delay_hours": execution_hours - source_hour_ids,
                "share": solution,
                "tx_cost_usd": source_tx_contribution,
            }
        )
        schedule = schedule[schedule["share"] > 1e-10].reset_index(drop=True)
    else:
        schedule = _empty_schedule()

    return movable_load, schedule, {
        "carbon_objective_tco2": float(np.dot(solution, hourly_carbon_cost)),
        "tx_cost_usd": float(np.dot(solution, tx_cost)),
        "migrated_transfer_gb": float(migrated_transfer_contribution.sum()),
        "source_tx_cost_usd": source_tx_cost_usd,
        "source_migrated_training_load": source_migrated_training_load,
        "source_migrated_transfer_gb": source_migrated_transfer_gb,
    }


def _solve_greedy(
    method_name: str,
    source_country_training_load: np.ndarray,
    source_country_transfer_gb: np.ndarray,
    allowed_destination_mask: np.ndarray,
    residual_capacity: np.ndarray,
    marginal_carbon_cost: np.ndarray,
    annual_marginal_carbon_cost: Optional[np.ndarray],
    tx_cost_per_gb: np.ndarray,
    delay_hours: int,
    return_schedule: bool,
    carbon_weight: float,
    tx_weight: float,
    objective_normalization: str,
    order_mode: str,
):
    n_countries, n_resources, n_hours = source_country_training_load.shape
    source_country_ids, source_hour_ids = _active_source_country_hours(source_country_training_load)
    carbon_scale, tx_cost_scale = _objective_scales_for_options(
        source_country_training_load=source_country_training_load,
        source_country_transfer_gb=source_country_transfer_gb,
        allowed_destination_mask=allowed_destination_mask,
        marginal_carbon_cost=marginal_carbon_cost,
        tx_cost_per_gb=tx_cost_per_gb,
        delay_hours=delay_hours,
        objective_normalization=objective_normalization,
    )
    group_order = _ordered_groups(
        method_name=method_name,
        source_country_ids=source_country_ids,
        source_hour_ids=source_hour_ids,
        allowed_destination_mask=allowed_destination_mask,
        n_hours=n_hours,
        delay_hours=delay_hours,
        order_mode=order_mode,
    )

    usage = np.zeros_like(residual_capacity)
    reservation_remaining = source_country_training_load.copy()
    movable_load = np.zeros((n_countries, n_resources, n_hours), dtype=np.float64)
    source_tx_cost_usd = np.zeros((n_countries,), dtype=np.float64)
    source_migrated_training_load = np.zeros((n_countries,), dtype=np.float64)
    source_migrated_transfer_gb = np.zeros((n_countries,), dtype=np.float64)
    rows = []
    objective = 0.0
    carbon_objective = 0.0
    tx_total = 0.0
    variables = 0
    gpu_id = RESOURCES.index("gpu")
    positive_tolerance = 1e-12

    for ordered_id in group_order:
        source_country = int(source_country_ids[ordered_id])
        source_hour = int(source_hour_ids[ordered_id])
        resources = source_country_training_load[source_country, :, source_hour]
        transfer_gb = float(source_country_transfer_gb[source_country, source_hour])
        reservation_remaining[source_country, :, source_hour] -= resources
        reservation_remaining[source_country, :, source_hour] = np.maximum(
            reservation_remaining[source_country, :, source_hour],
            0.0,
        )
        destination_ids, execution_hours, carbon_cost, tx_cost, option_objective = _candidate_arrays(
            method_name=method_name,
            source_country=source_country,
            source_hour=source_hour,
            allowed_destination_mask=allowed_destination_mask,
            n_hours=n_hours,
            delay_hours=delay_hours,
            resources=resources,
            transfer_gb=transfer_gb,
            marginal_carbon_cost=marginal_carbon_cost,
            annual_marginal_carbon_cost=annual_marginal_carbon_cost,
            tx_cost_per_gb=tx_cost_per_gb,
            carbon_scale=carbon_scale,
            tx_cost_scale=tx_cost_scale,
            carbon_weight=carbon_weight,
            tx_weight=tx_weight,
        )
        variables += len(destination_ids)
        if len(destination_ids) == 0:
            raise ValueError(
                f"{method_name} has no candidate for source_country={source_country}, "
                f"source_hour={source_hour}."
            )

        remaining = 1.0
        positive_resources = resources > positive_tolerance
        for destination, execution, candidate_carbon, candidate_tx, candidate_objective in zip(
            destination_ids,
            execution_hours,
            carbon_cost,
            tx_cost,
            option_objective,
        ):
            destination = int(destination)
            execution = int(execution)
            if np.any(positive_resources):
                slack = (
                    residual_capacity[destination, positive_resources, execution]
                    - usage[destination, positive_resources, execution]
                    - reservation_remaining[destination, positive_resources, execution]
                )
                max_fraction = float(np.min(slack / resources[positive_resources]))
            else:
                max_fraction = 1.0
            take = min(remaining, max(0.0, max_fraction))
            if take <= 1e-10:
                continue

            usage[destination, :, execution] += resources * take
            movable_load[destination, :, execution] += resources * take
            source_tx_cost_usd[source_country] += float(candidate_tx) * take
            if destination != source_country:
                source_migrated_training_load[source_country] += resources[gpu_id] * take
                source_migrated_transfer_gb[source_country] += transfer_gb * take
            objective += float(candidate_objective) * take
            carbon_objective += float(candidate_carbon) * take
            tx_total += float(candidate_tx) * take
            if return_schedule:
                rows.append(
                    {
                        "source_country_id": source_country,
                        "source_hour": source_hour,
                        "execution_hour": execution,
                        "execution_country_id": destination,
                        "delay_hours": execution - source_hour,
                        "share": take,
                        "tx_cost_usd": float(candidate_tx) * take,
                    }
                )
            remaining -= take
            if remaining <= 1e-9:
                break

        if remaining > 1e-8:
            raise ValueError(
                f"{method_name} could not place source_country={source_country}, "
                f"source_hour={source_hour}, remaining={remaining:.6g}."
            )

    schedule = pd.DataFrame(rows) if return_schedule and rows else _empty_schedule()
    return movable_load, schedule, {
        "status": 0,
        "message": f"{method_name} completed.",
        "objective": float(objective),
        "carbon_objective_tco2": float(carbon_objective),
        "tx_cost_usd": float(tx_total),
        "migrated_transfer_gb": float(source_migrated_transfer_gb.sum()),
        "source_tx_cost_usd": source_tx_cost_usd,
        "source_migrated_training_load": source_migrated_training_load,
        "source_migrated_transfer_gb": source_migrated_transfer_gb,
        "variables": int(variables),
        "windows": 1,
        "solver": "greedy",
        "solve_mode": "greedy",
    }


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
        last_hour = min(n_hours - 1, int(source_hour) + int(delay_hours))
        execution_hours = np.arange(int(source_hour), last_hour + 1, dtype=np.int32)
        n_local_variables = len(destinations) * len(execution_hours)
        source_country_parts.append(np.full(n_local_variables, int(source_country), dtype=np.int32))
        source_hour_parts.append(np.full(n_local_variables, int(source_hour), dtype=np.int32))
        destination_parts.append(np.repeat(destinations, len(execution_hours)))
        execution_parts.append(np.tile(execution_hours, len(destinations)))
        group_parts.append(np.full(n_local_variables, group_id, dtype=np.int32))

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


def _solve_single_capacity_window(
    source_country_training_load: np.ndarray,
    source_country_transfer_gb: np.ndarray,
    allowed_destination_mask: np.ndarray,
    residual_capacity: np.ndarray,
    marginal_carbon_cost: np.ndarray,
    tx_cost_per_gb: np.ndarray,
    delay_hours: int,
    return_schedule: bool,
    carbon_weight: float,
    tx_weight: float,
    objective_normalization: str,
    linprog_options: Optional[Mapping[str, Union[bool, float, int]]] = None,
):
    n_countries, n_resources, n_hours = source_country_training_load.shape
    source_country_base, source_hour_base = _active_source_country_hours(source_country_training_load)
    if len(source_hour_base) == 0:
        return np.zeros_like(source_country_training_load), _empty_schedule(), {
            "status": 0,
            "message": "No movable training workload.",
            "objective": 0.0,
            "carbon_objective_tco2": 0.0,
            "tx_cost_usd": 0.0,
            "migrated_transfer_gb": 0.0,
            "source_tx_cost_usd": np.zeros((n_countries,), dtype=np.float64),
            "source_migrated_training_load": np.zeros((n_countries,), dtype=np.float64),
            "source_migrated_transfer_gb": np.zeros((n_countries,), dtype=np.float64),
            "variables": 0,
        }

    source_country_ids, source_hour_ids, destination_ids, execution_hours, group_ids = _build_variable_index(
        active_source_country_ids=source_country_base,
        active_source_hour_ids=source_hour_base,
        allowed_destination_mask=allowed_destination_mask,
        n_hours=n_hours,
        delay_hours=delay_hours,
    )
    resources_by_variable = source_country_training_load[source_country_ids, :, source_hour_ids]
    carbon_cost = np.sum(
        marginal_carbon_cost[destination_ids, :, execution_hours] * resources_by_variable,
        axis=1,
    )
    transferred_gb = source_country_transfer_gb[source_country_ids, source_hour_ids]
    tx_cost = transferred_gb * tx_cost_per_gb[source_country_ids, destination_ids]
    benchmark_mask = (destination_ids == source_country_ids) & (execution_hours == source_hour_ids)

    if objective_normalization == "none":
        carbon_scale, tx_cost_scale = 1.0, 1.0
    else:
        carbon_scale = float(carbon_cost[benchmark_mask].sum())
        if carbon_scale <= 1e-12:
            group_max = np.zeros((len(source_country_base),), dtype=np.float64)
            np.maximum.at(group_max, group_ids, carbon_cost)
            carbon_scale = float(group_max.sum()) if group_max.sum() > 1e-12 else 1.0
        tx_cost_scale = float(tx_cost[benchmark_mask].sum())
        if tx_cost_scale <= 1e-12:
            group_max = np.zeros((len(source_country_base),), dtype=np.float64)
            np.maximum.at(group_max, group_ids, tx_cost)
            tx_cost_scale = float(group_max.sum()) if group_max.sum() > 1e-12 else 1.0

    cost = _combined_objective(
        carbon_cost=carbon_cost,
        tx_cost=tx_cost,
        carbon_scale=carbon_scale,
        tx_cost_scale=tx_cost_scale,
        carbon_weight=carbon_weight,
        tx_weight=tx_weight,
    )

    option_residual = residual_capacity[destination_ids, :, execution_hours]
    positive_resource = resources_by_variable > 1e-12
    infeasible_option = np.any(positive_resource & (option_residual <= 1e-12), axis=1)
    normalized_usage = np.divide(
        resources_by_variable,
        option_residual,
        out=np.zeros_like(resources_by_variable),
        where=option_residual > 1e-12,
    )
    capacity_coeff = np.max(normalized_usage, axis=1)
    infeasible_option |= capacity_coeff > (1.0 + 1e-8)
    keep = ~infeasible_option
    group_counts = np.bincount(group_ids[keep], minlength=len(source_country_base))
    missing_groups = np.flatnonzero(group_counts == 0)
    if len(missing_groups):
        preview = ", ".join(str(int(idx)) for idx in missing_groups[:10])
        raise ValueError(
            "single_capacity_min_cost_flow has no feasible option for "
            f"{len(missing_groups)} source groups; group preview: {preview}."
        )

    source_country_ids = source_country_ids[keep]
    source_hour_ids = source_hour_ids[keep]
    destination_ids = destination_ids[keep]
    execution_hours = execution_hours[keep]
    group_ids = group_ids[keep]
    resources_by_variable = resources_by_variable[keep]
    carbon_cost = carbon_cost[keep]
    transferred_gb = transferred_gb[keep]
    tx_cost = tx_cost[keep]
    cost = cost[keep]
    capacity_coeff = capacity_coeff[keep]
    n_variables = len(source_country_ids)
    variable_ids = np.arange(n_variables, dtype=np.int32)

    equality_matrix = sparse.coo_matrix(
        (
            np.ones((n_variables,), dtype=np.float64),
            (group_ids, variable_ids),
        ),
        shape=(len(source_country_base), n_variables),
    ).tocsr()
    equality_rhs = np.ones((len(source_country_base),), dtype=np.float64)

    positive = capacity_coeff > 1e-12
    if np.any(positive):
        capacity_rows = destination_ids[positive].astype(np.int64) * int(n_hours) + execution_hours[
            positive
        ].astype(np.int64)
        unique_capacity_rows, compact_rows = np.unique(capacity_rows, return_inverse=True)
        inequality_matrix = sparse.coo_matrix(
            (
                capacity_coeff[positive],
                (compact_rows, variable_ids[positive].astype(np.int64)),
            ),
            shape=(len(unique_capacity_rows), n_variables),
        ).tocsr()
        inequality_rhs = np.ones((len(unique_capacity_rows),), dtype=np.float64)
    else:
        inequality_matrix = sparse.csr_matrix((0, n_variables))
        inequality_rhs = np.zeros((0,), dtype=np.float64)

    if not np.all(np.isfinite(cost)):
        raise ValueError("single_capacity_min_cost_flow objective contains NaN or infinite values.")
    if not np.all(np.isfinite(inequality_matrix.data)):
        raise ValueError("single_capacity_min_cost_flow capacity matrix contains NaN or infinite values.")
    if not np.all(np.isfinite(equality_rhs)) or not np.all(np.isfinite(inequality_rhs)):
        raise ValueError("single_capacity_min_cost_flow right-hand side contains NaN or infinite values.")

    result = linprog(
        c=cost,
        A_ub=inequality_matrix,
        b_ub=np.maximum(inequality_rhs, 0.0),
        A_eq=equality_matrix,
        b_eq=equality_rhs,
        bounds=(0, None),
        method="highs",
        options={"presolve": True, **dict(linprog_options or {})},
    )
    if not result.success:
        positive_coeff = capacity_coeff[np.isfinite(capacity_coeff) & (capacity_coeff > 0)]
        coeff_summary = (
            f"coeff_min={float(positive_coeff.min()):.6g}, "
            f"coeff_max={float(positive_coeff.max()):.6g}"
            if len(positive_coeff)
            else "no_positive_coefficients"
        )
        raise RuntimeError(
            "single_capacity_min_cost_flow failed: "
            f"{result.message}; variables={n_variables}, "
            f"filtered_options={int((~keep).sum())}, {coeff_summary}"
        )

    solution = np.asarray(result.x, dtype=np.float64)
    movable_load, schedule, info = _decode_choice(
        solution=solution,
        source_country_ids=source_country_ids,
        source_hour_ids=source_hour_ids,
        destination_ids=destination_ids,
        execution_hours=execution_hours,
        hourly_carbon_cost=carbon_cost,
        tx_cost=tx_cost,
        source_country_training_load=source_country_training_load,
        transferred_gb=transferred_gb,
        return_schedule=return_schedule,
    )
    info.update(
        {
            "status": int(result.status),
            "message": str(result.message),
            "objective": float(result.fun),
            "variables": int(n_variables),
        }
    )
    return movable_load, schedule, info


def _solve_single_capacity_min_cost_flow(
    source_country_training_load: np.ndarray,
    source_country_transfer_gb: np.ndarray,
    allowed_destination_mask: np.ndarray,
    capacity_limit: np.ndarray,
    fixed_for_optimization: np.ndarray,
    baseline_movable_load: np.ndarray,
    marginal_carbon_cost: np.ndarray,
    tx_cost_per_gb: np.ndarray,
    delay_hours: int,
    commit_hours: int,
    return_schedule: bool,
    carbon_weight: float,
    tx_weight: float,
    objective_normalization: str,
    linprog_options: Optional[Mapping[str, Union[bool, float, int]]] = None,
):
    if commit_hours <= 0:
        raise ValueError("commit_hours must be positive.")
    n_countries, _, n_hours = source_country_training_load.shape
    effective_commit_hours = int(commit_hours)
    n_windows = int(np.ceil(n_hours / effective_commit_hours))
    scheduled_movable_load = np.zeros_like(baseline_movable_load)
    reservation_remaining = baseline_movable_load.copy()
    schedule_frames = []
    total_objective = 0.0
    total_carbon_objective = 0.0
    total_tx_cost_usd = 0.0
    total_migrated_transfer_gb = 0.0
    total_variables = 0
    source_tx_cost_usd = np.zeros((n_countries,), dtype=np.float64)
    source_migrated_training_load = np.zeros((n_countries,), dtype=np.float64)
    source_migrated_transfer_gb = np.zeros((n_countries,), dtype=np.float64)

    for source_start in range(0, n_hours, effective_commit_hours):
        source_end = min(n_hours, source_start + effective_commit_hours)
        reservation_remaining[:, :, source_start:source_end] = 0.0
        local_source_load = np.zeros_like(source_country_training_load)
        local_source_load[:, :, source_start:source_end] = source_country_training_load[
            :, :, source_start:source_end
        ]
        local_source_transfer_gb = np.zeros_like(source_country_transfer_gb)
        local_source_transfer_gb[:, source_start:source_end] = source_country_transfer_gb[
            :, source_start:source_end
        ]
        if not np.any(local_source_load > 1e-12):
            continue

        committed_load = fixed_for_optimization + scheduled_movable_load
        available_capacity = np.maximum(capacity_limit - committed_load, 0.0)
        effective_reservation = np.minimum(reservation_remaining, available_capacity)
        residual_capacity = np.maximum(capacity_limit - committed_load - effective_reservation, 0.0)

        window_load, window_schedule, window_info = _solve_single_capacity_window(
            source_country_training_load=local_source_load,
            source_country_transfer_gb=local_source_transfer_gb,
            allowed_destination_mask=allowed_destination_mask,
            residual_capacity=residual_capacity,
            marginal_carbon_cost=marginal_carbon_cost,
            tx_cost_per_gb=tx_cost_per_gb,
            delay_hours=delay_hours,
            return_schedule=return_schedule,
            carbon_weight=carbon_weight,
            tx_weight=tx_weight,
            objective_normalization=objective_normalization,
            linprog_options=linprog_options,
        )
        scheduled_movable_load += window_load
        total_objective += window_info["objective"]
        total_carbon_objective += window_info["carbon_objective_tco2"]
        total_tx_cost_usd += window_info["tx_cost_usd"]
        total_migrated_transfer_gb += window_info["migrated_transfer_gb"]
        total_variables += window_info["variables"]
        source_tx_cost_usd += window_info["source_tx_cost_usd"]
        source_migrated_training_load += window_info["source_migrated_training_load"]
        source_migrated_transfer_gb += window_info["source_migrated_transfer_gb"]
        if return_schedule and not window_schedule.empty:
            schedule_frames.append(window_schedule)

    schedule = pd.concat(schedule_frames, ignore_index=True) if schedule_frames else _empty_schedule()
    return scheduled_movable_load, schedule, {
        "status": 0,
        "message": "single_capacity_min_cost_flow completed.",
        "objective": float(total_objective),
        "carbon_objective_tco2": float(total_carbon_objective),
        "tx_cost_usd": float(total_tx_cost_usd),
        "migrated_transfer_gb": float(total_migrated_transfer_gb),
        "source_tx_cost_usd": source_tx_cost_usd,
        "source_migrated_training_load": source_migrated_training_load,
        "source_migrated_transfer_gb": source_migrated_transfer_gb,
        "variables": int(total_variables),
        "windows": int(n_windows),
        "commit_hours": int(effective_commit_hours),
        "solver": "single_capacity_min_cost_flow",
        "solve_mode": "windowed",
    }


def solve_training_migration_baseline_method(
    method_name: str,
    source_country_training_load: np.ndarray,
    source_country_transfer_gb: np.ndarray,
    allowed_destination_mask: np.ndarray,
    residual_capacity: np.ndarray,
    capacity_limit: np.ndarray,
    fixed_for_optimization: np.ndarray,
    baseline_movable_load: np.ndarray,
    marginal_carbon_cost: np.ndarray,
    annual_marginal_carbon_cost: Optional[np.ndarray],
    tx_cost_per_gb: np.ndarray,
    delay_hours: int,
    commit_hours: int,
    return_schedule: bool,
    carbon_weight: float,
    tx_weight: float,
    objective_normalization: str,
    linprog_options: Optional[Mapping[str, Union[bool, float, int]]] = None,
):
    if method_name in _GREEDY_METHODS:
        if method_name == HOURLY_ARRIVAL_GREEDY_FALLBACK:
            try:
                return _solve_greedy(
                    method_name=method_name,
                    source_country_training_load=source_country_training_load,
                    source_country_transfer_gb=source_country_transfer_gb,
                    allowed_destination_mask=allowed_destination_mask,
                    residual_capacity=residual_capacity,
                    marginal_carbon_cost=marginal_carbon_cost,
                    annual_marginal_carbon_cost=annual_marginal_carbon_cost,
                    tx_cost_per_gb=tx_cost_per_gb,
                    delay_hours=delay_hours,
                    return_schedule=return_schedule,
                    carbon_weight=carbon_weight,
                    tx_weight=tx_weight,
                    objective_normalization=objective_normalization,
                    order_mode="arrival",
                )
            except ValueError:
                result = _solve_greedy(
                    method_name=method_name,
                    source_country_training_load=source_country_training_load,
                    source_country_transfer_gb=source_country_transfer_gb,
                    allowed_destination_mask=allowed_destination_mask,
                    residual_capacity=residual_capacity,
                    marginal_carbon_cost=marginal_carbon_cost,
                    annual_marginal_carbon_cost=annual_marginal_carbon_cost,
                    tx_cost_per_gb=tx_cost_per_gb,
                    delay_hours=delay_hours,
                    return_schedule=return_schedule,
                    carbon_weight=carbon_weight,
                    tx_weight=tx_weight,
                    objective_normalization=objective_normalization,
                    order_mode="constrained",
                )
                result[2]["message"] = (
                    "hourly_arrival_greedy was infeasible; constrained fallback completed."
                )
                return result

        return _solve_greedy(
            method_name=method_name,
            source_country_training_load=source_country_training_load,
            source_country_transfer_gb=source_country_transfer_gb,
            allowed_destination_mask=allowed_destination_mask,
            residual_capacity=residual_capacity,
            marginal_carbon_cost=marginal_carbon_cost,
            annual_marginal_carbon_cost=annual_marginal_carbon_cost,
            tx_cost_per_gb=tx_cost_per_gb,
            delay_hours=delay_hours,
            return_schedule=return_schedule,
            carbon_weight=carbon_weight,
            tx_weight=tx_weight,
            objective_normalization=objective_normalization,
            order_mode="constrained",
        )

    if method_name == SINGLE_CAPACITY_MIN_COST_FLOW:
        return _solve_single_capacity_min_cost_flow(
            source_country_training_load=source_country_training_load,
            source_country_transfer_gb=source_country_transfer_gb,
            allowed_destination_mask=allowed_destination_mask,
            capacity_limit=capacity_limit,
            fixed_for_optimization=fixed_for_optimization,
            baseline_movable_load=baseline_movable_load,
            marginal_carbon_cost=marginal_carbon_cost,
            tx_cost_per_gb=tx_cost_per_gb,
            delay_hours=delay_hours,
            commit_hours=commit_hours,
            return_schedule=return_schedule,
            carbon_weight=carbon_weight,
            tx_weight=tx_weight,
            objective_normalization=objective_normalization,
            linprog_options=linprog_options,
        )

    raise ValueError(f"Unknown scheduling method: {method_name}.")
