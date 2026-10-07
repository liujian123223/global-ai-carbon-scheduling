#!/usr/bin/env python3
"""Validate one 831 output directory and the manuscript strategy boundaries."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_LABELS = OrderedDict(
    (
        ("local_immediate_training_overflow_constrained", "LI"),
        ("local_training_time_shift_greedy_constrained", "TM"),
        ("annual_average_immediate_overflow_greedy_constrained", "SM"),
        ("linear_programming", "JSM"),
    )
)

NEUTRAL_PARAMETERS = OrderedDict(
    (
        ("gpu_idle_power_fraction", 0.10),
        ("training_gpu_load_scale", 1.0),
        ("offline_inference_gpu_load_scale", 1.0),
        ("flexible_gpu_share_scale", 1.0),
        ("gpu_power_share", 0.50),
        ("hourly_carbon_variability_scale", 1.0),
        ("gpu_power_curve_alpha", 1.0),
        ("spatial_carbon_contrast_scale", 1.0),
        ("usable_residual_capacity_fraction", 1.0),
    )
)


def _find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one file matching {pattern!r} in {directory}; "
            f"found {len(matches)}."
        )
    if matches[0].stat().st_size == 0:
        raise AssertionError(f"Output file is empty: {matches[0]}")
    return matches[0]


def _as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def validate_output(
    output_dir: Path,
    expected_parameters: dict[str, float] | None = None,
) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        raise AssertionError(f"Output directory does not exist: {output_dir}")

    expected = dict(NEUTRAL_PARAMETERS)
    if expected_parameters:
        unknown = sorted(set(expected_parameters) - set(expected))
        if unknown:
            raise AssertionError(f"Unknown sensitivity parameters: {unknown}")
        expected.update(expected_parameters)

    annual_path = _find_one(output_dir, "Training_Migration_Annual_Summary_*.csv")
    annual = pd.read_csv(annual_path)
    required_annual = {
        "scenario",
        "year",
        "method",
        "paper_strategy",
        "solver_status",
        "migration_constraint",
        "transmission_policy",
        *expected.keys(),
    }
    missing = sorted(required_annual - set(annual.columns))
    if missing:
        raise AssertionError(f"Annual summary is missing required columns: {missing}")

    if len(annual) != len(METHOD_LABELS):
        raise AssertionError(
            f"Expected {len(METHOD_LABELS)} annual rows (LI/TM/SM/JSM); found {len(annual)}."
        )
    methods = annual["method"].astype(str)
    if set(methods) != set(METHOD_LABELS) or methods.duplicated().any():
        raise AssertionError(
            "Annual summary must contain each of LI, TM, SM, and JSM exactly once; "
            f"found {methods.tolist()}."
        )
    observed_labels = annual.set_index("method")["paper_strategy"].astype(str).to_dict()
    if observed_labels != dict(METHOD_LABELS):
        raise AssertionError(
            f"Paper-strategy labels differ from the manuscript mapping: {observed_labels}."
        )

    statuses = pd.to_numeric(annual["solver_status"], errors="coerce")
    if statuses.isna().any() or not (statuses == 0).all():
        failed = annual.loc[statuses != 0, ["method", "solver_status", "solver_message"]]
        raise AssertionError(f"At least one strategy failed:\n{failed.to_string(index=False)}")
    if set(annual["scenario"].astype(str)) != {"Base"}:
        raise AssertionError("831 sensitivity outputs must use the Base scenario only.")
    if set(pd.to_numeric(annual["year"], errors="coerce")) != {2030}:
        raise AssertionError("831 sensitivity outputs must use year 2030 only.")
    if set(annual["migration_constraint"].astype(str)) != {"global"}:
        raise AssertionError("831 sensitivity outputs must use the global migration constraint.")
    if set(annual["transmission_policy"].astype(str)) != {
        "global_transfer_with_safeguards"
    }:
        raise AssertionError(
            "831 sensitivity outputs must use global_transfer_with_safeguards."
        )

    for parameter, expected_value in expected.items():
        values = pd.to_numeric(annual[parameter], errors="coerce").to_numpy()
        if not np.isfinite(values).all() or not np.allclose(
            values,
            expected_value,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise AssertionError(
                f"Unexpected {parameter}: expected {expected_value}, observed {values.tolist()}."
            )

    flow_path = _find_one(output_dir, "Flexible_Migration_Flow_Summary_*.csv")
    flow = pd.read_csv(flow_path)
    required_flow = {"method", "is_cross_country", "average_delay_hours"}
    missing = sorted(required_flow - set(flow.columns))
    if missing:
        raise AssertionError(f"Flexible-flow summary is missing columns: {missing}")
    flow = flow[flow["method"].astype(str).isin(METHOD_LABELS)].copy()
    cross_country = _as_boolean(flow["is_cross_country"])
    delay = pd.to_numeric(flow["average_delay_hours"], errors="coerce").fillna(0.0)

    no_spatial = flow["method"].isin(
        {
            "local_immediate_training_overflow_constrained",
            "local_training_time_shift_greedy_constrained",
        }
    )
    if cross_country[no_spatial].any():
        raise AssertionError("LI or TM contains cross-country migration.")
    no_temporal = flow["method"].isin(
        {
            "local_immediate_training_overflow_constrained",
            "annual_average_immediate_overflow_greedy_constrained",
        }
    )
    if (np.abs(delay[no_temporal]) > 1.0e-10).any():
        raise AssertionError("LI or SM contains temporal migration.")

    return {"annual": annual_path, "flow": flow_path}


def _parse_expectations(items: list[str]) -> dict[str, float]:
    expectations: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--expect must use PARAMETER=VALUE syntax: {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key not in NEUTRAL_PARAMETERS:
            raise ValueError(f"Unknown sensitivity parameter: {key}")
        expectations[key] = float(raw_value)
    return expectations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="PARAMETER=VALUE",
        help="Override one neutral sensitivity value when validating a perturbed case.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    paths = validate_output(args.output_dir, _parse_expectations(args.expect))
    if not args.quiet:
        print(f"831 output validation passed: {args.output_dir.resolve()}")
        print(f"  annual: {paths['annual'].name}")
        print(f"  flow:   {paths['flow'].name}")


if __name__ == "__main__":
    main()
