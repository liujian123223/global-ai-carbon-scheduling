import argparse
import importlib.util
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Alibaba GPU v2026 training migration optimization with "
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
        help="global allows all selected countries; europe_only moves European training within Europe.",
    )
    parser.add_argument("--delay-hours", type=int, default=72)
    parser.add_argument("--solve-mode", choices=opt.SOLVE_MODES, default="windowed")
    parser.add_argument("--solver", choices=opt.SOLVERS, default="scipy")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[opt.LINEAR_PROGRAMMING_METHOD],
        choices=["all", *opt.SCHEDULING_METHODS],
    )
    parser.add_argument("--commit-hours", type=int, default=168)
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
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    linprog_options = {}
    if args.linprog_time_limit is not None:
        linprog_options["time_limit"] = args.linprog_time_limit

    results = opt.run_training_migration_optimization(
        renewable_energy_policy=args.policy,
        scenarios=args.scenarios,
        years=args.years,
        countries=args.countries,
        workload_profile_path=Path(args.workload_path),
        server_profile_path=Path(args.server_profile_path) if args.server_profile_path else None,
        network_profile_path=Path(args.network_profile_path) if args.network_profile_path else None,
        year_start=args.year_start,
        output_dir=args.output_dir,
        save_outputs=not args.no_save,
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
        workload_cache_dir=None if args.no_workload_cache else Path(args.workload_cache_dir),
        refresh_workload_cache=args.refresh_workload_cache,
        solve_mode=args.solve_mode,
        commit_hours=args.commit_hours,
        monolithic_variable_limit=args.monolithic_variable_limit,
        solver=args.solver,
        linprog_options=linprog_options or None,
        policy_transmission_matrix=Path(args.policy_transmission_matrix)
        if args.policy_transmission_matrix
        else None,
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
