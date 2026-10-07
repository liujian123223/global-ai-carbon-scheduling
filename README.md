# Global AI Carbon-Aware Scheduling

English | [简体中文](README.zh-CN.md)

Research code for hourly carbon-aware scheduling of AI workloads across 24 countries over 2026–2030. The experiments compare local immediate execution (LI), temporal migration (TM), spatial migration (SM), and joint spatiotemporal migration (JSM) under different demand, grid-decarbonization, and cross-border governance scenarios.

## Project structure

```text
global-ai-carbon-scheduling/
├── core/
│   ├── workload_component_model.py
│   ├── workload_scheduling_engine.py
│   ├── workload_training_migration_methods.py
│   ├── workload_training_migration_analysis_outputs.py
│   └── Carbon_water_footprint.py
├── dataset/
│   ├── Factors.py
│   ├── Installed_capacity_data.py
│   └── Azure/
├── scripts/
│   ├── main_experiments/
│   ├── workload_scope/
│   ├── time_sensitivity/
│   └── model_sensitivity/
├── requirements.txt
├── README.md
└── README.zh-CN.md
```

| Module or directory | Role |
|---|---|
| `core/workload_component_model.py` | Workload profiles and component power models |
| `core/workload_scheduling_engine.py` | Scheduling, electricity use, and carbon accounting |
| `core/workload_training_migration_methods.py` | Training-only scheduling methods |
| `core/workload_training_migration_analysis_outputs.py` | Country, component, and migration outputs |
| `core/Carbon_water_footprint.py` | Annual country-level footprint accounting |
| `dataset/Factors.py` | Annual carbon factors, PUE, and model parameters |
| `dataset/Installed_capacity_data.py` | Capacity shares and demand pathways |
| `dataset/Azure/` | Cross-border migration-permission matrices |
| `scripts/main_experiments/` | Main scenario experiments |
| `scripts/workload_scope/` | Training-only workload experiments |
| `scripts/time_sensitivity/` | Scheduling-window and delay sensitivity |
| `scripts/model_sensitivity/` | Model-parameter sensitivity and output validation |

## Data sources and preparation

### Workload traces

Download [Alibaba Cluster Trace GPU v2026](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2026) using the official [download instructions](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/data_download.md). The experiments use the `asi_opensource_pod_hourly` and `asi_opensource_server_hourly` Parquet tables. Extract them into `dataset/cluster-trace-gpu-v2026/data/`.

### Grid carbon intensity

Historical grid carbon-intensity data are obtained from [Electricity Maps](https://www.electricitymaps.com/); see its [dataset access guide](https://help.electricitymaps.com/en/articles/13169264-datasets-and-csv-downloads) and [API documentation](https://app.electricitymaps.com/docs).

The study uses annual average direct carbon intensities for 2017–2025 and hourly direct carbon intensities for 2025. Prepare the 2026–2030 hourly inputs for Current Policies (CP), Nationally Determined Contributions (NDC), and Net Zero (NZ) following the scenario-construction method in the paper. Store all three pathways in `dataset/EM-CPNDCNZ/`, with one CSV per country, pathway, and year: 24 countries × 3 pathways × 5 years, each containing 8,760 hourly records.

### Capacity, PUE, and scenario parameters

Country capacity shares and baseline PUE values are derived from [Data Center Map](https://www.datacentermap.com/research/explorer/). Demand pathways use the IEA's [Key Questions on Energy and AI](https://www.iea.org/reports/key-questions-on-energy-and-ai). The model parameters are provided in `dataset/Factors.py` and `dataset/Installed_capacity_data.py`. The `dataset/Azure/` directory contains the migration-permission matrices for the three governance scenarios defined in the paper.

### Local input layout

After preparing the workload and hourly carbon data, arrange them as follows:

```text
dataset/
├── EM-CPNDCNZ/
│   └── <country>/*-<CP|NDC|NZ>-<year>-hourly.csv
└── cluster-trace-gpu-v2026/
    └── data/
        ├── asi_opensource_pod_hourly/day=*/hour=*/*.parquet
        └── asi_opensource_server_hourly/day=*/hour=*/*.parquet
```

## Installation

Use Python 3.10 or later and Bash. Scheduling uses the HiGHS solver through SciPy.

```bash
git clone https://github.com/liujian123223/global-ai-carbon-scheduling.git
cd global-ai-carbon-scheduling
python -m pip install -r requirements.txt
```

## Running the experiments

Run the commands below from the repository root. Set the shared input paths; these can also point to external data directories.

```bash
export WORKLOAD_PATH="dataset/cluster-trace-gpu-v2026"
export HOURLY_CARBON_DIR="dataset/EM-CPNDCNZ"
```

### Main experiments

The main design comprises 180 configurations: five years (2026–2030), four demand pathways (Base, Lift-Off, High Efficiency, and Headwinds), three grid pathways (CP, NDC, and NZ), and three governance scenarios. Each configuration evaluates LI, TM, SM, and JSM.

```bash
bash scripts/main_experiments/run_main_experiments.sh
```

Results are written to `results/main_experiments/<configuration>/`. Each configuration produces CSV files for annual and country-level outcomes, workload execution, migration flows, and component energy consumption.

To inspect the experiment plan:

```bash
DRY_RUN=1 OUTPUT_ROOT=results/main_plan bash scripts/main_experiments/run_main_experiments.sh
```

Batch scripts run serially by default. Set `WORKERS` to run multiple configurations in parallel, for example:

```bash
WORKERS=4 bash scripts/main_experiments/run_main_experiments.sh
```

### Single-configuration example

Run the 2030–CP–Base case under the global-transfer governance scenario:

```bash
python scripts/main_experiments/run_scheduling.py \
  --policy CP --scenarios Base --years 1 --year-start 2030 \
  --workload-path "$WORKLOAD_PATH" \
  --hourly-carbon-dir "$HOURLY_CARBON_DIR" \
  --hourly-carbon-scope direct --strict-hourly-carbon \
  --azure-transmission-dir dataset/Azure \
  --azure-transmission-policy global_transfer_with_safeguards \
  --constraint global --solve-mode windowed --solver scipy \
  --commit-hours 24 --delay-hours 24 --offline-inference-delay-hours 6 \
  --baseline-scheduling-mode matched_window \
  --execution-policy capacity --inference-origin-fraction 0.75 --other-origin-fraction 0.50 \
  --capacity-quantile 0.96 --max-resource-utilization 1.0 \
  --max-intervals 8760 --ramping-trace-cycle-hours 4440 \
  --methods linear_programming \
    local_immediate_training_overflow_constrained \
    local_training_time_shift_greedy_constrained \
    annual_average_immediate_overflow_greedy_constrained \
  --output-dir results/main_experiments/2030_base_cp_global
```

### Workload scope and sensitivity analyses

| Experiment | Purpose |
|---|---|
| Workload scope | Training-only scheduling across the scenario grid; compare with the main experiments that schedule training and offline inference |
| Time sensitivity | Vary the scheduling window and maximum delays for training and offline inference under CP–Base–global transfer for 2026–2030 |
| Model sensitivity | Vary workload, component-power, carbon-intensity, and capacity parameters under 2030–CP–Base–global transfer |

Run each experiment with its own output directory:

```bash
OUTPUT_ROOT=results/workload_scope bash scripts/workload_scope/run_historical_training_grid.sh
OUTPUT_ROOT=results/time_sensitivity bash scripts/time_sensitivity/run_time_sensitivity_45_cases.sh
OUTPUT_ROOT=results/model_sensitivity bash scripts/model_sensitivity/run_model_sensitivity.sh
```

For individual Python options:

```bash
python scripts/main_experiments/run_scheduling.py --help
python scripts/workload_scope/run_training_only.py --help
python scripts/model_sensitivity/run_model_sensitivity.py --help
```
