# 全球 AI 工作负载碳感知调度

[English](README.md) | 简体中文

本仓库提供面向全球24个国家、2026—2030年的 AI 工作负载小时级碳感知调度代码。实验在不同需求路径、电网脱碳路径和跨境治理情景下，比较本地即时执行（LI）、时间迁移（TM）、空间迁移（SM）及联合时空迁移（JSM）。

## 项目目录

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

| 模块或目录 | 作用 |
|---|---|
| `core/workload_component_model.py` | 工作负载构建与组件功率模型 |
| `core/workload_scheduling_engine.py` | 调度求解、用电与碳排放核算 |
| `core/workload_training_migration_methods.py` | 仅训练工作负载调度方法 |
| `core/workload_training_migration_analysis_outputs.py` | 国家、组件与迁移结果输出 |
| `core/Carbon_water_footprint.py` | 国家层面年度足迹核算 |
| `dataset/Factors.py` | 年度碳因子、PUE及模型参数 |
| `dataset/Installed_capacity_data.py` | 容量份额与需求路径 |
| `dataset/Azure/` | 跨境迁移许可矩阵 |
| `scripts/main_experiments/` | 主情景实验 |
| `scripts/workload_scope/` | 仅训练工作负载实验 |
| `scripts/time_sensitivity/` | 调度窗口与延迟敏感性 |
| `scripts/model_sensitivity/` | 模型参数敏感性与输出校验 |

## 数据来源与准备

### 工作负载轨迹

根据官方[下载说明](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/data_download.md)获取 [Alibaba Cluster Trace GPU v2026](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2026)。实验使用 `asi_opensource_pod_hourly` 和 `asi_opensource_server_hourly` 两类 Parquet 表，解压后放入 `dataset/cluster-trace-gpu-v2026/data/`。

### 电网碳强度

历史电网碳强度数据来自 [Electricity Maps](https://www.electricitymaps.com/)，获取方式见其[数据集访问说明](https://help.electricitymaps.com/en/articles/13169264-datasets-and-csv-downloads)和 [API 文档](https://app.electricitymaps.com/docs)。

研究采用2017—2025年的年度平均直接碳强度和2025年的小时直接碳强度。按照论文中的情景构建方法，准备2026—2030年现行政策（CP）、国家自主贡献（NDC）和净零排放（NZ）三条路径的小时输入，并统一放入 `dataset/EM-CPNDCNZ/`。每个国家、路径和年份对应一个 CSV 文件，共24个国家 × 3条路径 × 5个年份，每个文件包含8,760条小时记录。

### 容量、PUE与情景参数

国家容量份额和基准 PUE 数据来源于 [Data Center Map](https://www.datacentermap.com/research/explorer/)。需求路径采用 IEA 的 [Key Questions on Energy and AI](https://www.iea.org/reports/key-questions-on-energy-and-ai)。模型参数保存在 `dataset/Factors.py` 和 `dataset/Installed_capacity_data.py` 中。`dataset/Azure/` 包含论文定义的三类治理情景对应的跨境迁移许可矩阵。

### 本地输入目录

准备好工作负载与小时碳强度数据后，按以下结构存放：

```text
dataset/
├── EM-CPNDCNZ/
│   └── <country>/*-<CP|NDC|NZ>-<year>-hourly.csv
└── cluster-trace-gpu-v2026/
    └── data/
        ├── asi_opensource_pod_hourly/day=*/hour=*/*.parquet
        └── asi_opensource_server_hourly/day=*/hour=*/*.parquet
```

## 安装

使用 Python 3.10 或更高版本及 Bash。调度优化通过 SciPy 调用 HiGHS 求解器。

```bash
git clone https://github.com/liujian123223/global-ai-carbon-scheduling.git
cd global-ai-carbon-scheduling
python -m pip install -r requirements.txt
```

## 运行实验

以下命令均从仓库根目录执行。先设置共用的数据路径，也可将其替换为外置数据目录。

```bash
export WORKLOAD_PATH="dataset/cluster-trace-gpu-v2026"
export HOURLY_CARBON_DIR="dataset/EM-CPNDCNZ"
```

### 主实验

主实验共180组配置：5个年份（2026—2030）、4条需求路径（Base、Lift-Off、High Efficiency和Headwinds）、3条电网脱碳路径（CP、NDC和NZ）及3类治理情景。每组配置运行 LI、TM、SM 和 JSM 四种方法。

```bash
bash scripts/main_experiments/run_main_experiments.sh
```

结果保存于 `results/main_experiments/<configuration>/`。每组配置输出年度及国家层面的结果、工作负载执行、迁移流和组件能耗等 CSV 文件。

查看实验运行计划：

```bash
DRY_RUN=1 OUTPUT_ROOT=results/main_plan bash scripts/main_experiments/run_main_experiments.sh
```

批处理默认串行运行。通过 `WORKERS` 设置并行配置数，例如：

```bash
WORKERS=4 bash scripts/main_experiments/run_main_experiments.sh
```

### 单组配置示例

运行全球迁移治理情景下的2030–CP–Base配置：

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

### 工作负载范围与敏感性分析

| 实验 | 作用 |
|---|---|
| 工作负载范围 | 在各情景配置下运行仅训练调度，与主实验中训练加离线推理的调度结果比较 |
| 时间约束敏感性 | 在2026—2030年 CP–Base–全球迁移情景下，改变调度窗口及训练、离线推理的最大延迟 |
| 模型参数敏感性 | 在2030–CP–Base–全球迁移情景下，改变工作负载、组件功率、碳强度及容量参数 |

分别运行，并指定各自的输出目录：

```bash
OUTPUT_ROOT=results/workload_scope bash scripts/workload_scope/run_historical_training_grid.sh
OUTPUT_ROOT=results/time_sensitivity bash scripts/time_sensitivity/run_time_sensitivity_45_cases.sh
OUTPUT_ROOT=results/model_sensitivity bash scripts/model_sensitivity/run_model_sensitivity.sh
```

查看各 Python 入口的参数说明：

```bash
python scripts/main_experiments/run_scheduling.py --help
python scripts/workload_scope/run_training_only.py --help
python scripts/model_sensitivity/run_model_sensitivity.py --help
```
