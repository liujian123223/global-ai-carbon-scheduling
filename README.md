# Global AI Carbon-Aware Scheduling

本仓库提供碳感知工作负载调度及敏感性分析所需的计算代码、年度因子及其他模型参数和运行说明。**Alibaba原始轨迹、历史及情景小时碳强度数据、论文绘图材料均不随仓库提供。**原始数据来源及本地输入准备要求见下文。

仓库地址：[liujian123223/global-ai-carbon-scheduling](https://github.com/liujian123223/global-ai-carbon-scheduling)。下载代码：

```bash
git clone https://github.com/liujian123223/global-ai-carbon-scheduling.git
cd global-ai-carbon-scheduling
```

所有运行命令均从**仓库根目录**执行。本地整理目录名为 `LTD`，克隆后的默认目录名为 `global-ai-carbon-scheduling`；原项目中的同名 `core/`、`dataset/` 和 `scripts/` 不是本仓库的隐含依赖。

## 目录与实验对应关系

```text
global-ai-carbon-scheduling/
├── core/                         # 工作负载映射、功率核算和调度算法
├── dataset/                      # 年度因子、容量、PUE及治理矩阵
├── scripts/
│   ├── main_experiments/          # 正文180组调度实验
│   ├── workload_scope/            # 仅训练工作负载对照
│   ├── time_sensitivity/          # 优化窗口和最大延迟敏感性
│   └── model_sensitivity/         # 组件参数、碳强度和容量敏感性，含运行所需的输出校验器
└── requirements.txt
```

本README包含各类实验的入口和运行方式。发布范围保留运行依赖；绘图材料、历史结果合并工具、独立开发测试及本地审计文件不纳入仓库。

四类实验共4个Bash入口，均默认 `WORKERS=1`，即串行运行。可在命令前设置 `WORKERS=N` 指定并行数，例如 `WORKERS=18 bash scripts/main_experiments/run_main_experiments.sh`。并行数不改变每组实验的参数和方法，并行工作进程日志位于对应输出目录的 `_worker_logs/`。并行启动前默认先准备共享工作负载缓存，避免并发写入同一缓存。

## 当前复现状态

**计算代码和模型参数已包含；完整重算需另行准备Alibaba小时轨迹和2026—2030年小时碳强度情景输入。**当前复现条件如下：

| 项目 | 当前状态及影响 |
|---|---|
| 核心计算代码 | 主调度、仅训练对照、模型参数敏感性入口及其核心模块均已包含 |
| 治理矩阵、容量及PUE参数 | 已包含 |
| 年度碳因子 | 已包含于 `dataset/Factors.py`，不属于逐小时数据 |
| 小时碳强度 | 未包含历史或2026—2030年情景小时数据；需按照论文方法准备24个国家、3条路径、5个年份共360个CSV，并统一放入本地 `dataset/EM-CPNDCNZ` 或指定外置目录 |
| Alibaba小时轨迹 | 未包含 pod/server 小时 Parquet；不能从任务汇总表替代构建 |
| 完整主实验批处理 | `run_main_experiments.sh` 已统一生成180组配置，每组运行四种论文方法 |
| 默认设置参照 | 时间敏感性不重算默认24/24/6 h点；模型敏感性不重算默认参数点，需另行运行或提供对应主实验/中性实验结果 |
| 绘图与参考结果 | 不随本仓库提供；运行脚本输出实验CSV，不直接生成论文图片和排版表格 |
| 历史运行环境 | 未提供原实验的精确依赖版本；`requirements.txt` 仅列运行依赖，不能作为历史环境锁文件 |

核心模块现名为 `core/workload_scheduling_engine.py`，负责工作负载构建、调度求解及电量/碳排放核算。年度因子改为从统一的 `dataset/Factors.py` 读取，小时数据目录检查已相应调整；因子合并未改变求解和核算逻辑。内部动态模块标识、函数名和输出字段保留，避免影响既有加载行为和结果核对。

## 原始数据来源与获取

### Alibaba Cluster Trace GPU v2026

本仓库不直接提供Alibaba GPU v2026原始轨迹。请从阿里巴巴官方 [Alibaba Cluster Trace Program：GPU v2026](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2026) 获取数据说明，并按其 [下载指南](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/data_download.md) 下载、解压数据。

本仓库的调度实验需要以下两类小时Parquet表：

- `asi_opensource_pod_hourly`：小时工作负载、资源请求及利用率。
- `asi_opensource_server_hourly`：小时服务器资源容量。

仅克隆Alibaba的GitHub仓库不会自动下载这些原始数据表；数据文件位于其下载指南给出的独立存储地址。将解压后的目录放到下文所示位置，或通过 `WORKLOAD_PATH` 指定外置路径。任务执行汇总表不能代替pod/server小时表。使用这些数据时，请遵循官方仓库的数据使用和引用说明。

### Electricity Maps历史碳强度

本研究的历史电网碳强度来自 [Electricity Maps](https://www.electricitymaps.com/)。原始历史小时序列不随本仓库分发。有关历史数据获取，请参阅官方 [数据集与CSV下载说明](https://help.electricitymaps.com/en/articles/13169264-datasets-and-csv-downloads) 和 [API文档](https://app.electricitymaps.com/docs)。具体可用年份、区域、时间粒度和访问权限以官方数据覆盖及账户权限为准。

研究输入中，2017—2025年的**年度平均直接碳因子**用于构建年度变化趋势，2025年的**小时直接碳强度**用于构建未来小时变化序列。`dataset/Factors.py` 保留了模型使用的2017—2025年年度值；本仓库不提供2017—2025年的原始逐小时数据全集。

用于调度实验的2026—2030年CP、NDC、NZ小时情景输入也不随仓库提供。请获取所需历史数据，并按照论文中的情景构建方法准备360个小时CSV，统一放到本地 `dataset/EM-CPNDCNZ`。这些文件是研究构造的情景数据，不是Electricity Maps对未来年份的观测记录。本仓库提供调度与核算代码，不包含从原始历史数据生成这些小时情景文件的脚本。

**Data availability (English).** This repository provides scheduling and carbon-accounting code, annual carbon-intensity factors, and other model parameters. It does not redistribute raw Alibaba workload traces, historical hourly carbon-intensity data, or the processed hourly scenario inputs for 2026–2030. Obtain the hourly pod and server tables from the official [Alibaba Cluster Trace Program](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2026), following its [download instructions](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/data_download.md). Historical grid-carbon data are provided by [Electricity Maps](https://www.electricitymaps.com/); consult its [dataset access guidance](https://help.electricitymaps.com/en/articles/13169264-datasets-and-csv-downloads) and [API documentation](https://app.electricitymaps.com/docs). The study uses annual mean direct carbon-intensity factors for 2017–2025 and hourly direct carbon-intensity profiles for 2025. Prepare the future hourly scenario inputs using the method described in the paper before running the experiments. Historical data availability depends on the requested region, year, temporal resolution, and access permissions.

## 输入数据

运行前需准备下列本地目录。其中 `Factors.py`、容量参数和 `Azure/` 随代码提供；`EM-CPNDCNZ/` 和Alibaba轨迹需另行准备，不上传GitHub：

```text
dataset/
├── Factors.py                    # PUE等模型参数及三条路径的年度碳因子
├── Installed_capacity_data.py
├── Azure/
│   ├── azure_global_transfer_with_safeguards.csv
│   ├── azure_eea_adequacy_trusted_zone.csv
│   └── azure_eea_only_regional_residency.csv
├── EM-CPNDCNZ/                    # 本地准备，不随仓库提供
│   └── <国家>/*-<CP或NDC或NZ>-<年份>-hourly.csv
└── cluster-trace-gpu-v2026/data/   # 从Alibaba官方来源下载
    ├── asi_opensource_pod_hourly/day=*/hour=*/*.parquet
    └── asi_opensource_server_hourly/day=*/hour=*/*.parquet
```

需要准备的小时碳强度覆盖24个国家、2026—2030年，每个国家—路径—年份文件包含8760个小时。四个实验Bash默认从相对于仓库根目录的 `dataset/EM-CPNDCNZ` 读取CP、NDC和NZ的小时CSV，年度因子和PUE等参数均从 `dataset/Factors.py` 读取。Alibaba的pod/server小时表可直接位于轨迹目录下，也可位于其 `data/` 子目录中。

外置工作负载通过 Shell 环境变量 `WORKLOAD_PATH` 或 Python 参数 `--workload-path` 指定。小时碳强度同样可通过 `HOURLY_CARBON_DIR` 或 `--hourly-carbon-dir` 指定。Shell脚本默认从 `dataset/cluster-trace-gpu-v2026` 读取轨迹；Python默认路径由项目根目录推导，不写入机器专属的绝对路径。`.gitignore` 排除Alibaba轨迹、小时碳强度目录及 `Origin_Figure_Data/`；保留这些本地目录不会将其加入正常的Git提交。

### 四个Bash入口共用同一数据位置

将准备好的CP、NDC、NZ小时CSV统一放到本地 `dataset/EM-CPNDCNZ`，并指定Alibaba轨迹位置。以下示例从仓库根目录执行；碳因子变量与默认值相同，可省略，也可改为外置 `EM-CPNDCNZ` 目录的绝对路径：

```bash
export HOURLY_CARBON_DIR="dataset/EM-CPNDCNZ"
export WORKLOAD_PATH="/absolute/path/to/cluster-trace-gpu-v2026"
```

四类实验均使用上述同一个小时碳因子目录，无需分别配置CP和NDC/NZ路径。`HOURLY_CARBON_DIR` 仅选择小时CSV位置；年度因子始终取自仓库内的 `dataset/Factors.py`。Alibaba目录应包含pod和server小时表。治理矩阵、容量和PUE参数继续使用LTD自带文件。

论文原实验的CP小时输入来自原项目 `dataset/EM-CPNDCNZ`，NDC/NZ小时输入来自原项目 `dataset/EM-IEA26_30/EM-CPNDCNZ`；这些来源路径仅用于说明原实验版本，对应小时文件不随本仓库提供。三组年度因子已原样迁入 `dataset/Factors.py`，其中CP年度因子与原版完全相同。NDC、NZ年度因子分别使用完整比例 `270/312`、`195/312`，2026—2030年的乘数为 `比例 ** ((年份-2025)/5)`。准备小时输入时应保留原实验的数据精度及时间戳口径：2028年原版CP包含2月29日并止于12月30日，原实验NDC/NZ跳过2月29日并止于12月31日，均为8760行。自行构建输入时如改变这些处理细节，不能保证与原结果逐位一致。

`dataset/Factors.py` 是唯一的因子参数文件。`CF_CP_electricity_map`、`CF_NDC_electricity_map`、`CF_NZ_electricity_map` 保存2017—2030年的调度年度因子；原有PUE及现有模块仍引用的其他参数保留各自名称、年份索引和数值。模型参数敏感性目录中的 `validate_results.py` 会被批处理直接调用，也属于运行依赖。

`run_plan.tsv` 固定保存生成时的路径和参数。更改环境变量后，需要重新生成清单再筛选运行；仅重新执行旧清单不会更新路径。工作负载缓存默认位于 `results/workload_profile_cache`，匹配的缓存可继续复用。

## 安装与轻量检查

使用 Python 3.10 或更高版本及 Bash。原论文运行记录使用 SciPy/HiGHS 求解；Gurobi不是默认依赖。

```bash
python -m pip install -r requirements.txt
python scripts/main_experiments/run_scheduling.py --help
python scripts/workload_scope/run_training_only.py --help
python scripts/model_sensitivity/run_model_sensitivity.py --help
```

上述帮助命令不要求原始轨迹。批处理的 `DRY_RUN=1` 只输出运行计划；主实验预演不要求原始数据，其他批处理仍可能检查碳强度目录。预演会写出命令清单、实验设计表或日志，请指定独立 `OUTPUT_ROOT`。

## 正文主实验

完整设计为5个年份 × 4条需求路径 × 3条脱碳路径 × 3种治理情景，共180组配置，每组包括LI、TM、SM、JSM。

| 论文方法 | 程序方法标识 |
|---|---|
| LI | `local_immediate_training_overflow_constrained` |
| TM | `local_training_time_shift_greedy_constrained` |
| SM | `annual_average_immediate_overflow_greedy_constrained` |
| JSM | `linear_programming` |

下面是一个完整的2030–CP–Base–GT单配置运行示例；**只运行此命令不等于完成全部180组配置**：

```bash
python scripts/main_experiments/run_scheduling.py \
  --policy CP --scenarios Base --years 1 --year-start 2030 \
  --workload-path dataset/cluster-trace-gpu-v2026 \
  --hourly-carbon-dir dataset/EM-CPNDCNZ \
  --hourly-carbon-scope direct --strict-hourly-carbon \
  --azure-transmission-dir dataset/Azure \
  --azure-transmission-policy global_transfer_with_safeguards \
  --constraint global --solve-mode windowed --solver scipy \
  --commit-hours 24 --delay-hours 24 --offline-inference-delay-hours 6 \
  --baseline-scheduling-mode matched_window \
  --capacity-quantile 0.96 --max-resource-utilization 1.0 \
  --max-intervals 8760 --ramping-trace-cycle-hours 4440 \
  --methods local_immediate_training_overflow_constrained \
    local_training_time_shift_greedy_constrained \
    annual_average_immediate_overflow_greedy_constrained linear_programming \
  --output-dir results/main_experiments/2030_base_cp_global
```

### 统一运行全部180组

```bash
# 无需原始数据，检查全部配置和命令；不进行数值计算。
DRY_RUN=1 OUTPUT_ROOT=results/main_plan bash scripts/main_experiments/run_main_experiments.sh

# 数据齐全后运行全部180组；默认串行。
bash scripts/main_experiments/run_main_experiments.sh

# 按需设置并行数，例如18。
WORKERS=18 bash scripts/main_experiments/run_main_experiments.sh
```

通过 `WORKLOAD_PATH` 指定外置Alibaba轨迹；三条脱碳路径均使用本地准备的 `dataset/EM-CPNDCNZ` 或 `HOURLY_CARBON_DIR` 指定的统一目录，无需分别指定输入目录。默认结果保存于 `results/main_experiments/<配置名称>/`，配置名称与原Result-V4对应；总运行计划为 `run_plan.tsv`。

统一批处理固定采用论文的24 h优化窗口、24/6 h训练/离线推理最大延迟、8760 h、matched_window基线及SciPy求解，并保持每个进程使用1个数值计算线程。`WORKERS` 只改变同时运行的配置数。原始CP全方法命令中JSM排在前面，NDC/NZ中JSM排在最后，新批处理沿用这一顺序；当前论文未使用的额外基线不再运行。

每组输出包含原Python入口生成的年度、国家、任务执行、迁移流、组件能耗及爬坡等CSV。断点续跑仅跳过命令一致且输出检查通过的配置；已有不同命令的结果不会自动覆盖。需要覆盖时显式设置 `FORCE=1`，或使用新的 `OUTPUT_ROOT`。输入数据文件、代码或环境改变后，应使用新的输出目录，不能仅凭相同路径认定结果可复用。

## 附录实验

- **可调度工作负载范围**：仅训练对照使用 `workload_scope/run_training_only.py`；训练加离线推理使用正文入口。在CP–Base–GT下比较2026—2030年的结果。`run_historical_training_grid.sh` 运行完整180组历史情景配置，包含2026–CP–Base的3种治理配置，已移除历史年份跳过规则；它不是仅运行论文5组对照的专用脚本。已有结果仍按断点续跑规则跳过，需重算时设置 `FORCE=1`。
- **时间约束敏感性**：`time_sensitivity/run_time_sensitivity_45_cases.sh` 运行45组非默认配置。优化窗口取12/48/96 h，训练最大延迟取0/12/48 h，离线推理最大延迟取0/2/12 h，分别覆盖2026—2030年。默认24/24/6 h结果来自正文CP–Base–GT配置。
- **模型参数敏感性**：`model_sensitivity/run_model_sensitivity.sh` 保留2030–CP–Base–GT的历史18组设置。默认串行，可通过 `WORKERS=N` 设置并行数；例如 `WORKERS=18` 时每个进程分配1组。当前论文图使用其中8个参数的16组非默认设置；另外2组GPU功率曲线实验不在当前图中。默认点需要独立运行或提供原 `_neutral_828_parity` 结果。具体参数及取值由该Bash文件中的实验设计定义，Python入口的 `--help` 列出参数含义。

三个附录批处理入口如下，均从仓库根目录运行并使用上述统一碳因子目录和 `WORKLOAD_PATH`：

```bash
bash scripts/workload_scope/run_historical_training_grid.sh
bash scripts/time_sensitivity/run_time_sensitivity_45_cases.sh
bash scripts/model_sensitivity/run_model_sensitivity.sh
```

## 数值一致性的边界

完成目录整理及轻量测试，仅能证明代码入口、引用关系及已测试的参数约束可用。缺少原始数据时，不能据此声称已重现论文数值。严格比对还应保留原实验命令、求解日志和软件版本：历史仅训练对照的部分年份记录了末尾24 h求解回退，不同求解器版本下的重新运行未必逐位一致。

## 合并前后的一致性核对

- 核心模块的因子合并改动仅涉及年度因子导入和小时数据目录检查；调度、功率、能耗及排放核算函数保持不变。
- 在使用原记录对应的输入目录时，合并后的主实验计划已与现有180份原始运行记录逐项比较，配置名称和有效计算参数一致。当前默认统一读取 `dataset/EM-CPNDCNZ`，实际数据版本由该目录的文件决定。原记录中的部分方法属于分批补跑，新入口为每组配置完整运行论文四方法。
- 主实验串行/并行计划均含180组唯一配置；模型参数敏感性串行/9进程计划均含原18组设置。
- 原18组模型参数敏感性命令与合并后的串行、并行命令逐项核对，有效参数一致。
- 两国、72小时合成输入的回归检查中，重命名前后四种方法的调度分配、目标值、设施用电和碳排放完全一致；JSM先运行或后运行的结果也一致。该检查使用实际求解函数，但不代表论文全规模数据重跑。
- 使用测试替身检查了180组批处理的任务分配、已完成配置跳过、不完整输出补跑及失败返回；测试文件均位于临时目录，不是实验结果。
- 原始轨迹和历史环境仍未补齐，上述命令核对不等同于已完整重算并验证论文全部数值。
