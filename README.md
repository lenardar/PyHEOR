# PyHEOR — Python Health Economics and Outcome Research

> **用 Python 做卫生经济学建模，像 R 的 hesim / DARTH 一样专业，但更简洁。**

PyHEOR 是一个面向卫生经济学研究的 Python 框架，支持：

| 功能                              | 说明                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------- |
| **Markov 队列模型**         | 离散时间状态转移模型 (cDTSTM)，时齐 / 时变转移矩阵                                          |
| **分区生存模型 (PSM)**      | 基于参数化生存曲线的状态概率划分                                                            |
| **微观模拟**                | 个体水平状态转移模型，支持患者异质性、事件处理器、双层 PSA                                  |
| **离散事件模拟 (DES)**     | 连续时间个体模拟，竞争风险、time-to-event 分布驱动、HR/AFT 集成                             |
| **IPD 生存曲线拟合**        | 6 种参数分布 MLE 拟合，AIC/BIC 比较，自动选优                                               |
| **KM 曲线数字化重建**       | Guyot method 从发表文献 KM 图反推 IPD，含数字化噪声预处理                                    |
| **参数化生存分布**          | Exponential, Weibull, Log-logistic, Log-normal, Gompertz, Generalized Gamma 等 10 种        |
| **基础分析**                | 确定性基线分析                                                                              |
| **单因素敏感性分析 (OWSA)** | 龙卷风图（INMB/ICER 排序），贴现率通过 `Param` 直接参与变动分析                                 |
| **概率敏感性分析 (PSA)**    | Monte Carlo 模拟，CE 散点图，CEAC                                                           |
| **多策略比较 & NMB**        | 效率前沿、支配/扩展支配检测、NMB 曲线、CEAF、EVPI                                           |
| **NMA 整合**                | 导入 R 后验样本，保留相关性，自动生成 PH/AFT 曲线                                           |
| **灵活的费用定义**          | 首周期费用、时间依赖函数、一次性费用、WLOS 方法、转移费用计划表、自定义费用函数               |
| **预算影响分析 (BIA)**      | 人群规模模型、市场份额演变、摄取曲线、情景/单因素敏感性分析                                  |
| **模型校准**                | 用观测数据反推未知参数：Nelder-Mead 多起点优化、LHS 随机搜索、SSE/WSSE/似然 GoF              |
| **可视化**                  | 28 种专业图表：状态转移图、前沿图、NMB 曲线、CEAF、EVPI、CEAC、KM+拟合曲线、BIA 影响图等    |
| **Excel 导出**              | 多 Sheet 导出，便于审核验证                                                                 |

---

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
  - [Markov 队列模型](#1-markov-队列模型) · [分区生存模型](#2-分区生存模型-psm) · [IPD 拟合](#3-ipd-生存曲线拟合) · [微观模拟](#4-微观模拟-microsimulation) · [多策略比较](#5-多策略比较--nmb-分析)
- [核心概念](#核心概念)
  - [参数系统](#参数系统) · [转移矩阵](#转移矩阵) · [费用定义](#灵活的费用定义) · [转移费用](#转移费用-transition-costs) · [自定义费用](#自定义费用-custom-costs) · [微观模拟设计](#微观模拟核心设计)
- [参数化生存分布](#参数化生存分布)
- [IPD 拟合功能详解](#ipd-拟合功能详解)
- [KM 曲线数字化重建](#km-曲线数字化重建)
- [离散事件模拟 (DES)](#离散事件模拟-des)
- [NMA 整合](#nma-整合)
- [Excel 导出](#excel-导出)
- [预算影响分析 (BIA)](#预算影响分析-bia)
- [模型校准](#模型校准)
- [可视化一览](#可视化一览)
- [项目结构](#项目结构) · [设计理念](#设计理念) · [路线图](#路线图)

---

## 安装

```bash
# 从源码安装
git clone <repo-url>
cd pyheor
pip install -e .
```

依赖：`numpy`, `pandas`, `matplotlib`, `scipy`（可选：`openpyxl` 用于 Excel 导出）

---

## 快速开始

### 1. Markov 队列模型

```python
import pyheor as ph

# 定义模型
model = ph.MarkovModel(
    states=["Healthy", "Sick", "Dead"],
    strategies=["Standard", "New Treatment"],
    n_cycles=40,
    cycle_length=1,          # 年
    dr_cost=ph.Param(0.03, low=0.0, high=0.08, label="费用贴现率"),
    dr_qaly=ph.Param(0.03, low=0.0, high=0.05, label="效用贴现率"),
    half_cycle_correction=True,
)

# 添加参数（带 PSA 分布）
model.add_param("p_HS", base=0.15, low=0.10, high=0.20, dist=ph.Beta(mean=0.15, sd=0.03))
model.add_param("p_HD", base=0.02, low=0.01, high=0.03, dist=ph.Beta(mean=0.02, sd=0.005))
model.add_param("p_SD", base=0.10, low=0.05, high=0.15, dist=ph.Beta(mean=0.10, sd=0.02))
model.add_param("c_drug", base=2000, low=1500, high=2500, dist=ph.Gamma(mean=2000, sd=400))
model.add_param("u_healthy", base=0.95, low=0.90, high=1.0, dist=ph.Beta(mean=0.95, sd=0.02))

# 设置转移矩阵（ph.C = 补数）
model.set_transitions("Standard", lambda p, t: [
    [ph.C,  p["p_HS"], p["p_HD"]],
    [0,     ph.C,      p["p_SD"]],
    [0,     0,         1        ],
])

model.set_transitions("New Treatment", lambda p, t: [
    [ph.C,  p["p_HS"] * 0.7, p["p_HD"]],
    [0,     ph.C,             p["p_SD"] * 0.8],
    [0,     0,                1              ],
])

# 设置费用和效用
model.set_state_cost("Healthy", "Standard", lambda p, t: 500)
model.set_state_cost("Healthy", "New Treatment", lambda p, t: 500 + p["c_drug"])
model.set_state_cost("Sick", "Standard", lambda p, t: 3000)
model.set_state_cost("Sick", "New Treatment", lambda p, t: 3000 + p["c_drug"])

model.set_utility("Healthy", lambda p, t: p["u_healthy"])
model.set_utility("Sick", lambda p, t: 0.6)

# 运行分析
result = model.run_base_case()
print(result.summary())
print(result.icer())

# OWSA（贴现率已通过 Param 自动注册，无需额外 add_param）
owsa = model.run_owsa()
print(owsa.summary(outcome="icer"))  # 按 ICER 影响幅度排序

# PSA (1000 次 Monte Carlo)
psa = model.run_psa(n_sim=1000)
print(psa.summary())
```

### 2. 分区生存模型 (PSM)

```python
import pyheor as ph

# 创建 PSM
psm = ph.PSMModel(
    states=["PFS", "Progressed", "Dead"],
    strategies=["SOC", "New Drug"],
    n_cycles=120,
    cycle_length=1/12,  # 月周期
)

# 设置参数化生存曲线
psm.set_survival_all("SOC", [
    ph.Weibull(shape=1.2, scale=36),     # OS
    ph.LogLogistic(shape=1.5, scale=18), # PFS
])

psm.set_survival_all("New Drug", [
    ph.ProportionalHazards(ph.Weibull(shape=1.2, scale=36), hr=0.7),
    ph.AcceleratedFailureTime(ph.LogLogistic(shape=1.5, scale=18), af=1.3),
])

# 设置费用
psm.set_state_cost("PFS", "SOC", lambda p, t: 1000)
psm.set_state_cost("PFS", "New Drug", lambda p, t: 1000 + 5000)
psm.set_state_cost("Progressed", "SOC", lambda p, t: 2500)
psm.set_state_cost("Progressed", "New Drug", lambda p, t: 2500)

psm.set_utility("PFS", lambda p, t: 0.80)
psm.set_utility("Progressed", lambda p, t: 0.55)

# 运行
result = psm.run_base_case()
print(result.summary())
print(result.icer())

# 可视化
result.plot_survival()
result.plot_state_area()
```

### 3. IPD 生存曲线拟合

```python
import pyheor as ph
import pandas as pd

# 读取 IPD 数据
df = pd.read_csv("patient_data.csv")  # 需要 time 和 event 列

# 创建拟合器并拟合 6 种分布
fitter = ph.SurvivalFitter(
    time=df["time"],
    event=df["event"],
    label="Overall Survival",
)
fitter.fit()

# 查看 AIC/BIC 比较表
print(fitter.summary())
#      Distribution          Parameters  k  Log-Likelihood    AIC    BIC  ΔAIC  ΔBIC  AIC Weight
#           Weibull  shape=1.55, sc...   2        -490.11  984.22  990.81  0.00  0.00   0.666
# Generalized Gamma  mu=3.18, sigma...   3        -490.10  986.21  996.10  1.99  5.29   0.246
#          Gompertz  shape=0.05, ra...   2        -492.45  988.90  995.49  4.68  4.68   0.064
#         ...

# 自动选择最优模型
best = fitter.best_model()          # 默认 AIC
best_bic = fitter.best_model("bic") # 也可用 BIC

# 获取拟合后的分布对象（可直接用于 PSM）
dist = best.distribution
print(f"Median survival: {dist.quantile(0.5):.1f}")

# 模型选择报告（详细解读）
print(fitter.selection_report())

# KM + 所有拟合曲线图
fig = fitter.plot_fits()
fig.savefig("km_fitted_curves.png", dpi=150)

# 风险函数图
fitter.plot_hazard()

# 诊断图：log累积风险 vs log时间
fitter.plot_cumhazard_diagnostic()

# Q-Q 图
fitter.plot_qq()

# 导出到 Excel
fitter.to_excel("fitting_results.xlsx")
```

### 4. 微观模拟 (Microsimulation)

```python
import pyheor as ph
import numpy as np

# 定义个体水平模型
model = ph.MicroSimModel(
    states=["Healthy", "Sick", "Sicker", "Dead"],
    strategies=["SOC", "Treatment"],
    n_cycles=30,
    n_patients=5000,
    cycle_length=1.0,
    dr_cost=0.03,
    dr_qaly=0.03,
    half_cycle_correction=True,
    seed=42,
)

# 添加参数（与 Markov 模型完全一致）
model.add_param("p_HS", base=0.15, dist=ph.Beta(mean=0.15, sd=0.03))
model.add_param("p_SD", base=0.05, dist=ph.Beta(mean=0.05, sd=0.01))
model.add_param("hr_trt", base=0.70, dist=ph.LogNormal(mean=0.70, sd=0.10))

# 转移矩阵（同 Markov 模型语法）
model.set_transitions("SOC", lambda p, t: [
    [ph.C,  p["p_HS"],                0,  0.005],
    [0,     ph.C,                      0.10, p["p_SD"]],
    [0,     0,                         ph.C, 0.10],
    [0,     0,                         0,    1],
])

model.set_transitions("Treatment", lambda p, t: [
    [ph.C,  p["p_HS"] * p["hr_trt"], 0,  0.005],
    [0,     ph.C,                      0.10 * p["hr_trt"], p["p_SD"]],
    [0,     0,                         ph.C, 0.10],
    [0,     0,                         0,    1],
])

# 费用 & 效用
model.set_state_cost("medical", {"Healthy": 500, "Sick": 3000, "Sicker": 8000, "Dead": 0})
model.set_state_cost("treatment", {
    "SOC": {"Healthy": 0, "Sick": 0, "Sicker": 0, "Dead": 0},
    "Treatment": {"Healthy": 5000, "Sick": 5000, "Sicker": 5000, "Dead": 0},
})
model.set_utility({"Healthy": 0.95, "Sick": 0.75, "Sicker": 0.50, "Dead": 0.0})

# 事件处理器：进入 Sicker 时一次性住院费
model.on_state_enter("Sicker", lambda idx, t, attrs: {"cost": 15000})

# 运行分析
result = model.run_base_case(verbose=True)
print(result.summary())   # 含 SD 和 95% 百分位数
print(result.icer())

# 可视化
result.plot_trace()             # 状态占比轨迹
result.plot_survival()          # 经验生存曲线
result.plot_outcomes_histogram() # 患者结局分布

# PSA: 外层参数不确定性 × 内层个体随机性
psa = model.run_psa(n_outer=500, n_inner=2000, seed=42)
print(psa.summary())
psa.plot_ceac(wtp_range=(0, 150000))
psa.plot_scatter(wtp=50000)
```

#### 患者异质性（年龄依赖转移概率）

```python
# 定义异质人群
pop = ph.PatientProfile(
    n_patients=5000,
    attributes={
        "age": np.random.normal(55, 12, 5000).clip(20, 90),
        "female": np.random.binomial(1, 0.52, 5000),
    }
)
model.set_population(pop)

# 3 参数 lambda：(params, cycle, attrs) → 依据个体属性调整转移概率
model.set_transitions("SOC", lambda p, t, attrs: [
    [ph.C,  p["p_HS"] * (1 + (attrs["age"] - 55) * 0.02), 0.005],
    [0,     ph.C,  p["p_SD"]],
    [0,     0,     1],
])

result = model.run_base_case(verbose=True)
# patient_outcomes 返回每位患者的费用、QALYs、LYs
print(result.patient_outcomes.head())
```

### 5. 多策略比较 & NMB 分析

```python
import pyheor as ph

# 从确定性结果创建CEAnalysis
result = model.run_base_case()
cea = ph.CEAnalysis.from_result(result)

# 效率前沿：顺序 ICER + 支配/扩展支配检测
print(cea.frontier())
# Strategy  Cost  QALYs  Inc_Cost  Inc_QALYs   ICER  Status
# SOC       38K   7.18   -        -           -      Ref
# Drug A    73K   7.92   35K      0.74        47K    ND
# Drug C   104K   7.44   -        -           -      D     ← 强支配
# Drug B   154K   8.80   81K      0.87        92K    ND

# NMB 排名
print(cea.nmb(wtp=100000))
print(f"最优策略: {cea.optimal_strategy(wtp=100000)}")

# NMB 曲线 & 前沿图
cea.plot_frontier(wtp=100000)
cea.plot_nmb_curve(wtp_range=(0, 200000))
```

#### PSA → CEAF & EVPI

```python
# 从 PSA 结果创建（含每次模拟数据）
psa_result = model.run_psa(n_sim=2000)
cea_psa = ph.CEAnalysis.from_psa(psa_result)

# CEAF: 前沿策略的成本效果可接受概率
cea_psa.plot_ceaf(wtp_range=(0, 200000))

# EVPI: 完美信息期望价值
print(f"EVPI at WTP=$100K: ${cea_psa.evpi_single(100000):,.0f}")
cea_psa.plot_evpi(wtp_range=(0, 200000), population=100000)
```

#### IPD → PSM 一体化流程

```python
# 拟合 IPD → 获得最优分布 → 直接用于 PSM
fitter_os = ph.SurvivalFitter(time=df_os["time"], event=df_os["event"], label="OS")
fitter_pfs = ph.SurvivalFitter(time=df_pfs["time"], event=df_pfs["event"], label="PFS")
fitter_os.fit()
fitter_pfs.fit()

psm = ph.PSMModel(states=["PFS", "Progressed", "Dead"], ...)
psm.set_survival_all("SOC", [
    fitter_os.best_model().distribution,   # 拟合得到的 OS 曲线
    fitter_pfs.best_model().distribution,  # 拟合得到的 PFS 曲线
])
```

---

## 核心概念

### 参数系统

每个参数通过 `add_param()` 定义，包含：

| 属性               | 说明                                                                              |
| ------------------ | --------------------------------------------------------------------------------- |
| `base`           | 基线值（确定性分析）                                                              |
| `low` / `high` | OWSA 范围                                                                         |
| `dist`           | PSA 分布（Beta, Gamma, Normal, LogNormal, Uniform, Triangular, Dirichlet, Fixed） |

```python
model.add_param("p_progression",
    base=0.15,           # 基线分析用
    low=0.10, high=0.20, # OWSA 范围
    dist=ph.Beta(mean=0.15, sd=0.03),  # PSA 用
    label="疾病进展概率",  # 用于图表显示
)
```

### 转移矩阵

使用 `ph.C`（补数哨兵）自动计算对角线元素：

```python
# 时齐矩阵
model.set_transitions("Strategy", lambda p, t: [
    [ph.C,  p["p_AB"], p["p_AD"]],
    [0,     ph.C,      p["p_BD"]],
    [0,     0,         1        ],
])

# 时变矩阵（t 为周期数）
model.set_transitions("Strategy", lambda p, t: [
    [ph.C,  p["p_AB"] * (1 + 0.01 * t), p["p_AD"]],
    [0,     ph.C,                        p["p_BD"] + 0.001 * t],
    [0,     0,                           1],
])
```

### 灵活的费用定义

```python
# 基础状态费用
model.set_state_cost("Sick", "Treatment", lambda p, t: 3000)

# 时间依赖费用
model.set_state_cost("Sick", "Treatment", lambda p, t: 3000 if t < 5 else 2000)

# 首周期一次性费用
model.set_state_cost("Sick", "Treatment", lambda p, t: 50000,
                     first_cycle_only=True)

# 限定应用周期
model.set_state_cost("Sick", "Treatment", lambda p, t: p["c_drug"],
                     apply_cycles=(0, 24))  # 仅前 24 个周期

# WLOS (Weighted Length of Stay) 方法
model.set_state_cost("Sick", "Treatment", lambda p, t: 5000,
                     method="wlos")
```

### 转移费用 (Transition Costs)

状态转移时触发的费用（如疾病进展时的手术费、转入 ICU 时的住院费）。基于每周期的**转移流量**自动计算：`trace[t-1, from] × P[from→to] × 单位费用`。

```python
# 从 Healthy 进入 Sick 时的手术费
model.set_transition_cost("surgery", "Healthy", "Sick", 50000)

# 参数引用
model.set_transition_cost("surgery", "Healthy", "Sick", "c_surgery")

# 策略特异性
model.set_transition_cost("icu", "Sick", "Dead", {
    "SOC": 20000,
    "Treatment": 15000,
})

# 多个转移可以共享类别名
model.set_transition_cost("hospital", "Healthy", "Sick", 10000)
model.set_transition_cost("hospital", "Sick", "Sicker", 25000)
```

#### 费用计划表 (Cost Schedule)

当转移后需要跨多个周期产生费用时（如手术 + 随访），可以传入**列表**。引擎通过卷积自动处理多批次转入患者的费用叠加：

```python
# 进展时手术 50000，下一周期随访 10000 → 共 2 周期
model.set_transition_cost("surgery", "PFS", "Progressed", [50000, 10000])

# 第 1 周期 100、第 2 周期跳过、第 3 周期 200 → 稀疏计划表
model.set_transition_cost("treatment", "Healthy", "Sick", [100, 0, 200])

# 参数引用也可以在列表中使用
model.set_transition_cost("chemo", "PFS", "Progressed",
    ["c_chemo_init", "c_chemo_maint", "c_chemo_maint"])

# 策略特异性 + 计划表混用
model.set_transition_cost("rescue", "PFS", "Progressed", {
    "SOC": [30000, 5000],       # 计划表
    "New Drug": 15000,           # 标量
})
```

> **与 `first_cycle_only` 的区别**：`first_cycle_only` 只在 cycle 0 生效（仅一次）；transition cost 在**每个周期**只要有人转移就会产生费用。Transition cost 不受半周期校正影响（事件型费用）。

### 自定义费用 (Custom Costs)

当 `set_transition_cost` 按单个状态对定义费用不够灵活时，可以用 `set_custom_cost` 传入自定义函数，直接基于转移矩阵和状态分布计算费用。支持 MarkovModel 和 PSMModel。

```python
# 函数签名
# func(strategy, params, t, state_prev, state_curr, P, states) -> float
#   strategy:   当前策略名
#   params:     参数字典 {name: value}
#   t:          当前周期 (1-based)
#   state_prev: t-1 时刻各状态占比 (np.array)
#   state_curr: t 时刻各状态占比 (np.array)
#   P:          转移概率矩阵 (MarkovModel) 或 None (PSMModel)
#   states:     状态名列表

# MarkovModel: 基于转移流量计算手术费
def surgery_cost(strategy, params, t, state_prev, state_curr, P, states):
    i_from = states.index("PFS")
    i_to = states.index("Progressed")
    flow = state_prev[i_from] * P[i_from, i_to]
    return flow * params["c_surgery"]

model.set_custom_cost("surgery", surgery_cost)

# PSMModel: 基于状态变化量计算进展费用 (无转移矩阵，P=None)
def progression_cost(strategy, params, t, state_prev, state_curr, P, states):
    i_prog = states.index("Progressed")
    new_prog = max(0, state_curr[i_prog] - state_prev[i_prog])
    return new_prog * params["c_progression"]

psm.set_custom_cost("progression", progression_cost)
```

> 自定义费用不受半周期校正影响（与转移费用一致）。函数通过 `params` 接收参数值，OWSA/PSA 的参数变化和抽样会自然传导。

### 微观模拟核心设计

**与队列模型的异同**：MicroSimModel 使用与 MarkovModel 完全一致的 API（`add_param`, `set_transitions`, `set_state_cost`, `set_utility`），区别在于每位患者独立从转移概率中采样，产生个体水平的异质性结局。

```python
# 事件处理器：状态进入/退出时触发一次性费用
model.on_state_enter("ICU", lambda idx, t, attrs: {"cost": 50000})
model.on_state_exit("Surgery", lambda idx, t, attrs: {"cost": 20000})

# 患者异质性：3 参数 lambda(params, cycle, attrs) 自动检测
model.set_transitions("SOC", lambda p, t, attrs: [
    [ph.C,  p["p_HS"] * (1 + (attrs["age"] - 55) * 0.02), p["p_HD"]],
    [0,     ph.C,  p["p_SD"]],
    [0,     0,     1],
])

# PSA 双层结构
# 外层: 参数不确定性 (从 dist 抽样)
# 内层: 个体随机性 (每位患者独立模拟)
psa = model.run_psa(n_outer=500, n_inner=2000)  # 500 × 2000 次模拟
```

**性能优化**：当转移矩阵不依赖个体属性（2 参数 lambda）时，引擎自动使用向量化批量采样，速度接近队列模型。

### 贴现率

所有模型（Markov / PSM / MicroSim / DES）均通过 `dr_cost` 和 `dr_qaly` 两个独立参数设置贴现率。**默认值为 0（不贴现）**，未设置的一方不会被贴现。

```python
# 基础用法：固定贴现率
model = ph.MarkovModel(
    ...,
    dr_cost=0.03,  # 费用年贴现率 3%
    dr_qaly=0.03,  # 效用年贴现率 3%
)

# 只贴现费用，不贴现效用
model = ph.MarkovModel(
    ...,
    dr_cost=0.06,  # 费用贴现 6%
    # dr_qaly 默认 0，效用不贴现
)
```

#### 贴现率敏感性分析

传入 `Param` 对象即可将贴现率纳入 OWSA / PSA，无需额外调用 `add_param()`：

```python
model = ph.MarkovModel(
    ...,
    dr_cost=ph.Param(0.03, low=0.0, high=0.08, label="费用贴现率"),
    dr_qaly=ph.Param(0.03, low=0.0, high=0.05, label="效用贴现率"),
)

# dr_cost 和 dr_qaly 自动注册到 model.params，直接参与 OWSA
owsa = model.run_owsa()
owsa.summary()       # 龙卷风图数据中包含贴现率
owsa.plot_tornado()  # 可视化

# 也可以只对其中一个做敏感性分析
model = ph.MarkovModel(
    ...,
    dr_cost=0.03,                                          # 固定
    dr_qaly=ph.Param(0.03, low=0.0, high=0.05),           # 变动
)
```

> **设计原则**：贴现率的基准值和敏感性分析范围在同一处定义，避免重复指定。`float` 表示固定值，`Param` 表示可变动值。

### 半周期校正

| 值               | 说明                                            |
| ---------------- | ----------------------------------------------- |
| `True` / `"trapezoidal"` | 梯形法：首尾周期权重 ×0.5（默认）        |
| `"life-table"`   | 生命表法：相邻 trace 行取均值（与 R heemod 一致）|
| `False` / `None` | 不校正                                          |

```python
# 运行时可动态切换
model.half_cycle_correction = "life-table"
model.half_cycle_correction = "trapezoidal"
model.half_cycle_correction = False
```

---

## 参数化生存分布

10 种内置生存分布：

| 分布                               | 参数      | 风险形状特征                         |
| ---------------------------------- | --------- | ------------------------------------ |
| `Exponential(rate)`              | λ        | 常数风险                             |
| `Weibull(shape, scale)`          | α, λ    | shape>1 递增，<1 递减                |
| `LogLogistic(shape, scale)`      | α, λ    | shape>1 先升后降                     |
| `SurvLogNormal(meanlog, sdlog)`  | μ, σ    | 先升后降                             |
| `Gompertz(shape, rate)`          | a, b      | shape>0 递增，<0 递减                |
| `GeneralizedGamma(mu, sigma, Q)` | μ, σ, Q | 灵活（含 Weibull、LogNormal 为特例） |

辅助分布：

| 分布                                         | 说明                            |
| -------------------------------------------- | ------------------------------- |
| `ProportionalHazards(baseline, hr)`        | 等比例风险：h(t) = h₀(t) × HR |
| `AcceleratedFailureTime(baseline, af)`     | 加速失效：S(t) = S₀(t/AF)      |
| `KaplanMeier(times, probs)`                | 经验分布 + 外推                 |
| `PiecewiseExponential(breakpoints, rates)` | 分段常数风险                    |

每个分布都提供 `survival(t)`, `hazard(t)`, `pdf(t)`, `quantile(p)`, `cumulative_hazard(t)`, `restricted_mean(t_max)` 方法。

---

## IPD 拟合功能详解

### SurvivalFitter 类

```python
fitter = ph.SurvivalFitter(
    time=...,           # 观测时间
    event=...,          # 事件指示 (1=事件, 0=删失)
    distributions=None, # 默认拟合全部 6 种，也可指定子集
    label="OS",         # 标签（用于图表标题）
)
```

### 拟合方法

- **MLE（最大似然估计）**：对每种分布使用 `scipy.optimize.minimize` 求解
- **初始值**：基于数据特征自动选择（中位时间、事件率等）
- **参数化**：内部使用对数变换确保参数为正

### 模型选择标准

| 指标       | 公式                  | 说明                            |
| ---------- | --------------------- | ------------------------------- |
| AIC        | 2k - 2ln(L)           | 偏好拟合好+简洁的模型；适合预测 |
| BIC        | k·ln(n) - 2ln(L)     | 比 AIC 更惩罚复杂度；适合大样本 |
| ΔAIC      | AIC - AIC_min         | <2 差异不显著，>10 决定性差异   |
| AIC Weight | exp(-0.5·ΔAIC) / Σ | 模型的相对可能性权重            |

#### ΔAIC 解读规则

| ΔAIC | 证据强度           |
| ----- | ------------------ |
| 0–2  | 弱：两个模型差不多 |
| 2–6  | 中等：有所偏好     |
| 6–10 | 强：明确偏好       |
| >10   | 决定性：压倒性偏好 |

### 诊断图

1. **KM + 拟合曲线图** (`plot_fits()`): KM 阶梯函数叠加所有参数曲线，★ 标记最优
2. **风险函数图** (`plot_hazard()`): 各分布的瞬时风险率对比
3. **log 累积风险图** (`plot_cumhazard_diagnostic()`): 若 Weibull 正确则为直线
4. **Q-Q 图** (`plot_qq()`): 理论分位数 vs 经验分位数

---

## KM 曲线数字化重建

从发表文献的 Kaplan-Meier 曲线图反推个体患者数据 (IPD)，实现「文献 KM 图 → IPD → 参数拟合 → 建模」的完整流程。

基于 Guyot et al. (2012) 算法，参考 R 包 `IPDfromKM` 的预处理策略。

### 基本流程

```python
import pyheor as ph

# 1. 从数字化工具（如 WebPlotDigitizer）获取 KM 坐标
t_digitized = [0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24]
s_digitized = [1.0, 0.92, 0.83, 0.74, 0.66, 0.58, 0.50, 0.40, 0.32, 0.25, 0.20]

# 2. 从文献中读取 number-at-risk 表
t_risk = [0, 6, 12, 18, 24]
n_risk = [120, 88, 60, 38, 22]

# 3. 重建 IPD
ipd_time, ipd_event = ph.guyot_reconstruct(
    t_digitized, s_digitized,
    t_risk, n_risk,
    tot_events=96,  # 可选：文献报告的总事件数
)

# 4. 直接喂入 SurvivalFitter 拟合参数分布
fitter = ph.SurvivalFitter(ipd_time, ipd_event, label="OS")
fitter.fit()
print(fitter.summary())
best = fitter.best_model()

# 5. 用拟合结果构建 PSM 模型
psm = ph.PSMModel(...)
psm.set_survival_all("SOC", [best.distribution, ...])
```

### 数字化坐标预处理

手动数字化的坐标难免有噪声（手抖、非单调点、重复时间等），`clean_digitized_km` 提供自动清洗：

```python
# 独立调用预处理（guyot_reconstruct 内部也会自动调用）
t_clean, s_clean = ph.clean_digitized_km(t_raw, s_raw)
```

预处理步骤：

1. 按时间排序
2. 移除越界点（time < 0 或 survival 超出 [0, 1]）
3. 确保起点为 (0, 1.0)
4. 异常值检测：移除导致连续两次大跳变的「夹心」点（top 1% 跳变法，来自 IPDfromKM）
5. 重复时间处理：同一时间保留最大和最小值（KM 阶梯的上下沿）
6. 强制单调非递增
7. 去除连续重复值

### 参考文献

- Guyot P, Ades AE, Ouwens MJ, Welton NJ (2012). Enhanced secondary analysis of survival data: reconstructing the data from published Kaplan-Meier survival curves. *BMC Med Res Methodol*, 12:9.
- Liu N, Zhou Y, Lee JJ (2021). IPDfromKM: reconstruct individual patient data from published Kaplan-Meier survival curves. *BMC Med Res Methodol*.

---

## 离散事件模拟 (DES)

DES 模型在**连续时间**下模拟个体患者，事件时间直接从生存分布中抽样，无需固定周期长度。

### 核心特点

- **连续时间**：无周期长度伪影，精度任意
- **竞争风险**：同一状态多个事件竞赛，最早先发
- **天然适配生存数据**：直接使用参数化分布 + HR/AFT
- **逆 CDF 抽样**：`SurvivalDistribution.quantile(U)` 生成事件时间

### 基本用法

```python
import pyheor as ph

# 定义模型
model = ph.DESModel(
    states=["PFS", "Progressed", "Dead"],
    strategies={"SOC": "Standard of Care", "TRT": "New Treatment"},
    time_horizon=40,
    dr_cost=0.03,
    dr_qaly=0.03,# 添加参数
model.add_param("hr_pfs", base=0.70,
                dist=ph.LogNormal(mean=-0.36, sd=0.15))

# 基线生存分布
baseline_pfs2prog = ph.Weibull(shape=1.2, scale=5.0)
baseline_pfs2dead = ph.Weibull(shape=1.0, scale=20.0)
baseline_prog2dead = ph.Weibull(shape=1.5, scale=3.0)

# SOC: 直接使用基线分布
model.set_event("SOC", "PFS", "Progressed", baseline_pfs2prog)
model.set_event("SOC", "PFS", "Dead",       baseline_pfs2dead)
model.set_event("SOC", "Progressed", "Dead", baseline_prog2dead)

# TRT: HR 应用于 PFS→Progressed
model.set_event("TRT", "PFS", "Progressed",
    lambda p: ph.ProportionalHazards(baseline_pfs2prog, p["hr_pfs"]))
model.set_event("TRT", "PFS", "Dead",       baseline_pfs2dead)
model.set_event("TRT", "Progressed", "Dead", baseline_prog2dead)

# 费用 (连续时间费率: 元/年)
model.set_state_cost("drug", {
    "SOC": {"PFS": 500, "Progressed": 200, "Dead": 0},
    "TRT": {"PFS": 3000, "Progressed": 200, "Dead": 0},
})
model.set_state_cost("medical", {"PFS": 1000, "Progressed": 5000, "Dead": 0})
model.set_entry_cost("surgery", "Progressed", 50000)  # 一次性进入费用

# 效用
model.set_utility({"PFS": 0.85, "Progressed": 0.50, "Dead": 0})
```

### 运行分析

```python
# 基线分析 (3000 患者/策略)
result = model.run(n_patients=3000, seed=42)
result.summary()       # 费用、QALYs、LYs 汇总 (含 95% CI)
result.icer()          # ICER
result.nmb(wtp=100000) # NMB

# 丰富的结果属性
result.event_log        # 所有患者事件日志 (时间、来源状态、目标状态)
result.time_in_state    # 各状态平均停留时间
result.costs_by_category # 分类费用明细
result.patient_outcomes  # 个体患者结果
result.survival_curve()  # 经验生存曲线

# PSA (外层参数采样 × 内层患者模拟)
psa = model.run_psa(n_sim=200, n_patients=1000, seed=123)
psa.summary()  # PSA 汇总
psa.icer()     # PSA ICER (含 95% CI)
psa.ce_table   # 所有 PSA 迭代的 CE 表
psa.ceac_data() # CEAC 数据
```

### 与 NMA 集成

```python
# 从 NMA 后验样本构建 HR 分布
nma = ph.load_nma_samples("nma_hr_samples.csv", log_scale=True)
nma.add_params_to_model(model, param_prefix="hr")

# 在事件中使用后验 HR
baseline = ph.Weibull(shape=1.2, scale=5.0)
model.set_event("Drug_A", "PFS", "Progressed",
    lambda p: ph.ProportionalHazards(baseline, p["hr_Drug_A"]))
```

### DES vs 其他模型类型

| 特性 | MarkovModel | MicroSimModel | DESModel |
|------|-------------|---------------|----------|
| 时间轴 | 离散周期 | 离散周期 | 连续时间 |
| 分析层级 | 队列 | 个体 | 个体 |
| 转移机制 | 转移矩阵 | 转移概率 | time-to-event 分布 |
| 竞争风险 | 需手动处理 | 需手动处理 | 天然支持 |
| 周期伪影 | 有 (需半周期校正) | 有 | 无 |
| 速度 | 最快 | 中等 | 较慢 |
| 适用场景 | 简单模型 | 复杂异质性 | 事件驱动的复杂模型 |

---

## NMA 整合

PyHEOR 的 NMA 模块负责**导入和使用** R 包（gemtc / multinma / bnma）产生的后验样本，而非重新实现 NMA 算法。

### 加载后验样本

```python
import pyheor as ph

# 宽格式 CSV (一列一个治疗)
nma = ph.load_nma_samples("nma_hr_samples.csv", log_scale=True)

# 长格式 CSV (treatment / value 列)
nma = ph.load_nma_samples(
    "nma_long.csv",
    treatment_col="treatment",
    value_col="d",
    log_scale=True,  # log(HR) → HR 自动转换
)

# 查看后验汇总
print(nma.summary())
#   treatment   mean  median     sd   q2.5  q97.5
#    Drug_A    0.75    0.74   0.12   0.55   1.02
#    Drug_B    0.61    0.60   0.13   0.40   0.90
```

### 注入模型参数

```python
# 方式 1: 逐个添加
model.add_param("hr_A", base=nma.median("Drug_A"), dist=nma.dist("Drug_A"))

# 方式 2: 批量添加 (自动命名 hr_Drug_A, hr_Drug_B)
nma.add_params_to_model(model, param_prefix="hr",
                        treatments=["Drug_A", "Drug_B"])

# 在转移概率中使用
model.set_transitions("Drug_A", lambda p, t: [
    [ph.C,          p["p_prog"] * p["hr_Drug_A"],  0.01],
    [0,             ph.C,                           p["p_death"]],
    [0,             0,                              1],
])
```

### 快速构建生存曲线

```python
# PH 模型: 基准曲线 × HR
baseline = ph.Weibull(shape=1.2, scale=8.0)
curves = ph.make_ph_curves(baseline, nma)
# curves["Drug_A"] → ProportionalHazards(baseline, HR=0.75)

# AFT 模型: 加速因子
curves_aft = ph.make_aft_curves(baseline, nma_af)
```

### 核心类

| 类 / 函数 | 说明 |
|---|---|
| `load_nma_samples()` | 从 CSV/Excel/Feather 加载后验（宽/长格式，支持 log 转换） |
| `NMAPosterior` | 后验容器，提供 `dist()` / `correlated()` / `summary()` / `add_params_to_model()` |
| `PosteriorDist` | `Distribution` 子类，从后验列中有放回抽样 |
| `CorrelatedPosterior` | 联合后验，同行抽样保留相关性 |
| `make_ph_curves()` | 从 NMA HR + 基线曲线 → `ProportionalHazards` 字典 |
| `make_aft_curves()` | 从 NMA AF + 基线曲线 → `AcceleratedFailureTime` 字典 |

---

## Excel 导出

### 1. 结果数据导出

将分析结果导出为多 Sheet 的 Excel 文件，便于审核和验证：

```python
# Markov / PSM 结果
ph.export_to_excel(result, "base_case.xlsx")
ph.export_to_excel(owsa, "owsa.xlsx")
ph.export_to_excel(psa, "psa.xlsx")

# 多策略比较
ph.export_comparison_excel({"Strategy A": result_a, "Strategy B": result_b}, "comparison.xlsx")

# IPD 拟合结果
fitter.to_excel("fitting_results.xlsx")
```

### 2. Excel 公式验证模型 ⭐

导出一个**用 Excel 公式独立计算**的完整模型文件，用于交叉验证 Python 结果：

```python
# 从结果对象导出
result = model.run_base_case()
ph.export_excel_model(result, "verification.xlsx")

# 或直接从模型导出 (使用基线参数)
ph.export_excel_model(model, "verification.xlsx")
```

Excel 验证模型包含：

| 区域 | 内容 |
|------|------|
| **输入区** (黄色底色) | 转移概率矩阵、状态费用向量、效用权重、贴现率 |
| **计算区** (公式) | `SUMPRODUCT` 计算 Trace, 费用, QALY; `SUM` 计算贴现总值 |
| **Summary sheet** | Excel 公式结果 vs Python 结果 vs 差异 (应为 ~0) |

**支持的模型类型**：

| 模型 | Trace | 费用/QALY/贴现 | ICER |
|------|-------|----------------|------|
| Markov (时齐) | ✅ Excel 公式 | ✅ Excel 公式 | ✅ Excel 公式 |
| Markov (时变) | Python 值 | ✅ Excel 公式 | ✅ Excel 公式 |
| PSM | Python 生存值 → ✅ 状态概率公式 | ✅ Excel 公式 | ✅ Excel 公式 |

### Excel Sheet 内容

| 分析类型    | Sheet 内容                                                            |
| ----------- | --------------------------------------------------------------------- |
| Base Case   | Summary, State Trace, Cost/QALY by Cycle, ICER                        |
| OWSA        | Tornado Data, Per-Parameter Results                                   |
| PSA         | Summary Stats, All Simulations, CEAC Data                             |
| PSM Base    | Summary, State Probabilities, Survival Data                           |
| IPD Fitting | Model Comparison, KM Data, Per-Distribution Details, Selection Report |
| **验证模型** | **Summary (含差异), 每策略计算Sheet (公式+输入)** |

---

## 预算影响分析 (BIA)

预算影响分析估计在短期时间范围内（通常 1–5 年）引入新技术对医疗支付方预算的财务影响。
遵循 ISPOR BIA 良好实践指南 (Sullivan et al. 2014, Mauskopf et al. 2007)。

### 基本用法

```python
import pyheor as ph

bia = ph.BudgetImpactAnalysis(
    strategies=["Drug A", "Drug B", "Drug C"],
    per_patient_costs={"Drug A": 5000, "Drug B": 12000, "Drug C": 8000},
    population=10000,
    market_share_current={"Drug A": 0.6, "Drug B": 0.1, "Drug C": 0.3},
    market_share_new={"Drug A": 0.4, "Drug B": 0.3, "Drug C": 0.3},
    time_horizon=5,
)

bia.summary()             # 年度预算影响汇总表
bia.cost_by_strategy()    # 按策略分解的费用表
bia.detail()              # 按情景/策略/年份的详细明细
```

### 人群模型

支持多种人群规模设定方式：

```python
population=10000                                    # 固定人群
population=[10000, 10500, 11000, 11500, 12000]      # 逐年指定
population={"base": 10000, "growth_rate": 0.05}      # 复合增长
population={"base": 10000, "annual_increase": 500}   # 线性增长
```

### 市场份额摄取曲线

提供线性和 S 型摄取曲线工具：

```python
# 线性: 0% → 40% over 5 years
ph.BudgetImpactAnalysis.linear_uptake(0.0, 0.4, 5)
# [0.0, 0.1, 0.2, 0.3, 0.4]

# S 型: 0% → 40% over 5 years (steepness=1.5)
ph.BudgetImpactAnalysis.sigmoid_uptake(0.0, 0.4, 5, steepness=1.5)
# [0.0, 0.035, 0.200, 0.365, 0.400]

# 在 BIA 中使用
bia = ph.BudgetImpactAnalysis(
    strategies=["SoC", "New"],
    per_patient_costs={"SoC": 3000, "New": 15000},
    population={"base": 50000, "growth_rate": 0.03},
    market_share_current={"SoC": 1.0, "New": 0.0},
    market_share_new={
        "SoC": ph.BudgetImpactAnalysis.linear_uptake(1.0, 0.6, 5),
        "New": ph.BudgetImpactAnalysis.linear_uptake(0.0, 0.4, 5),
    },
    time_horizon=5,
    cost_inflation=0.02,   # 年通胀率
)
```

### 从模型结果创建

```python
result = model.run_base_case()

bia = ph.BudgetImpactAnalysis.from_result(
    result,
    population=10000,
    market_share_current={"SoC": 0.8, "New": 0.2},
    market_share_new={"SoC": 0.5, "New": 0.5},
    time_horizon=5,
    annualize_years=20,   # 模型总年数，用于年化总费用
)
```

### 情景分析

```python
bia.scenario_analysis({
    "Base Case": {},
    "High Population": {"population": 15000},
    "Low Population": {"population": 7000},
    "Fast Uptake": {"market_share_new": {"SoC": 0.3, "New": 0.7}},
})
```

### 单因素敏感性分析

```python
# 人群规模敏感性
bia.one_way_sensitivity("population", values=[8000, 9000, 10000, 11000, 12000])

# 某策略费用敏感性
bia.one_way_sensitivity("Drug B", values=[10000, 12000, 15000])
```

### 龙卷风图数据

```python
tornado_df = bia.tornado({
    "population": (8000, 12000),
    "Drug B": (10000, 15000),
    "cost_inflation": (0.0, 0.05),
})
```

### BIA 可视化

| 图表 | 方法 | 说明 |
|------|------|------|
| 预算影响柱状图 | `bia.plot_budget_impact()` | 年度影响 + 累计曲线 |
| 预算对比图 | `bia.plot_budget_comparison()` | 当前 vs 新情景 总费用 |
| 市场份额图 | `bia.plot_market_share()` | 双面板堆叠面积图 |
| 费用明细图 | `bia.plot_detail()` | 按策略堆叠柱状图 |
| BIA 龙卷风图 | `bia.plot_tornado({...})` | 单因素敏感性龙卷风 |

---

## 模型校准

模型校准用观测数据（如发病率、死亡率、患病率）反推模型中无法直接观测的参数（如自然史转移概率），使模型输出匹配经验数据。

基于 Vanni et al. (2011) 七步校准框架和 Alarid-Escudero et al. (2018) 教程。

### 基本用法

```python
import pyheor as ph

# 1. 构建模型（参数值待校准）
model = ph.MarkovModel(
    states=["Healthy", "Sick", "Dead"],
    strategies=["SOC"],
    n_cycles=20,
    half_cycle_correction=False,
)
model.add_param("p_HS", base=0.10)   # 初始猜测
model.add_param("p_SD", base=0.05)
model.set_transitions("SOC", lambda p, t: [
    [ph.C,  p["p_HS"], 0.01],
    [0,     ph.C,      p["p_SD"]],
    [0,     0,         1],
])
model.set_utility({"Healthy": 1.0, "Sick": 0.7, "Dead": 0.0})

# 2. 定义校准目标（从文献/数据中获得的观测值）
targets = [
    ph.CalibrationTarget(
        name="10yr_healthy",
        observed=0.42,           # 10 年后健康状态占比
        se=0.05,                 # 标准误
        extract_fn=lambda sim: sim["SOC"]["trace"][10, 0],
    ),
    ph.CalibrationTarget(
        name="10yr_dead",
        observed=0.25,
        se=0.04,
        extract_fn=lambda sim: sim["SOC"]["trace"][10, 2],
    ),
]

# 3. 定义待校准参数及搜索范围
calib_params = [
    ph.CalibrationParam("p_HS", lower=0.01, upper=0.30),
    ph.CalibrationParam("p_SD", lower=0.01, upper=0.20),
]

# 4. 运行校准
result = ph.calibrate(
    model,
    targets,
    calib_params,
    gof="wsse",              # 加权 SSE（按 1/SE² 加权）
    method="nelder_mead",    # 多起点 Nelder-Mead
    n_restarts=10,
    seed=42,
)

# 5. 查看结果
print(result.summary())              # 最优参数值
print(result.target_comparison())    # 观测 vs 预测对比
result.apply_to_model(model)         # 将校准结果写回模型
```

### 搜索方法

| 方法 | 参数 | 特点 |
|------|------|------|
| `nelder_mead` | `n_restarts=10` | 多起点无导数优化，精确但较慢 |
| `random_search` | `n_samples=1000` | LHS 采样逐一评估，简单直观 |

### GoF 度量

| 度量 | 公式 | 适用场景 |
|------|------|----------|
| `sse` | Σ(obs - pred)² | 默认，简单快速 |
| `wsse` | Σ(obs - pred)²/SE² | 多目标量纲不同时 |
| `loglik_normal` | -Σ log N(obs \| pred, SE²) | 统计原则化 |

### CalibrationResult 方法

```python
result.summary()              # 参数汇总表
result.target_comparison()    # 观测 vs 预测
result.apply_to_model(model)  # 写回模型参数
result.plot_gof()             # GoF 值散点图
result.plot_pairs(top_n=100)  # 参数对图（top-n 最优组合）
```

### 参考文献

- Vanni T et al. (2011). Calibrating models in economic evaluation: a seven-step approach. *PharmacoEconomics*, 29(1), 35-49.
- Alarid-Escudero F et al. (2018). A Tutorial on Calibration of Health Decision Models. *Medical Decision Making*, 38(8), 980-990.

---

## 可视化一览

PyHEOR 共提供 **28 种**专业图表，覆盖全部模型类型和分析流程：

### Markov 模型 (8 种)

| 函数                          | 说明                              |
| ----------------------------- | --------------------------------- |
| `plot_transition_diagram()` | 状态转移图                        |
| `plot_model_diagram()`      | TreeAge 风格模型图                |
| `plot_trace()`              | Markov trace（队列轨迹）          |
| `plot_tornado()`            | OWSA 龙卷风图                     |
| `plot_owsa_param()`         | 单参数 OWSA 线图                  |
| `plot_scatter()`            | CE 散点图（增量成本 vs 增量效果） |
| `plot_ceac()`               | 成本-效果可接受曲线               |
| `plot_convergence()`        | PSA 收敛诊断图                    |

### PSM 模型 (4 种)

| 函数                       | 说明                 |
| -------------------------- | -------------------- |
| `plot_survival_curves()` | 参数化生存曲线       |
| `plot_state_area()`      | 面积图（各状态占比） |
| `plot_psm_trace()`       | PSM 状态轨迹         |
| `plot_psm_comparison()`  | 多策略生存曲线对比   |

### 微观模拟 (3 种)

| 函数                         | 说明                                      |
| ---------------------------- | ----------------------------------------- |
| `plot_microsim_trace()`    | 个体模拟状态占比轨迹                      |
| `plot_microsim_survival()` | 经验生存曲线（基于模拟数据）              |
| `plot_microsim_outcomes()` | 患者结局分布（QALYs / 费用 / LYs 直方图） |

### IPD 拟合 (4 种)

| 方法                                   | 说明                    |
| -------------------------------------- | ----------------------- |
| `fitter.plot_fits()`                 | KM + 所有参数拟合曲线   |
| `fitter.plot_hazard()`               | 各分布风险函数          |
| `fitter.plot_cumhazard_diagnostic()` | log(H) vs log(t) 诊断图 |
| `fitter.plot_qq()`                   | Q-Q 分位数图            |

### CEA / 多策略比较 (4 种)

| 函数                   | 说明                               |
| ---------------------- | ---------------------------------- |
| `plot_ce_frontier()`   | 效率前沿图 + WTP 线 + ICER 标注   |
| `plot_nmb_curve()`     | NMB 曲线（多策略随 WTP 变化）      |
| `plot_ceaf()`          | 成本效果可接受前沿曲线 (CEAF)     |
| `plot_evpi()`          | 完美信息期望价值 (EVPI) 曲线      |

### 预算影响分析 (5 种)

| 函数                         | 说明                             |
| ---------------------------- | -------------------------------- |
| `plot_budget_impact()`       | 年度预算影响柱状图 + 累计曲线    |
| `plot_budget_comparison()`   | 当前 vs 新情景总费用对比         |
| `plot_market_share()`        | 双面板市场份额堆叠面积图         |
| `plot_detail()`              | 按策略堆叠费用明细图             |
| `plot_tornado()`             | BIA 敏感性龙卷风图               |

---

## 项目结构

```
pyheor/
├── pyproject.toml
├── README.md
├── src/pyheor/              # 包源码 (src layout)
│   ├── __init__.py          # 包入口，统一导出
│   ├── utils.py             # 工具函数 (C 补数, 贴现, 验证)
│   ├── distributions.py     # PSA 概率分布 (Beta, Gamma, ...)
│   ├── survival.py          # 10 种参数化生存分布
│   ├── plotting.py          # 可视化 (28 种图表)
│   │
│   ├── models/              # ── 建模引擎 ──
│   │   ├── markov.py        #  Markov 队列模型 (MarkovModel)
│   │   ├── psm.py           #  分区生存模型 (PSMModel)
│   │   ├── microsim.py      #  微观模拟 (MicroSimModel)
│   │   └── des.py           #  离散事件模拟 (DESModel)
│   │
│   ├── analysis/            # ── 分析与决策 ──
│   │   ├── results.py       #  结果类 (BaseResult, OWSAResult, PSAResult, ...)
│   │   ├── comparison.py    #  多策略比较 / CEA (CEAnalysis)
│   │   ├── bia.py           #  预算影响分析 (BudgetImpactAnalysis)
│   │   └── calibration.py   #  模型校准 (Nelder-Mead, 随机搜索)
│   │
│   ├── evidence/            # ── 数据与证据合成 ──
│   │   ├── fitting.py       #  IPD 生存曲线拟合 (SurvivalFitter)
│   │   ├── digitize.py      #  KM 曲线数字化重建 (Guyot method)
│   │   └── nma.py           #  NMA 后验样本整合 (NMAPosterior)
│   │
│   └── export/              # ── 导出 ──
│       ├── excel.py         #  Excel 结果数据导出
│       └── excel_model.py   #  Excel 公式验证模型导出
│
├── tests/                   # pytest 测试套件 (243 个测试)
└── examples/
    ├── demo_hiv_model.py    #  Markov 模型示例 (HIV)
    ├── demo_psm_model.py    #  PSM 模型示例 (肿瘤)
    ├── demo_ipd_fitting.py  #  IPD 拟合示例
    ├── demo_microsim.py     #  微观模拟示例
    └── demo_comparison.py   #  多策略比较示例
```

---

## 设计理念

- **简洁的 API**：一个模型对象搞定 base case / OWSA / PSA，不需要分开调用
- **灵活的参数系统**：`ph.C` 自动补数，lambda 函数定义时变概率/费用
- **与 R 生态对齐**：分布参数化、方法命名参考 hesim / flexsurv / DARTH
- **生产级可视化**：所有图表开箱即用，配色统一，支持自定义
- **可验证性**：Excel 导出 trace 数据，方便与 TreeAge / Excel 模型交叉验证

---

## 路线图

- [X] Markov 队列模型 (cDTSTM)
- [X] 单因素敏感性分析 (OWSA) + 龙卷风图
- [X] 概率敏感性分析 (PSA) + CEAC + CE 散点图
- [X] 灵活费用系统（首周期、时变、WLOS、自定义费用函数）
- [X] 半周期校正多方法（梯形法 / 生命表法 / 无校正）& 可配置贴现率
- [X] OWSA 龙卷风图 ICER 排序 & 贴现率通过 `Param` 直接参与敏感性分析
- [X] 分区生存模型 (PSM)
- [X] 10 种参数化生存分布
- [X] Excel 多 Sheet 导出
- [X] IPD 生存曲线拟合 + AIC/BIC 模型比较
- [X] KM + 拟合曲线可视化 + 诊断图
- [X] 微观模拟 (Individual-level simulation)
- [X] 多队列比较 + NMB 分析 + CEAF + EVPI
- [X] 网络 Meta 分析 (NMA) 整合
- [X] 离散事件模拟 (DES) — 连续时间、竞争风险、HR/AFT 集成
- [X] 预算影响分析 (BIA) — 人群模型、市场份额演变、摄取曲线、情景/敏感性分析
- [X] 数字化 KM 曲线重建 (Guyot method)
- [X] 模型校准 (Nelder-Mead 多起点优化, LHS 随机搜索, SSE/WSSE/似然 GoF)
- [X] 正式测试套件 (pytest, 243 个测试覆盖全部模块)

---

## 许可证

MIT License
