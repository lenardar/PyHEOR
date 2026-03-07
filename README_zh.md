# PyHEOR — Python Health Economics and Outcome Research

[English](README.md) | **中文** | [Français](README_fr.md)

> **用 Python 做卫生经济学建模，像 R 的 hesim / DARTH 一样专业，但更简洁。**

PyHEOR 是一个面向卫生经济学研究的 Python 框架，支持：

| 功能                              | 说明                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------- |
| **Markov 队列模型**               | 离散时间状态转移模型 (cDTSTM)，时齐 / 时变转移矩阵                                          |
| **分区生存模型 (PSM)**            | 基于参数化生存曲线的状态概率划分                                                            |
| **微观模拟**                      | 个体水平状态转移模型，支持患者异质性、事件处理器、双层 PSA                                  |
| **离散事件模拟 (DES)**            | 连续时间个体模拟，竞争风险、time-to-event 分布驱动、HR/AFT 集成                             |
| **参数化生存分布**                | Exponential, Weibull, Log-logistic, Log-normal, Gompertz, Generalized Gamma 等 10 种        |
| **灵活的费用定义**                | 首周期费用、时间依赖函数、一次性费用、WLOS 方法、转移费用计划表、自定义费用函数               |
| **基础分析 / OWSA / PSA**         | 确定性分析、龙卷风图 (INMB/ICER)、Monte Carlo + CE 散点图 + CEAC                            |
| **多策略比较 & NMB**              | 效率前沿、支配/扩展支配检测、NMB 曲线、CEAF、EVPI                                           |
| **IPD 生存曲线拟合**              | 6 种参数分布 MLE 拟合，AIC/BIC 比较，自动选优                                               |
| **KM 曲线数字化重建**             | Guyot method 从发表文献 KM 图反推 IPD，含数字化噪声预处理                                    |
| **NMA 整合**                      | 导入 R 后验样本，保留相关性，自动生成 PH/AFT 曲线                                           |
| **预算影响分析 (BIA)**            | 人群规模模型、市场份额演变、摄取曲线、情景/单因素敏感性分析                                  |
| **模型校准**                      | 用观测数据反推未知参数：Nelder-Mead 多起点优化、LHS 随机搜索、SSE/WSSE/似然 GoF              |
| **可视化**                        | 28 种专业图表：状态转移图、前沿图、NMB 曲线、CEAF、EVPI、CEAC、KM+拟合曲线、BIA 影响图等    |
| **导出**                          | Excel 多 Sheet 导出、Excel 公式验证模型、Markdown 一键报告                                   |

---

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [用户指南](#用户指南)
  - [模型类型](#模型类型) · [参数系统](#参数系统) · [转移矩阵](#转移矩阵) · [费用与效用](#费用与效用) · [生存分析](#生存分析) · [敏感性分析与报告](#敏感性分析与报告) · [高级功能](#高级功能) · [导出](#导出)
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

依赖：`numpy`, `pandas`, `matplotlib`, `scipy`（可选：`openpyxl` 用于 Excel 导出，`tabulate` 用于 Markdown 报告）

---

## 快速开始

```python
import pyheor as ph

# ── 定义模型 ──
model = ph.MarkovModel(
    states=["Healthy", "Sick", "Dead"],
    strategies=["SOC", "Treatment"],
    n_cycles=40,
    cycle_length=1,
    dr_cost=ph.Param(0.03, low=0.0, high=0.08, label="费用贴现率"),
    dr_qaly=ph.Param(0.03, low=0.0, high=0.05, label="效用贴现率"),
    half_cycle_correction=True,
)

# ── 参数 ──
model.add_param("p_HS", base=0.15, low=0.10, high=0.20,
    dist=ph.Beta(mean=0.15, sd=0.03))
model.add_param("c_drug", base=2000, low=1500, high=2500,
    dist=ph.Gamma(mean=2000, sd=400))

# ── 转移矩阵 (ph.C = 补数) ──
model.set_transitions("SOC", lambda p, t: [
    [ph.C,  p["p_HS"], 0.02],
    [0,     ph.C,      0.10],
    [0,     0,         1   ],
])
model.set_transitions("Treatment", lambda p, t: [
    [ph.C,  p["p_HS"] * 0.7, 0.02],
    [0,     ph.C,             0.08],
    [0,     0,                1   ],
])

# ── 费用 & 效用 ──
model.set_state_cost("medical", {"Healthy": 500, "Sick": 3000, "Dead": 0})
model.set_state_cost("drug", {
    "SOC": {"Healthy": 0, "Sick": 0, "Dead": 0},
    "Treatment": {
        "Healthy": lambda p, t: p["c_drug"],
        "Sick": lambda p, t: p["c_drug"],
        "Dead": 0,
    },
})
model.set_utility({"Healthy": 0.95, "Sick": 0.60, "Dead": 0.0})

# ── 运行分析 ──
result = model.run_base_case()
print(result.summary())
print(result.icer())

owsa = model.run_owsa()       # 贴现率通过 Param 自动参与 OWSA
owsa.plot_tornado()

psa = model.run_psa(n_sim=1000)
psa.plot_ceac()

# ── 一键生成 Markdown 报告 ──
ph.generate_report(model, "report.md")
```

---

## 用户指南

### 模型类型

#### Markov 队列模型

离散时间队列模型 (cDTSTM)，适用于状态转移概率已知的简单模型。完整示例见 [快速开始](#快速开始)。

#### 分区生存模型 (PSM)

基于参数化生存曲线推导状态占比，适用于肿瘤经济学中常见的 PFS/OS 分析框架。

```python
import pyheor as ph

psm = ph.PSMModel(
    states=["PFS", "Progressed", "Dead"],
    survival_endpoints=["PFS", "OS"],
    strategies=["SOC", "New Drug"],
    n_cycles=120,
    cycle_length=1/12,
    dr_cost=0.03,
    dr_qaly=0.03,
)

# 基线生存曲线
baseline_pfs = ph.LogLogistic(shape=1.5, scale=18)
baseline_os = ph.Weibull(shape=1.2, scale=36)

# SOC: 直接使用基线
psm.set_survival("SOC", "PFS", baseline_pfs)
psm.set_survival("SOC", "OS", baseline_os)

# New Drug: HR / AFT 修饰
psm.set_survival("New Drug", "PFS",
    lambda p: ph.AcceleratedFailureTime(baseline_pfs, af=1.3))
psm.set_survival("New Drug", "OS",
    lambda p: ph.ProportionalHazards(baseline_os, hr=0.7))

# 费用 & 效用
psm.set_state_cost("treatment", {
    "SOC": {"PFS": 1000, "Progressed": 2500, "Dead": 0},
    "New Drug": {"PFS": 6000, "Progressed": 2500, "Dead": 0},
})
psm.set_utility({"PFS": 0.80, "Progressed": 0.55, "Dead": 0.0})

result = psm.run_base_case()
print(result.summary())
result.plot_survival()
result.plot_state_area()
```

#### 微观模拟 (Microsimulation)

个体水平状态转移模型，与 MarkovModel 共享同一套 API（`add_param`, `set_transitions`, `set_state_cost`, `set_utility`），但每位患者独立采样，产生个体异质性结局。

```python
import pyheor as ph

model = ph.MicroSimModel(
    states=["Healthy", "Sick", "Sicker", "Dead"],
    strategies=["SOC", "Treatment"],
    n_cycles=30,
    n_patients=5000,
    cycle_length=1.0,
    dr_cost=0.03,
    dr_qaly=0.03,
    seed=42,
)

model.add_param("p_HS", base=0.15, dist=ph.Beta(mean=0.15, sd=0.03))
model.add_param("hr_trt", base=0.70, dist=ph.LogNormal(mean=0.70, sd=0.10))

model.set_transitions("SOC", lambda p, t: [
    [ph.C,  p["p_HS"],                0,     0.005],
    [0,     ph.C,                     0.10,  0.05 ],
    [0,     0,                        ph.C,  0.10 ],
    [0,     0,                        0,     1    ],
])
model.set_transitions("Treatment", lambda p, t: [
    [ph.C,  p["p_HS"] * p["hr_trt"], 0,     0.005],
    [0,     ph.C,                     0.10 * p["hr_trt"], 0.05],
    [0,     0,                        ph.C,  0.10 ],
    [0,     0,                        0,     1    ],
])

model.set_state_cost("medical", {"Healthy": 500, "Sick": 3000, "Sicker": 8000, "Dead": 0})
model.set_state_cost("drug", {
    "SOC": {"Healthy": 0, "Sick": 0, "Sicker": 0, "Dead": 0},
    "Treatment": {"Healthy": 5000, "Sick": 5000, "Sicker": 5000, "Dead": 0},
})
model.set_utility({"Healthy": 0.95, "Sick": 0.75, "Sicker": 0.50, "Dead": 0.0})

# 事件处理器：进入 Sicker 时一次性住院费
model.on_state_enter("Sicker", lambda idx, t, attrs: {"cost": 15000})

result = model.run_base_case(verbose=True)
print(result.summary())   # 含 SD 和 95% 百分位数

# PSA: 外层参数不确定性 × 内层个体随机性
psa = model.run_psa(n_outer=500, n_inner=2000, seed=42)
psa.plot_ceac(wtp_range=(0, 150000))
```

**患者异质性**：转移概率支持 3 参数 lambda `(params, cycle, attrs)`，可基于个体属性（年龄、性别等）调整：

```python
import numpy as np

pop = ph.PatientProfile(
    n_patients=5000,
    attributes={
        "age": np.random.normal(55, 12, 5000).clip(20, 90),
        "female": np.random.binomial(1, 0.52, 5000),
    }
)
model.set_population(pop)

model.set_transitions("SOC", lambda p, t, attrs: [
    [ph.C,  p["p_HS"] * (1 + (attrs["age"] - 55) * 0.02), 0.005],
    [0,     ph.C,  0.05],
    [0,     0,     1],
])
```

**性能优化**：当转移矩阵不依赖个体属性（2 参数 lambda）时，引擎自动使用向量化批量采样，速度接近队列模型。

#### 离散事件模拟 (DES)

DES 在**连续时间**下模拟个体患者，事件时间直接从生存分布中抽样，无需固定周期长度。

```python
import pyheor as ph

model = ph.DESModel(
    states=["PFS", "Progressed", "Dead"],
    strategies={"SOC": "Standard of Care", "TRT": "New Treatment"},
    time_horizon=40,
    dr_cost=0.03,
    dr_qaly=0.03,
)

model.add_param("hr_pfs", base=0.70,
    dist=ph.LogNormal(mean=-0.36, sd=0.15))

baseline_pfs2prog = ph.Weibull(shape=1.2, scale=5.0)
baseline_pfs2dead = ph.Weibull(shape=1.0, scale=20.0)
baseline_prog2dead = ph.Weibull(shape=1.5, scale=3.0)

# SOC: 直接使用基线
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
model.set_entry_cost("surgery", "Progressed", 50000)

model.set_utility({"PFS": 0.85, "Progressed": 0.50, "Dead": 0})

# 运行
result = model.run(n_patients=3000, seed=42)
result.summary()
result.icer()

# PSA
psa = model.run_psa(n_sim=200, n_patients=1000, seed=123)
psa.summary()
```

**DES vs 其他模型类型**：

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

#### 贴现率

所有模型均通过 `dr_cost` 和 `dr_qaly` 两个独立参数设置贴现率。**默认值为 0（不贴现）**，未设置的一方不会被贴现。

```python
# 固定贴现率
model = ph.MarkovModel(..., dr_cost=0.03, dr_qaly=0.03)

# 只贴现费用
model = ph.MarkovModel(..., dr_cost=0.06)  # dr_qaly 默认 0
```

传入 `Param` 对象即可将贴现率纳入 OWSA / PSA，无需额外调用 `add_param()`：

```python
model = ph.MarkovModel(
    ...,
    dr_cost=ph.Param(0.03, low=0.0, high=0.08, label="费用贴现率"),
    dr_qaly=ph.Param(0.03, low=0.0, high=0.05, label="效用贴现率"),
)

owsa = model.run_owsa()
owsa.plot_tornado()  # 龙卷风图中包含贴现率

# 也可以只对其中一个做敏感性分析
model = ph.MarkovModel(
    ...,
    dr_cost=0.03,                                        # 固定
    dr_qaly=ph.Param(0.03, low=0.0, high=0.05),          # 变动
)
```

> **设计原则**：贴现率的基准值和敏感性分析范围在同一处定义，避免重复指定。`float` = 固定值，`Param` = 可变动值。

#### 半周期校正

| 值                         | 说明                                            |
| -------------------------- | ----------------------------------------------- |
| `True` / `"trapezoidal"` | 梯形法：首尾周期权重 ×0.5（默认）                |
| `"life-table"`            | 生命表法：相邻 trace 行取均值（与 R heemod 一致）|
| `False` / `None`          | 不校正                                          |

```python
model.half_cycle_correction = "life-table"
model.half_cycle_correction = "trapezoidal"
model.half_cycle_correction = False
```

---

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

---

### 费用与效用

#### 状态费用

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

#### 转移费用 (Transition Costs)

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
```

**费用计划表**：当转移后需要跨多个周期产生费用时（如手术 + 随访），传入列表。引擎通过卷积自动处理多批次转入患者的费用叠加：

```python
# 进展时手术 50000，下一周期随访 10000 → 共 2 周期
model.set_transition_cost("surgery", "PFS", "Progressed", [50000, 10000])

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

#### 自定义费用 (Custom Costs)

当 `set_transition_cost` 按单个状态对定义费用不够灵活时，可以用 `set_custom_cost` 传入自定义函数，直接基于转移矩阵和状态分布计算费用。支持 MarkovModel 和 PSMModel。

```python
# 函数签名
# func(strategy, params, t, state_prev, state_curr, P, states) -> float

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

---

### 生存分析

#### 参数化生存分布

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

#### IPD 生存曲线拟合

```python
import pyheor as ph
import pandas as pd

df = pd.read_csv("patient_data.csv")
fitter = ph.SurvivalFitter(
    time=df["time"],
    event=df["event"],
    label="Overall Survival",
)
fitter.fit()

# AIC/BIC 比较表
print(fitter.summary())

# 自动选择最优模型
best = fitter.best_model()           # 默认 AIC
dist = best.distribution             # 可直接用于 PSM
print(fitter.selection_report())     # 模型选择详细解读

# 诊断图
fitter.plot_fits()                   # KM + 所有拟合曲线
fitter.plot_hazard()                 # 风险函数
fitter.plot_cumhazard_diagnostic()   # log(H) vs log(t)
fitter.plot_qq()                     # Q-Q 图

# 导出
fitter.to_excel("fitting_results.xlsx")
```

**模型选择标准**：

| 指标       | 公式                  | 说明                            |
| ---------- | --------------------- | ------------------------------- |
| AIC        | 2k - 2ln(L)           | 偏好拟合好+简洁的模型；适合预测 |
| BIC        | k·ln(n) - 2ln(L)     | 比 AIC 更惩罚复杂度；适合大样本 |
| ΔAIC      | AIC - AIC_min         | <2 差异不显著，>10 决定性差异   |
| AIC Weight | exp(-0.5·ΔAIC) / Σ | 模型的相对可能性权重            |

**IPD → PSM 一体化流程**：

```python
fitter_os = ph.SurvivalFitter(time=df_os["time"], event=df_os["event"], label="OS")
fitter_pfs = ph.SurvivalFitter(time=df_pfs["time"], event=df_pfs["event"], label="PFS")
fitter_os.fit()
fitter_pfs.fit()

psm = ph.PSMModel(...)
psm.set_survival("SOC", "OS", fitter_os.best_model().distribution)
psm.set_survival("SOC", "PFS", fitter_pfs.best_model().distribution)
```

#### KM 曲线数字化重建

从发表文献的 KM 曲线图反推 IPD，实现「文献 KM 图 → IPD → 参数拟合 → 建模」的完整流程。基于 Guyot et al. (2012) 算法。

```python
# 1. 从数字化工具（如 WebPlotDigitizer）获取 KM 坐标
t_digitized = [0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24]
s_digitized = [1.0, 0.92, 0.83, 0.74, 0.66, 0.58, 0.50, 0.40, 0.32, 0.25, 0.20]

# 2. 从文献中读取 number-at-risk 表
t_risk = [0, 6, 12, 18, 24]
n_risk = [120, 88, 60, 38, 22]

# 3. 重建 IPD
ipd_time, ipd_event = ph.guyot_reconstruct(
    t_digitized, s_digitized, t_risk, n_risk, tot_events=96,
)

# 4. 直接喂入 SurvivalFitter
fitter = ph.SurvivalFitter(ipd_time, ipd_event, label="OS")
fitter.fit()
```

**数字化坐标预处理**：`clean_digitized_km` 提供自动清洗（排序、去越界、异常值检测、强制单调等），`guyot_reconstruct` 内部也会自动调用。

参考文献：

- Guyot P, Ades AE, Ouwens MJ, Welton NJ (2012). Enhanced secondary analysis of survival data. *BMC Med Res Methodol*, 12:9.
- Liu N, Zhou Y, Lee JJ (2021). IPDfromKM. *BMC Med Res Methodol*.

---

### 敏感性分析与报告

#### OWSA & PSA

```python
# OWSA（贴现率通过 Param 自动注册）
owsa = model.run_owsa(wtp=50000)
print(owsa.summary(outcome="icer"))   # 按 ICER 影响幅度排序
owsa.plot_tornado(outcome="nmb", max_params=10)

# PSA (Monte Carlo)
psa = model.run_psa(n_sim=1000, seed=42)
print(psa.summary())
print(psa.icer())
psa.plot_scatter(wtp=50000)
psa.plot_ceac()
psa.plot_convergence()
```

#### 一键报告 (`generate_report`)

模型参数设置完毕后，一键运行全部分析并生成 Markdown 报告 + 配套图片：

```python
ph.generate_report(
    model,
    "report.md",       # 输出路径，图片存入 report_files/
    wtp=50000,          # WTP 阈值
    n_sim=1000,         # PSA 模拟次数
    max_params=10,      # 龙卷风图最多显示参数数
    run_psa=None,       # None=自动检测（有 dist 就跑）
)
```

报告内容包含：模型概述、参数表、基础分析结果、ICER、OWSA 龙卷风图及排序表、PSA 汇总统计及增量分析、CE 平面散点图、CEAC 曲线。所有模型类型（Markov / PSM / MicroSim / DES）均支持。

---

### 高级功能

#### 多策略比较 & NMB 分析

```python
# 从确定性结果创建 CEAnalysis
result = model.run_base_case()
cea = ph.CEAnalysis.from_result(result)

# 效率前沿：顺序 ICER + 支配/扩展支配检测
print(cea.frontier())

# NMB 排名
print(cea.nmb(wtp=100000))
print(f"最优策略: {cea.optimal_strategy(wtp=100000)}")

# 可视化
cea.plot_frontier(wtp=100000)
cea.plot_nmb_curve(wtp_range=(0, 200000))
```

**PSA → CEAF & EVPI**：

```python
psa_result = model.run_psa(n_sim=2000)
cea_psa = ph.CEAnalysis.from_psa(psa_result)

cea_psa.plot_ceaf(wtp_range=(0, 200000))
print(f"EVPI at WTP=$100K: ${cea_psa.evpi_single(100000):,.0f}")
cea_psa.plot_evpi(wtp_range=(0, 200000), population=100000)
```

#### NMA 整合

PyHEOR 的 NMA 模块负责**导入和使用** R 包（gemtc / multinma / bnma）产生的后验样本。

```python
# 加载后验样本 (支持宽/长格式 CSV)
nma = ph.load_nma_samples("nma_hr_samples.csv", log_scale=True)
print(nma.summary())

# 批量注入模型参数
nma.add_params_to_model(model, param_prefix="hr",
                        treatments=["Drug_A", "Drug_B"])

# 快速构建生存曲线
baseline = ph.Weibull(shape=1.2, scale=8.0)
curves = ph.make_ph_curves(baseline, nma)      # PH
curves_aft = ph.make_aft_curves(baseline, nma)  # AFT
```

| 类 / 函数 | 说明 |
|---|---|
| `load_nma_samples()` | 从 CSV/Excel/Feather 加载后验（宽/长格式，支持 log 转换） |
| `NMAPosterior` | 后验容器，提供 `dist()` / `correlated()` / `summary()` / `add_params_to_model()` |
| `PosteriorDist` | `Distribution` 子类，从后验列中有放回抽样 |
| `CorrelatedPosterior` | 联合后验，同行抽样保留相关性 |
| `make_ph_curves()` / `make_aft_curves()` | 从 NMA 后验 + 基线曲线 → PH/AFT 曲线字典 |

#### 预算影响分析 (BIA)

预算影响分析估计在短期时间范围内（通常 1–5 年）引入新技术对预算的财务影响。遵循 ISPOR BIA 良好实践指南。

```python
bia = ph.BudgetImpactAnalysis(
    strategies=["Drug A", "Drug B", "Drug C"],
    per_patient_costs={"Drug A": 5000, "Drug B": 12000, "Drug C": 8000},
    population=10000,
    market_share_current={"Drug A": 0.6, "Drug B": 0.1, "Drug C": 0.3},
    market_share_new={"Drug A": 0.4, "Drug B": 0.3, "Drug C": 0.3},
    time_horizon=5,
)

bia.summary()
bia.cost_by_strategy()
```

**人群模型**：

```python
population=10000                                    # 固定人群
population=[10000, 10500, 11000, 11500, 12000]      # 逐年指定
population={"base": 10000, "growth_rate": 0.05}      # 复合增长
population={"base": 10000, "annual_increase": 500}   # 线性增长
```

**市场份额摄取曲线**：

```python
ph.BudgetImpactAnalysis.linear_uptake(0.0, 0.4, 5)           # 线性
ph.BudgetImpactAnalysis.sigmoid_uptake(0.0, 0.4, 5, steepness=1.5)  # S 型
```

**从模型结果创建 / 情景分析 / 敏感性分析**：

```python
# 从模型结果
bia = ph.BudgetImpactAnalysis.from_result(result, population=10000, ...)

# 情景分析
bia.scenario_analysis({
    "Base Case": {},
    "High Population": {"population": 15000},
    "Fast Uptake": {"market_share_new": {"SoC": 0.3, "New": 0.7}},
})

# 单因素敏感性
bia.one_way_sensitivity("population", values=[8000, 9000, 10000, 11000, 12000])

# 龙卷风图
bia.tornado({"population": (8000, 12000), "Drug B": (10000, 15000)})
```

#### 模型校准

模型校准用观测数据反推模型中无法直接观测的参数。基于 Vanni et al. (2011) 和 Alarid-Escudero et al. (2018)。

```python
# 定义校准目标
targets = [
    ph.CalibrationTarget(
        name="10yr_healthy",
        observed=0.42, se=0.05,
        extract_fn=lambda sim: sim["SOC"]["trace"][10, 0],
    ),
]

# 定义待校准参数
calib_params = [
    ph.CalibrationParam("p_HS", lower=0.01, upper=0.30),
    ph.CalibrationParam("p_SD", lower=0.01, upper=0.20),
]

# 运行校准
result = ph.calibrate(
    model, targets, calib_params,
    gof="wsse",
    method="nelder_mead",
    n_restarts=10,
    seed=42,
)

print(result.summary())
print(result.target_comparison())
result.apply_to_model(model)
```

| 搜索方法 | 参数 | 特点 |
|------|------|------|
| `nelder_mead` | `n_restarts=10` | 多起点无导数优化，精确但较慢 |
| `random_search` | `n_samples=1000` | LHS 采样逐一评估，简单直观 |

| GoF 度量 | 公式 | 适用场景 |
|------|------|----------|
| `sse` | Σ(obs - pred)² | 默认，简单快速 |
| `wsse` | Σ(obs - pred)²/SE² | 多目标量纲不同时 |
| `loglik_normal` | -Σ log N(obs \| pred, SE²) | 统计原则化 |

---

### 导出

#### Excel 导出

```python
# 结果数据导出 (多 Sheet)
ph.export_to_excel(result, "base_case.xlsx")
ph.export_to_excel(owsa, "owsa.xlsx")
ph.export_to_excel(psa, "psa.xlsx")

# 多策略比较
ph.export_comparison_excel({"Strategy A": result_a, "Strategy B": result_b}, "comparison.xlsx")

# IPD 拟合结果
fitter.to_excel("fitting_results.xlsx")
```

#### Excel 公式验证模型

导出一个**用 Excel 公式独立计算**的完整模型文件，用于交叉验证 Python 结果：

```python
result = model.run_base_case()
ph.export_excel_model(result, "verification.xlsx")

# 或直接从模型导出
ph.export_excel_model(model, "verification.xlsx")
```

| 区域 | 内容 |
|------|------|
| **输入区** (黄色底色) | 转移概率矩阵、状态费用向量、效用权重、贴现率 |
| **计算区** (公式) | `SUMPRODUCT` 计算 Trace, 费用, QALY; `SUM` 计算贴现总值 |
| **Summary sheet** | Excel 公式结果 vs Python 结果 vs 差异 (应为 ~0) |

**支持的模型类型**：

| 模型 | Trace | 费用/QALY/贴现 | ICER |
|------|-------|----------------|------|
| Markov (时齐) | Excel 公式 | Excel 公式 | Excel 公式 |
| Markov (时变) | Python 值 | Excel 公式 | Excel 公式 |
| PSM | Python 生存值 → 状态概率公式 | Excel 公式 | Excel 公式 |

#### Excel Sheet 内容

| 分析类型    | Sheet 内容                                                            |
| ----------- | --------------------------------------------------------------------- |
| Base Case   | Summary, State Trace, Cost/QALY by Cycle, ICER                        |
| OWSA        | Tornado Data, Per-Parameter Results                                   |
| PSA         | Summary Stats, All Simulations, CEAC Data                             |
| PSM Base    | Summary, State Probabilities, Survival Data                           |
| IPD Fitting | Model Comparison, KM Data, Per-Distribution Details, Selection Report |
| 验证模型     | Summary (含差异), 每策略计算 Sheet (公式+输入)                          |

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

```text
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
│       ├── excel_model.py   #  Excel 公式验证模型导出
│       └── report.py        #  Markdown 一键报告
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
- [X] Excel 多 Sheet 导出 + Excel 公式验证模型
- [X] IPD 生存曲线拟合 + AIC/BIC 模型比较
- [X] KM + 拟合曲线可视化 + 诊断图
- [X] 微观模拟 (Individual-level simulation)
- [X] 多队列比较 + NMB 分析 + CEAF + EVPI
- [X] 网络 Meta 分析 (NMA) 整合
- [X] 离散事件模拟 (DES) — 连续时间、竞争风险、HR/AFT 集成
- [X] 预算影响分析 (BIA) — 人群模型、市场份额演变、摄取曲线、情景/敏感性分析
- [X] 数字化 KM 曲线重建 (Guyot method)
- [X] 模型校准 (Nelder-Mead 多起点优化, LHS 随机搜索, SSE/WSSE/似然 GoF)
- [X] Markdown 一键报告 (`generate_report`)
- [X] 正式测试套件 (pytest, 243 个测试覆盖全部模块)
- [ ] 结构化输出 (`to_dict` / `to_json`)，面向 LLM 的机器可读结果
- [ ] 自动解读 (`interpret(wtp)`)——标准化结论文本生成
- [ ] 自然语言建模接口——JSON Schema 模型定义，自动构建与执行

---

## 许可证

MIT License
