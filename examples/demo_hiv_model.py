# %% [markdown]
# # PyHEOR 使用示例：HIV 模型 (Chancellor et al.)
# 
# 复现 hesim 教程中的 HIV 简单 Markov 模型。该模型比较两种治疗策略：
# - **单药治疗** (Monotherapy): 齐多夫定单药
# - **联合治疗** (Combination therapy): 齐多夫定 + 拉米夫定（仅前2年使用拉米夫定）
# 
# 模型包含 4 个健康状态：A (CD4 200-500), B (CD4 <200), C (AIDS), D (Death)
# 
# 特色功能展示：
# 1. 基础分析（确定性）
# 2. 单因素敏感性分析（OWSA + 龙卷风图）
# 3. 概率敏感性分析（PSA + CEAC + CE散点图）
# 4. 状态转移关系图 & TreeAge风格模型图

# %%
import sys
sys.path.insert(0, '/Users/xuzhiyuan/code/research')

import numpy as np
import pyheor as ph

print(f"PyHEOR v{ph.__version__}")

# %% [markdown]
# ## 1. 定义模型结构
# 
# 4 个健康状态，2 个治疗策略，20 个周期（年）

# %%
model = ph.MarkovModel(
    states=["State A", "State B", "State C", "Death"],
    strategies={
        "Mono": "Monotherapy (ZDV)",
        "Combo": "Combination (ZDV+LAM)",
    },
    n_cycles=20,
    cycle_length=1.0,  # 1 year per cycle
    discount_rate={"costs": 0.06, "qalys": 0.0},  # 6% for costs, 0% for QALYs
    half_cycle_correction=False,  # 与原始文献保持一致（右端点求和）
)

print(model)

# %% [markdown]
# ## 2. 定义参数
# 
# 参数包括：
# - **转移概率** (from Dirichlet posterior on monotherapy transition counts)
# - **相对风险** (combination vs monotherapy)
# - **药物成本** (ZDV, lamivudine)
# - **医疗成本** (direct + community medical)

# %%
# Transition count matrix (monotherapy) — from Chelsea & Westminster Hospital
# Row i → Col j counts
# A→A:1251, A→B:350, A→C:116, A→D:17
# B→B:731,  B→C:512, B→D:15
# C→C:1312, C→D:437
trans_counts = np.array([
    [1251, 350, 116, 17],
    [0, 731, 512, 15],
    [0, 0, 1312, 437],
    [0, 0, 0, 469],
])

# Convert counts to probabilities (point estimates = row-normalized)
row_sums = trans_counts.sum(axis=1, keepdims=True)
trans_probs = trans_counts / row_sums

print("Transition probabilities (Monotherapy):")
print(np.round(trans_probs, 4))

# %%
# Relative risk of progression for combination therapy
rr_mean = 0.509
rr_lower = 0.365
rr_upper = 0.710

# Log-scale parameters for the relative risk
lrr_mean = np.log(rr_mean)
lrr_se = (np.log(rr_upper) - np.log(rr_lower)) / (2 * 1.96)

print(f"RR = {rr_mean} (95% CI: {rr_lower} - {rr_upper})")
print(f"Log(RR) mean = {lrr_mean:.4f}, SE = {lrr_se:.4f}")

# %%
# Define all parameters
model.add_params({
    # Transition probabilities (monotherapy)
    "p_AB": ph.Param(trans_probs[0, 1], label="P(A→B)"),
    "p_AC": ph.Param(trans_probs[0, 2], label="P(A→C)"),
    "p_AD": ph.Param(trans_probs[0, 3], label="P(A→D)"),
    "p_BC": ph.Param(trans_probs[1, 2], label="P(B→C)"),
    "p_BD": ph.Param(trans_probs[1, 3], label="P(B→D)"),
    "p_CD": ph.Param(trans_probs[2, 3], label="P(C→D)"),
    
    # Relative risk (combination therapy)
    "rr": ph.Param(
        rr_mean,
        dist=ph.LogNormal(meanlog=lrr_mean, sdlog=lrr_se),
        label="Relative Risk",
        low=rr_lower,
        high=rr_upper,
    ),
    
    # Drug costs (per year)
    "c_zido": ph.Param(2278, label="Cost: Zidovudine"),
    "c_lam": ph.Param(2086.50, label="Cost: Lamivudine"),
    
    # Direct medical costs
    "c_dmed_A": ph.Param(1701, dist=ph.Gamma(mean=1701, sd=1701), label="Direct Medical (A)"),
    "c_dmed_B": ph.Param(1774, dist=ph.Gamma(mean=1774, sd=1774), label="Direct Medical (B)"),
    "c_dmed_C": ph.Param(6948, dist=ph.Gamma(mean=6948, sd=6948), label="Direct Medical (C)"),
    
    # Community medical costs    
    "c_cmed_A": ph.Param(1055, dist=ph.Gamma(mean=1055, sd=1055), label="Community Medical (A)"),
    "c_cmed_B": ph.Param(1278, dist=ph.Gamma(mean=1278, sd=1278), label="Community Medical (B)"),
    "c_cmed_C": ph.Param(2059, dist=ph.Gamma(mean=2059, sd=2059), label="Community Medical (C)"),
    
    # Utility (life-years only in this model)
    "u": ph.Param(1.0, label="Utility"),
})

print(model.info())

# %% [markdown]
# ## 3. 定义转移概率矩阵
# 
# 关键特性：
# - 联合治疗的相对风险 `rr` 降低向更严重状态转移的概率
# - `rr` 仅在前2年有效（拉米夫定使用期限）
# - 使用 `ph.C` 自动计算补数（complement）

# %%
# Monotherapy: constant transition matrix
model.set_transitions("Mono", lambda p, t: [
    [ph.C,  p["p_AB"],         p["p_AC"],         p["p_AD"]],
    [0,     ph.C,              p["p_BC"],          p["p_BD"]],
    [0,     0,                 ph.C,               p["p_CD"]],
    [0,     0,                 0,                  1],
])

# Combination therapy: RR applied only for first 2 years
model.set_transitions("Combo", lambda p, t: [
    [ph.C,  p["p_AB"] * (p["rr"] if t <= 2 else 1.0),
            p["p_AC"] * (p["rr"] if t <= 2 else 1.0),
            p["p_AD"] * (p["rr"] if t <= 2 else 1.0)],
    [0,     ph.C,
            p["p_BC"] * (p["rr"] if t <= 2 else 1.0),
            p["p_BD"] * (p["rr"] if t <= 2 else 1.0)],
    [0,     0,
            ph.C,
            p["p_CD"] * (p["rr"] if t <= 2 else 1.0)],
    [0,     0,                 0,                  1],
])

# %% [markdown]
# ## 4. 定义成本
# 
# 三类成本：
# - **药物成本**: 单药用ZDV，联合用ZDV+LAM（LAM仅前2年）
# - **直接医疗成本**: 按状态分
# - **社区医疗成本**: 按状态分

# %%
# Drug costs (time-dependent: lamivudine only for first 2 years)
model.set_state_cost("drug", lambda p, t: {
    "Mono": {
        "State A": p["c_zido"],
        "State B": p["c_zido"],
        "State C": p["c_zido"],
    },
    "Combo": {
        "State A": p["c_zido"] + (p["c_lam"] if t <= 2 else 0),
        "State B": p["c_zido"] + (p["c_lam"] if t <= 2 else 0),
        "State C": p["c_zido"] + (p["c_lam"] if t <= 2 else 0),
    },
})

# Direct medical costs
model.set_state_cost("direct_medical", {
    "State A": "c_dmed_A",
    "State B": "c_dmed_B",
    "State C": "c_dmed_C",
    "Death": 0,
})

# Community medical costs
model.set_state_cost("community_medical", {
    "State A": "c_cmed_A",
    "State B": "c_cmed_B",
    "State C": "c_cmed_C",
    "Death": 0,
})

# %% [markdown]
# ## 5. 定义效用
# 
# 本模型使用生命年（LYs）作为结果，所有存活状态效用为1

# %%
model.set_utility({
    "State A": "u",
    "State B": "u",
    "State C": "u",
    "Death": 0,
})

# %% [markdown]
# ## 6. 运行基础分析

# %%
base = model.run_base_case()

print("=" * 60)
print("BASE CASE RESULTS")
print("=" * 60)
print("\n--- Summary ---")
print(base.summary().to_string(index=False))
print("\n--- ICER ---")
print(base.icer().to_string(index=False))
print("\n--- NMB (WTP=$50,000/QALY) ---")
print(base.nmb(wtp=50000).to_string(index=False))

# %% [markdown]
# ## 7. 可视化：Markov Trace

# %%
fig = base.plot_trace(style="area")

# %%
fig = base.plot_trace(style="line")

# %% [markdown]
# ## 8. 可视化：状态转移关系图

# %%
# Monotherapy transition diagram
fig = base.plot_transition_diagram(strategy="Mono")

# %%
# Combination therapy transition diagram (at cycle 1)
fig = base.plot_transition_diagram(strategy="Combo")

# %% [markdown]
# ## 9. 可视化：TreeAge 风格模型结构图

# %%
fig = base.plot_model_diagram()

# %% [markdown]
# ## 10. 单因素敏感性分析 (OWSA)

# %%
owsa = model.run_owsa(
    params=["rr", "c_dmed_A", "c_dmed_B", "c_dmed_C", 
            "c_cmed_A", "c_cmed_B", "c_cmed_C"],
    wtp=50000,
)

print("OWSA Summary:")
print(owsa.summary().to_string(index=False))

# %%
fig = owsa.plot_tornado()

# %% [markdown]
# ## 11. 概率敏感性分析 (PSA)

# %%
psa = model.run_psa(n_sim=1000, seed=42)

print("\n--- PSA Summary ---")
print(psa.summary().to_string(index=False))
print("\n--- PSA ICER ---")
print(psa.icer().to_string(index=False))

# %%
# Cost-effectiveness acceptability curve
fig = psa.plot_ceac(wtp_range=(0, 50000))

# %%
# CE scatter plot
fig = psa.plot_scatter(wtp=50000)

# %%
# PSA convergence
fig = psa.plot_convergence(wtp=50000)

# %% [markdown]
# ## 12. 高级特性：灵活成本定义示例
# 
# 演示 PyHEOR 超越 hesim 的灵活成本定义能力

# %%
# 创建一个新模型来演示灵活成本功能
model2 = ph.MarkovModel(
    states=["PFS", "Progressed", "Dead"],
    strategies={"SOC": "Standard of Care", "New": "New Treatment"},
    n_cycles=40,
    cycle_length=0.5,  # 6-month cycles
    discount_rate=0.03,
)

model2.add_params({
    "p_prog": ph.Param(0.08, dist=ph.Beta(mean=0.08, sd=0.02), label="P(PFS→Prog)"),
    "p_death_pfs": ph.Param(0.01, label="P(PFS→Death)"),
    "p_death_prog": ph.Param(0.15, dist=ph.Beta(mean=0.15, sd=0.03), label="P(Prog→Death)"),
    "hr": ph.Param(0.65, dist=ph.LogNormal(mean=0.65, sd=0.1), label="Hazard Ratio"),
    "c_soc": ph.Param(3000, label="SOC Drug Cost (annual)"),
    "c_new": ph.Param(15000, dist=ph.Gamma(mean=15000, sd=2000), label="New Drug Cost (annual)"),
    "c_prog_care": ph.Param(8000, dist=ph.Gamma(mean=8000, sd=1000), label="Progression Care (annual)"),
    "c_ae": ph.Param(5000, dist=ph.Gamma(mean=5000, sd=800), label="AE Management Cost"),
    "u_pfs": ph.Param(0.82, dist=ph.Beta(mean=0.82, sd=0.03), label="Utility: PFS"),
    "u_prog": ph.Param(0.52, dist=ph.Beta(mean=0.52, sd=0.05), label="Utility: Progressed"),
})

# Transitions
model2.set_transitions("SOC", lambda p, t: [
    [ph.C,  p["p_prog"],          p["p_death_pfs"]],
    [0,     ph.C,                  p["p_death_prog"]],
    [0,     0,                     1],
])

model2.set_transitions("New", lambda p, t: [
    [ph.C,  p["p_prog"] * p["hr"],  p["p_death_pfs"]],
    [0,     ph.C,                     p["p_death_prog"]],
    [0,     0,                        1],
])

# 药物成本（按策略不同）
model2.set_state_cost("drug", {
    "SOC": {"PFS": "c_soc", "Progressed": "c_soc"},
    "New": {"PFS": "c_new", "Progressed": "c_soc"},  # 进展后转回SOC
})

# 进展期护理成本
model2.set_state_cost("progression_care", {
    "PFS": 0, "Progressed": "c_prog_care", "Dead": 0,
})

# ★ 关键特性：不良反应成本仅在第一周期计算
model2.set_state_cost("adverse_events", {
    "New": {"PFS": "c_ae", "Progressed": "c_ae"},
}, first_cycle_only=True)

# ★ 关键特性：时间依赖的监测成本函数
model2.set_state_cost("monitoring", lambda p, t: {
    "PFS": 2000 if t <= 4 else 800,      # 前2年密集监测，之后减少
    "Progressed": 3000 if t <= 4 else 1500,
    "Dead": 0,
})

model2.set_utility({
    "PFS": "u_pfs",
    "Progressed": "u_prog",
    "Dead": 0,
})

# Run full analysis
print("=" * 60)
print("ADVANCED MODEL: Oncology CEA with Flexible Costs")
print("=" * 60)

base2 = model2.run_base_case()
print("\n--- Base Case ---")
print(base2.summary().to_string(index=False))
print("\n--- ICER ---")
print(base2.icer().to_string(index=False))

# %%
fig = base2.plot_trace()

# %%
fig = base2.plot_transition_diagram(strategy="New")

# %%
fig = base2.plot_model_diagram()

# %%
owsa2 = model2.run_owsa(
    params=["hr", "c_new", "c_ae", "c_prog_care", "u_pfs", "u_prog", "p_prog", "p_death_prog"],
    wtp=100000,
)
fig = owsa2.plot_tornado()

# %%
psa2 = model2.run_psa(n_sim=1000, seed=123)
print(psa2.summary().to_string(index=False))

# %%
fig = psa2.plot_ceac(wtp_range=(0, 200000))

# %%
fig = psa2.plot_scatter(wtp=100000)

# %% [markdown]
# ## 总结
# 
# PyHEOR 相比 R hesim 的改进：
# 
# | 特性 | hesim (R) | PyHEOR (Python) |
# |------|-----------|-----------------|
# | 基础分析 | ✗ (仅PSA) | ✓ 确定性分析 |
# | 单因素敏感性分析 | ✗ | ✓ OWSA + 龙卷风图 |
# | 概率敏感性分析 | ✓ | ✓ PSA + CEAC + CE平面 |
# | 成本定义 | 固定/按状态 | 灵活：首周期、时间函数、一次性 |
# | 转移关系图 | ✗ | ✓ 自动生成美观的状态转移图 |
# | TreeAge风格图 | ✗ | ✓ 模型结构可视化 |
# | API风格 | R6 Classes | Pythonic 链式调用 |
