"""
PSM (Partitioned Survival Model) Demo — Oncology Cost-Effectiveness Analysis

This example demonstrates a typical oncology PSM comparing:
- SOC (Standard of Care): chemotherapy alone
- TRT (New Treatment): immunotherapy + chemotherapy

The model uses:
- Weibull PFS and OS curves for SOC (from clinical trial fitting)
- Hazard ratios from a phase III trial for TRT
- 3 states: PFS, Progressed, Dead
- Monthly cycle (cycle_length = 1/12 year) over 20 years
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyheor as ph

print("=" * 70)
print("  PSM Demo: Oncology Cost-Effectiveness Analysis")
print("=" * 70)

# =========================================================================
# 1. Define the PSM Model
# =========================================================================

model = ph.PSMModel(
    states=["PFS", "Progressed", "Dead"],
    survival_endpoints=["PFS", "OS"],
    strategies={"SOC": "Chemotherapy", "TRT": "Immuno + Chemo"},
    n_cycles=240,             # 240 months = 20 years
    cycle_length=1/12,        # Monthly cycles
    dr_cost=0.03,
    dr_qaly=0.03,
    half_cycle_correction=True,
)

print("\n📋 Model structure:")
print(model.info())

# =========================================================================
# 2. Define Parameters
# =========================================================================

model.add_params({
    # Baseline survival parameters (Weibull for SOC)
    "pfs_shape": ph.Param(1.0, label="PFS Weibull shape"),
    "pfs_scale": ph.Param(0.8, label="PFS Weibull scale (years)",
                          dist=ph.LogNormal(mean=0.8, sd=0.1)),
    "os_shape":  ph.Param(1.2, label="OS Weibull shape"),
    "os_scale":  ph.Param(2.5, label="OS Weibull scale (years)",
                          dist=ph.LogNormal(mean=2.5, sd=0.3)),

    # Treatment effect (hazard ratios)
    "hr_pfs": ph.Param(0.65, dist=ph.LogNormal(mean=0.65, sd=0.08),
                        label="HR (PFS)", low=0.50, high=0.80),
    "hr_os":  ph.Param(0.75, dist=ph.LogNormal(mean=0.75, sd=0.10),
                        label="HR (OS)", low=0.60, high=0.90),

    # Costs (monthly)
    "c_chemo":     ph.Param(3000, dist=ph.Gamma(mean=3000, sd=300),
                             label="Chemo cost/month"),
    "c_immuno":    ph.Param(8000, dist=ph.Gamma(mean=8000, sd=800),
                             label="Immuno cost/month"),
    "c_prog_care": ph.Param(5000, dist=ph.Gamma(mean=5000, sd=500),
                             label="Progressed care/month"),
    "c_bsc":       ph.Param(1500, dist=ph.Gamma(mean=1500, sd=150),
                             label="BSC cost/month"),
    "c_ae":        ph.Param(2000, label="AE cost (one-time)", low=1000, high=3000),

    # Utilities
    "u_pfs":  ph.Param(0.78, dist=ph.Beta(mean=0.78, sd=0.05), label="Utility PFS"),
    "u_prog": ph.Param(0.55, dist=ph.Beta(mean=0.55, sd=0.08), label="Utility Progressed"),
})

# =========================================================================
# 3. Set Survival Curves
# =========================================================================

# SOC: baseline Weibull curves
model.set_survival_all("SOC", {
    "PFS": lambda p: ph.Weibull(shape=p["pfs_shape"], scale=p["pfs_scale"]),
    "OS":  lambda p: ph.Weibull(shape=p["os_shape"], scale=p["os_scale"]),
})

# TRT: PH (proportional hazards) treatment effect on baseline
model.set_survival_all("TRT", {
    "PFS": lambda p: ph.ProportionalHazards(
        ph.Weibull(shape=p["pfs_shape"], scale=p["pfs_scale"]),
        hr=p["hr_pfs"]
    ),
    "OS": lambda p: ph.ProportionalHazards(
        ph.Weibull(shape=p["os_shape"], scale=p["os_scale"]),
        hr=p["hr_os"]
    ),
})

# =========================================================================
# 4. Set Costs
# =========================================================================

# Drug costs (during PFS only)
model.set_state_cost("drug", {
    "SOC": {"PFS": "c_chemo", "Progressed": 0, "Dead": 0},
    "TRT": {"PFS": lambda p, t: p["c_chemo"] + p["c_immuno"],
             "Progressed": 0, "Dead": 0},
})

# Progressed disease care
model.set_state_cost("prog_care", {
    "PFS": 0, "Progressed": "c_prog_care", "Dead": 0,
})

# Best supportive care (all alive states)
model.set_state_cost("bsc", {
    "PFS": "c_bsc", "Progressed": "c_bsc", "Dead": 0,
})

# Adverse event cost (first cycle only for TRT)
model.set_state_cost("ae", {
    "TRT": {"PFS": "c_ae"},
}, first_cycle_only=True)

# =========================================================================
# 5. Set Utility
# =========================================================================

model.set_utility({
    "PFS": "u_pfs",
    "Progressed": "u_prog",
    "Dead": 0.0,
})

# =========================================================================
# 6. Run Base Case
# =========================================================================

print("\n" + "=" * 70)
print("  BASE CASE ANALYSIS")
print("=" * 70)

base = model.run_base_case()

print("\n📊 Summary:")
print(base.summary().to_string(index=False))

print("\n💰 ICER:")
print(base.icer().to_string(index=False))

print("\n📈 NMB (WTP = $100,000):")
print(base.nmb(wtp=100000).to_string(index=False))

# State trace
print("\n📋 State occupancy (first 12 months):")
trace = base.state_trace
soc_trace = trace[trace['Strategy'] == 'Chemotherapy'].head(13)
print(soc_trace[['Cycle', 'Time', 'PFS', 'Progressed', 'Dead']].to_string(index=False))

# =========================================================================
# 7. Survival Data
# =========================================================================

print("\n📈 Survival data (selected timepoints):")
surv = base.survival_data
for t in [0, 0.5, 1, 2, 3, 5]:
    row = surv[(surv['Time'].between(t - 0.01, t + 0.01))]
    if len(row) > 0:
        for _, r in row.iterrows():
            print(f"  t={r['Time']:.1f}y, {r['Strategy']}: S({r['Endpoint']})={r['Survival']:.3f}")

# =========================================================================
# 8. OWSA
# =========================================================================

print("\n" + "=" * 70)
print("  ONE-WAY SENSITIVITY ANALYSIS")
print("=" * 70)

owsa = model.run_owsa(
    params=["hr_pfs", "hr_os", "c_immuno", "c_chemo", "c_prog_care", "u_pfs", "u_prog", "c_ae"],
    wtp=100000,
)

print("\n🌪️ OWSA Summary (Top parameters):")
owsa_df = owsa.summary()
print(owsa_df[['Parameter', 'Base Value', 'Low Value', 'High Value',
               'INMB (Low)', 'INMB (High)', 'Range']].head(8).to_string(index=False))

# =========================================================================
# 9. PSA
# =========================================================================

print("\n" + "=" * 70)
print("  PROBABILISTIC SENSITIVITY ANALYSIS")
print("=" * 70)

psa = model.run_psa(n_sim=100, seed=42)

print("\n📊 PSA Summary:")
print(psa.summary().to_string(index=False))

print("\n💰 PSA ICER:")
print(psa.icer().to_string(index=False))

# =========================================================================
# 10. Excel Export
# =========================================================================

print("\n" + "=" * 70)
print("  EXCEL VERIFICATION EXPORT")
print("=" * 70)

# Export base case
ph.export_to_excel(base, "/Users/xuzhiyuan/code/research/pyheor/examples/psm_base_case.xlsx")

# Export OWSA
ph.export_to_excel(owsa, "/Users/xuzhiyuan/code/research/pyheor/examples/psm_owsa.xlsx")

# Export PSA
ph.export_to_excel(psa, "/Users/xuzhiyuan/code/research/pyheor/examples/psm_psa.xlsx",
                   include_psa=True)

# =========================================================================
# 11. Also export the Markov model for comparison
# =========================================================================

print("\n" + "=" * 70)
print("  MARKOV MODEL EXCEL EXPORT (for comparison)")
print("=" * 70)

# Simple 3-state Markov model
markov = ph.MarkovModel(
    states=["Healthy", "Sick", "Dead"],
    strategies={"SOC": "Standard Care", "New": "New Drug"},
    n_cycles=20,
    cycle_length=1.0,
    dr_cost=0.03,
    dr_qaly=0.03,
)

markov.add_params({
    "p_HS": ph.Param(0.15, dist=ph.Beta(mean=0.15, sd=0.03), label="P(H→S)"),
    "p_HD": ph.Param(0.02, label="P(H→D)"),
    "p_SD": ph.Param(0.10, dist=ph.Beta(mean=0.10, sd=0.02), label="P(S→D)"),
    "hr":   ph.Param(0.70, dist=ph.LogNormal(mean=0.70, sd=0.10), label="HR (New)"),
    "c_H":  ph.Param(500, dist=ph.Gamma(mean=500, sd=50), label="Cost Healthy"),
    "c_S":  ph.Param(3000, dist=ph.Gamma(mean=3000, sd=300), label="Cost Sick"),
    "c_drug": ph.Param(5000, label="Drug Cost (New)"),
    "u_H":  ph.Param(0.90, dist=ph.Beta(mean=0.90, sd=0.03), label="Util Healthy"),
    "u_S":  ph.Param(0.60, dist=ph.Beta(mean=0.60, sd=0.05), label="Util Sick"),
})

markov.set_transitions("SOC", lambda p, t: [
    [ph.C,  p["p_HS"],              p["p_HD"]],
    [0,     ph.C,                   p["p_SD"]],
    [0,     0,                      1        ],
])

markov.set_transitions("New", lambda p, t: [
    [ph.C,  p["p_HS"] * p["hr"],   p["p_HD"]],
    [0,     ph.C,                   p["p_SD"] * p["hr"]],
    [0,     0,                      1        ],
])

markov.set_state_cost("medical", {"Healthy": "c_H", "Sick": "c_S", "Dead": 0})
markov.set_state_cost("drug", {"New": {"Healthy": "c_drug", "Sick": "c_drug"}})
markov.set_utility({"Healthy": "u_H", "Sick": "u_S", "Dead": 0})

markov_base = markov.run_base_case()
ph.export_to_excel(markov_base, "/Users/xuzhiyuan/code/research/pyheor/examples/markov_base_case.xlsx")

print("\n📊 Markov Summary:")
print(markov_base.summary().to_string(index=False))
print("\n💰 Markov ICER:")
print(markov_base.icer().to_string(index=False))

# =========================================================================
# 12. Plots
# =========================================================================

print("\n" + "=" * 70)
print("  GENERATING PLOTS")
print("=" * 70)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyheor.plotting import (
    plot_survival_curves, plot_state_area, plot_psm_trace,
    plot_psm_comparison,
)

plot_dir = "/Users/xuzhiyuan/code/research/pyheor/examples/plots"
os.makedirs(plot_dir, exist_ok=True)

# Survival curves
fig = plot_survival_curves(base)
fig.savefig(f"{plot_dir}/psm_survival_curves.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  ✅ Survival curves plot saved")

# State area plots
for strategy in model.strategy_names:
    fig = plot_state_area(base, strategy=strategy)
    label = model.strategy_labels[strategy].replace(' ', '_').replace('+', '')
    fig.savefig(f"{plot_dir}/psm_area_{label}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
print("  ✅ State area plots saved")

# State trace comparison
fig = plot_psm_trace(base, figsize=(15, 5))
fig.savefig(f"{plot_dir}/psm_trace.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  ✅ State trace plot saved")

# OS comparison
fig = plot_psm_comparison(base, "OS")
fig.savefig(f"{plot_dir}/psm_os_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  ✅ OS comparison plot saved")

# PFS comparison
fig = plot_psm_comparison(base, "PFS")
fig.savefig(f"{plot_dir}/psm_pfs_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  ✅ PFS comparison plot saved")

# Tornado
fig = owsa.plot_tornado()
fig.savefig(f"{plot_dir}/psm_tornado.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  ✅ Tornado plot saved")

# PSA CE scatter
fig = psa.plot_scatter()
fig.savefig(f"{plot_dir}/psm_ce_scatter.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  ✅ CE scatter plot saved")

# PSA CEAC
fig = psa.plot_ceac(wtp_range=(0, 200000))
fig.savefig(f"{plot_dir}/psm_ceac.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  ✅ CEAC plot saved")

print("\n" + "=" * 70)
print("  ALL DONE! 🎉")
print("=" * 70)
