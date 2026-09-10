"""
PSM (Partitioned Survival Model) Demo — Oncology Cost-Effectiveness Analysis

This example demonstrates a typical oncology PSM comparing:
- SOC (Standard of Care): chemotherapy alone
- TRT (New Treatment): immunotherapy + chemotherapy

The model uses:
- An exponential OS curve with PFS defined through an excess hazard
- Treatment effects that preserve PFS <= OS for every PSA draw
- 3 states: PFS, Progressed, Dead
- Monthly cycle (cycle_length = 1/12 year) over 20 years
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyheor as ph

EXAMPLE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = EXAMPLE_DIR / "figures"
WORKBOOK_DIR = EXAMPLE_DIR / "workbooks"
FIGURE_DIR.mkdir(exist_ok=True)
WORKBOOK_DIR.mkdir(exist_ok=True)


def save_figure(fig, filename):
    """Save beside this script, independent of the working directory."""
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path.relative_to(EXAMPLE_DIR)}")

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

# =========================================================================
# 2. Define Parameters
# =========================================================================

model.add_params({
    # Survival parameters. A positive excess hazard guarantees PFS <= OS.
    "os_rate": ph.Param(
        0.40,
        dist=ph.Gamma(mean=0.40, sd=0.05),
        label="OS event rate/year",
        low=0.32,
        high=0.48,
    ),
    "pfs_excess_rate": ph.Param(
        0.85,
        dist=ph.Gamma(mean=0.85, sd=0.12),
        label="PFS excess event rate/year",
        low=0.65,
        high=1.05,
    ),

    # Treatment effect. The gap ratio is bounded between 0 and 1.
    "hr_os":  ph.Param(0.75, dist=ph.LogNormal(mean=0.75, sd=0.10),
                        label="HR (OS)", low=0.60, high=0.90),
    "pfs_gap_ratio": ph.Param(
        0.55,
        dist=ph.Beta(mean=0.55, sd=0.08),
        label="Treatment PFS/OS hazard-gap ratio",
        low=0.40,
        high=0.70,
    ),

    # Annual cost rates. The model multiplies these by the 1/12-year cycle.
    "c_chemo":     ph.Param(36000, dist=ph.Gamma(mean=36000, sd=3600),
                             label="Chemo cost/year"),
    "c_immuno":    ph.Param(96000, dist=ph.Gamma(mean=96000, sd=9600),
                             label="Immuno cost/year"),
    "c_prog_care": ph.Param(60000, dist=ph.Gamma(mean=60000, sd=6000),
                             label="Progressed care/year"),
    "c_bsc":       ph.Param(18000, dist=ph.Gamma(mean=18000, sd=1800),
                             label="BSC cost/year"),
    "c_ae":        ph.Param(2000, label="AE cost (one-time)", low=1000, high=3000),

    # Utilities
    "u_pfs":  ph.Param(0.78, dist=ph.Beta(mean=0.78, sd=0.05), label="Utility PFS"),
    "u_prog": ph.Param(0.55, dist=ph.Beta(mean=0.55, sd=0.08), label="Utility Progressed"),
})

# =========================================================================
# 3. Set Survival Curves
# =========================================================================

# SOC: PFS has a strictly greater event rate than OS.
model.set_survival_all("SOC", {
    "PFS": lambda p: ph.Exponential(
        rate=p["os_rate"] + p["pfs_excess_rate"],
    ),
    "OS": lambda p: ph.Exponential(rate=p["os_rate"]),
})

# TRT: apply the OS effect first, then a still-positive PFS excess hazard.
model.set_survival_all("TRT", {
    "PFS": lambda p: ph.Exponential(
        rate=(p["os_rate"] * p["hr_os"]
              + p["pfs_excess_rate"] * p["pfs_gap_ratio"]),
    ),
    "OS": lambda p: ph.Exponential(rate=p["os_rate"] * p["hr_os"]),
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

print("\n📋 Complete model specification:")
print(model.info())

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
    params=["pfs_gap_ratio", "hr_os", "c_immuno", "c_chemo",
            "c_prog_care", "u_pfs", "u_prog", "c_ae"],
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
print("  EXCEL EXPORTS")
print("=" * 70)

# This workbook contains formulas, not just a snapshot of Python results.
ph.export_excel_model(base, WORKBOOK_DIR / "base_case.xlsx")


# =========================================================================
# 11. Plots
# =========================================================================

print("\n" + "=" * 70)
print("  GENERATING PLOTS")
print("=" * 70)

from pyheor.plotting import plot_state_area, plot_survival_curves

# Survival curves
fig = plot_survival_curves(base)
save_figure(fig, "survival_curves.png")

# State area plots
for strategy in model.strategy_names:
    fig = plot_state_area(base, strategy=strategy)
    filename = {
        "SOC": "state_area_chemotherapy.png",
        "TRT": "state_area_immuno_chemo.png",
    }[strategy]
    save_figure(fig, filename)

# Tornado
fig = owsa.plot_tornado()
save_figure(fig, "tornado.png")

# PSA CEAC
fig = psa.plot_ceac(wtp_range=(0, 200000))
save_figure(fig, "ceac.png")

print("\n" + "=" * 70)
print("  ALL DONE! 🎉")
print("=" * 70)
