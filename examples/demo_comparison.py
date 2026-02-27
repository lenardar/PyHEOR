"""
Demo: Multi-Strategy Comparison & NMB Analysis
===============================================

This example demonstrates CEAnalysis with:
1. Efficiency frontier with dominance detection (4 strategies)
2. NMB analysis across WTP thresholds
3. CEAF and EVPI from PSA data
4. All 4 new visualization types

Uses a 4-strategy oncology cost-effectiveness example:
  - SOC (Standard of Care)
  - Drug A (moderate cost, moderate benefit)
  - Drug B (high cost, high benefit)
  - Drug C (dominated — high cost, low benefit)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pyheor as ph

np.random.seed(42)
os.makedirs("plots", exist_ok=True)

# =============================================================================
# 1. Build a 4-Strategy Markov Model
# =============================================================================
print("=" * 70)
print("Part 1: Build 4-Strategy Model")
print("=" * 70)

model = ph.MarkovModel(
    states=["Stable", "Progressed", "Dead"],
    strategies=["SOC", "Drug A", "Drug B", "Drug C"],
    n_cycles=40,
    cycle_length=1.0,
    discount_rate={"costs": 0.03, "qalys": 0.03},
    half_cycle_correction=True,
)

# Parameters
model.add_param("p_SP",   base=0.15, low=0.10, high=0.20,
                dist=ph.Beta(mean=0.15, sd=0.03), label="P(Stable→Progressed)")
model.add_param("p_SD",   base=0.02, low=0.01, high=0.03,
                dist=ph.Beta(mean=0.02, sd=0.005), label="P(Stable→Dead)")
model.add_param("p_PD",   base=0.10, low=0.05, high=0.15,
                dist=ph.Beta(mean=0.10, sd=0.02), label="P(Progressed→Dead)")
model.add_param("hr_a",   base=0.75, low=0.60, high=0.90,
                dist=ph.LogNormal(mean=0.75, sd=0.10), label="HR Drug A")
model.add_param("hr_b",   base=0.55, low=0.40, high=0.70,
                dist=ph.LogNormal(mean=0.55, sd=0.10), label="HR Drug B")
model.add_param("hr_c",   base=0.90, low=0.75, high=1.05,
                dist=ph.LogNormal(mean=0.90, sd=0.10), label="HR Drug C")
model.add_param("c_soc",  base=2000)
model.add_param("c_a",    base=8000, dist=ph.Gamma(mean=8000, sd=1500))
model.add_param("c_b",    base=18000, dist=ph.Gamma(mean=18000, sd=3000))
model.add_param("c_c",    base=15000, dist=ph.Gamma(mean=15000, sd=2500))
model.add_param("c_prog", base=5000, dist=ph.Gamma(mean=5000, sd=1000))
model.add_param("u_stable", base=0.85, dist=ph.Beta(mean=0.85, sd=0.03))
model.add_param("u_prog",   base=0.55, dist=ph.Beta(mean=0.55, sd=0.05))

# Transitions
model.set_transitions("SOC", lambda p, t: [
    [ph.C,  p["p_SP"],               p["p_SD"]],
    [0,     ph.C,                    p["p_PD"]],
    [0,     0,                       1],
])
model.set_transitions("Drug A", lambda p, t: [
    [ph.C,  p["p_SP"] * p["hr_a"],  p["p_SD"]],
    [0,     ph.C,                    p["p_PD"]],
    [0,     0,                       1],
])
model.set_transitions("Drug B", lambda p, t: [
    [ph.C,  p["p_SP"] * p["hr_b"],  p["p_SD"]],
    [0,     ph.C,                    p["p_PD"]],
    [0,     0,                       1],
])
model.set_transitions("Drug C", lambda p, t: [
    [ph.C,  p["p_SP"] * p["hr_c"],  p["p_SD"]],
    [0,     ph.C,                    p["p_PD"]],
    [0,     0,                       1],
])

# Costs — strategy-specific
model.set_state_cost("treatment", {
    "SOC":    {"Stable": "c_soc",  "Progressed": "c_prog", "Dead": 0},
    "Drug A": {"Stable": "c_a",    "Progressed": "c_prog", "Dead": 0},
    "Drug B": {"Stable": "c_b",    "Progressed": "c_prog", "Dead": 0},
    "Drug C": {"Stable": "c_c",    "Progressed": "c_prog", "Dead": 0},
})

# Utilities
model.set_utility({
    "Stable":     "u_stable",
    "Progressed": "u_prog",
    "Dead":       0.0,
})

# =============================================================================
# 2. Run Base Case & Efficiency Frontier
# =============================================================================
print("\n--- Base Case ---")
result = model.run_base_case()
print(result.summary().to_string(index=False))

# Create CEAnalysis from deterministic result
cea = ph.CEAnalysis.from_result(result)
print("\n" + cea.info())

print("\n--- Efficiency Frontier ---")
frontier = cea.frontier()
print(frontier.to_string(index=False))
print(f"\nFrontier strategies: {cea.frontier_strategies()}")
print(f"Drug C dominated? {cea.is_dominated('Drug C')}")

# NMB ranking at WTP=50,000
print("\n--- NMB Ranking (WTP=$50,000) ---")
print(cea.nmb(wtp=50000).to_string(index=False))
print(f"\nOptimal strategy at WTP=$50,000: {cea.optimal_strategy(50000)}")
print(f"Optimal strategy at WTP=$100,000: {cea.optimal_strategy(100000)}")
print(f"Optimal strategy at WTP=$200,000: {cea.optimal_strategy(200000)}")

# INMB
print("\n--- Incremental NMB vs SOC (WTP=$100,000) ---")
print(cea.inmb(wtp=100000, comparator="SOC").to_string(index=False))

# =============================================================================
# 3. Visualizations — Deterministic
# =============================================================================
print("\n--- Generating Plots ---")

# CE Frontier plot
fig1 = cea.plot_frontier(wtp=100000)
fig1.savefig("plots/ce_frontier.png", dpi=150, bbox_inches='tight')
print("✅ plots/ce_frontier.png")

# NMB curve
fig2 = cea.plot_nmb_curve(wtp_range=(0, 200000))
fig2.savefig("plots/nmb_curves.png", dpi=150, bbox_inches='tight')
print("✅ plots/nmb_curves.png")

# =============================================================================
# 4. PSA → CEAF + EVPI
# =============================================================================
print("\n" + "=" * 70)
print("Part 2: PSA → CEAF & EVPI")
print("=" * 70)

psa_result = model.run_psa(n_sim=2000, seed=42)

# Create CEAnalysis from PSA data
cea_psa = ph.CEAnalysis.from_psa(psa_result)
print(cea_psa.info())

# CEAF data
print("\n--- CEAF (sample) ---")
ceaf_data = cea_psa.ceaf(wtp_range=(0, 200000), n_wtp=21)
print(ceaf_data[["WTP", "Optimal_Strategy", "CEAF"]].to_string(index=False))

# EVPI
print("\n--- EVPI (sample) ---")
evpi_data = cea_psa.evpi(wtp_range=(0, 200000), n_wtp=21)
print(evpi_data[["WTP", "EVPI"]].to_string(index=False))
print(f"\nEVPI at WTP=$100,000: ${cea_psa.evpi_single(100000):,.0f}")

# Plots
fig3 = cea_psa.plot_ceaf(wtp_range=(0, 200000))
fig3.savefig("plots/ceaf.png", dpi=150, bbox_inches='tight')
print("✅ plots/ceaf.png")

fig4 = cea_psa.plot_evpi(wtp_range=(0, 200000), population=100000)
fig4.savefig("plots/evpi.png", dpi=150, bbox_inches='tight')
print("✅ plots/evpi.png")

# Also show the PSA-based frontier
print("\n--- PSA Frontier (mean values) ---")
print(cea_psa.frontier().to_string(index=False))

print("\n✅ Multi-strategy comparison demo complete!")
