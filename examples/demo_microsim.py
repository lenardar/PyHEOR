"""
Demo: Microsimulation — Sick-Sicker Model
==========================================

This example demonstrates MicroSimModel with:
1. Basic microsimulation (3-state Sick-Sicker model)
2. Patient heterogeneity (age-dependent transitions)
3. State entry event handlers (one-time costs)
4. Comparing results with the cohort MarkovModel
5. PSA with outer (parameter) × inner (patient) uncertainty
6. Visualizations: trace, survival, outcome distributions

The Sick-Sicker model (Krijkamp et al. 2018):
  Healthy → Sick → Sicker → Dead
  with direct Healthy→Dead and Sick→Dead transitions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import pyheor as ph

np.random.seed(42)
os.makedirs("plots", exist_ok=True)

# =============================================================================
# 1. Define the Microsimulation Model
# =============================================================================
print("=" * 70)
print("Part 1: Basic Microsimulation — Sick-Sicker Model")
print("=" * 70)

model = ph.MicroSimModel(
    states=["Healthy", "Sick", "Sicker", "Dead"],
    strategies=["Standard of Care", "New Treatment"],
    n_cycles=30,
    n_patients=5000,
    cycle_length=1.0,        # Annual cycles
    discount_rate={"costs": 0.03, "qalys": 0.03},
    half_cycle_correction=True,
    seed=42,
)

# Add parameters
model.add_param("p_HS",     base=0.15, low=0.10, high=0.20,
                dist=ph.Beta(mean=0.15, sd=0.03), label="P(Healthy→Sick)")
model.add_param("p_SS",     base=0.10, low=0.05, high=0.15,
                dist=ph.Beta(mean=0.10, sd=0.02), label="P(Sick→Sicker)")
model.add_param("p_HD",     base=0.005, low=0.002, high=0.01,
                dist=ph.Beta(mean=0.005, sd=0.001), label="P(Healthy→Dead)")
model.add_param("p_SD",     base=0.02, low=0.01, high=0.05,
                dist=ph.Beta(mean=0.02, sd=0.005), label="P(Sick→Dead)")
model.add_param("p_SSD",    base=0.10, low=0.05, high=0.15,
                dist=ph.Beta(mean=0.10, sd=0.02), label="P(Sicker→Dead)")
model.add_param("hr_trt",   base=0.70, low=0.50, high=0.90,
                dist=ph.LogNormal(mean=0.70, sd=0.10), label="HR (Treatment)")
model.add_param("c_trt",    base=5000, low=3000, high=7000,
                dist=ph.Gamma(mean=5000, sd=1000), label="Treatment Cost")
model.add_param("c_sick",   base=3000, low=2000, high=4000,
                dist=ph.Gamma(mean=3000, sd=500), label="Sick State Cost")
model.add_param("c_sicker", base=8000, low=5000, high=11000,
                dist=ph.Gamma(mean=8000, sd=1500), label="Sicker State Cost")
model.add_param("u_healthy",base=0.95, low=0.90, high=1.0,
                dist=ph.Beta(mean=0.95, sd=0.02), label="Utility: Healthy")
model.add_param("u_sick",   base=0.75, low=0.60, high=0.85,
                dist=ph.Beta(mean=0.75, sd=0.05), label="Utility: Sick")
model.add_param("u_sicker", base=0.50, low=0.35, high=0.65,
                dist=ph.Beta(mean=0.50, sd=0.05), label="Utility: Sicker")

# Transitions for Standard of Care
model.set_transitions("Standard of Care", lambda p, t: [
    [ph.C,  p["p_HS"],              0,  p["p_HD"]],
    [0,     ph.C,                   p["p_SS"],  p["p_SD"]],
    [0,     0,                      ph.C,  p["p_SSD"]],
    [0,     0,                      0,  1],
])

# Treatment reduces progression to Sick by HR
model.set_transitions("New Treatment", lambda p, t: [
    [ph.C,  p["p_HS"] * p["hr_trt"], 0,  p["p_HD"]],
    [0,     ph.C,                     p["p_SS"] * p["hr_trt"],  p["p_SD"]],
    [0,     0,                        ph.C,  p["p_SSD"]],
    [0,     0,                        0,  1],
])

# Costs
model.set_state_cost("medical", {
    "Healthy": 500,
    "Sick": "c_sick",
    "Sicker": "c_sicker",
    "Dead": 0,
})

model.set_state_cost("treatment", {
    "Standard of Care": {"Healthy": 0, "Sick": 0, "Sicker": 0, "Dead": 0},
    "New Treatment": {"Healthy": "c_trt", "Sick": "c_trt", "Sicker": "c_trt", "Dead": 0},
})

# One-time cost when entering Sicker state (hospitalization)
model.on_state_enter("Sicker", lambda idx, t, attrs: {"cost": 15000})

# Utilities
model.set_utility({
    "Healthy": "u_healthy",
    "Sick": "u_sick",
    "Sicker": "u_sicker",
    "Dead": 0.0,
})

# Print model info
print(model.info())
print()

# =============================================================================
# 2. Run Base Case
# =============================================================================
print("\n--- Running Base Case ---")
result = model.run_base_case(verbose=True)

print("\nSummary:")
print(result.summary().to_string(index=False))
print("\nICER:")
print(result.icer().to_string(index=False))
print("\nNMB (WTP=$50,000):")
print(result.nmb(wtp=50000).to_string(index=False))

# =============================================================================
# 3. Visualizations
# =============================================================================
print("\n--- Generating Plots ---")

# State occupancy trace
fig1 = result.plot_trace(figsize=(14, 5))
fig1.savefig("plots/microsim_trace.png", dpi=150, bbox_inches='tight')
print("✅ plots/microsim_trace.png")

# Survival curves
fig2 = result.plot_survival(figsize=(10, 7))
fig2.savefig("plots/microsim_survival.png", dpi=150, bbox_inches='tight')
print("✅ plots/microsim_survival.png")

# QALY distribution
fig3 = result.plot_outcomes_histogram(outcome="qalys", figsize=(10, 6))
fig3.savefig("plots/microsim_qaly_dist.png", dpi=150, bbox_inches='tight')
print("✅ plots/microsim_qaly_dist.png")

# Cost distribution
fig4 = result.plot_outcomes_histogram(outcome="cost", figsize=(10, 6))
fig4.savefig("plots/microsim_cost_dist.png", dpi=150, bbox_inches='tight')
print("✅ plots/microsim_cost_dist.png")

# =============================================================================
# 4. Compare with Cohort Model
# =============================================================================
print("\n" + "=" * 70)
print("Part 2: Comparison — Microsim vs Cohort Model")
print("=" * 70)

cohort = ph.MarkovModel(
    states=["Healthy", "Sick", "Sicker", "Dead"],
    strategies=["Standard of Care", "New Treatment"],
    n_cycles=30,
    cycle_length=1.0,
    discount_rate={"costs": 0.03, "qalys": 0.03},
    half_cycle_correction=True,
)

# Same parameters
for name, param in model.params.items():
    cohort.add_param(name, base=param.base, dist=param.dist,
                     label=param.label, low=param.low, high=param.high)

cohort.set_transitions("Standard of Care", lambda p, t: [
    [ph.C,  p["p_HS"],              0,  p["p_HD"]],
    [0,     ph.C,                   p["p_SS"],  p["p_SD"]],
    [0,     0,                      ph.C,  p["p_SSD"]],
    [0,     0,                      0,  1],
])
cohort.set_transitions("New Treatment", lambda p, t: [
    [ph.C,  p["p_HS"] * p["hr_trt"], 0,  p["p_HD"]],
    [0,     ph.C,                     p["p_SS"] * p["hr_trt"],  p["p_SD"]],
    [0,     0,                        ph.C,  p["p_SSD"]],
    [0,     0,                        0,  1],
])

cohort.set_state_cost("medical", {
    "Healthy": 500, "Sick": "c_sick", "Sicker": "c_sicker", "Dead": 0,
})
cohort.set_state_cost("treatment", {
    "Standard of Care": {"Healthy": 0, "Sick": 0, "Sicker": 0, "Dead": 0},
    "New Treatment": {"Healthy": "c_trt", "Sick": "c_trt", "Sicker": "c_trt", "Dead": 0},
})
cohort.set_utility({
    "Healthy": "u_healthy", "Sick": "u_sick", "Sicker": "u_sicker", "Dead": 0.0,
})

cohort_result = cohort.run_base_case()

print("\nCohort results:")
print(cohort_result.summary().to_string(index=False))
print("\nMicrosim results:")
print(result.summary()[['Strategy', 'Mean Cost', 'Mean QALYs']].to_string(index=False))

print("\n(Microsimulation should approximate the cohort model")
print(" with differences due to stochastic sampling.)")
print(" Note: Microsim also includes ¥15,000 hospitalization on entering Sicker")

# =============================================================================
# 5. Patient Heterogeneity
# =============================================================================
print("\n" + "=" * 70)
print("Part 3: Patient Heterogeneity (Age-Dependent Transitions)")
print("=" * 70)

hetero_model = ph.MicroSimModel(
    states=["Healthy", "Sick", "Dead"],
    strategies=["SOC", "Treatment"],
    n_cycles=30,
    n_patients=5000,
    cycle_length=1.0,
    discount_rate=0.03,
    seed=42,
)

# Heterogeneous population
pop = ph.PatientProfile(
    n_patients=5000,
    attributes={
        "age": np.random.normal(55, 12, 5000).clip(20, 90),
        "female": np.random.binomial(1, 0.52, 5000),
    }
)
hetero_model.set_population(pop)

hetero_model.add_param("p_HS_base", base=0.10, dist=ph.Beta(mean=0.10, sd=0.02))
hetero_model.add_param("p_HD_base", base=0.005, dist=ph.Beta(mean=0.005, sd=0.001))
hetero_model.add_param("p_SD",     base=0.05, dist=ph.Beta(mean=0.05, sd=0.01))
hetero_model.add_param("hr_trt",   base=0.65, dist=ph.LogNormal(mean=0.65, sd=0.10))

# Age-dependent transitions!
hetero_model.set_transitions("SOC", lambda p, t, attrs: [
    [ph.C,  p["p_HS_base"] * (1 + (attrs.get("age", 55) - 55) * 0.02),
            p["p_HD_base"] * (1 + max(attrs.get("age", 55) - 50, 0) * 0.03)],
    [0,     ph.C,  p["p_SD"]],
    [0,     0,     1],
])

hetero_model.set_transitions("Treatment", lambda p, t, attrs: [
    [ph.C,  p["p_HS_base"] * (1 + (attrs.get("age", 55) - 55) * 0.02) * p["hr_trt"],
            p["p_HD_base"] * (1 + max(attrs.get("age", 55) - 50, 0) * 0.03)],
    [0,     ph.C,  p["p_SD"]],
    [0,     0,     1],
])

hetero_model.set_state_cost("medical", {"Healthy": 500, "Sick": 5000, "Dead": 0})
hetero_model.set_state_cost("drug", {
    "SOC": {"Healthy": 0, "Sick": 0},
    "Treatment": {"Healthy": 3000, "Sick": 3000},
})
hetero_model.set_utility({"Healthy": 0.95, "Sick": 0.65, "Dead": 0.0})

print("\nRunning heterogeneous microsimulation...")
hetero_result = hetero_model.run_base_case(verbose=True)

print("\nSummary:")
print(hetero_result.summary().to_string(index=False))
print("\nICER:")
print(hetero_result.icer().to_string(index=False))

# Age subgroup analysis
outcomes = hetero_result.patient_outcomes
outcomes_soc = outcomes[outcomes['Strategy'] == 'SOC'].copy()
ages = pop.get_all("age")
outcomes_soc['Age Group'] = pd.cut(ages, bins=[20, 45, 60, 75, 90],
                                    labels=["20-45", "45-60", "60-75", "75-90"])
print("\nQALYs by Age Group (SOC):")
age_summary = outcomes_soc.groupby('Age Group', observed=True)['Total QALYs'].agg(['mean', 'std'])
print(age_summary.to_string())

# =============================================================================
# 6. PSA (Outer × Inner)
# =============================================================================
print("\n" + "=" * 70)
print("Part 4: PSA — 100 outer × 2000 inner")
print("=" * 70)

psa_result = model.run_psa(
    n_outer=100,
    n_inner=2000,
    seed=42,
    verbose=True,
)

print("\nPSA Summary:")
print(psa_result.summary().to_string(index=False))
print("\nPSA ICER:")
print(psa_result.icer().to_string(index=False))

# CEAC
fig5 = psa_result.plot_ceac(wtp_range=(0, 150000))
fig5.savefig("plots/microsim_ceac.png", dpi=150, bbox_inches='tight')
print("✅ plots/microsim_ceac.png")

# CE Scatter
fig6 = psa_result.plot_scatter(wtp=50000)
fig6.savefig("plots/microsim_ce_scatter.png", dpi=150, bbox_inches='tight')
print("✅ plots/microsim_ce_scatter.png")

print("\n✅ Microsimulation demo complete!")
