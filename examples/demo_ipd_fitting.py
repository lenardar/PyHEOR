"""
Demo: IPD Survival Curve Fitting
=================================

This example demonstrates using SurvivalFitter to:
1. Simulate individual patient data (IPD)
2. Fit all 6 parametric distributions
3. Compare AIC/BIC
4. Select best model
5. Plot KM + all fitted curves
6. Generate diagnostic plots
7. Export results to Excel
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pyheor as ph

np.random.seed(42)

# =============================================================================
# 1. Simulate IPD data (Weibull with shape=1.5, scale=24 months)
# =============================================================================
n_patients = 200
true_shape = 1.5
true_scale = 24

# Generate Weibull event times
u = np.random.uniform(0, 1, n_patients)
event_times = true_scale * (-np.log(u)) ** (1 / true_shape)

# Random censoring (~30%)
censor_times = np.random.uniform(0, 50, n_patients)
time = np.minimum(event_times, censor_times)
event = (event_times <= censor_times).astype(int)

print(f"Simulated data: n={n_patients}, events={event.sum()}, "
      f"censored={n_patients - event.sum()}")
print(f"True model: Weibull(shape={true_shape}, scale={true_scale})")
print()

# =============================================================================
# 2. Fit all parametric models
# =============================================================================
fitter = ph.SurvivalFitter(time=time, event=event, label="Overall Survival")
fitter.fit(verbose=True)

# =============================================================================
# 3. Model comparison table
# =============================================================================
print("\n" + "=" * 70)
print("Model Comparison Table:")
print("=" * 70)
summary = fitter.summary()
print(summary.to_string(index=False))

# =============================================================================
# 4. Best model
# =============================================================================
best = fitter.best_model()
print(f"\nBest model (AIC): {best.name}")
print(f"  Parameters: {best.params}")
print(f"  AIC: {best.aic:.2f}, BIC: {best.bic:.2f}")

best_bic = fitter.best_model("bic")
print(f"\nBest model (BIC): {best_bic.name}")

# =============================================================================
# 5. Selection report
# =============================================================================
print("\n")
print(fitter.selection_report())

# =============================================================================
# 6. Plots
# =============================================================================
os.makedirs("plots", exist_ok=True)

# 6a. KM + fitted curves
fig1 = fitter.plot_fits(figsize=(12, 8))
fig1.savefig("plots/ipd_km_fitted_curves.png", dpi=150, bbox_inches='tight')
print("\n✅ Saved: plots/ipd_km_fitted_curves.png")

# 6b. Hazard functions
fig2 = fitter.plot_hazard(figsize=(10, 6))
fig2.savefig("plots/ipd_hazard_functions.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plots/ipd_hazard_functions.png")

# 6c. Log-cumulative hazard diagnostic
fig3 = fitter.plot_cumhazard_diagnostic(figsize=(10, 6))
fig3.savefig("plots/ipd_cumhazard_diagnostic.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plots/ipd_cumhazard_diagnostic.png")

# 6d. Q-Q plot
fig4 = fitter.plot_qq(figsize=(7, 7))
fig4.savefig("plots/ipd_qq_plot.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plots/ipd_qq_plot.png")

# =============================================================================
# 7. Export to Excel
# =============================================================================
fitter.to_excel("ipd_fitting_results.xlsx")

# =============================================================================
# 8. Use fitted distribution in PSM model
# =============================================================================
print("\n" + "=" * 70)
print("Using fitted distribution in PSM model:")
print("=" * 70)

best_dist = fitter.best_model().distribution
print(f"Best distribution: {best_dist}")
print(f"Survival at t=12: {best_dist.survival(12):.4f}")
print(f"Survival at t=24: {best_dist.survival(24):.4f}")
print(f"Survival at t=36: {best_dist.survival(36):.4f}")
print(f"Median survival: {best_dist.quantile(0.5):.2f}")

print("\n✅ IPD fitting demo complete!")
