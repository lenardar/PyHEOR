# PyHEOR — Python Health Economics and Outcome Research

**English** | [中文](README_zh.md) | [Français](README_fr.md)

> **Health economics modeling in Python — as professional as R's hesim / DARTH, but more concise.**

PyHEOR is a Python framework for health economics research, supporting:

| Feature                           | Description                                                                                  |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| **Markov Cohort Model**           | Discrete-time state-transition model (cDTSTM), time-homogeneous / time-dependent transition matrices |
| **Partitioned Survival Model (PSM)** | State probability partitioning based on parametric survival curves                          |
| **Microsimulation**               | Individual-level state-transition model with patient heterogeneity, event handlers, two-level PSA |
| **Discrete Event Simulation (DES)** | Continuous-time individual simulation, competing risks, time-to-event distribution driven, HR/AFT integration |
| **Parametric Survival Distributions** | Exponential, Weibull, Log-logistic, Log-normal, Gompertz, Generalized Gamma, and 10 others |
| **Flexible Cost Definitions**     | First-cycle costs, time-dependent functions, one-time costs, WLOS method, transition cost schedules, custom cost functions |
| **Base Case / OWSA / PSA**        | Deterministic analysis, tornado diagrams (INMB/ICER), Monte Carlo + CE scatter plot + CEAC    |
| **Multi-Strategy Comparison & NMB** | Efficiency frontier, dominance/extended dominance detection, NMB curves, CEAF, EVPI          |
| **IPD Survival Curve Fitting**    | MLE fitting with 6 parametric distributions, AIC/BIC comparison, automatic best model selection |
| **KM Curve Digitization & Reconstruction** | Guyot method to reconstruct IPD from published KM plots, with digitization noise preprocessing |
| **NMA Integration**               | Import R posterior samples, preserve correlations, auto-generate PH/AFT curves               |
| **Budget Impact Analysis (BIA)**  | Population size models, market share evolution, uptake curves, scenario/one-way sensitivity analysis |
| **Model Calibration**             | Estimate unknown parameters from observed data: Nelder-Mead multi-start optimization, LHS random search, SSE/WSSE/likelihood GoF |
| **Visualization**                 | 28 professional charts: state transition diagrams, frontier plots, NMB curves, CEAF, EVPI, CEAC, KM + fitted curves, BIA impact plots, etc. |
| **Export**                         | Multi-sheet Excel export, Excel formula-based verification model, one-click Markdown reports  |

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [User Guide](#user-guide)
  - [Model Types](#model-types) · [Parameter System](#parameter-system) · [Transition Matrix](#transition-matrix) · [Costs and Utilities](#costs-and-utilities) · [Survival Analysis](#survival-analysis) · [Sensitivity Analysis and Reporting](#sensitivity-analysis-and-reporting) · [Advanced Features](#advanced-features) · [Export](#export)
- [Visualization Gallery](#visualization-gallery)
- [Project Structure](#project-structure) · [Design Philosophy](#design-philosophy) · [Roadmap](#roadmap)

---

## Installation

```bash
# Install from source
git clone <repo-url>
cd pyheor
pip install -e .
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy` (optional: `openpyxl` for Excel export, `tabulate` for Markdown reports)

---

## Quick Start

```python
import pyheor as ph

# ── Define Model ──
model = ph.MarkovModel(
    states=["Healthy", "Sick", "Dead"],
    strategies=["SOC", "Treatment"],
    n_cycles=40,
    cycle_length=1,
    dr_cost=ph.Param(0.03, low=0.0, high=0.08, label="Cost discount rate"),
    dr_qaly=ph.Param(0.03, low=0.0, high=0.05, label="Utility discount rate"),
    half_cycle_correction=True,
)

# ── Parameters ──
model.add_param("p_HS", base=0.15, low=0.10, high=0.20,
    dist=ph.Beta(mean=0.15, sd=0.03))
model.add_param("c_drug", base=2000, low=1500, high=2500,
    dist=ph.Gamma(mean=2000, sd=400))

# ── Transition Matrix (ph.C = complement) ──
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

# ── Costs & Utilities ──
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

# ── Run Analysis ──
result = model.run_base_case()
print(result.summary())
print(result.icer())

owsa = model.run_owsa()       # Discount rates auto-included in OWSA via Param
owsa.plot_tornado()

psa = model.run_psa(n_sim=1000)
psa.plot_ceac()

# ── One-click Markdown Report ──
ph.generate_report(model, "report.md")
```

---

## User Guide

### Model Types

#### Markov Cohort Model

Discrete-time cohort model (cDTSTM), suitable for simple models with known state-transition probabilities. See the full example in [Quick Start](#quick-start).

#### Partitioned Survival Model (PSM)

Derives state proportions from parametric survival curves, suitable for the PFS/OS analysis framework commonly used in oncology health economics.

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

# Baseline survival curves
baseline_pfs = ph.LogLogistic(shape=1.5, scale=18)
baseline_os = ph.Weibull(shape=1.2, scale=36)

# SOC: use baseline directly
psm.set_survival("SOC", "PFS", baseline_pfs)
psm.set_survival("SOC", "OS", baseline_os)

# New Drug: HR / AFT modification
psm.set_survival("New Drug", "PFS",
    lambda p: ph.AcceleratedFailureTime(baseline_pfs, af=1.3))
psm.set_survival("New Drug", "OS",
    lambda p: ph.ProportionalHazards(baseline_os, hr=0.7))

# Costs & Utilities
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

#### Microsimulation

Individual-level state-transition model that shares the same API as MarkovModel (`add_param`, `set_transitions`, `set_state_cost`, `set_utility`), but each patient is sampled independently, producing heterogeneous individual-level outcomes.

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

# Event handler: one-time hospitalization cost upon entering Sicker
model.on_state_enter("Sicker", lambda idx, t, attrs: {"cost": 15000})

result = model.run_base_case(verbose=True)
print(result.summary())   # Includes SD and 95% percentiles

# PSA: outer parameter uncertainty x inner individual stochasticity
psa = model.run_psa(n_outer=500, n_inner=2000, seed=42)
psa.plot_ceac(wtp_range=(0, 150000))
```

**Patient Heterogeneity**: Transition probabilities support a 3-argument lambda `(params, cycle, attrs)`, enabling adjustments based on individual attributes (age, sex, etc.):

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

**Performance Optimization**: When the transition matrix does not depend on individual attributes (2-argument lambda), the engine automatically uses vectorized batch sampling, achieving speeds comparable to the cohort model.

#### Discrete Event Simulation (DES)

DES simulates individual patients in **continuous time**, with event times sampled directly from survival distributions, eliminating the need for fixed cycle lengths.

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

# SOC: use baseline directly
model.set_event("SOC", "PFS", "Progressed", baseline_pfs2prog)
model.set_event("SOC", "PFS", "Dead",       baseline_pfs2dead)
model.set_event("SOC", "Progressed", "Dead", baseline_prog2dead)

# TRT: HR applied to PFS->Progressed
model.set_event("TRT", "PFS", "Progressed",
    lambda p: ph.ProportionalHazards(baseline_pfs2prog, p["hr_pfs"]))
model.set_event("TRT", "PFS", "Dead",       baseline_pfs2dead)
model.set_event("TRT", "Progressed", "Dead", baseline_prog2dead)

# Costs (continuous-time rates: $/year)
model.set_state_cost("drug", {
    "SOC": {"PFS": 500, "Progressed": 200, "Dead": 0},
    "TRT": {"PFS": 3000, "Progressed": 200, "Dead": 0},
})
model.set_state_cost("medical", {"PFS": 1000, "Progressed": 5000, "Dead": 0})
model.set_entry_cost("surgery", "Progressed", 50000)

model.set_utility({"PFS": 0.85, "Progressed": 0.50, "Dead": 0})

# Run
result = model.run(n_patients=3000, seed=42)
result.summary()
result.icer()

# PSA
psa = model.run_psa(n_sim=200, n_patients=1000, seed=123)
psa.summary()
```

**DES vs Other Model Types**:

| Feature | MarkovModel | MicroSimModel | DESModel |
|---------|-------------|---------------|----------|
| Time axis | Discrete cycles | Discrete cycles | Continuous time |
| Analysis level | Cohort | Individual | Individual |
| Transition mechanism | Transition matrix | Transition probabilities | Time-to-event distributions |
| Competing risks | Requires manual handling | Requires manual handling | Natively supported |
| Cycle artifacts | Present (requires half-cycle correction) | Present | None |
| Speed | Fastest | Moderate | Slower |
| Use case | Simple models | Complex heterogeneity | Event-driven complex models |

---

### Parameter System

Each parameter is defined via `add_param()`, containing:

| Attribute          | Description                                                                            |
| ------------------ | -------------------------------------------------------------------------------------- |
| `base`           | Baseline value (deterministic analysis)                                                |
| `low` / `high` | OWSA range                                                                             |
| `dist`           | PSA distribution (Beta, Gamma, Normal, LogNormal, Uniform, Triangular, Dirichlet, Fixed) |

```python
model.add_param("p_progression",
    base=0.15,           # For base case analysis
    low=0.10, high=0.20, # OWSA range
    dist=ph.Beta(mean=0.15, sd=0.03),  # For PSA
    label="Disease progression probability",  # For chart display
)
```

#### Discount Rates

All models set discount rates via two independent parameters, `dr_cost` and `dr_qaly`. **The default is 0 (no discounting)**; whichever is not set will not be discounted.

```python
# Fixed discount rates
model = ph.MarkovModel(..., dr_cost=0.03, dr_qaly=0.03)

# Discount costs only
model = ph.MarkovModel(..., dr_cost=0.06)  # dr_qaly defaults to 0
```

Pass a `Param` object to include discount rates in OWSA / PSA without needing an additional `add_param()` call:

```python
model = ph.MarkovModel(
    ...,
    dr_cost=ph.Param(0.03, low=0.0, high=0.08, label="Cost discount rate"),
    dr_qaly=ph.Param(0.03, low=0.0, high=0.05, label="Utility discount rate"),
)

owsa = model.run_owsa()
owsa.plot_tornado()  # Tornado diagram includes discount rates

# You can also apply sensitivity analysis to only one
model = ph.MarkovModel(
    ...,
    dr_cost=0.03,                                        # Fixed
    dr_qaly=ph.Param(0.03, low=0.0, high=0.05),          # Variable
)
```

> **Design Principle**: The baseline value and sensitivity analysis range for discount rates are defined in the same place, avoiding redundant specification. `float` = fixed value, `Param` = variable value.

#### Half-Cycle Correction

| Value                      | Description                                               |
| -------------------------- | --------------------------------------------------------- |
| `True` / `"trapezoidal"` | Trapezoidal method: first and last cycle weights x0.5 (default) |
| `"life-table"`            | Life-table method: average of adjacent trace rows (consistent with R heemod) |
| `False` / `None`          | No correction                                             |

```python
model.half_cycle_correction = "life-table"
model.half_cycle_correction = "trapezoidal"
model.half_cycle_correction = False
```

---

### Transition Matrix

Use `ph.C` (complement sentinel) to auto-calculate diagonal elements:

```python
# Time-homogeneous matrix
model.set_transitions("Strategy", lambda p, t: [
    [ph.C,  p["p_AB"], p["p_AD"]],
    [0,     ph.C,      p["p_BD"]],
    [0,     0,         1        ],
])

# Time-dependent matrix (t is the cycle number)
model.set_transitions("Strategy", lambda p, t: [
    [ph.C,  p["p_AB"] * (1 + 0.01 * t), p["p_AD"]],
    [0,     ph.C,                        p["p_BD"] + 0.001 * t],
    [0,     0,                           1],
])
```

---

### Costs and Utilities

#### State Costs

```python
# Basic state cost
model.set_state_cost("Sick", "Treatment", lambda p, t: 3000)

# Time-dependent cost
model.set_state_cost("Sick", "Treatment", lambda p, t: 3000 if t < 5 else 2000)

# First-cycle one-time cost
model.set_state_cost("Sick", "Treatment", lambda p, t: 50000,
                     first_cycle_only=True)

# Restricted to specific cycles
model.set_state_cost("Sick", "Treatment", lambda p, t: p["c_drug"],
                     apply_cycles=(0, 24))  # First 24 cycles only

# WLOS (Weighted Length of Stay) method
model.set_state_cost("Sick", "Treatment", lambda p, t: 5000,
                     method="wlos")
```

#### Transition Costs

Costs triggered upon state transitions (e.g., surgery costs upon disease progression, hospitalization costs upon ICU transfer). Automatically calculated based on per-cycle **transition flows**: `trace[t-1, from] x P[from->to] x unit cost`.

```python
# Surgery cost upon transitioning from Healthy to Sick
model.set_transition_cost("surgery", "Healthy", "Sick", 50000)

# Parameter reference
model.set_transition_cost("surgery", "Healthy", "Sick", "c_surgery")

# Strategy-specific
model.set_transition_cost("icu", "Sick", "Dead", {
    "SOC": 20000,
    "Treatment": 15000,
})
```

**Cost Schedules**: When a transition triggers costs spanning multiple cycles (e.g., surgery + follow-up), pass a list. The engine automatically handles cost stacking from multiple cohorts of transitioning patients via convolution:

```python
# Progression: surgery 50000, next cycle follow-up 10000 -> spans 2 cycles
model.set_transition_cost("surgery", "PFS", "Progressed", [50000, 10000])

# Parameter references can also be used within lists
model.set_transition_cost("chemo", "PFS", "Progressed",
    ["c_chemo_init", "c_chemo_maint", "c_chemo_maint"])

# Strategy-specific + schedule mixed usage
model.set_transition_cost("rescue", "PFS", "Progressed", {
    "SOC": [30000, 5000],       # Schedule
    "New Drug": 15000,           # Scalar
})
```

> **Difference from `first_cycle_only`**: `first_cycle_only` only applies at cycle 0 (once only); transition costs are incurred in **every cycle** whenever patients transition. Transition costs are not affected by half-cycle correction (event-type costs).

#### Custom Costs

When `set_transition_cost` with per-state-pair definitions is not flexible enough, use `set_custom_cost` to pass a custom function that calculates costs directly based on the transition matrix and state distribution. Supported by MarkovModel and PSMModel.

```python
# Function signature
# func(strategy, params, t, state_prev, state_curr, P, states) -> float

# MarkovModel: calculate surgery cost based on transition flows
def surgery_cost(strategy, params, t, state_prev, state_curr, P, states):
    i_from = states.index("PFS")
    i_to = states.index("Progressed")
    flow = state_prev[i_from] * P[i_from, i_to]
    return flow * params["c_surgery"]

model.set_custom_cost("surgery", surgery_cost)

# PSMModel: calculate progression cost based on state changes (no transition matrix, P=None)
def progression_cost(strategy, params, t, state_prev, state_curr, P, states):
    i_prog = states.index("Progressed")
    new_prog = max(0, state_curr[i_prog] - state_prev[i_prog])
    return new_prog * params["c_progression"]

psm.set_custom_cost("progression", progression_cost)
```

> Custom costs are not affected by half-cycle correction (consistent with transition costs). The function receives parameter values via `params`, and OWSA/PSA parameter variations and sampling propagate naturally.

---

### Survival Analysis

#### Parametric Survival Distributions

10 built-in survival distributions:

| Distribution                       | Parameters | Hazard Shape Characteristics             |
| ---------------------------------- | ---------- | ---------------------------------------- |
| `Exponential(rate)`              | lambda     | Constant hazard                          |
| `Weibull(shape, scale)`          | alpha, lambda | shape>1 increasing, <1 decreasing     |
| `LogLogistic(shape, scale)`      | alpha, lambda | shape>1 rises then falls             |
| `SurvLogNormal(meanlog, sdlog)`  | mu, sigma  | Rises then falls                         |
| `Gompertz(shape, rate)`          | a, b       | shape>0 increasing, <0 decreasing        |
| `GeneralizedGamma(mu, sigma, Q)` | mu, sigma, Q | Flexible (includes Weibull, LogNormal as special cases) |

Auxiliary distributions:

| Distribution                                 | Description                          |
| -------------------------------------------- | ------------------------------------ |
| `ProportionalHazards(baseline, hr)`        | Proportional hazards: h(t) = h0(t) x HR |
| `AcceleratedFailureTime(baseline, af)`     | Accelerated failure time: S(t) = S0(t/AF) |
| `KaplanMeier(times, probs)`                | Empirical distribution + extrapolation |
| `PiecewiseExponential(breakpoints, rates)` | Piecewise constant hazard            |

Each distribution provides `survival(t)`, `hazard(t)`, `pdf(t)`, `quantile(p)`, `cumulative_hazard(t)`, `restricted_mean(t_max)` methods.

#### IPD Survival Curve Fitting

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

# AIC/BIC comparison table
print(fitter.summary())

# Automatically select the best model
best = fitter.best_model()           # Default: AIC
dist = best.distribution             # Can be used directly in PSM
print(fitter.selection_report())     # Detailed model selection report

# Diagnostic plots
fitter.plot_fits()                   # KM + all fitted curves
fitter.plot_hazard()                 # Hazard functions
fitter.plot_cumhazard_diagnostic()   # log(H) vs log(t)
fitter.plot_qq()                     # Q-Q plot

# Export
fitter.to_excel("fitting_results.xlsx")
```

**Model Selection Criteria**:

| Metric     | Formula               | Description                                          |
| ---------- | --------------------- | ---------------------------------------------------- |
| AIC        | 2k - 2ln(L)           | Favors good fit + parsimony; suitable for prediction |
| BIC        | k*ln(n) - 2ln(L)      | Penalizes complexity more than AIC; suitable for large samples |
| Delta AIC  | AIC - AIC_min          | <2 not significant, >10 decisive difference          |
| AIC Weight | exp(-0.5*Delta AIC) / Sum | Relative likelihood weight of the model           |

**IPD to PSM Integrated Workflow**:

```python
fitter_os = ph.SurvivalFitter(time=df_os["time"], event=df_os["event"], label="OS")
fitter_pfs = ph.SurvivalFitter(time=df_pfs["time"], event=df_pfs["event"], label="PFS")
fitter_os.fit()
fitter_pfs.fit()

psm = ph.PSMModel(...)
psm.set_survival("SOC", "OS", fitter_os.best_model().distribution)
psm.set_survival("SOC", "PFS", fitter_pfs.best_model().distribution)
```

#### KM Curve Digitization & Reconstruction

Reconstruct IPD from published KM curve plots, enabling the complete workflow: "Published KM plot -> IPD -> Parametric fitting -> Modeling." Based on the Guyot et al. (2012) algorithm.

```python
# 1. Obtain KM coordinates from a digitization tool (e.g., WebPlotDigitizer)
t_digitized = [0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24]
s_digitized = [1.0, 0.92, 0.83, 0.74, 0.66, 0.58, 0.50, 0.40, 0.32, 0.25, 0.20]

# 2. Read the number-at-risk table from the publication
t_risk = [0, 6, 12, 18, 24]
n_risk = [120, 88, 60, 38, 22]

# 3. Reconstruct IPD
ipd_time, ipd_event = ph.guyot_reconstruct(
    t_digitized, s_digitized, t_risk, n_risk, tot_events=96,
)

# 4. Feed directly into SurvivalFitter
fitter = ph.SurvivalFitter(ipd_time, ipd_event, label="OS")
fitter.fit()
```

**Digitized Coordinate Preprocessing**: `clean_digitized_km` provides automatic cleaning (sorting, out-of-bounds removal, outlier detection, enforced monotonicity, etc.). `guyot_reconstruct` also calls it internally.

References:

- Guyot P, Ades AE, Ouwens MJ, Welton NJ (2012). Enhanced secondary analysis of survival data. *BMC Med Res Methodol*, 12:9.
- Liu N, Zhou Y, Lee JJ (2021). IPDfromKM. *BMC Med Res Methodol*.

---

### Sensitivity Analysis and Reporting

#### OWSA & PSA

```python
# OWSA (discount rates auto-registered via Param)
owsa = model.run_owsa(wtp=50000)
print(owsa.summary(outcome="icer"))   # Sorted by ICER impact magnitude
owsa.plot_tornado(outcome="nmb", max_params=10)

# PSA (Monte Carlo)
psa = model.run_psa(n_sim=1000, seed=42)
print(psa.summary())
print(psa.icer())
psa.plot_scatter(wtp=50000)
psa.plot_ceac()
psa.plot_convergence()
```

#### One-Click Report (`generate_report`)

After model parameters are configured, run all analyses and generate a Markdown report + accompanying figures with a single call:

```python
ph.generate_report(
    model,
    "report.md",       # Output path; figures saved to report_files/
    wtp=50000,          # WTP threshold
    n_sim=1000,         # Number of PSA simulations
    max_params=10,      # Max parameters shown in tornado diagram
    run_psa=None,       # None = auto-detect (runs if dist is defined)
)
```

The report includes: model overview, parameter table, base case results, ICER, OWSA tornado diagram and ranking table, PSA summary statistics and incremental analysis, CE plane scatter plot, and CEAC curve. All model types (Markov / PSM / MicroSim / DES) are supported.

---

### Advanced Features

#### Multi-Strategy Comparison & NMB Analysis

```python
# Create CEAnalysis from deterministic results
result = model.run_base_case()
cea = ph.CEAnalysis.from_result(result)

# Efficiency frontier: sequential ICER + dominance/extended dominance detection
print(cea.frontier())

# NMB ranking
print(cea.nmb(wtp=100000))
print(f"Optimal strategy: {cea.optimal_strategy(wtp=100000)}")

# Visualization
cea.plot_frontier(wtp=100000)
cea.plot_nmb_curve(wtp_range=(0, 200000))
```

**PSA -> CEAF & EVPI**:

```python
psa_result = model.run_psa(n_sim=2000)
cea_psa = ph.CEAnalysis.from_psa(psa_result)

cea_psa.plot_ceaf(wtp_range=(0, 200000))
print(f"EVPI at WTP=$100K: ${cea_psa.evpi_single(100000):,.0f}")
cea_psa.plot_evpi(wtp_range=(0, 200000), population=100000)
```

#### NMA Integration

PyHEOR's NMA module is responsible for **importing and using** posterior samples produced by R packages (gemtc / multinma / bnma).

```python
# Load posterior samples (supports wide/long format CSV)
nma = ph.load_nma_samples("nma_hr_samples.csv", log_scale=True)
print(nma.summary())

# Batch inject into model parameters
nma.add_params_to_model(model, param_prefix="hr",
                        treatments=["Drug_A", "Drug_B"])

# Quickly build survival curves
baseline = ph.Weibull(shape=1.2, scale=8.0)
curves = ph.make_ph_curves(baseline, nma)      # PH
curves_aft = ph.make_aft_curves(baseline, nma)  # AFT
```

| Class / Function | Description |
|---|---|
| `load_nma_samples()` | Load posteriors from CSV/Excel/Feather (wide/long format, supports log transformation) |
| `NMAPosterior` | Posterior container providing `dist()` / `correlated()` / `summary()` / `add_params_to_model()` |
| `PosteriorDist` | `Distribution` subclass, samples with replacement from the posterior column |
| `CorrelatedPosterior` | Joint posterior, same-row sampling to preserve correlations |
| `make_ph_curves()` / `make_aft_curves()` | NMA posterior + baseline curve -> PH/AFT curve dictionary |

#### Budget Impact Analysis (BIA)

Budget impact analysis estimates the financial impact of introducing a new technology on the budget over a short time horizon (typically 1-5 years). Follows the ISPOR BIA good practice guidelines.

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

**Population Models**:

```python
population=10000                                    # Fixed population
population=[10000, 10500, 11000, 11500, 12000]      # Specified per year
population={"base": 10000, "growth_rate": 0.05}      # Compound growth
population={"base": 10000, "annual_increase": 500}   # Linear growth
```

**Market Share Uptake Curves**:

```python
ph.BudgetImpactAnalysis.linear_uptake(0.0, 0.4, 5)           # Linear
ph.BudgetImpactAnalysis.sigmoid_uptake(0.0, 0.4, 5, steepness=1.5)  # Sigmoid
```

**Creating from Model Results / Scenario Analysis / Sensitivity Analysis**:

```python
# From model results
bia = ph.BudgetImpactAnalysis.from_result(result, population=10000, ...)

# Scenario analysis
bia.scenario_analysis({
    "Base Case": {},
    "High Population": {"population": 15000},
    "Fast Uptake": {"market_share_new": {"SoC": 0.3, "New": 0.7}},
})

# One-way sensitivity
bia.one_way_sensitivity("population", values=[8000, 9000, 10000, 11000, 12000])

# Tornado diagram
bia.tornado({"population": (8000, 12000), "Drug B": (10000, 15000)})
```

#### Model Calibration

Model calibration uses observed data to estimate model parameters that cannot be directly observed. Based on Vanni et al. (2011) and Alarid-Escudero et al. (2018).

```python
# Define calibration targets
targets = [
    ph.CalibrationTarget(
        name="10yr_healthy",
        observed=0.42, se=0.05,
        extract_fn=lambda sim: sim["SOC"]["trace"][10, 0],
    ),
]

# Define parameters to calibrate
calib_params = [
    ph.CalibrationParam("p_HS", lower=0.01, upper=0.30),
    ph.CalibrationParam("p_SD", lower=0.01, upper=0.20),
]

# Run calibration
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

| Search Method | Parameters | Characteristics |
|---------------|------------|-----------------|
| `nelder_mead` | `n_restarts=10` | Multi-start derivative-free optimization, precise but slower |
| `random_search` | `n_samples=1000` | LHS sampling with individual evaluation, simple and intuitive |

| GoF Metric | Formula | Use Case |
|------------|---------|----------|
| `sse` | Sum(obs - pred)^2 | Default, simple and fast |
| `wsse` | Sum(obs - pred)^2/SE^2 | When multiple targets have different scales |
| `loglik_normal` | -Sum log N(obs \| pred, SE^2) | Statistically principled |

---

### Export

#### Excel Export

```python
# Result data export (multi-sheet)
ph.export_to_excel(result, "base_case.xlsx")
ph.export_to_excel(owsa, "owsa.xlsx")
ph.export_to_excel(psa, "psa.xlsx")

# Multi-strategy comparison
ph.export_comparison_excel({"Strategy A": result_a, "Strategy B": result_b}, "comparison.xlsx")

# IPD fitting results
fitter.to_excel("fitting_results.xlsx")
```

#### Excel Formula-Based Verification Model

Export a complete model file that **independently computes using Excel formulas**, for cross-validating Python results:

```python
result = model.run_base_case()
ph.export_excel_model(result, "verification.xlsx")

# Or export directly from the model
ph.export_excel_model(model, "verification.xlsx")
```

| Section | Content |
|---------|---------|
| **Input Area** (yellow background) | Transition probability matrix, state cost vector, utility weights, discount rates |
| **Calculation Area** (formulas) | `SUMPRODUCT` computes Trace, Costs, QALYs; `SUM` computes discounted totals |
| **Summary sheet** | Excel formula results vs Python results vs difference (should be ~0) |

**Supported Model Types**:

| Model | Trace | Costs/QALYs/Discounting | ICER |
|-------|-------|-------------------------|------|
| Markov (time-homogeneous) | Excel formulas | Excel formulas | Excel formulas |
| Markov (time-dependent) | Python values | Excel formulas | Excel formulas |
| PSM | Python survival values -> state probability formulas | Excel formulas | Excel formulas |

#### Excel Sheet Contents

| Analysis Type   | Sheet Contents                                                        |
| --------------- | --------------------------------------------------------------------- |
| Base Case       | Summary, State Trace, Cost/QALY by Cycle, ICER                        |
| OWSA            | Tornado Data, Per-Parameter Results                                   |
| PSA             | Summary Stats, All Simulations, CEAC Data                             |
| PSM Base        | Summary, State Probabilities, Survival Data                           |
| IPD Fitting     | Model Comparison, KM Data, Per-Distribution Details, Selection Report |
| Verification Model | Summary (with differences), Per-Strategy Calculation Sheet (formulas + inputs) |

---

## Visualization Gallery

PyHEOR provides **28** professional charts, covering all model types and analysis workflows:

### Markov Model (8 types)

| Function                        | Description                                  |
| ------------------------------- | -------------------------------------------- |
| `plot_transition_diagram()`   | State transition diagram                     |
| `plot_model_diagram()`        | TreeAge-style model diagram                  |
| `plot_trace()`                | Markov trace (cohort trajectory)             |
| `plot_tornado()`              | OWSA tornado diagram                         |
| `plot_owsa_param()`           | Single-parameter OWSA line plot              |
| `plot_scatter()`              | CE scatter plot (incremental cost vs effect)  |
| `plot_ceac()`                 | Cost-effectiveness acceptability curve        |
| `plot_convergence()`          | PSA convergence diagnostic plot              |

### PSM Model (4 types)

| Function                     | Description                       |
| ---------------------------- | --------------------------------- |
| `plot_survival_curves()`   | Parametric survival curves        |
| `plot_state_area()`        | Area chart (state proportions)    |
| `plot_psm_trace()`         | PSM state trajectory              |
| `plot_psm_comparison()`    | Multi-strategy survival curve comparison |

### Microsimulation (3 types)

| Function                       | Description                                         |
| ------------------------------ | --------------------------------------------------- |
| `plot_microsim_trace()`      | Individual simulation state proportion trajectory   |
| `plot_microsim_survival()`   | Empirical survival curve (from simulated data)      |
| `plot_microsim_outcomes()`   | Patient outcome distributions (QALYs / Costs / LYs histograms) |

### IPD Fitting (4 types)

| Method                                 | Description                  |
| -------------------------------------- | ---------------------------- |
| `fitter.plot_fits()`                 | KM + all parametric fit curves |
| `fitter.plot_hazard()`               | Hazard functions by distribution |
| `fitter.plot_cumhazard_diagnostic()` | log(H) vs log(t) diagnostic plot |
| `fitter.plot_qq()`                   | Q-Q quantile plot            |

### CEA / Multi-Strategy Comparison (4 types)

| Function               | Description                                    |
| ---------------------- | ---------------------------------------------- |
| `plot_ce_frontier()`   | Efficiency frontier + WTP line + ICER labels   |
| `plot_nmb_curve()`     | NMB curve (multiple strategies across WTP)     |
| `plot_ceaf()`          | Cost-effectiveness acceptability frontier (CEAF) |
| `plot_evpi()`          | Expected value of perfect information (EVPI) curve |

### Budget Impact Analysis (5 types)

| Function                       | Description                                    |
| ------------------------------ | ---------------------------------------------- |
| `plot_budget_impact()`       | Annual budget impact bar chart + cumulative curve |
| `plot_budget_comparison()`   | Current vs new scenario total cost comparison  |
| `plot_market_share()`        | Dual-panel market share stacked area chart     |
| `plot_detail()`              | Stacked cost breakdown by strategy             |
| `plot_tornado()`             | BIA sensitivity tornado diagram                |

---

## Project Structure

```text
pyheor/
├── pyproject.toml
├── README.md
├── src/pyheor/              # Package source (src layout)
│   ├── __init__.py          # Package entry, unified exports
│   ├── utils.py             # Utility functions (C complement, discounting, validation)
│   ├── distributions.py     # PSA probability distributions (Beta, Gamma, ...)
│   ├── survival.py          # 10 parametric survival distributions
│   ├── plotting.py          # Visualization (28 chart types)
│   │
│   ├── models/              # ── Modeling Engine ──
│   │   ├── markov.py        #  Markov cohort model (MarkovModel)
│   │   ├── psm.py           #  Partitioned survival model (PSMModel)
│   │   ├── microsim.py      #  Microsimulation (MicroSimModel)
│   │   └── des.py           #  Discrete event simulation (DESModel)
│   │
│   ├── analysis/            # ── Analysis & Decision ──
│   │   ├── results.py       #  Result classes (BaseResult, OWSAResult, PSAResult, ...)
│   │   ├── comparison.py    #  Multi-strategy comparison / CEA (CEAnalysis)
│   │   ├── bia.py           #  Budget impact analysis (BudgetImpactAnalysis)
│   │   └── calibration.py   #  Model calibration (Nelder-Mead, random search)
│   │
│   ├── evidence/            # ── Data & Evidence Synthesis ──
│   │   ├── fitting.py       #  IPD survival curve fitting (SurvivalFitter)
│   │   ├── digitize.py      #  KM curve digitization & reconstruction (Guyot method)
│   │   └── nma.py           #  NMA posterior sample integration (NMAPosterior)
│   │
│   └── export/              # ── Export ──
│       ├── excel.py         #  Excel result data export
│       ├── excel_model.py   #  Excel formula-based verification model export
│       └── report.py        #  One-click Markdown report
│
├── tests/                   # pytest test suite (243 tests)
└── examples/
    ├── demo_hiv_model.py    #  Markov model example (HIV)
    ├── demo_psm_model.py    #  PSM model example (oncology)
    ├── demo_ipd_fitting.py  #  IPD fitting example
    ├── demo_microsim.py     #  Microsimulation example
    └── demo_comparison.py   #  Multi-strategy comparison example
```

---

## Design Philosophy

- **Concise API**: A single model object handles base case / OWSA / PSA without separate calls
- **Flexible Parameter System**: `ph.C` auto-complement, lambda functions define time-dependent probabilities/costs
- **Aligned with R Ecosystem**: Distribution parameterization and method naming reference hesim / flexsurv / DARTH
- **Production-Quality Visualization**: All charts work out of the box, consistent color scheme, customizable
- **Verifiability**: Excel export of trace data for easy cross-validation with TreeAge / Excel models

---

## Roadmap

- [X] Markov cohort model (cDTSTM)
- [X] One-way sensitivity analysis (OWSA) + tornado diagram
- [X] Probabilistic sensitivity analysis (PSA) + CEAC + CE scatter plot
- [X] Flexible cost system (first-cycle, time-dependent, WLOS, custom cost functions)
- [X] Multi-method half-cycle correction (trapezoidal / life-table / none) & configurable discount rates
- [X] OWSA tornado ICER ranking & discount rates directly included in sensitivity analysis via `Param`
- [X] Partitioned survival model (PSM)
- [X] 10 parametric survival distributions
- [X] Multi-sheet Excel export + Excel formula-based verification model
- [X] IPD survival curve fitting + AIC/BIC model comparison
- [X] KM + fitted curve visualization + diagnostic plots
- [X] Microsimulation (individual-level simulation)
- [X] Multi-cohort comparison + NMB analysis + CEAF + EVPI
- [X] Network meta-analysis (NMA) integration
- [X] Discrete event simulation (DES) -- continuous time, competing risks, HR/AFT integration
- [X] Budget impact analysis (BIA) -- population models, market share evolution, uptake curves, scenario/sensitivity analysis
- [X] Digitized KM curve reconstruction (Guyot method)
- [X] Model calibration (Nelder-Mead multi-start optimization, LHS random search, SSE/WSSE/likelihood GoF)
- [X] One-click Markdown report (`generate_report`)
- [X] Formal test suite (pytest, 243 tests covering all modules)
- [ ] Structured output (`to_dict` / `to_json`) for LLM-ready results
- [ ] Auto-interpretation (`interpret(wtp)`) — standardized conclusion text generation
- [ ] Natural language modeling interface — JSON Schema model definition, auto-build & execute

---

## License

MIT License
