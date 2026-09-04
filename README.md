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
| **Visualization**                 | 19 professional charts: state transition diagrams, frontier plots, NMB curves, CEAF, EVPI, CEAC, etc. |
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

Every model has a consistent full class name and a concise alias for everyday use:

| Full class name | Concise alias |
|---|---|
| `CohortStateTransitionModel` | `MarkovModel` |
| `PartitionedSurvivalModel` | `PSMModel` |
| `IndividualStateTransitionModel` | `MicroSimModel` |
| `DiscreteEventSimulationModel` | `DESModel` |

Each pair references the same class and can be used interchangeably. The examples below use the concise aliases.

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
| `True` / `"trapezoidal"` | Trapezoidal method: use the average occupancy of adjacent trace time points for each interval (default) |
| `"life-table"`            | Compatibility alias for `"trapezoidal"`; produces identical results |
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
model.set_state_cost("medical", {"Treatment": {"Sick": 3000}})

# Time-dependent cost
model.set_state_cost("medical", lambda p, t: {
    "Treatment": {"Sick": 3000 if t < 5 else 2000}
})

# Cost rate incurred only during the first time interval
model.set_state_cost("induction", {"Treatment": {"Sick": 50000}},
                     first_cycle_only=True)

# One-time cost at model start
model.set_state_cost("init", {"Sick": 50000}, method="starting")

# Restricted to specific cycles
model.set_state_cost("drug", {"Treatment": {"Sick": "c_drug"}},
                     apply_cycles=range(24))  # First 24 time intervals only

# WLOS (Weighted Length of Stay) method
model.set_state_cost("medical", {"Treatment": {"Sick": 5000}},
                     method="wlos")
```

#### Transition Costs

Costs triggered upon state transitions (e.g., surgery costs upon disease progression, hospitalization costs upon ICU transfer). Automatically calculated from each interval's **transition flow**: `trace[i, from] x P_i[from->to] x unit cost`.

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

> **Difference from `first_cycle_only`**: `first_cycle_only` is a rate incurred only during interval 0; a transition cost is a lump sum incurred when the transition occurs. Transition costs are not scaled by cycle length and are not affected by half-cycle correction.

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


### Export

#### Excel Export

```python
# Result data export (multi-sheet)
ph.export_to_excel(result, "base_case.xlsx")
ph.export_to_excel(owsa, "owsa.xlsx")
ph.export_to_excel(psa, "psa.xlsx")

# Multi-strategy comparison
ph.export_comparison_excel({"Strategy A": result_a, "Strategy B": result_b}, "comparison.xlsx")

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
| **Input Area** (yellow background) | Transition matrix, state and transition costs, survival parameters, utility weights, discount settings |
| **Calculation Area** (formulas) | Trace/state probabilities, interval occupancy, event flows, costs, QALYs, discounting, and totals |
| **Summary sheet** | Excel formula results vs Python results vs difference (should be ~0) |

**Supported Model Types**:

| Model | Trace | Costs/QALYs/Discounting | ICER |
|-------|-------|-------------------------|------|
| Markov (time-homogeneous) | Excel formulas from one editable matrix | Excel formulas | Excel formulas |
| Markov (time-dependent) | Excel formulas from an editable matrix for each interval | Excel formulas | Excel formulas |
| PSM | Excel formulas for common parametric curves; clearly marked external inputs for unsupported curves | Excel formulas | Excel formulas |

Time-varying matrices are exposed as explicit Excel inputs rather than hidden Python trace values. Custom Python cost callbacks and other logic that cannot be translated faithfully are rejected with an error.

#### Excel Sheet Contents

| Analysis Type   | Sheet Contents                                                        |
| --------------- | --------------------------------------------------------------------- |
| Base Case       | Summary, State Trace, Cost/QALY by Cycle, ICER                        |
| OWSA            | Tornado Data, Per-Parameter Results                                   |
| PSA             | Summary Stats, All Simulations, CEAC Data                             |
| PSM Base        | Summary, State Probabilities, Survival Data                           |
| Verification Model | Summary (with differences), Per-Strategy Calculation Sheet (formulas + inputs) |

---

## Visualization Gallery

PyHEOR provides **19** professional charts, covering all model types and analysis workflows:

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


### CEA / Multi-Strategy Comparison (4 types)

| Function               | Description                                    |
| ---------------------- | ---------------------------------------------- |
| `plot_ce_frontier()`   | Efficiency frontier + WTP line + ICER labels   |
| `plot_nmb_curve()`     | NMB curve (multiple strategies across WTP)     |
| `plot_ceaf()`          | Cost-effectiveness acceptability frontier (CEAF) |
| `plot_evpi()`          | Expected value of perfect information (EVPI) curve |


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
│   ├── plotting.py          # Visualization (19 chart types)
│   │
│   ├── models/              # ── Modeling Engine ──
│   │   ├── markov.py        #  Markov cohort model (MarkovModel)
│   │   ├── psm.py           #  Partitioned survival model (PSMModel)
│   │   ├── microsim.py      #  Microsimulation (MicroSimModel)
│   │   └── des.py           #  Discrete event simulation (DESModel)
│   │
│   ├── analysis/            # ── Analysis & Decision ──
│   │   ├── results.py       #  Result classes (BaseResult, OWSAResult, PSAResult, ...)
│   │   └── comparison.py    #  Multi-strategy comparison / CEA (CEAnalysis)
│   │
│   └── export/              # ── Export ──
│       ├── excel.py         #  Excel result data export
│       ├── excel_model.py   #  Excel formula-based verification model export
│       └── report.py        #  One-click Markdown report
│
├── tests/                   # pytest test suite
└── examples/
    ├── demo_hiv_model.py    #  Markov model example (HIV)
    ├── demo_psm_model.py    #  PSM model example (oncology)
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
- [X] Microsimulation (individual-level simulation)
- [X] Multi-cohort comparison + NMB analysis + CEAF + EVPI
- [X] Discrete event simulation (DES) -- continuous time, competing risks, HR/AFT integration
- [X] One-click Markdown report (`generate_report`)
- [X] Formal test suite (pytest)
- [ ] Structured output (`to_dict` / `to_json`) for LLM-ready results
- [ ] Auto-interpretation (`interpret(wtp)`) — standardized conclusion text generation
- [ ] Natural language modeling interface — JSON Schema model definition, auto-build & execute
- [ ] HEOR Agent (`pyheor-agent`) — natural-language model definition, execution, and report generation, available as both Python API (`HEORAgent`) and CLI
- [ ] Rust core acceleration (low priority) — PyO3 + maturin bindings to accelerate microsimulation patient loops, DES event queues, and PSA parallelism

---

## License

GNU Affero General Public License v3.0 (AGPL-3.0-or-later)

Copyright (C) 2025 lenardar
