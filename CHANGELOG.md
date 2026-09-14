# Changelog

## Unreleased

### MicroSim calculation semantics

This release aligns the individual-level engine with the conventions the
cohort engines already follow. Existing microsimulation analyses should be
rerun and reviewed: totals change wherever the previous behavior counted an
extra reward period, discounted at interval starts, or left event costs
undiscounted. Deterministic transitions now reproduce the Markov totals
exactly.

- `n_cycles=N` means N reward intervals. The reward loop previously accrued
  over all N+1 observation points, returning 11 life-years for ten one-year
  intervals. Enabling half-cycle correction happened to mask this, because the
  first and last endpoint weights cancelled the extra period.
- Half-cycle correction averages the value at both interval endpoints rather
  than weighting the first and last observation by one half, so time-varying
  costs and utilities behave as they do in Markov and PSM.
- State rewards discount at interval midpoints and event costs at the end of
  the interval that produced them. Event costs were previously undiscounted.
- `discount_convention` is accepted, selecting annual-effective or
  continuously compounded discounting.
- Transition callbacks receive 0-based interval indices, matching the cost and
  utility callbacks. They previously received 1-based cycle numbers, so one
  model exposed two different meanings of `t`.
- Invalid transition matrices raise instead of being clipped and renormalised,
  matching the cohort engine.
- Multi-strategy runs draw per-patient common random numbers, so a strategy
  difference no longer carries unrelated Monte Carlo noise.
- Constructor arguments, mapping keys, handler state names, cost methods and
  `apply_cycles` are validated as they are in the other engines. An unknown
  state name in a cost or utility mapping is an error rather than a silent
  zero.
- `alive_cycles` is replaced by `time_alive`, measured in years on the same
  footing as the life-year accrual; the patient outcome column is now
  `Years Alive`.

### Report generation guards

- `generate_report` raises a clear error for a single-strategy model instead
  of failing inside `plot_tornado` partway through report generation.
- The OWSA section only runs when at least one parameter has a distribution
  or explicit bounds. `dr_cost`/`dr_qaly` are auto-registered even when the
  caller adds no parameters, so checking `model.params` alone ran OWSA (and
  surfaced the discount rate in the tornado) whether or not the caller
  intended any sensitivity analysis.
- The `_files` image directory is only created when a figure will actually
  be written to it.

### DES survival curve at large time horizons

- `DESResult.survival_curve` compared each grid point to `time_horizon` with
  `np.isclose`, whose default relative tolerance scales with the horizon's
  magnitude. At `time_horizon=1e6` that tolerance covered roughly the last
  10 units of follow-up, so a death shortly before the horizon could still
  read as at risk over several trailing grid points instead of only the
  final one. Compare against the grid's own last value instead, which is
  exactly `time_horizon` by construction.

### Excel export fixes

- Sheet names no longer collide after Excel's 31-character truncation.
  Strategy labels agreeing on their first ~25 characters previously produced
  the same sheet name, and the second strategy's data silently overwrote the
  first's.
- The Markov QALY sheet's `Time (yrs)` column now uses the same interval
  midpoint as its `Discount Factor` column; it previously used the interval
  start, contradicting the factor computed two columns over.
- `export_excel_model`'s PH/AFT wrappers resolve the baseline curve before
  writing the hazard-ratio/acceleration-factor input cell. If the baseline
  cannot be translated to a formula, the wrapper now falls back cleanly
  instead of leaving an editable cell that nothing references.
- Renamed the `(raw)`/`QALY(raw)`/`LY(raw)` columns in `export_excel_model`
  workbooks to `(occ)`/`QALY(occ)`/`LY(occ)`: they hold the half-cycle-
  corrected occupancy-weighted reward, not a pre-correction raw value, and
  `export_to_excel` uses `(raw)` for the actual pre-correction figure.
- `_write_transition_matrices` in `export_to_excel` only reflects interval 0
  of a time-varying model; documented the limitation and removed a dead
  import.

### DES consistency fixes

- `run()` and `run_psa()` accept `np.integer` for `n_patients`/`n_sim`,
  matching `PSMModel.run_psa`.
- `set_events_from()` forwards `clock` to every event it registers instead
  of always falling back to the model default.
- The arity check used to decide whether a distribution callable accepts
  patient attributes now counts every declared parameter, not just those
  without a default; `lambda p, a=None: ...` previously never received
  `attrs`.
- State-cost and utility mappings reject keys that mix state and strategy
  names, matching the cohort engines. A state name colliding with a
  strategy name previously resolved by strategy-name priority silently.
- `on_state_enter` handlers can return `{"cost": amount, "category": name}`
  to book the cost under a category of their choosing instead of always
  `"event"`, which could collide with an existing state-cost category.
- Documented that competing-event ties break by declaration order; relevant
  only to degenerate or point-mass distributions.

### OWSA with more than two strategies

- `OWSAResult.summary()`, `plot_tornado()` and `plot_owsa_param()` accept an
  explicit `intervention` argument and require one once a model has more
  than two strategies. They previously picked the first non-comparator
  strategy silently, which is ambiguous with three or more.
- A parameter missing its low or high bound now raises a clear error instead
  of a bare `StopIteration`.

### Incremental analysis: pairing, tolerance, and stale data

- `classify_incremental` accepts separate `cost_tol` and `effect_tol` instead
  of one tolerance for both currency and QALY scales. `icer()` on every
  result class now derives both from the magnitudes actually involved, the
  same way `calculate_icers` already scaled its frontier tolerance.
- PSA `icer()` methods pair each strategy's per-simulation rows by
  `sim` explicitly rather than relying on `ce_table`'s row order, matching
  `ceac_data()` and the plotting code, which already did.
- `CEAnalysis.from_result` and `.from_psa` both key strategies by their
  display label. `from_psa` previously used the internal strategy name, so
  `is_dominated()` expected different arguments depending on how the
  `CEAnalysis` was constructed from the same model.
- An extendedly dominated (`ED`) row's `ICER`, `Inc_Cost`, `Inc_QALYs` and
  `Ref` are cleared to NaN/empty instead of keeping the values computed
  before it was eliminated from the frontier, which read as a valid
  sequential ICER.

### Shared parameter handling

- `add_param`, `add_params`, discount-rate registration, the attribute
  override used by sensitivity analysis, and the Markov/PSM `run_owsa` now
  live in `models.common` rather than being copied into each engine. The
  copies had already drifted apart in their error messages.
- `run_owsa(range_pct=...)` has an effect again. `Param` fills `low` and
  `high` in for reporting, so the sweep could not tell an explicit bound from
  a default one and always used ±20%. An explicit bound still wins;
  `range_pct` now governs the parameters without one.
- `run_owsa(params=[...])` rejects unknown parameter names.
- An invalid discount rate names the argument at fault.

### Result table consistency

- `icer()` returns the same columns for every engine. `ICER` is numeric and is
  NaN whenever the quadrant admits no ratio; `ICER Classification` carries the
  readable verdict. Previously `ICER` held a formatted string on base-case
  results but a float on PSA results, and only some classes offered the
  classification column.
- `nmb()` reports `Incremental NMB` everywhere; MicroSim and DES called it
  `INMB`. The absolute columns stay engine-specific (`QALYs`/`Total Cost` for
  cohort models, `Mean QALYs`/`Mean Cost` for individual-level ones) because
  the distinction is meaningful.
- An unknown `comparator` raises instead of failing with a `KeyError` deeper
  in the call.
- `summary()`, `ceac_data()` and `plot_ceac()` on the PSA result classes no
  longer accept a `comparator`. It was parsed and discarded: a CEAC reports
  each strategy's probability of having the highest net benefit, so it is
  defined across all strategies at once.
- The four base-case result classes share one implementation of `icer()` and
  `nmb()` instead of four near-identical copies, which is what allowed the
  MicroSim variants to drift in the first place.

### Incremental analysis

- MicroSim base-case and PSA ICERs classify the incremental quadrant before
  dividing, so a strategy costing more for fewer QALYs is reported as
  `Dominated` rather than as a negative ratio. The PSA variant gained the
  `ICER Classification` column its counterparts already returned.
- Efficiency frontier comparisons use a tolerance proportional to the scale of
  the inputs. Floating-point noise no longer registers as strong dominance,
  and two strategies that cannot be told apart are labelled `EQ` instead of
  dominated.
- `calculate_icers` rejects non-finite costs or effects, naming the offending
  strategies, instead of silently ranking them last.
- ICER tornado diagrams rank a parameter whose bound is dominant or dominated
  first. The guard tested for infinity while the classifier returns NaN, so
  such a parameter sorted last and could fall outside `max_params` entirely.

### Reproducibility

- DES derives per-patient streams from a `SeedSequence` instead of seeding
  NumPy globally, so a seeded run no longer disturbs the caller's RNG state.
  Per-patient streams are used for single-strategy runs too.
- MicroSim PSA draws parameters through the seeded generator it already
  created, making `run_psa(seed=...)` reproducible.
- A `**kwargs` signature no longer counts as support for an explicit `rng`
  argument, which previously left such draws unseeded.

### Input validation

- `Normal`, `LogNormal`, `Uniform`, `Triangular` and `Dirichlet` validate their
  parameters at construction. `LogNormal(mean=-1, sd=0.5)` previously produced
  a positive-valued distribution centred near +1, because the moment match
  squares the mean.
- Vector-valued draws are rejected where a scalar parameter is expected, so
  using `Dirichlet` as a `Param` distribution explains the problem.
- `KaplanMeier` validates lengths, bounds, monotonicity and the extrapolation
  name; `PiecewiseExponential` validates rates and breakpoints.
- The default quantile widens its search bracket until it contains the root
  and returns infinity when a quantile is genuinely unreachable, such as with
  a Gompertz cure fraction. It previously searched a fixed `[0, 1e6]` interval
  and returned NaN on failure.
- `CEAnalysis` validates that its columns and PSA matrices match the strategy
  count, and pairs PSA draws by simulation id rather than by position.

### Documentation and dead code

- The PSM module docstring's state formula was reversed relative to the
  implementation (`State_k = S_{k-1} - S_k` vs. the actual `S_k - S_{k-1}`).
- Removed the unused `_return_raw`/`survival_curves_raw` path in PSM. It
  returned the same values as `survival_curves`, because the curve-crossing
  check raises rather than clamping; there was nothing left to compare
  against once that check was added.
- Removed the unused Tunnels bullet from the MicroSim module docstring; the
  feature was never implemented.
- `export_to_excel`'s docstring described a `Discounting` sheet that does not
  exist; discount factors are a column within the cost and QALY sheets.
- Removed a dead import and an unreachable branch in the Excel exporters.

### Excel export

- Weibull survival formulas parenthesise the power. Excel binds unary minus
  more tightly than `^`, so `=EXP(-(t/s)^k)` evaluated as `EXP((-(t/s))^k)`:
  `#NUM!` for fractional shapes, and a silently wrong survival probability for
  even integer ones. Proportional-hazards and accelerated-failure-time
  wrappers around a Weibull baseline inherited the defect.

### Plotting

- Importing `pyheor` no longer switches the Matplotlib backend to `Agg`, which
  silenced `plt.show()` for the caller. Report generation switches only while
  rendering and restores the previous backend.

### Shared model definitions

- `Param` and the internal cycle-based cost definition now live in
  `models.common`, so PSM, MicroSim, and DES no longer depend on the Markov
  module for model-independent concepts. Use the public `pyheor.Param` API;
  model-specific module aliases such as `pyheor.models.markov.Param` have
  been removed before the first stable release.

### DES clock semantics

- `DESModel(clock="reset")` preserves the existing state-entry clock.
- `clock="forward"` samples residual event times from cumulative hazards at
  absolute study time, allowing calendar-time risk without a new event API.
- `set_event(..., clock=...)` can override the model default for individual
  transitions, allowing clock-forward and clock-reset risks in one model.
- Invalid horizons, patient counts, attribute lengths, and event TTEs now fail
  explicitly instead of producing non-finite results or silently recycling data.
- DES state-entry handlers now follow the existing MicroSim contract and record
  returned one-time costs at the actual entry time.
- DES multi-strategy runs now use per-patient common random-number streams, so
  strategy differences are not inflated by unrelated random-stream offsets.
- DES supports explicit annual-effective (`"discrete"`) and continuously
  compounded (`"continuous"`) discounting for both lump sums and continuous
  state accruals; invalid rates fail before simulation.
- DES base-case and PSA ICERs classify incremental quadrants before division,
  and survival curves keep right-censored patients in the risk set at the
  study horizon.
- DES rejects empty or duplicate state/strategy definitions, invalid
  `state_type` values, unknown mapping keys, and missing parameter references.
- DES accepts an explicit fixed `initial_state`; initial entry effects occur at
  time zero, and unsupported self-loops or runaway zero-time cycles fail
  explicitly.

### Consistent figure layout

- Plot defaults now use an installed CJK-capable font when available and are
  scoped to each plotting call, so they do not alter unrelated Matplotlib work.
- Strategy colours are stable across PSA, CEAC, convergence, survival, and
  multi-strategy plots; the palette no longer repeats after six strategies.
- Long multi-strategy Markov, PSM, and microsimulation traces use compact
  multi-row panels with one shared legend instead of squeezing legends into
  every subplot. Model-structure diagrams grow vertically with strategy count.
- Monetary plots accept `currency` for their axis labels and tick formatting.

### Tornado plot readability

- Show 10 parameters by default, wrap long labels, and size automatic-height
  plots according to label line counts. Installed CJK fonts are used when available.
- Parameter bound annotations are opt-in (`show_values=True`) and appear in
  separate columns. Low/high scenario markers preserve their input direction
  even when a higher parameter produces a lower outcome.
- Draw each endpoint range once, including scenarios on the same side of the
  base case. Plot style changes stay local to the figure.
- `label_width`, `font_family`, and `currency` allow label and unit customization.

### Follow-up calculation review

- Excel retains the original state cost rate when its first applicable
  interval is later than zero, and honors cost/QALY discount-rate overrides.
- Export-time choices (discount convention, interval count, HCC and initial
  state) are marked as fixed metadata rather than yellow editable inputs.
  Change these in Python and regenerate the workbook.
- PSA and OWSA use the same incremental-quadrant classification as base case.
  ICER tornado plots reject non-numeric scenarios and direct users to NMB.
- Non-finite state costs and utilities raise an error with strategy and interval.
- Beta/Gamma reject zero, negative and non-finite standard deviations. Use
  `Fixed(mean)` to represent zero uncertainty; no artificial variance is added.

This release tightens the calculation semantics for Markov and PSM models.
Existing analyses should be rerun and reviewed because totals may change where
the previous behavior counted an extra reward period or used ambiguous timing.

### Calculation semantics

- `n_cycles=N` now means N time intervals. Traces and survival tables contain
  N+1 observation points, while cost, QALY, and LY arrays contain N intervals.
- Time-dependent callbacks receive 0-based interval indices: `0` through
  `N-1`.
- Half-cycle correction uses one explicit name, `"trapezoidal"`; the former
  `"life-table"` compatibility alias has been removed.
- State rewards accrue at interval midpoints, transition/custom event costs at
  interval ends, and `method="starting"` costs at time zero.
- Discounting can use annual-effective (`"discrete"`) or continuously
  compounded (`"continuous"`) rates.

### Error handling

- Invalid transition matrices, non-finite values, impossible PSM curves, and
  curve crossings now raise contextual errors instead of being clipped or
  repaired.
- Dominant and dominated ICER quadrants are classified before division, so a
  negative numeric ICER is not presented as an ordinary ratio.
- Report generation no longer silently drops a failed table or chart.

### Excel review model

- `export_to_excel()` is a result-data export. It no longer claims that static
  values are an independently recalculable model.
- `export_excel_model()` builds editable Excel calculation chains for Markov
  traces, time-varying transition matrices, state and transition costs, cost
  schedules, PSM survival curves, QALYs, LYs, discounting, and ICERs.
- Workbook validation formulas display `ERROR` when edited probabilities,
  matrix sums, survival ordering, or state probabilities become invalid.
- Python callbacks that cannot be translated faithfully fail explicitly.

MicroSim calculation behavior is unchanged in this release; DES changes are
listed above under the follow-up calculation review.
