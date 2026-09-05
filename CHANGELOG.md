# Changelog

## Unreleased

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
- `"life-table"` is a compatibility alias for trapezoidal half-cycle
  correction; both average adjacent trace observations for each interval.
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
