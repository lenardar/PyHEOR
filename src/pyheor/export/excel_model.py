"""Auditable cycle-reward Excel models; Python callbacks expand to explicit inputs."""
import numpy as np
from openpyxl import Workbook
from openpyxl.utils import get_column_letter as CL
from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.workbook.properties import CalcProperties
from .excel import _unique_sheet_name

_HEADER_FONT = Font(bold=True)
_NOTE_FONT = Font(italic=True, color="999999")
_INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
_FMT_PROB = "0.000000"


def export_excel_model(model_or_result, filepath, params=None):
    from ..models.markov import MarkovModel
    from ..models.psm import PSMModel
    model = getattr(model_or_result, "model", model_or_result)
    if not isinstance(model, (MarkovModel, PSMModel)):
        raise TypeError("Excel formula models support MarkovModel and PSMModel")
    if any(model._custom_rewards.values()):
        raise NotImplementedError("Excel cannot translate custom reward callbacks")
    params = (dict(model_or_result.params) if hasattr(model_or_result, "results")
              else dict(model._get_base_params(), **(params or {})))
    with model._attr_param_override(params):
        results = model._simulate_single(params)
        wb = Workbook()
        wb.calculation = CalcProperties(fullCalcOnLoad=True, forceFullCalc=True, calcMode="auto")
        summary = wb.active
        summary.title = "Summary"
        summary.append(["Strategy", "Excel Cost", "Excel QALYs", "Python Cost", "Python QALYs",
                        "Cost Difference", "QALY Difference", "ICER vs first", "Classification"])
        used = {"Summary"}
        for index, strategy in enumerate(model.strategy_names, 2):
            ws = wb.create_sheet(_unique_sheet_name(f"Calc_{model.strategy_labels[strategy]}", used))
            cost_ref, qaly_ref = _build_cycle_sheet(ws, model, strategy, params)
            reference = "'" + ws.title.replace("'", "''") + "'!"
            result = results[strategy]
            summary.append([model.strategy_labels[strategy], f"={reference}{cost_ref}",
                            f"={reference}{qaly_ref}", sum(result["total_costs"].values()),
                            result["total_qalys"], f"=B{index}-D{index}", f"=C{index}-E{index}"])
            if index > 2:
                dc, dq = f"(B{index}-$B$2)", f"(C{index}-$C$2)"
                summary.cell(index, 8, f'=IF(OR(ABS({dq})<1E-10,AND({dc}<0,{dq}>0),AND({dc}>0,{dq}<0)),"",{dc}/{dq})')
                summary.cell(index, 9, f'=IF(ABS({dq})<1E-10,"No effect difference",IF(AND({dc}<=0,{dq}>0),"Dominant",IF(AND({dc}>=0,{dq}<0),"Dominated","Trade-off")))')
        for ws in wb:
            ws.freeze_panes = "B2"
            ws.column_dimensions["A"].width = 32
            for col in range(2, min(ws.max_column, 20) + 1):
                ws.column_dimensions[CL(col)].width = 18
        wb.save(filepath)


def _build_cycle_sheet(ws, model, strategy, params):
    from ..models.markov import MarkovModel
    is_markov = isinstance(model, MarkovModel)
    N, S = model.n_cycles, model.n_states
    ws.append(["Cycle length (years)", model.cycle.years])
    ws.append(["Cost discount rate per cycle", model.dr_cost])
    ws.append(["QALY discount rate per cycle", model.dr_qaly])
    ws.append(["Occupancy method", model.method])
    validation = DataValidation(type="list", formula1='"beginning,end,life-table"')
    ws.add_data_validation(validation)
    validation.add(ws["B4"])
    ws.append(["Input convention", "State costs and QALYs are per cycle; first cycle undiscounted"])
    ws.append(["Dynamic inputs", "Callbacks expanded per cycle; changing source params requires re-export"])
    ws.append(["Total cost"])
    ws.append(["Total QALYs"])
    for cell in ("B1", "B2", "B3", "B4"):
        ws[cell].fill = _INPUT_FILL
    row = 11
    def input_row(label, values):
        nonlocal row
        ws.cell(row, 1, label)
        refs = []
        for j, value in enumerate(values, 2):
            ws.cell(row, j, float(value)).fill = _INPUT_FILL
            refs.append(f"${CL(j)}${row}")
        row += 1
        return refs
    matrices = []
    if is_markov:
        for k in range(1, N + 1):
            P = model._get_transition_matrix(strategy, params, k)
            matrices.append([input_row(f"P cycle {k} / {state}", P[j]) for j, state in enumerate(model.states)])
    else:
        specs, external = [], []
        values = model._resolve_survival_values(strategy, params)
        for j, endpoint in enumerate(model.survival_endpoints):
            curve = model._resolve_curve(strategy, endpoint, params)
            spec, row = _write_survival_curve_inputs(ws, row, endpoint, curve)
            specs.append(spec)
            external.append([input_row(f"{endpoint} external survival boundary {k}", [values[k, j]])[0]
                             for k in range(N + 1)] if spec is None else None)
    state_inputs = {}
    for kind, definitions in (("cost", model._costs), ("qaly", model._qalys)):
        state_inputs[kind] = {category: [input_row(f"{kind} / {category} / cycle {k}",
                            model._state_reward_vector(d, strategy, params, k)) for k in range(1, N + 1)]
                              for category, d in definitions.items()}
    starting = {kind: {c: input_row(f"Starting {kind} / {c}", [model._scalar_reward(v, strategy, params, 0)])[0]
                      for c, v in model._starting_rewards[kind].items()} for kind in ("cost", "qaly")}
    events = {kind: [(c, source, target, [input_row(f"{kind} event / {c} / cycle {k}",
                      [model._scalar_reward(v, strategy, params, k)])[0] for k in range(1, N + 1)])
                     for c, source, target, v in model._event_rewards[kind]] for kind in ("cost", "qaly")}
    trace_start = row + 3
    ws.cell(trace_start - 1, 1, "Boundary")
    for j, name in enumerate(model.states, 2):
        ws.cell(trace_start - 1, j, name)
    trace = []
    curves = []
    for k in range(N + 1):
        r = trace_start + k
        ws.cell(r, 1, k)
        refs = [f"{CL(j + 2)}{r}" for j in range(S)]
        trace.append(refs)
        if is_markov:
            for j in range(S):
                if k == 0:
                    ws.cell(r, j + 2, int(j == model.initial_state_idx))
                else:
                    terms = [f"{trace[k-1][source]}*{matrices[k-1][source][j]}" for source in range(S)]
                    ws.cell(r, j + 2, "=" + "+".join(terms))
        else:
            curve_refs = []
            for j, spec in enumerate(specs):
                col = S + 3 + j
                ref = f"{CL(col)}{r}"
                curve_refs.append(ref)
                ws.cell(r, col, _survival_formula(spec, f"A{r}") if spec is not None else f"={external[j][k]}")
            curves.append(curve_refs)
            ws.cell(r, 2, f"={curve_refs[0]}")
            for j in range(1, len(specs)):
                ws.cell(r, j + 2, f"={curve_refs[j]}-{curve_refs[j-1]}")
            if model.terminal_state is None:
                ws.cell(r, S + 1, f"=1-{curve_refs[-1]}")
            else:
                ws.cell(r, S, 0 if k == 0 else f"={curves[k-1][-1]}-{curve_refs[-1]}")
                ws.cell(r, S + 1, 0 if k == 0 else f"=1-{curves[k-1][-1]}")
    row = trace_start + N + 4
    raw_flows, corrected = [], []
    if is_markov and any(events.values()):
        for i in range(N):
            block = []
            for source in range(S):
                ws.cell(row, 1, f"Raw flow cycle {i+1} / {model.states[source]}")
                refs = []
                for target in range(S):
                    ws.cell(row, target + 2, f"={trace[i][source]}*{matrices[i][source][target]}")
                    refs.append(f"{CL(target+2)}{row}")
                block.append(refs)
                row += 1
            raw_flows.append(block)
        for i in range(N):
            block = []
            for source in range(S):
                ws.cell(row, 1, f"Corrected flow cycle {i+1} / {model.states[source]}")
                refs = []
                for target in range(S):
                    previous = raw_flows[i-1][source][target] if i else "0"
                    current = raw_flows[i][source][target]
                    ws.cell(row, target+2, f'=IF($B$4="life-table",({previous}+{current})/2,IF($B$4="beginning",{previous},{current}))')
                    refs.append(f"{CL(target+2)}{row}")
                block.append(refs)
                row += 1
            corrected.append(block)
    row += 3
    cost_refs, qaly_refs = [], []
    for i in range(N):
        ws.cell(row, 1, i + 1)
        occ = []
        for j in range(S):
            prev, curr = trace[i][j], trace[i+1][j]
            ws.cell(row, j+2, f'=IF($B$4="life-table",({prev}+{curr})/2,IF($B$4="beginning",{prev},{curr}))')
            occ.append(f"{CL(j+2)}{row}")
        col = S + 3
        for kind in ("cost", "qaly"):
            terms = []
            for category, inputs in state_inputs[kind].items():
                terms.append("(" + "+".join(f"{occ[j]}*{inputs[i][j]}" for j in range(S)) + ")")
            if i == 0:
                terms.extend(starting[kind].values())
            for category, source, target, inputs in events[kind]:
                j = model.states.index(target)
                if source is None:
                    weight = trace[0][j] if i == 0 else "(" + "+".join(corrected[i][s][j] for s in range(S) if s != j) + ")"
                else:
                    weight = corrected[i][model.states.index(source)][j]
                terms.append(f"{weight}*{inputs[i]}")
            rate = "$B$2" if kind == "cost" else "$B$3"
            ws.cell(row, col, "=(" + ("+".join(terms) or "0") + f")/(1+{rate})^({i})")
            (cost_refs if kind == "cost" else qaly_refs).append(f"{CL(col)}{row}")
            col += 1
        row += 1
    ws['B7'] = "=" + "+".join(cost_refs)
    ws['B8'] = "=" + "+".join(qaly_refs)
    return "B7", "B8"


def _write_survival_curve_inputs(ws, row, endpoint, curve):
    """Write supported curve parameters and return a formula specification."""
    from ..survival import (
        AcceleratedFailureTime,
        Exponential,
        GeneralizedGamma,
        Gompertz,
        KaplanMeier,
        LogLogistic,
        PiecewiseExponential,
        ProportionalHazards,
        SurvLogNormal,
        Weibull,
    )

    def write_param(label, value):
        nonlocal row
        ws.cell(row, 1, label)
        cell = ws.cell(row, 2, value)
        cell.fill = _INPUT_FILL
        ref = f"$B${row}"
        row += 1
        return ref

    from ..survival_tools import ScaledSurvival
    prefix = endpoint
    if isinstance(curve, ScaledSurvival):
        base, row = _write_survival_curve_inputs(ws, row, f"{prefix} / baseline", curve.baseline)
        if base is None:
            return None, row
        factor = write_param(f"{prefix} / Time scaling factor", curve.factor)
        return {"type": "scaled", "baseline": base, "factor": factor}, row
    if isinstance(curve, ProportionalHazards):
        # Resolve the baseline first: writing the HR cell before knowing
        # whether the baseline is representable would otherwise leave an
        # editable cell that no formula ever references.
        base, row = _write_survival_curve_inputs(
            ws, row, f"{prefix} / baseline", curve.baseline,
        )
        if base is None:
            return None, row
        hr_ref = write_param(f"{prefix} / PH hazard ratio", curve.hr)
        return {"type": "ph", "baseline": base, "hr": hr_ref}, row
    if isinstance(curve, AcceleratedFailureTime):
        base, row = _write_survival_curve_inputs(
            ws, row, f"{prefix} / baseline", curve.baseline,
        )
        if base is None:
            return None, row
        af_ref = write_param(f"{prefix} / AFT acceleration factor", curve.af)
        return {"type": "aft", "baseline": base, "af": af_ref}, row
    if isinstance(curve, Exponential):
        return {
            "type": "exponential",
            "rate": write_param(f"{prefix} / Exponential rate", curve.rate),
        }, row
    if isinstance(curve, Weibull):
        return {
            "type": "weibull",
            "shape": write_param(f"{prefix} / Weibull shape", curve.shape),
            "scale": write_param(f"{prefix} / Weibull scale", curve.scale),
        }, row
    if isinstance(curve, LogLogistic):
        return {
            "type": "loglogistic",
            "shape": write_param(f"{prefix} / Log-logistic shape", curve.shape),
            "scale": write_param(f"{prefix} / Log-logistic scale", curve.scale),
        }, row
    if isinstance(curve, SurvLogNormal):
        return {
            "type": "lognormal",
            "meanlog": write_param(f"{prefix} / Log-normal meanlog", curve.meanlog),
            "sdlog": write_param(f"{prefix} / Log-normal sdlog", curve.sdlog),
        }, row
    if isinstance(curve, Gompertz):
        return {
            "type": "gompertz",
            "shape": write_param(f"{prefix} / Gompertz shape", curve.shape),
            "rate": write_param(f"{prefix} / Gompertz rate", curve.rate),
        }, row
    if isinstance(curve, GeneralizedGamma):
        return {
            "type": "generalized_gamma",
            "mu": write_param(f"{prefix} / Generalized gamma mu", curve.mu),
            "sigma": write_param(
                f"{prefix} / Generalized gamma sigma", curve.sigma,
            ),
            "q": write_param(f"{prefix} / Generalized gamma Q", curve.Q),
        }, row
    if isinstance(curve, PiecewiseExponential):
        breakpoints = [
            write_param(f"{prefix} / Breakpoint {i + 1}", float(value))
            for i, value in enumerate(curve.breakpoints)
        ]
        rates = [
            write_param(f"{prefix} / Rate {i + 1}", float(value))
            for i, value in enumerate(curve.rates)
        ]
        return {
            "type": "piecewise_exponential",
            "breakpoints": breakpoints,
            "rates": rates,
        }, row
    if isinstance(curve, KaplanMeier):
        ws.cell(row, 1, f"{prefix} / Kaplan-Meier data")
        ws.cell(row, 2, "Time").font = _HEADER_FONT
        ws.cell(row, 3, "Survival").font = _HEADER_FONT
        row += 1
        first_data_row = row
        for time, survival in zip(curve.times, curve.surv):
            for column, value in ((2, time), (3, survival)):
                cell = ws.cell(row, column, float(value))
                cell.fill = _INPUT_FILL
                cell.number_format = _FMT_PROB
            row += 1
        last_data_row = row - 1
        tail_rate = None
        if curve.extrapolation == "exponential":
            tail_rate = write_param(
                f"{prefix} / Exponential tail rate", curve._tail_rate,
            )
        return {
            "type": "kaplan_meier",
            "times": f"$B${first_data_row}:$B${last_data_row}",
            "survival": f"$C${first_data_row}:$C${last_data_row}",
            "last_time": f"$B${last_data_row}",
            "last_survival": f"$C${last_data_row}",
            "extrapolation": curve.extrapolation,
            "tail_rate": tail_rate,
        }, row

    ws.cell(row, 1, f"{prefix} / {type(curve).__name__}")
    ws.cell(row, 2, "External survival inputs below").font = _NOTE_FONT
    return None, row + 1


def _survival_formula(spec, time_ref):
    """Translate a supported survival specification into one Excel formula."""
    kind = spec["type"]
    if kind == "scaled":
        return _survival_formula(spec["baseline"], f"({time_ref}*{spec['factor']})")
    if kind == "ph":
        base = _survival_formula(spec["baseline"], time_ref)[1:]
        return f"=({base})^{spec['hr']}"
    if kind == "aft":
        return _survival_formula(
            spec["baseline"], f"({time_ref}/{spec['af']})",
        )
    if kind == "exponential":
        return f"=EXP(-{spec['rate']}*{time_ref})"
    if kind == "weibull":
        # Excel binds unary minus tighter than '^', so the power needs its own
        # parentheses or '-(t/s)^k' would evaluate as '(-(t/s))^k'.
        return f"=EXP(-(({time_ref}/{spec['scale']})^{spec['shape']}))"
    if kind == "loglogistic":
        return f"=1/(1+({time_ref}/{spec['scale']})^{spec['shape']})"
    if kind == "lognormal":
        return (
            f"=IF({time_ref}=0,1,1-_xlfn.NORM.S.DIST("
            f"(LN({time_ref})-{spec['meanlog']})/{spec['sdlog']},TRUE))"
        )
    if kind == "gompertz":
        return (
            f"=IF(ABS({spec['shape']})<1E-12,"
            f"EXP(-{spec['rate']}*{time_ref}),"
            f"EXP(-{spec['rate']}/{spec['shape']}*"
            f"(EXP({spec['shape']}*{time_ref})-1)))"
        )
    if kind == "generalized_gamma":
        q = spec["q"]
        mu = spec["mu"]
        sigma = spec["sigma"]
        gamma_scale = f"EXP({mu}+{sigma}*LN({q}^2)/{q})"
        u = f"({time_ref}/({gamma_scale}))^({q}/{sigma})"
        gamma_cdf = f"_xlfn.GAMMA.DIST({u},1/({q}^2),1,TRUE)"
        lognormal = (
            f"1-_xlfn.NORM.S.DIST((LN({time_ref})-{mu})/{sigma},TRUE)"
        )
        return (
            f"=IF({time_ref}=0,1,IF(ABS({q})<1E-10,{lognormal},"
            f"IF({q}>0,1-{gamma_cdf},{gamma_cdf})))"
        )
    if kind == "piecewise_exponential":
        terms = []
        previous = "0"
        for index, rate in enumerate(spec["rates"]):
            if index < len(spec["breakpoints"]):
                breakpoint = spec["breakpoints"][index]
                duration = f"MAX(0,MIN({time_ref},{breakpoint})-{previous})"
                previous = breakpoint
            else:
                duration = f"MAX(0,{time_ref}-{previous})"
            terms.append(f"{rate}*{duration}")
        return f"=EXP(-({'+'.join(terms)}))"
    if kind == "kaplan_meier":
        if spec["extrapolation"] == "exponential":
            beyond = f"EXP(-{spec['tail_rate']}*{time_ref})"
        else:
            beyond = spec["last_survival"]
        return (
            f"=IF({time_ref}>{spec['last_time']},{beyond},"
            f"LOOKUP({time_ref},{spec['times']},{spec['survival']}))"
        )
    raise ValueError(f"Unsupported survival formula specification: {kind!r}")
