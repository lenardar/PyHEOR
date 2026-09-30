"""Tests that distinguish result exports from auditable Excel models."""
import pyheor as _ph
from tests.input_helpers import _method, _cycle_values, _starting_values



from pathlib import Path
import pytest
from openpyxl import load_workbook
from pyheor import Exponential, GeneralizedGamma, KaplanMeier, MarkovModel, PSMModel, PiecewiseExponential, export_excel_model, export_to_excel

def _markov_model(n_cycles=2, hcc=True):
    model = MarkovModel(states=['Alive', 'Dead'], strategies=['S1'], n_cycles=n_cycles, method=_method(hcc), dr_cost=0.03, dr_qaly=0.03, cycle=_ph.Cycle(1, 'year'))
    model.set_transitions('S1', [[1, 0], [0, 1]])
    model.set_state_cost('care', _cycle_values(model, {'Alive': 100, 'Dead': 0}))
    model.set_state_qaly('health', _cycle_values(model, {'Alive': 1, 'Dead': 0}))
    return model

def _formula_cells(worksheet):
    return [cell.value for row in worksheet.iter_rows() for cell in row if isinstance(cell.value, str) and cell.value.startswith('=')]

def test_result_export_has_one_row_per_interval(tmp_path):
    model = _markov_model(n_cycles=2)
    path = tmp_path / 'results.xlsx'
    export_to_excel(model.run_base_case(), path)
    workbook = load_workbook(path, data_only=False)
    costs = workbook['Costs_S1']
    assert costs['A2'].value == 1
    assert costs['A3'].value == 2
    assert costs['A4'].value == 'TOTAL'

def test_result_export_rejects_formula_claim(tmp_path):
    path = tmp_path / 'not-a-model.xlsx'
    with pytest.raises(ValueError, match='export_excel_model'):
        export_to_excel(_markov_model().run_base_case(), path, include_formulas=True)
    assert not path.exists()

def test_markov_audit_workbook_has_editable_cycle_inputs(tmp_path):
    path = tmp_path / "markov-model.xlsx"
    export_excel_model(_markov_model(), path)
    workbook = load_workbook(path)
    calculation = workbook["Calc_S1"]
    assert workbook.calculation.calcMode == "auto"
    assert workbook.calculation.fullCalcOnLoad
    assert calculation["B4"].value == "life-table"
    assert calculation["B7"].data_type == "f"
    assert any('IF($B$4="life-table"' in f for f in _formula_cells(calculation))


def test_time_varying_markov_expands_one_based_matrix_inputs(tmp_path):
    model = _markov_model()
    model.set_transitions("S1", lambda p, k: [[1-k*.1, k*.1], [0, 1]])
    path = tmp_path / "dynamic.xlsx"
    export_excel_model(model, path)
    ws = load_workbook(path)["Calc_S1"]
    inputs = {row[0].value: row[1].value for row in ws.iter_rows()}
    assert inputs["P cycle 1 / Alive"] == pytest.approx(.9)
    assert inputs["P cycle 2 / Alive"] == pytest.approx(.8)


def test_callable_transition_reward_expands_cycle_values(tmp_path):
    model = _markov_model()
    model.set_transition_cost("event", "Alive", "Dead", lambda p, k: 100/k)
    path = tmp_path / "events.xlsx"
    export_excel_model(model, path)
    ws = load_workbook(path)["Calc_S1"]
    inputs = {row[0].value: row[1].value for row in ws.iter_rows()}
    assert inputs["cost event / event / cycle 1"] == 100
    assert inputs["cost event / event / cycle 2"] == 50


def test_custom_reward_is_rejected_before_execution(tmp_path):
    model = _markov_model()
    def unsupported(ctx):
        raise AssertionError("callback must not run during failed export")
    model.set_custom_cost("custom", unsupported)
    path = tmp_path / "custom.xlsx"
    with pytest.raises(NotImplementedError, match="custom reward"):
        export_excel_model(model, path)
    assert not path.exists()


def test_psm_parametric_survival_uses_excel_formulas(tmp_path):
    model = PSMModel(states=['Alive', 'Dead'], survival_endpoints=['OS'], strategies=['S1'], n_cycles=2, cycle=_ph.Cycle(1, 'year'))
    model.set_survival('S1', 'OS', Exponential(rate=0.1))
    model.set_state_cost('care', _cycle_values(model, {'Alive': 100, 'Dead': 0}))
    model.set_state_qaly('health', _cycle_values(model, {'Alive': 1, 'Dead': 0}))
    path = tmp_path / 'psm-model.xlsx'
    export_excel_model(model, path)
    calculation = load_workbook(path, data_only=False)['Calc_S1']
    formulas = _formula_cells(calculation)
    section_cells = [cell.value for row in calculation.iter_rows() for cell in row]
    assert 'OS / Exponential rate' in section_cells
    assert any(('EXP(-' in formula for formula in formulas))
    assert not any((formula.startswith('=MAX(') for formula in formulas))

@pytest.mark.parametrize(('curve', 'formula_token'), [(GeneralizedGamma(mu=0, sigma=1, Q=1), '_xlfn.GAMMA.DIST'), (PiecewiseExponential(breakpoints=[1], rates=[0.1, 0.2]), 'MIN('), (KaplanMeier(times=[0, 1, 2], survival_probs=[1, 0.9, 0.8]), 'LOOKUP(')])
def test_psm_library_survival_types_use_excel_formulas(tmp_path, curve, formula_token):
    model = PSMModel(states=['Alive', 'Dead'], survival_endpoints=['OS'], strategies=['S1'], n_cycles=2, cycle=_ph.Cycle(1, 'year'))
    model.set_survival('S1', 'OS', curve)
    path = tmp_path / 'psm-library-curve.xlsx'
    export_excel_model(model, path)
    calculation = load_workbook(path, data_only=False)['Calc_S1']
    formulas = _formula_cells(calculation)
    assert any((formula_token in formula for formula in formulas))

class _UnsupportedCurve:
    """A survival curve export_excel_model cannot translate to a formula."""

    def survival(self, t):
        import numpy as np
        return np.exp(-0.1 * np.asarray(t, dtype=float))

    def hazard(self, t):
        return 0.1

def test_ph_wrapping_an_unsupported_baseline_leaves_no_orphan_input(tmp_path):
    from pyheor.survival import ProportionalHazards
    model = PSMModel(states=['Alive', 'Dead'], survival_endpoints=['OS'], strategies=['S1'], n_cycles=2, cycle=_ph.Cycle(1, 'year'))
    model.set_survival('S1', 'OS', ProportionalHazards(_UnsupportedCurve(), hr=0.8))
    path = tmp_path / 'ph-unsupported.xlsx'
    export_excel_model(model, path)
    calculation = load_workbook(path, data_only=False)['Calc_S1']
    labels = [cell.value for row in calculation.iter_rows() for cell in row if isinstance(cell.value, str)]
    assert not any(('hazard ratio' in label for label in labels))

def test_colliding_sheet_names_do_not_overwrite_each_other(tmp_path):
    long_prefix = 'Adjuvant chemotherapy plus targeted maintenance therapy'
    model = MarkovModel(states=['Alive', 'Dead'], strategies={'A': f'{long_prefix} A', 'B': f'{long_prefix} B'}, n_cycles=2, cycle=_ph.Cycle(1, 'year'))
    for strategy in ('A', 'B'):
        model.set_transitions(strategy, [[1, 0], [0, 1]])
    result = model.run_base_case()
    path = tmp_path / 'collision.xlsx'
    export_to_excel(result, path)
    sheet_names = load_workbook(path).sheetnames
    trace_sheets = [name for name in sheet_names if name.startswith('Trace_')]
    assert len(trace_sheets) == len(set(trace_sheets)) == 2

def test_colliding_calc_sheet_names_in_excel_model(tmp_path):
    long_prefix = 'Adjuvant chemotherapy plus targeted maintenance therapy'
    model = MarkovModel(states=['Alive', 'Dead'], strategies={'A': f'{long_prefix} A', 'B': f'{long_prefix} B'}, n_cycles=2, cycle=_ph.Cycle(1, 'year'))
    for strategy in ('A', 'B'):
        model.set_transitions(strategy, [[1, 0], [0, 1]])
    path = tmp_path / 'collision-model.xlsx'
    export_excel_model(model, path)
    sheet_names = load_workbook(path).sheetnames
    calc_sheets = [name for name in sheet_names if name.startswith('Calc_')]
    assert len(calc_sheets) == len(set(calc_sheets)) == 2
