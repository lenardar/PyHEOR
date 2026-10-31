"""Ordered PSM curve correction in calculations, plots and exports."""
import numpy as np
import pytest
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import pyheor as ph
from pyheor.plotting import _psm_occupancy_plot_data


def model_with_crossings(terminal=False):
    states = ['PF', 'PD', 'Terminal', 'Dead'] if terminal else ['PF', 'PD1', 'PD2', 'Dead']
    endpoints = ['PFS', 'OS'] if terminal else ['PFS', 'PFS2', 'OS']
    model = ph.PSMModel(states=states, survival_endpoints=endpoints, strategies=['A'],
                        cycle=ph.Cycle(1, 'year'), n_cycles=2,
                        terminal_state='Terminal' if terminal else None)
    values = [[1, .9, .8], [1, .7, .4]] if terminal else [[1, .9, .3], [1, .8, .6], [1, .7, .5]]
    for endpoint, survival in zip(endpoints, values):
        model.set_survival('A', endpoint, ph.KaplanMeier([0, 1, 2], survival))
    return model


def test_reverse_correction_preserves_os_and_reward_totals():
    model = model_with_crossings()
    model.set_state_cost('care', {'PF': 10, 'PD1': 20, 'PD2': 30})
    model.set_state_qaly('health', {'PF': .8, 'PD1': .6, 'PD2': .4})
    result = model.run_base_case().results['A']
    np.testing.assert_allclose(result['trace'], [[1, 0, 0, 0], [.7, 0, 0, .3], [.3, .2, 0, .5]], atol=1e-15)
    np.testing.assert_allclose(result['survival_curves']['OS'], [1, .7, .5])
    assert result['total_costs']['care'] == pytest.approx(15.5)
    assert result['total_qalys'] == pytest.approx(1.14)
    # The fitted input distribution is retained, without modifying its parameters.
    np.testing.assert_allclose(result['survival_distributions']['PFS'].survival([1]), [.9])


def test_terminal_counts_use_unchanged_os():
    result = model_with_crossings(terminal=True).run_base_case().results['A']
    np.testing.assert_allclose(result['trace'], [[1, 0, 0, 0], [.7, 0, .3, 0], [.4, 0, .3, .3]], atol=1e-15)
    np.testing.assert_allclose(result['trace'].sum(axis=1), 1)


def test_dense_curves_and_occupancy_match_corrected_boundaries():
    result = model_with_crossings().run_base_case()
    data = result.survival_at([0, .5, 1, 1.5, 2]).pivot(index='Time', columns='Endpoint', values='Survival')
    np.testing.assert_allclose(data['PFS'], [1, 1, .7, .7, .3])
    np.testing.assert_allclose(data['PFS2'], [1, 1, .7, .7, .5])
    times, trace, _ = _psm_occupancy_plot_data(result, 'A', 21)
    assert np.all(trace >= 0)
    np.testing.assert_allclose(trace.sum(axis=1), 1)
    np.testing.assert_allclose(trace[np.searchsorted(times, [0, 1, 2])], result.results['A']['trace'])
    fig = result.plot_survival(n_points=21)
    try:
        for line in fig.axes[0].lines:
            assert np.all((line.get_ydata() >= 0) & (line.get_ydata() <= 1))
    finally:
        plt.close(fig)


def test_crossing_between_boundaries_is_capped_on_plot_grid():
    model = ph.PSMModel(states=['PF','PD','Dead'], survival_endpoints=['PFS','OS'],
                        strategies=['A'], cycle=ph.Cycle(1,'year'), n_cycles=2)
    model.set_survival('A','PFS',ph.KaplanMeier([0,.7,1.7],[1,.6,.2]))
    model.set_survival('A','OS',ph.KaplanMeier([0,.4,1.4],[1,.7,.3]))
    result = model.run_base_case()
    _, trace, _ = _psm_occupancy_plot_data(result,'A',21)
    assert np.all(trace >= 0)
    np.testing.assert_allclose(trace.sum(axis=1), 1)


def test_invalid_raw_probability_is_not_hidden_by_minimum():
    class Invalid(ph.Exponential):
        def survival(self, t):
            return np.full_like(np.asarray(t, dtype=float), 1.1)
    model = model_with_crossings()
    model.set_survival('A', 'PFS', Invalid(.1))
    with pytest.raises(ValueError, match='outside'):
        model.run_base_case()


def test_formula_export_links_to_corrected_successor(tmp_path):
    model = model_with_crossings()
    path = tmp_path / 'corrected.xlsx'
    ph.export_excel_model(model, path)
    ws = load_workbook(path)['Calc_A']
    row = next(cell.row + 1 for row in ws for cell in row if cell.value == 'Boundary')
    # Four states occupy B:E; the three corrected endpoint curves occupy G:I.
    for r in range(row, row + 3):
        assert ws[f'G{r}'].value.startswith('=MIN(')
        assert ws[f'G{r}'].value.endswith(f',H{r})')
        assert ws[f'H{r}'].value.endswith(f',I{r})')
        assert not ws[f'I{r}'].value.startswith('=MIN(')
        assert ws[f'C{r}'].value == f'=H{r}-G{r}'
        assert ws[f'D{r}'].value == f'=I{r}-H{r}'


def test_external_excel_inputs_retain_raw_curve_for_recalculation(tmp_path):
    from pyheor.survival import SurvivalDistribution
    class ExternalCurve(SurvivalDistribution):
        def survival(self, t):
            return np.interp(t, [0, 1, 2], [1, .9, .3])
        def hazard(self, t):
            raise NotImplementedError
        def __repr__(self):
            return "ExternalCurve()"
    model = model_with_crossings()
    model.set_survival('A', 'PFS', ExternalCurve())
    path = tmp_path / 'external.xlsx'
    ph.export_excel_model(model, path)
    ws = load_workbook(path)['Calc_A']
    input_cell = next(cell for row in ws for cell in row
                      if cell.value == 'PFS external survival boundary 1')
    assert ws.cell(input_cell.row, 2).value == .9
    formulas = [cell.value for row in ws for cell in row if cell.data_type == 'f']
    assert any(f.startswith(f'=MIN($B${input_cell.row},') for f in formulas)
