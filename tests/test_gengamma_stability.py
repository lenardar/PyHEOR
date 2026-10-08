"""Generalized Gamma regressions against synthetic flexsurv 2.3.2 output.

The CSV was generated with pgengamma(..., lower.tail=FALSE) and dgengamma
on the reported parameter grid. At |Q| < 1e-5, Python uses the log-normal
limit: flexsurv's density also suffers cancellation for very small Q.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import pyheor as ph
from openpyxl import load_workbook
from pyheor.export.excel_model import _survival_formula


_REFERENCE = pd.read_csv(Path(__file__).parent / 'data/gengamma_flexsurv_2.3.2.csv')
_GROUPS = list(_REFERENCE.groupby(['sigma', 'Q'], sort=True))


@pytest.mark.parametrize('parameters,reference', _GROUPS,
                         ids=[f'sigma={sigma},Q={q}' for (sigma, q), _ in _GROUPS])
def test_against_flexsurv(parameters, reference):
    sigma, q = parameters
    t = reference['t'].to_numpy()
    curve = ph.GeneralizedGamma(mu=0.8, sigma=sigma, Q=q)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        survival = curve.survival(t)
        density = curve.pdf(t)
        hazard = curve.hazard(t)
    assert np.all(np.isfinite(survival))
    assert np.all(np.isfinite(density))
    assert np.all(np.isfinite(hazard))
    assert np.all((survival >= 0) & (survival <= 1))
    assert np.all(np.diff(survival) <= 0)
    # The limiting branch differs from R's nonzero-Q curve by O(Q).
    atol = 2e-7 if 0 < abs(q) < 1e-5 else 2e-12
    np.testing.assert_allclose(survival, reference['survival'], rtol=0, atol=atol)
    if abs(q) < 1e-5:
        expected = stats.norm.pdf((np.log(t) - 0.8) / sigma) / (sigma * t)
        np.testing.assert_allclose(density, expected, rtol=1e-13)
    else:
        rtol = 2e-6 if abs(q) < 0.005 else 5e-10
        np.testing.assert_allclose(density, reference['pdf'], rtol=rtol)
    np.testing.assert_allclose(hazard * survival, density, rtol=1e-13)
    assert float(curve.survival(t[0])) == pytest.approx(survival[0])
    assert float(curve.pdf(t[0])) == pytest.approx(density[0])


@pytest.mark.parametrize('sigma', [0.5, 1, 2])
@pytest.mark.parametrize('q', [-0.01, -0.005, 0.005, 0.01])
def test_reported_lognormal_limit(sigma, q):
    t = np.array([0.5, 1, 2, 5, 10])
    actual = ph.GeneralizedGamma(0.8, sigma, q).survival(t)
    expected = stats.norm.sf((np.log(t) - 0.8) / sigma)
    assert np.max(np.abs(actual - expected)) < 0.01


@pytest.mark.parametrize('q', [-1.01e-5, -1e-5, -0.99e-5, 0, 0.99e-5, 1e-5, 1.01e-5])
def test_density_at_limit_threshold(q):
    t = np.array([0.5, 1, 2, 5, 10])
    curve = ph.GeneralizedGamma(0.8, 1, q)
    expected = stats.norm.pdf(np.log(t) - 0.8) / t
    np.testing.assert_allclose(curve.pdf(t), expected, rtol=1e-4)


def test_upper_tail_retains_probability():
    # Q=1, sigma=1 is exponential: 1-CDF would round this to zero.
    curve = ph.GeneralizedGamma(0, 1, 1)
    assert curve.survival(50) == pytest.approx(np.exp(-50), rel=1e-13, abs=0)
    assert ph.GeneralizedGamma(0, 1, 0).survival(np.exp(10)) > 0


@pytest.mark.parametrize('q', [-0.005, 0.005])
def test_monthly_psm_life_years(q):
    cycle = ph.Cycle(1, 'month')
    model = ph.PSMModel(states=['Alive', 'Dead'], survival_endpoints=['OS'],
                        strategies=['GG', 'LN'], n_cycles=240, cycle=cycle)
    for strategy, curve in [('GG', ph.GeneralizedGamma(0.8, 1, q)),
                            ('LN', ph.SurvLogNormal(0.8, 1))]:
        model.set_survival(strategy, 'OS',
                           ph.rescale_survival(curve, from_unit='year', to_period=cycle))
    result = model.run_base_case()
    gg = result.results['GG']['total_lys']
    ln = result.results['LN']['total_lys']
    assert 3 < gg < 4
    assert gg == pytest.approx(ln, rel=0.01)


@pytest.mark.parametrize('q', [-0.005, 0, 0.005])
def test_export_uses_stable_argument_and_matching_limit(tmp_path, q):
    model = ph.PSMModel(states=['Alive', 'Dead'], survival_endpoints=['OS'],
                        strategies=['A'], n_cycles=3, cycle=ph.Cycle(1, 'month'))
    model.set_survival('A', 'OS', ph.GeneralizedGamma(0.8, 2, q))
    path = tmp_path / 'gengamma.xlsx'
    ph.export_excel_model(model, path)
    workbook = load_workbook(path)
    formulas = [cell.value for sheet in workbook for row in sheet for cell in row
                if cell.data_type == 'f' and '_xlfn.GAMMA.DIST' in cell.value]
    assert formulas
    for formula in formulas:
        assert '<1E-05' in formula
        assert 'LN(' in formula and 'EXP(' in formula
        assert 'NORM.S.DIST(-((' in formula
        assert 'TRUE)' in formula
    # Named inputs make the transformation readable and guard the exact formula
    # that originally overflowed, independent of generated cell addresses.
    formula = _survival_formula(
        dict(type='generalized_gamma', q='q', mu='mu', sigma='sigma'), 't')
    assert 'EXP(q*((LN(t)-mu)/sigma))/(q)^2' in formula
    assert 'EXP(mu+sigma*LN(q^2)/q)' not in formula
