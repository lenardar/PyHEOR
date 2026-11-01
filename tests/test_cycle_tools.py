"""Unit conversions and the documented natural-scale R parameterizations."""
import numpy as np
import pytest
import pyheor as ph
from scipy.stats import gamma, lognorm


def test_explicit_duration_and_qaly_units():
    month = ph.Cycle(1, 'month')
    assert month.years == 1/12
    assert month.in_unit('month') == 1
    assert month.time(12, 'year') == 1
    assert ph.qaly(.8, month) == pytest.approx(.8/12)
    assert ph.qaly(-.1, 3, 'week') == pytest.approx(-.3/52)
    np.testing.assert_allclose(ph.qaly([.8, .4], month), [.8/12, .4/12])
    with pytest.raises(ValueError): ph.qaly(.8, 1)
    with pytest.raises(ValueError): ph.qaly(.8, month, 'year')


@pytest.mark.parametrize('length,unit', [(0,'year'),(-1,'month'),(np.inf,'day'),(True,'year'),(1,'fortnight')])
def test_invalid_cycle(length, unit):
    with pytest.raises(ValueError): ph.Cycle(length, unit)


def test_effective_discount_rate_preserves_yearly_discount():
    monthly = ph.rescale_discount_rate(.045, ph.Cycle(1,'year'), ph.Cycle(1,'month'))
    assert (1+monthly)**12 == pytest.approx(1.045)
    assert ph.rescale_discount_rate(monthly, 1, 12) == pytest.approx(.045)
    with pytest.raises(TypeError): ph.rescale_discount_rate(.03, 1, ph.Cycle(1,'month'))
    with pytest.raises(ValueError): ph.rescale_discount_rate(-1, 12, 1)


def test_survival_scaling_includes_hazard_and_quantile():
    curve = ph.rescale_survival(ph.Exponential(rate=.2), from_unit='year', to_period=ph.Cycle(1,'month'))
    assert curve.survival(12) == pytest.approx(np.exp(-.2))
    assert curve.hazard(12) == pytest.approx(.2/12)
    assert curve.quantile(.5) == pytest.approx(np.log(2)/.2*12)


@pytest.mark.parametrize('name,parameters,expected', [
    ('exp', dict(rate=.2), lambda t: np.exp(-.2*t)),
    ('weibull', dict(shape=1.3,scale=4), lambda t: np.exp(-(t/4)**1.3)),
    ('weibullPH',dict(shape=1.3,scale=.2), lambda t: np.exp(-.2*t**1.3)),
    ('llogis',dict(shape=1.3,scale=4), lambda t: 1/(1+(t/4)**1.3)),
    ('lnorm',dict(meanlog=.7,sdlog=.4), lambda t: lognorm.sf(t,s=.4,scale=np.exp(.7))),
    ('gompertz',dict(shape=.1,rate=.2), lambda t: np.exp(-.2*np.expm1(.1*t)/.1)),
    ('gengamma.orig',dict(shape=.5,scale=4,k=3), lambda t: gamma.sf((t/4)**.5,a=3)),
    ('gengamma',dict(mu=.7,sigma=.4,Q=0), lambda t: lognorm.sf(t,s=.4,scale=np.exp(.7))),
])
def test_r_parameterizations_against_independent_cdfs(name, parameters, expected):
    t=np.array([.01,.5,1,3,8])
    np.testing.assert_allclose(ph.from_flexsurv(name,**parameters).survival(t), expected(t), rtol=1e-12)


def test_r_adapter_rejects_ambiguous_parameters():
    with pytest.raises(ValueError): ph.from_flexsurv('weibull',shape=2,rate=1)
    with pytest.raises(ValueError): ph.from_flexsurv('excel_weibull',shape=2,scale=1)
    with pytest.raises(ValueError): ph.from_flexsurv('gengamma.orig',shape=0,scale=1,k=2)


@pytest.mark.parametrize('utility', [0, True, -.2, .8, np.float32(.8),
                                    np.float64(.8), np.int64(1), np.array(.8)])
@pytest.mark.parametrize('numeric_duration', [False, True])
def test_qaly_scalar_and_array_paths_agree(utility, numeric_duration):
    args = (3, 'month') if numeric_duration else (ph.Cycle(3, 'month'),)
    result = ph.qaly(utility, *args)
    expected = ph.qaly(np.array([utility], dtype=float), *args)[0]
    assert type(result) is float
    assert result == expected


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('wrap', [float, np.float32, np.asarray,
                                 lambda x: [0, x], lambda x: [[x], [0]]])
def test_qaly_rejects_nonfinite_scalar_and_array_utilities(value, wrap):
    with pytest.raises(ValueError, match='Utility must be finite'):
        ph.qaly(wrap(value), ph.Cycle(1, 'month'))


def test_qaly_preserves_array_shape_and_input():
    values = np.array([[.8, -.2], [0, 1]])
    before = values.copy()
    result = ph.qaly(values, 3, 'month')
    np.testing.assert_array_equal(result, values * .25)
    np.testing.assert_array_equal(values, before)
    assert result.shape == values.shape
    assert ph.qaly(np.empty((0, 2)), ph.Cycle(1, 'month')).shape == (0, 2)
