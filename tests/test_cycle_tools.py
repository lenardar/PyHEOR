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
