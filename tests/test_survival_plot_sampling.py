"""Plot fitted curves directly; retain empirical steps and actual PSA data."""
import numpy as np
import pytest
import matplotlib.pyplot as plt
import pyheor as ph
from pyheor.plotting import plot_psm_comparison


def psm(curves):
    m=ph.PSMModel(states=['Alive','Dead'],survival_endpoints=['OS'],strategies=list(curves),
                  cycle=ph.Cycle(1,'month'),n_cycles=2)
    for arm,curve in curves.items():m.set_survival(arm,'OS',curve)
    return m.run_base_case()


def test_dense_survival_uses_the_fitted_curve_and_retains_base_totals():
    result=psm({'SOC':ph.Exponential(.4)})
    total=result.summary().copy()
    fig=result.plot_survival(n_points=501)
    try:
        line=fig.axes[0].lines[0]
        assert len(line.get_xdata())==501
        np.testing.assert_allclose(line.get_ydata(),np.exp(-.4*line.get_xdata()*12))
        assert result.summary().equals(total)
        assert len(result.survival_data)==3
        values=result.survival_at([0,1,2],unit='month')
        np.testing.assert_allclose(values.Survival,np.exp(-.4*np.arange(3)))
    finally:plt.close(fig)


def test_kaplan_meier_preserves_off_grid_jump_times_and_steps():
    result=psm({'SOC':ph.KaplanMeier([0,.37,1.23],[1,.8,.5])})
    fig=result.plot_survival(n_points=10)
    try:
        line=fig.axes[0].lines[0]
        assert line.get_drawstyle()=='steps-post'
        assert any(np.isclose(line.get_xdata(),.37/12))
        assert any(np.isclose(line.get_xdata(),1.23/12))
        assert set(line.get_ydata())=={1,.8,.5}
    finally:plt.close(fig)


def test_comparison_uses_a_common_grid_for_different_step_curves():
    result=psm({'SOC':ph.KaplanMeier([0,.37],[1,.8]),'TRT':ph.KaplanMeier([0,.51],[1,.9])})
    fig=plot_psm_comparison(result,'OS',n_points=10)
    try:
        first,second=fig.axes[0].lines
        np.testing.assert_equal(first.get_xdata(),second.get_xdata())
    finally:plt.close(fig)


def test_des_empirical_survival_labels_declared_time_unit():
    m=ph.DESModel(states=['Alive','Dead'],strategies=['SOC'],time_horizon=2,time_unit='month')
    result=m.run(n_patients=2,seed=42,progress=False)
    fig=result.plot_survival()
    try:
        assert fig.axes[0].get_xlabel()=='Time (months)'
        assert fig.axes[0].lines[0].get_drawstyle()=='steps-post'
    finally:plt.close(fig)


def test_ceac_plot_uses_unsmoothed_monte_carlo_probabilities():
    m=psm({'SOC':ph.Exponential(.4),'TRT':ph.Exponential(.3)}).model
    m.add_param('cost',100,dist=ph.Uniform(low=50,high=150))
    m.set_state_cost('care',{'SOC':{'Alive':50},'TRT':{'Alive':'cost'}})
    m.set_state_qaly('health',{'Alive':ph.qaly(.8,m.cycle)})
    psa=m.run_psa(n_sim=10,seed=42,progress=False)
    expected=psa.ceac_data(n_wtp=501)
    fig=psa.plot_ceac(smooth=False)
    try:
        for arm,line in zip(m.strategy_names,fig.axes[0].lines):
            np.testing.assert_equal(line.get_ydata(),expected[expected.strategy==arm]['Prob CE'])
    finally:plt.close(fig)


def test_des_default_survival_uses_actual_event_times():
    m=ph.DESModel(states=['Alive','Dead'],strategies=['SOC'],time_horizon=2,time_unit='month')
    m.set_event('SOC','Alive','Dead',ph.KaplanMeier([0,.37],[1,0]))
    result=m.run(n_patients=2,seed=42,progress=False)
    curve=result.survival_curve()
    np.testing.assert_allclose(curve.Time,[0,.37,2])
    np.testing.assert_allclose(curve.Survival,[1,0,0])


def test_micro_default_smooth_curve_preserves_cycle_points_and_monotonicity():
    m=ph.MicroSimModel(states=['Alive','Dead'],strategies=['SOC'],
                      n_cycles=8,n_patients=200,cycle=ph.Cycle(1,'year'))
    m.set_transitions('SOC',[[.8,.2],[0,1]])
    result=m.run_base_case(seed=42,verbose=False)
    original=result.survival_curve().copy()
    summary=result.summary().copy()
    fig=result.plot_survival()
    try:
        line=fig.axes[0].lines[0]
        times,probabilities=line.get_xdata(),line.get_ydata()
        assert line.get_drawstyle()=='default'
        assert len(times)>len(original)
        indices=np.searchsorted(times,original.Time)
        np.testing.assert_equal(times[indices],original.Time)
        np.testing.assert_allclose(probabilities[indices],original.Survival,atol=1e-15)
        assert np.all(np.diff(probabilities)<=1e-14)
        assert np.all((probabilities>=0)&(probabilities<=1))
        assert result.survival_curve().equals(original)
        assert result.summary().equals(summary)
    finally:plt.close(fig)


@pytest.mark.parametrize('style,drawstyle',[('line','default'),('step','steps-post')])
def test_micro_explicit_survival_styles(style,drawstyle):
    m=ph.MicroSimModel(states=['Alive','Dead'],strategies=['SOC'],
                      n_cycles=3,n_patients=20,cycle=ph.Cycle(1,'year'))
    m.set_transitions('SOC',[[.8,.2],[0,1]])
    result=m.run_base_case(seed=42,verbose=False)
    fig=result.plot_survival(style=style)
    try:
        line=fig.axes[0].lines[0]
        assert line.get_drawstyle()==drawstyle
        assert len(line.get_xdata())==4
    finally:plt.close(fig)


def test_ceac_smoothing_preserves_probability_constraints_and_raw_data():
    from pyheor.plotting import _ceac_display
    times=np.linspace(0,100,101)
    probability=np.where(times<50,1.0,0.0)
    values=np.column_stack([probability,1-probability])
    original=values.copy()
    display,smoothed=_ceac_display(times,values,smooth=True,bandwidth=3)
    assert len(display)>len(times)
    assert np.all((smoothed>=0)&(smoothed<=1))
    np.testing.assert_allclose(smoothed.sum(axis=1),1,atol=1e-15)
    assert np.max(np.abs(np.diff(smoothed[:,0])))<.01
    assert np.all(np.diff(smoothed[:,0])<=1e-14)
    np.testing.assert_equal(values,original)


def test_multistrategy_ceac_display_keeps_probabilities_summing_to_one():
    from pyheor.plotting import _ceac_display
    times=np.arange(10,dtype=float)
    values=np.array([[1,0,0],[1,0,0],[.8,.2,0],[.5,.5,0],[.3,.6,.1],
                     [.2,.6,.2],[.1,.5,.4],[0,.4,.6],[0,.3,.7],[0,.2,.8]])
    _,smoothed=_ceac_display(times,values,smooth=True,bandwidth=.5)
    np.testing.assert_allclose(smoothed.sum(axis=1),1,atol=1e-15)
    assert np.all((smoothed>=0)&(smoothed<=1))


def test_ceac_plot_marks_display_smoothing_and_does_not_change_analysis():
    m=psm({'SOC':ph.Exponential(.4),'TRT':ph.Exponential(.3)}).model
    m.add_param('cost',100,dist=ph.Uniform(low=50,high=150))
    m.set_state_cost('care',{'SOC':{'Alive':50},'TRT':{'Alive':'cost'}})
    m.set_state_qaly('health',{'Alive':ph.qaly(.8,m.cycle)})
    psa=m.run_psa(n_sim=10,seed=42,progress=False)
    original=psa.ceac_data().copy()
    summary=psa.summary().copy()
    fig=psa.plot_ceac()
    try:
        assert 'smoothed display' in fig.axes[0].get_title()
        assert len(fig.axes[0].lines[0].get_xdata())==2001
        assert psa.ceac_data().equals(original)
        assert psa.summary().equals(summary)
    finally:plt.close(fig)


@pytest.mark.parametrize('smooth',[False,True])
def test_ceaf_display_preserves_raw_analysis_and_strategy_choice(smooth):
    costs=np.array([[0,10,25],[0,20,30],[0,30,45]],dtype=float)
    qalys=np.array([[1,2,3],[1,2.5,3],[1,2,4]],dtype=float)
    cea=ph.CEAnalysis(['A','B','C'],costs.mean(axis=0),qalys.mean(axis=0),
                      psa_costs=costs,psa_qalys=qalys)
    original=cea.ceaf(wtp_range=(0,50),n_wtp=101)
    fig=cea.plot_ceaf(wtp_range=(0,50),n_wtp=101,smooth=smooth)
    try:
        line=fig.axes[0].lines[-1]
        if not smooth:
            np.testing.assert_equal(line.get_ydata(),original.CEAF)
        else:
            assert 'smoothed display' in fig.axes[0].get_title()
            assert len(line.get_xdata())==2001
            assert np.all((line.get_ydata()>=0)&(line.get_ydata()<=1))
        assert cea.ceaf(wtp_range=(0,50),n_wtp=101).equals(original)
    finally:plt.close(fig)
