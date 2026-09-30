"""Curve displays preserve model nodes, probability constraints and real steps."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
import pyheor as ph
from pyheor.plotting import _occupancy_display, _psm_occupancy_plot_data


def test_joint_occupancy_interpolation_preserves_nodes_bounds_and_direction():
    times=np.array([0.,1.,2.,4.])
    trace=np.array([[1,0,0],[.7,.2,.1],[.3,.4,.3],[0,.2,.8]])
    original=trace.copy()
    x,y=_occupancy_display(times,trace,smooth=True,n_points=501)
    np.testing.assert_equal(y[np.searchsorted(x,times)],trace)
    np.testing.assert_allclose(y.sum(axis=1),1,atol=1e-15)
    assert np.all((y>=0)&(y<=1))
    assert np.all(np.diff(y[:,0])<=1e-15)
    assert np.all(np.diff(y[:,-1])>=-1e-15)
    for i in range(len(times)-1):
        section=y[(x>=times[i])&(x<=times[i+1])]
        assert np.all(section>=np.minimum(trace[i],trace[i+1])-1e-15)
        assert np.all(section<=np.maximum(trace[i],trace[i+1])+1e-15)
    np.testing.assert_equal(trace,original)


def test_constant_state_does_not_flatten_other_state_curves():
    times=np.array([0.,1.,2.])
    trace=np.array([[.8,.2,0],[.5,.2,.3],[.3,.2,.5]])
    x,y=_occupancy_display(times,trace,smooth=True,n_points=2001)
    i=np.searchsorted(x,1.)
    before=(y[i]-y[i-1])/(x[i]-x[i-1])
    after=(y[i+1]-y[i])/(x[i+1]-x[i])
    np.testing.assert_allclose(before,after,atol=.001)
    assert before[0]<-.1 and after[2]>.1
    np.testing.assert_allclose(y[:,1],.2,atol=1e-15)


@pytest.mark.parametrize('engine',[ph.MarkovModel,ph.MicroSimModel])
@pytest.mark.parametrize('smooth',[True,False])
def test_cycle_trace_display_retains_original_data(engine,smooth):
    options={'n_patients':100} if engine is ph.MicroSimModel else {}
    m=engine(states=['Alive','Dead'],strategies=['SOC'],cycle=ph.Cycle(1,'year'),
             n_cycles=3,**options)
    m.set_transitions('SOC',[[.7,.3],[0,1]])
    result=m.run_base_case(seed=42,verbose=False) if options else m.run_base_case()
    original=result.results['SOC']['trace'].copy()
    summary=result.summary().copy()
    fig=result.plot_trace(smooth=smooth,n_points=501,**({'style':'line'} if not options else {}))
    try:
        x=fig.axes[0].lines[0].get_xdata()
        values=np.column_stack([line.get_ydata() for line in fig.axes[0].lines])
        np.testing.assert_equal(values[np.searchsorted(x,np.arange(4))],original)
        np.testing.assert_allclose(values.sum(axis=1),1,atol=1e-15)
        assert len(x)>4 if smooth else len(x)==4
        np.testing.assert_equal(result.results['SOC']['trace'],original)
        assert result.summary().equals(summary)
    finally:plt.close(fig)


def partition_result(terminal=False,empirical=False):
    states=['PF','PD','Terminal','Dead'] if terminal else ['PF','PD','Dead']
    m=ph.PSMModel(states=states,survival_endpoints=['PFS','OS'],strategies=['SOC'],
                  cycle=ph.Cycle(1,'month'),n_cycles=3,
                  terminal_state='Terminal' if terminal else None)
    if empirical:
        m.set_survival('SOC','PFS',ph.KaplanMeier([0,.37,1.23],[1,.6,.2]))
        m.set_survival('SOC','OS',ph.KaplanMeier([0,.51,1.71],[1,.8,.4]))
    else:
        m.set_survival('SOC','PFS',ph.Exponential(.4))
        m.set_survival('SOC','OS',ph.Exponential(.2))
    return m.run_base_case()


@pytest.mark.parametrize('terminal',[False,True])
def test_psm_trace_and_area_share_valid_dense_partitions(terminal):
    result=partition_result(terminal=terminal)
    original=result.results['SOC']['trace'].copy()
    summary=result.summary().copy()
    x,y,style=_psm_occupancy_plot_data(result,'SOC',501)
    assert len(x)>4 and style=='default'
    np.testing.assert_allclose(y.sum(axis=1),1,atol=1e-15)
    np.testing.assert_allclose(y[np.searchsorted(x,np.arange(4)/12)],original,atol=1e-15)
    if not terminal:
        np.testing.assert_allclose(y[:,0],np.exp(-.4*x*12))
        np.testing.assert_allclose(y[:,1],np.exp(-.2*x*12)-np.exp(-.4*x*12))
    fig=result.plot_trace()
    area=result.plot_state_area()
    try:
        for j,ax in enumerate(fig.axes):
            np.testing.assert_allclose(ax.lines[0].get_ydata(),y[:,j])
        assert len(area.axes[0].collections)==result.model.n_states
        np.testing.assert_equal(result.results['SOC']['trace'],original)
        assert result.summary().equals(summary)
    finally:
        plt.close(fig)
        plt.close(area)


@pytest.mark.parametrize('terminal',[False,True])
def test_psm_empirical_occupancy_keeps_steps(terminal):
    result=partition_result(terminal=terminal,empirical=True)
    x,y,style=_psm_occupancy_plot_data(result,'SOC',501)
    assert style=='steps-post'
    np.testing.assert_allclose(y.sum(axis=1),1,atol=1e-15)
    if not terminal:
        assert np.any(np.isclose(x,.37/12))
        assert np.any(np.isclose(x,.51/12))
    else:
        np.testing.assert_equal(y,result.results['SOC']['trace'])
    fig=result.plot_trace()
    try:
        assert all(ax.lines[0].get_drawstyle()=='steps-post' for ax in fig.axes)
    finally:plt.close(fig)


def test_psm_raw_trace_retains_cycle_nodes():
    result=partition_result()
    fig=result.plot_trace(smooth=False)
    try:
        for j,ax in enumerate(fig.axes):
            np.testing.assert_equal(ax.lines[0].get_ydata(),result.results['SOC']['trace'][:,j])
    finally:plt.close(fig)


def test_nmb_and_evpi_displays_retain_calculated_values_and_corners():
    costs=np.array([[0,10,25],[0,20,30],[0,30,45]],dtype=float)
    qalys=np.array([[1,2,3],[1,2.5,3],[1,2,4]],dtype=float)
    cea=ph.CEAnalysis(['A','B','C'],costs.mean(axis=0),qalys.mean(axis=0),
                      psa_costs=costs,psa_qalys=qalys)
    expected_nmb=cea.nmb_curve(wtp_range=(0,50),n_wtp=501)
    expected_evpi=cea.evpi(wtp_range=(0,50),n_wtp=501)
    nmb=cea.plot_nmb_curve(wtp_range=(0,50))
    evpi=cea.plot_evpi(wtp_range=(0,50),population=1000)
    try:
        for i,strategy in enumerate(cea.strategies):
            np.testing.assert_equal(nmb.axes[0].lines[i].get_ydata(),expected_nmb[strategy])
        np.testing.assert_equal(evpi.axes[0].lines[0].get_ydata(),expected_evpi.EVPI)
        np.testing.assert_equal(evpi.axes[1].lines[0].get_ydata(),1000*expected_evpi.EVPI)
        assert np.all(expected_evpi.EVPI>=-1e-12)
    finally:
        plt.close(nmb)
        plt.close(evpi)
