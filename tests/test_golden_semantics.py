"""Hand-calculated goldens for explicit cycles and heemod reward conventions."""
import numpy as np
import pytest
import pyheor as ph


def identity(kind='markov', *, method='life-table', cycle=None, n_cycles=3, **kwargs):
    cycle = cycle or ph.Cycle(1, 'year')
    shared = dict(states=['Alive', 'Dead'], strategies=['S'], n_cycles=n_cycles,
                  cycle=cycle, method=method, **kwargs)
    if kind == 'psm':
        model = ph.PSMModel(**shared, survival_endpoints=['OS'])
        model.set_survival('S', 'OS', ph.KaplanMeier([0], [1]))
    else:
        model = (ph.MarkovModel(**shared) if kind == 'markov'
                 else ph.MicroSimModel(**shared, n_patients=4))
        model.set_transitions('S', [[1, 0], [0, 1]])
    return model


def totals(model):
    if isinstance(model, ph.MicroSimModel):
        r = model.run_base_case(seed=7, verbose=False).results['S']
        return r['mean_cost'], r['mean_qalys'], r['mean_lys']
    r = model.run_base_case().results['S']
    return sum(r['total_costs'].values()), r['total_qalys'], r['total_lys']


@pytest.mark.parametrize('kind', ['markov', 'psm', 'micro'])
@pytest.mark.parametrize('method', ['beginning', 'end', 'life-table'])
def test_monthly_cycle_costs_are_not_scaled_again(kind, method):
    m = identity(kind, cycle=ph.Cycle(1, 'month'), n_cycles=12, method=method)
    m.set_state_cost('care', {'Alive': 100})
    m.set_state_qaly('health', {'Alive': ph.qaly(.8, m.cycle)})
    assert totals(m) == pytest.approx((1200, .8, 1))


@pytest.mark.parametrize('kind', ['markov', 'psm', 'micro'])
def test_first_cycle_undiscounted_and_later_cycles_discounted(kind):
    m = identity(kind, dr_cost=.1, dr_qaly=.2)
    m.set_state_cost('care', {'Alive': 100})
    m.set_state_qaly('health', {'Alive': 1})
    assert totals(m) == pytest.approx((sum(100 / 1.1**i for i in range(3)),
                                      sum(1 / 1.2**i for i in range(3)),
                                      sum(1 / 1.2**i for i in range(3))))


@pytest.mark.parametrize('kind', ['markov', 'psm', 'micro'])
def test_starting_rewards_differ_from_first_cycle_state_rewards(kind):
    m = identity(kind, n_cycles=1)
    if kind == 'psm':
        m.set_survival('S', 'OS', ph.KaplanMeier([0, 1], [1, 0]))
    else:
        m.set_transitions('S', [[0, 1], [0, 1]])
    m.set_state_cost('care', {'Alive': 100}, cycles=1)
    m.set_starting_cost('test', 100)
    m.set_starting_qaly('ae', -.2)
    m.set_state_qaly('baseline', {'Alive': 1})
    assert totals(m) == pytest.approx((150, .3, .5))


@pytest.mark.parametrize('method,expected', [('beginning', 1), ('end', 0), ('life-table', .5)])
@pytest.mark.parametrize('kind', ['markov', 'psm', 'micro'])
def test_death_after_first_interval(method, expected, kind):
    m = identity(kind, method=method, n_cycles=1)
    if kind == 'psm':
        m.set_survival('S', 'OS', ph.KaplanMeier([0, 1], [1, 0]))
    else:
        m.set_transitions('S', [[0, 1], [0, 1]])
    m.set_state_qaly('baseline', {'Alive': 1})
    assert totals(m)[1:] == pytest.approx((expected, expected))


@pytest.mark.parametrize('kind', ['markov', 'micro'])
@pytest.mark.parametrize('method,expected', [('beginning', 100/1.1), ('end', 100), ('life-table', 50+50/1.1)])
def test_transition_rewards_follow_heemod_flow_correction(kind, method, expected):
    m = identity(kind, method=method, n_cycles=2, dr_cost=.1, dr_qaly=.1)
    m.set_transitions('S', [[0, 1], [0, 1]])
    m.set_transition_cost('procedure', 'Alive', 'Dead', 100)
    m.set_transition_qaly('procedure', 'Alive', 'Dead', -.1)
    cost, health, _ = totals(m)
    assert cost == pytest.approx(expected)
    assert health == pytest.approx(-expected / 1000)


@pytest.mark.parametrize('kind', ['markov', 'psm', 'micro'])
def test_named_components_cycle_filters_and_dynamic_params(kind):
    m = identity(kind, n_cycles=4)
    m.add_param('cost', 10, low=5, high=20)
    m.add_param('u', .8, low=.4, high=1)
    m.set_state_cost('drug', lambda p, k: {'Alive': p['cost'] * k}, cycles=[1, 3])
    m.set_state_qaly('health', lambda p, k: {'Alive': ph.qaly(p['u'], m.cycle)})
    m.set_state_qaly('ae', {'Alive': -.1}, cycles=2)
    assert totals(m)[:2] == pytest.approx((40, 3.1))
    p = m._get_base_params(); p['cost'] = 5; p['u'] = .4
    if kind != 'micro':
        r = m._simulate_single(p)['S']
        assert sum(r['total_costs'].values()) == pytest.approx(20)
        assert r['total_qalys'] == pytest.approx(1.5)


@pytest.mark.parametrize('kind', ['markov', 'psm', 'micro'])
def test_discount_conversion_expression_recomputed_for_draws(kind):
    m = identity(kind, n_cycles=2,
                 dr_cost=lambda p: ph.rescale_discount_rate(p['r'], 1, .5))
    m.add_param('r', .21, low=0, high=.44)
    m.set_state_cost('care', {'Alive': 100})
    assert totals(m)[0] == pytest.approx(100 + 100/1.1)
    if kind != 'micro':
        r = m._simulate_single({'r': .44})['S']
        assert sum(r['total_costs'].values()) == pytest.approx(100 + 100/1.2)


def test_context_flows_and_probabilities_are_explicit_and_readonly():
    m = identity(n_cycles=2)
    m.set_transitions('S', [[.8, .2], [0, 1]])
    seen = []
    def cost(ctx):
        seen.append(ctx.cycle_index)
        assert ctx.transition_matrix[0, 1] == .2
        with pytest.raises(ValueError):
            ctx.state_prev[0] = 0
        return ctx.flow('Alive', 'Dead') * 100
    m.set_custom_cost('procedure', cost)
    m.set_custom_qaly('loss', lambda ctx: -.1 * ctx.flow('Alive', 'Dead'))
    assert totals(m)[:2] == pytest.approx((36, -.036))
    assert seen == [1, 2]


def test_psm_does_not_invent_flows():
    m = identity('psm')
    with pytest.raises(ValueError, match='PSM'):
        m.set_entry_cost('rescue', 'Dead', 100)
    m.set_custom_cost('bad', lambda ctx: ctx.flow('Alive', 'Dead'))
    with pytest.raises(ValueError, match='does not identify'):
        m.run_base_case()


@pytest.mark.parametrize('cycles', [0, [0], [4], [True], [1, 1], [.5]])
def test_invalid_cycles_rejected(cycles):
    with pytest.raises((TypeError, ValueError)):
        identity().set_state_cost('care', {'Alive': 1}, cycles=cycles)


def test_removed_interfaces_fail_explicitly():
    m = identity()
    assert not hasattr(m, 'set_utility')
    with pytest.raises(TypeError):
        m.set_state_cost('care', {'Alive': 100}, first_cycle_only=True)
    with pytest.raises(TypeError):
        m.set_transition_cost('care', 'Alive', 'Dead', [100, 20])
    with pytest.raises(TypeError):
        ph.MarkovModel(['Alive', 'Dead'], ['S'], 2, cycle_length=.5)


def test_des_continuous_rates_and_event_rewards_in_months(monkeypatch):
    m = ph.DESModel(states=['Alive', 'Dead'], strategies=['S'], time_horizon=2,
                    time_unit='month', dr_cost=.1)
    m.set_event('S', 'Alive', 'Dead', ph.Exponential(1))
    monkeypatch.setattr(m, '_sample_tte', lambda dist, rng=None: 1)
    m.set_state_cost('drug', {'Alive': 100})
    m.set_state_qaly('health', {'Alive': ph.qaly(.8, duration=1, unit='month')})
    m.set_starting_cost('test', 10)
    m.set_entry_cost('eol', 'Dead', 110)
    m.set_transition_qaly('loss', 'Alive', 'Dead', -.01)
    r = m.run(n_patients=2, progress=False).results['S']
    assert r['total_cost'] == pytest.approx([10+100*(1-1/1.1)/np.log(1.1)+100]*2)
    assert r['total_qalys'] == pytest.approx([.8/12-.01]*2)
    assert r['total_lys'] == pytest.approx([1/12]*2)


def test_des_time_varying_state_rewards_are_integrated():
    m = ph.DESModel(states=['Alive','Dead'], strategies=['S'], time_horizon=2)
    m.set_state_cost('care', lambda p, time: {'Alive': 100*time})
    m.set_state_qaly('health', lambda p, time: {'Alive': .8-.1*time})
    r = m.run(n_patients=2, progress=False).results['S']
    assert r['total_cost'] == pytest.approx([200]*2)
    assert r['total_qalys'] == pytest.approx([1.4]*2)


def test_terminal_bookkeeping_with_synthetic_counts():
    # Synthetic hand-calculated example; contains no research data.
    cycle = ph.Cycle(1, 'month')
    m = ph.PSMModel(states=['PFS','PD','Terminal','Dead'],
                    survival_endpoints=['PFS','OS'],strategies=['SOC'],
                    n_cycles=2,cycle=cycle,terminal_state='Terminal')
    m.set_survival('SOC','PFS',ph.KaplanMeier([0,1,2],[1,.8,.6]))
    m.set_survival('SOC','OS',ph.KaplanMeier([0,1,2],[1,.9,.7]))
    m.set_state_cost('care',{'PFS':10,'PD':20,'Terminal':100})
    m.set_starting_cost('initial',5)
    m.set_state_qaly('health',{'PFS':ph.qaly(.6,cycle),'PD':ph.qaly(.3,cycle)})
    m.set_starting_qaly('loss',-.01)
    r=m.run_base_case().results['SOC']
    np.testing.assert_allclose(r['trace'],[[1,0,0,0],[.8,.1,.1,0],[.6,.1,.2,.1]],atol=1e-15)
    assert sum(r['total_costs'].values())==pytest.approx(44)
    assert r['total_qalys']==pytest.approx(.07375)
    assert r['total_lys']==pytest.approx(1.75/12)


@pytest.mark.parametrize('kind', ['markov', 'psm', 'micro'])
def test_psa_recomputes_qaly_survival_and_discount_conversions(kind):
    cycle = ph.Cycle(1, 'month')
    m = identity(kind, cycle=cycle, n_cycles=2,
                 dr_cost=lambda p: ph.rescale_discount_rate(p['r'], 12, 1))
    m.add_param('r', .03, dist=ph.Uniform(low=0,high=.1))
    m.add_param('u', .8, dist=ph.Uniform(low=.5,high=.9))
    m.set_state_cost('care', {'Alive': 100})
    m.set_state_qaly('health', {'Alive': lambda p,k: ph.qaly(p['u'],cycle)})
    if kind=='psm':
        m.add_param('hazard', .1, dist=ph.Uniform(low=.05,high=.2))
        m.set_survival('S','OS',lambda p: ph.rescale_survival(ph.Exponential(p['hazard']),from_unit='year',to_period=cycle))
    psa=(m.run_psa(n_outer=3,n_inner=4,seed=19,verbose=False) if kind=='micro'
         else m.run_psa(n_sim=3,seed=19,progress=False))
    for params, results in zip(psa.sampled_params,psa.psa_results):
        occupancy=np.ones(2)
        if kind=='psm':
            endpoints=np.exp(-params['hazard']*np.arange(3)/12)
            occupancy=(endpoints[:-1]+endpoints[1:])/2
        rate=(1+params['r'])**(1/12)-1
        expected_cost=100*(occupancy[0]+occupancy[1]/(1+rate))
        expected_qaly=params['u']/12*occupancy.sum()
        result=results['S']
        actual_cost=result['mean_cost'] if kind=='micro' else sum(result['total_costs'].values())
        actual_qaly=result['mean_qalys'] if kind=='micro' else result['total_qalys']
        assert actual_cost==pytest.approx(expected_cost)
        assert actual_qaly==pytest.approx(expected_qaly)


@pytest.mark.parametrize('method', ['beginning', 'end', 'life-table'])
def test_micro_vectorized_rewards_match_individual_context_path(method):
    m=identity('micro',method=method,n_cycles=3,dr_cost=.02)
    m.set_transitions('S',[[.7,.3],[0,1]])
    m.set_state_cost('care',{'Alive':lambda p,k:100*k})
    m.set_state_qaly('health',{'Alive':.8})
    m.set_starting_cost('start',30)
    m.set_starting_qaly('ae',-.02)
    m.set_entry_cost('entry','Alive',20)
    m.set_transition_cost('terminal','Alive','Dead',500)
    m.set_entry_qaly('loss','Dead',-.1)
    fast=m.run_base_case(seed=42,verbose=False).results['S']
    m.set_custom_cost('force_context',lambda ctx:0)
    slow=m.run_base_case(seed=42,verbose=False).results['S']
    for key in ('cost_history','qaly_history','ly_history','state_history'):
        np.testing.assert_allclose(fast[key],slow[key],rtol=1e-12)
