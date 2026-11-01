"""Process PSA preserves complete draws, ordering and validation."""
import numpy as np
import pandas as pd
import pytest
import pyheor as ph


def build(kind):
    args = dict(states=['Alive', 'Dead'], strategies=['A', 'B'],
                n_cycles=4, cycle=ph.Cycle(1, 'month'),
                dr_cost=ph.Param(.01, dist=ph.Beta(mean=.01, sd=.001)))
    if kind == 'psm':
        # Local classes and lambdas must survive both model and result transfer.
        class LocalExponential(ph.Exponential):
            pass
        m = ph.PSMModel(**args, survival_endpoints=['OS'])
        for strategy in m.strategy_names:
            m.set_survival(strategy, 'OS', lambda p: LocalExponential(p['rate']))
    else:
        m = ph.MarkovModel(**args)
        for strategy in m.strategy_names:
            m.set_transitions(strategy, lambda p, t: [[1-p['rate'], p['rate']], [0, 1]])
    m.add_param('rate', .1, dist=ph.Beta(mean=.1, sd=.01))
    m.set_state_cost('care', {'Alive': lambda p, t: 10 * t})
    m.set_state_qaly('health', {'Alive': ph.qaly(.8, m.cycle)})
    return m


def assert_same(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            if key == 'survival_distributions':
                for endpoint in left[key]:
                    np.testing.assert_array_equal(left[key][endpoint].survival([0, 1, 2]),
                                                  right[key][endpoint].survival([0, 1, 2]))
            else:
                assert_same(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize('kind', ['markov', 'psm'])
@pytest.mark.parametrize('jobs', [2, 4])
def test_parallel_matches_serial_complete_results(kind, jobs):
    m = build(kind)
    rate = m.dr_cost
    serial = m.run_psa(25, seed=123, progress=False)
    parallel = m.run_psa(25, seed=123, progress=False, n_jobs=jobs)
    assert_same(serial.sampled_params, parallel.sampled_params)
    assert_same(serial.psa_results, parallel.psa_results)
    pd.testing.assert_frame_equal(serial.ce_table, parallel.ce_table, check_exact=True)
    assert m.dr_cost == rate


@pytest.mark.parametrize('kind', ['markov', 'psm'])
@pytest.mark.parametrize('jobs', [True, False, 0, -1, 1.5, '2'])
def test_invalid_worker_count(kind, jobs):
    with pytest.raises(ValueError, match='n_jobs must be a positive integer'):
        build(kind).run_psa(2, n_jobs=jobs, progress=False)


@pytest.mark.parametrize('kind', ['markov', 'psm'])
def test_worker_errors_propagate_and_model_can_run_again(kind):
    m = build(kind)
    m.set_state_cost('care', {'Alive': lambda p, t: np.nan if t == 2 else 1})
    with pytest.raises(ValueError, match='Reward must be finite'):
        m.run_psa(2, seed=42, n_jobs=2, progress=False)
    m.set_state_cost('care', {'Alive': 1})
    assert m.run_psa(1, seed=42, n_jobs=2, progress=False).n_sim == 1


@pytest.mark.parametrize('kind', ['markov', 'psm'])
def test_serial_default_does_not_create_workers(kind, monkeypatch):
    from concurrent.futures import ProcessPoolExecutor
    def fail(*args, **kwargs):
        raise AssertionError('Serial PSA must not create a process pool')
    monkeypatch.setattr(ProcessPoolExecutor, '__init__', fail)
    m = build(kind)
    assert_same(m.run_psa(2, seed=42, progress=False).psa_results,
                m.run_psa(2, seed=42, n_jobs=1, progress=False).psa_results)


@pytest.mark.parametrize('kind', ['markov', 'psm'])
def test_legacy_sampling_keeps_rng_order_and_caller_state(kind, capsys):
    class LegacyDistribution(ph.Distribution):
        def sample(self, n=1):
            return np.random.uniform(.05, .15, size=n)
        def __repr__(self):
            return 'LegacyDistribution()'
    m = build(kind)
    m.params['rate'].dist = LegacyDistribution()
    state = np.random.get_state()
    serial = m.run_psa(9, seed=77, progress=False)
    parallel = m.run_psa(9, seed=77, n_jobs=np.int64(2), progress=True)
    assert_same(serial.sampled_params, parallel.sampled_params)
    assert_same(serial.psa_results, parallel.psa_results)
    current = np.random.get_state()
    assert state[0] == current[0]
    np.testing.assert_array_equal(state[1], current[1])
    assert state[2:] == current[2:]
    assert 'PSA complete: 9 simulations' in capsys.readouterr().out
