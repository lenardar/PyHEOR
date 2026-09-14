"""Scale-aware tolerance in classify_incremental and its callers."""

import numpy as np
import pytest

from pyheor import DESModel, MarkovModel
from pyheor.analysis.comparison import CEAnalysis
from pyheor.analysis.results import classify_incremental


class TestClassifyIncremental:
    def test_default_tolerance_is_symmetric_and_absolute(self):
        # classify_incremental alone has no notion of "dollars" vs. "QALYs";
        # a caller with scale context (e.g. total cost/QALY levels) is
        # expected to pass cost_tol/effect_tol explicitly, which the
        # StrategyOutcomeResult and PSA icer() methods do.
        value, _ = classify_incremental(100.0, 1e-9)
        assert value == pytest.approx(1e11)

    def test_explicit_effect_tolerance_treats_the_effect_as_unchanged(self):
        # A 1e-9 QALY gap is noise once the caller supplies a tolerance on
        # the QALY scale; the $100 cost gap is still real, so the strategy
        # is Dominated (costlier, no better) rather than assigned a 1e11
        # ICER as if the tiny effect gap were meaningful.
        value, label = classify_incremental(100.0, 1e-9, effect_tol=1e-6)
        assert np.isnan(value)
        assert label == "Dominated"

    def test_default_tolerance_matches_previous_behaviour(self):
        # cost_tol/effect_tol default to 1e-10 when the caller passes no
        # scale, so classify_incremental(delta_cost, delta_effect) alone is
        # unchanged for callers that have not been updated.
        value, label = classify_incremental(100.0, 0.5)
        assert value == pytest.approx(200.0)
        assert label == "200"

    def test_genuine_small_cost_difference_is_still_dominant(self):
        value, label = classify_incremental(1e-11, 0.5, cost_tol=1e-6)
        assert np.isnan(value)
        assert label == "Dominant"


class TestRealisticScaleNoiseIsAbsorbed:
    """CRN can leave a strategy pair 1e-9-scale apart on an O(1e5) budget."""

    def test_des_common_random_numbers_leave_negligible_qaly_noise(self):
        model = DESModel(
            states=["Alive", "Dead"], strategies=["A", "B"], time_horizon=10.0
        )
        from pyheor.survival import Exponential
        for strategy in ("A", "B"):
            model.set_event(strategy, "Alive", "Dead", Exponential(rate=0.1))
        model.set_state_cost("care", {"Alive": 10000})
        model.set_utility({"Alive": 1.0})

        result = model.run(n_patients=500, seed=1, progress=False)
        row = result.icer().iloc[0]

        # Identical strategies under CRN: any leftover difference is
        # floating-point noise, not a real 1e10-scale ICER.
        assert row["ICER Classification"] == "No difference"
        assert np.isnan(row["ICER"])


class TestPsaIcerToleranceAndPairing:
    def build_ce_table(self, sim_ids, soc_cost, trt_cost, soc_qaly, trt_qaly):
        import pandas as pd
        rows = []
        for sim, sc, tc, sq, tq in zip(sim_ids, soc_cost, trt_cost, soc_qaly, trt_qaly):
            rows.append({'sim': sim, 'strategy': 'SOC', 'total_cost': sc, 'qalys': sq})
            rows.append({'sim': sim, 'strategy': 'TRT', 'total_cost': tc, 'qalys': tq})
        return pd.DataFrame(rows)

    def test_icer_is_insensitive_to_row_order(self):
        from pyheor.analysis.results import _paired_psa_icer

        ordered = self.build_ce_table(
            [1, 2, 3], [100, 110, 90], [150, 140, 160], [1.0, 1.1, 0.9], [1.5, 1.4, 1.6]
        )
        shuffled = ordered.sample(frac=1, random_state=0).reset_index(drop=True)

        _, _, cost_a, qaly_a = _paired_psa_icer(ordered, 'TRT', 'SOC')
        _, _, cost_b, qaly_b = _paired_psa_icer(shuffled, 'TRT', 'SOC')

        np.testing.assert_array_equal(np.sort(cost_a), np.sort(cost_b))
        np.testing.assert_array_equal(np.sort(qaly_a), np.sort(qaly_b))
        assert cost_a.mean() == pytest.approx(cost_b.mean())


class TestCeaFactoryConsistency:
    """from_result and from_psa must agree on what identifies a strategy."""

    def test_both_factories_use_display_labels(self):
        strategies = {"soc": "Standard of Care", "trt": "New Treatment"}

        class FakeResult:
            def summary(self):
                import pandas as pd
                return pd.DataFrame([
                    {"Strategy": "Standard of Care", "Total Cost": 100.0, "QALYs": 1.0},
                    {"Strategy": "New Treatment", "Total Cost": 200.0, "QALYs": 1.5},
                ])

        class FakePsaResult:
            model = None

            @property
            def ce_table(self):
                import pandas as pd
                rows = []
                for sim in range(1, 4):
                    for name, label, cost, qaly in (
                        ("soc", "Standard of Care", 100.0, 1.0),
                        ("trt", "New Treatment", 200.0, 1.5),
                    ):
                        rows.append({
                            "sim": sim, "strategy": name,
                            "strategy_label": label,
                            "total_cost": cost, "qalys": qaly,
                        })
                return pd.DataFrame(rows)

        cea_det = CEAnalysis.from_result(FakeResult())
        cea_psa = CEAnalysis.from_psa(FakePsaResult())

        assert cea_det.strategies == cea_psa.strategies == [
            "Standard of Care", "New Treatment"
        ]
        assert cea_det.is_dominated("New Treatment") == cea_psa.is_dominated(
            "New Treatment"
        )
