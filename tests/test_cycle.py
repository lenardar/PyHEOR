"""Cycle length and time unit handling shared by the cohort and individual engines."""

import warnings

import numpy as np
import pytest

import pyheor as ph
from pyheor.utils import Cycle, normalize_time_unit, resolve_cycle


def _markov(**kwargs):
    model = ph.MarkovModel(
        states=["Alive", "Dead"], strategies=["S"], n_cycles=12,
        dr_cost=0.035, **kwargs,
    )
    model.set_transitions("S", lambda p, t: [[ph.C, 0.01], [0, 1]])
    model.set_state_cost("c", {"Alive": 1200.0})
    return model


class TestCycle:
    @pytest.mark.parametrize("unit, years", [
        ("year", 1.0),
        ("month", 1 / 12),
        ("week", 7 / 365.25),
        ("day", 1 / 365.25),
    ])
    def test_unit_conversion_to_years(self, unit, years):
        assert Cycle(1, unit).years == pytest.approx(years)

    @pytest.mark.parametrize("alias, canonical", [
        ("Months", "month"), ("mo", "month"), ("yr", "year"), ("Y", "year"),
        ("weeks", "week"), ("wk", "week"), ("days", "day"), ("d", "day"),
    ])
    def test_unit_aliases(self, alias, canonical):
        assert normalize_time_unit(alias) == canonical

    def test_unknown_unit_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown time unit"):
            Cycle(1, "fortnight")

    @pytest.mark.parametrize("length", [0, -1, np.nan, np.inf])
    def test_invalid_length_is_rejected(self, length):
        with pytest.raises(ValueError, match="cycle_length"):
            Cycle(length, "month")

    def test_boolean_length_is_rejected(self):
        with pytest.raises(TypeError, match="cycle_length"):
            Cycle(True, "month")

    @pytest.mark.parametrize("text, expected", [
        ("1 month", Cycle(1, "month")),
        ("3 months", Cycle(3, "month")),
        ("week", Cycle(1, "week")),
        ("0.5 year", Cycle(0.5, "year")),
    ])
    def test_parse(self, text, expected):
        assert Cycle.parse(text) == expected

    @pytest.mark.parametrize("text", ["", "1 2 3", "abc month", "month 1"])
    def test_parse_rejects_malformed_text(self, text):
        with pytest.raises(ValueError):
            Cycle.parse(text)

    def test_string_form(self):
        assert str(Cycle(1, "month")) == "1 month"
        assert str(Cycle(4, "weeks")) == "4 weeks"
        assert str(Cycle(0.5, "year")) == "0.5 years"


class TestResolveCycle:
    def test_bare_number_is_years_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_cycle(0.5) == Cycle(0.5, "year")

    def test_number_with_unit(self):
        assert resolve_cycle(2, "weeks") == Cycle(2, "week")

    def test_unit_alone_means_one_unit(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_cycle(None, "month") == Cycle(1, "month")

    def test_omitted_warns(self):
        with pytest.warns(FutureWarning, match="cycle_length was not specified"):
            assert resolve_cycle() == Cycle(1, "year")

    def test_unit_cannot_be_combined_with_self_describing_forms(self):
        with pytest.raises(ValueError, match="time_unit"):
            resolve_cycle("1 month", "year")
        with pytest.raises(ValueError, match="time_unit"):
            resolve_cycle(Cycle(1, "month"), "year")


class TestModels:
    def test_string_and_number_forms_agree(self):
        by_string = _markov(cycle_length="1 month").run_base_case()
        by_unit = _markov(cycle_length=1, time_unit="month").run_base_case()
        by_years = _markov(cycle_length=1 / 12).run_base_case()
        by_object = _markov(cycle_length=Cycle(1, "month")).run_base_case()
        reference = by_years.summary()["Total Cost"].iloc[0]
        for result in (by_string, by_unit, by_object):
            assert result.summary()["Total Cost"].iloc[0] == pytest.approx(reference)

    def test_attributes_expose_specification(self):
        model = _markov(cycle_length="4 weeks")
        assert model.time_unit == "week"
        assert model.cycle == Cycle(4, "week")
        assert model.cycle_length == pytest.approx(4 * 7 / 365.25)

    def test_omitted_cycle_length_warns_and_uses_one_year(self):
        with pytest.warns(FutureWarning, match="cycle_length was not specified"):
            model = _markov()
        assert model.cycle_length == 1.0
        assert model.time_unit == "year"

    def test_explicit_cycle_length_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _markov(cycle_length="1 year")

    def test_info_reports_the_unit(self):
        assert "12 × 1 month" in _markov(cycle_length="1 month").info()

    def test_monthly_costs_are_annual_rates(self):
        yearly = _markov(cycle_length="1 year").run_base_case()
        monthly = _markov(cycle_length="1 month").run_base_case()
        undiscounted = 1200.0
        assert yearly.summary()["Total Cost"].iloc[0] < 12 * undiscounted
        assert monthly.summary()["Total Cost"].iloc[0] < undiscounted

    def test_psm_and_microsim_accept_units(self):
        psm = ph.PSMModel(
            states=["PFS", "Prog", "Dead"], survival_endpoints=["PFS", "OS"],
            strategies=["S"], n_cycles=12, cycle_length="1 month",
        )
        micro = ph.MicroSimModel(
            states=["Alive", "Dead"], strategies=["S"], n_cycles=12,
            n_patients=10, cycle_length=1, time_unit="month",
        )
        assert psm.cycle == micro.cycle == Cycle(1, "month")

    def test_public_export(self):
        assert ph.Cycle is Cycle
