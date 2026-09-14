"""Operator-precedence safety for generated Excel formulas.

Excel binds unary minus tighter than ``^``, so ``-(a)^b`` evaluates as
``(-(a))^b``. With a fractional exponent that yields ``#NUM!``; with an even
integer exponent it silently returns the wrong magnitude and sign. Formulas
must therefore wrap the power explicitly: ``-((a)^b)``.

Python has the opposite precedence (``-a**b`` is ``-(a**b)``), so a formula
that looks correct when read as Python can still be wrong in Excel. These
tests check the generated strings directly instead of evaluating them.
"""

import pytest
from openpyxl import load_workbook

from pyheor import MarkovModel, PSMModel, C
from pyheor.export.excel_model import export_excel_model
from pyheor.survival import (
    AcceleratedFailureTime,
    Exponential,
    GeneralizedGamma,
    Gompertz,
    KaplanMeier,
    LogLogistic,
    PiecewiseExponential,
    ProportionalHazards,
    SurvLogNormal,
    Weibull,
)

OPERATORS = set("(+-*/^,=<>")


def negated_groups_raised_to_a_power(formula):
    """Return every ``-(...)^`` substring in an Excel formula."""
    hits = []
    for start, char in enumerate(formula):
        if char != "-" or formula[start + 1:start + 2] != "(":
            continue
        before = formula[:start].rstrip()
        if before and before[-1] not in OPERATORS:
            continue  # binary minus, not negation
        depth = 0
        for index in range(start + 1, len(formula)):
            if formula[index] == "(":
                depth += 1
            elif formula[index] == ")":
                depth -= 1
                if depth == 0:
                    if formula[index + 1:].lstrip().startswith("^"):
                        hits.append(formula[start:index + 2])
                    break
    return hits


def formulas_in(path):
    workbook = load_workbook(path)
    for sheet in workbook:
        for row in sheet.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    yield sheet.title, cell.coordinate, cell.value


def assert_precedence_safe(path):
    offenders = [
        (sheet, coord, hit)
        for sheet, coord, formula in formulas_in(path)
        for hit in negated_groups_raised_to_a_power(formula)
    ]
    assert not offenders, (
        "Excel applies '^' to the negated group in these formulas: "
        f"{offenders[:5]}"
    )


class TestDetector:
    """The detector itself, so a silent false negative cannot hide a bug."""

    @pytest.mark.parametrize("formula", [
        "=EXP(-(A1/B1)^C1)",
        "=EXP(-((A1+A2)/B1)^C1)",
        "=1*-(A1)^2",
    ])
    def test_flags_negated_power(self, formula):
        assert negated_groups_raised_to_a_power(formula)

    @pytest.mark.parametrize("formula", [
        "=EXP(-((A1/B1)^C1))",     # explicitly wrapped
        "=EXP(-A1*B1)",            # negation over a product
        "=1/(1+(A1/B1)^C1)",       # no leading negation
        "=(A1)^B1",                # no negation at all
        "=A1-(B1)^C1",             # binary minus
        "=EXP(-(A1+B1))",          # negated group, no power
    ])
    def test_accepts_safe_formulas(self, formula):
        assert not negated_groups_raised_to_a_power(formula)


def psm_with(curve):
    model = PSMModel(
        states=["Alive", "Dead"],
        survival_endpoints=["OS"],
        strategies=["S1"],
        n_cycles=5,
    )
    model.set_survival("S1", "OS", curve)
    model.set_state_cost("care", {"Alive": 1000, "Dead": 0})
    model.set_utility({"Alive": 0.8, "Dead": 0.0})
    return model


class TestGeneratedWorkbooks:
    @pytest.mark.parametrize("curve", [
        Exponential(rate=0.1),
        Weibull(shape=1.5, scale=5.0),
        LogLogistic(shape=1.5, scale=5.0),
        SurvLogNormal(meanlog=1.0, sdlog=0.5),
        Gompertz(shape=0.1, rate=0.05),
        GeneralizedGamma(mu=1.0, sigma=0.5, Q=0.5),
        PiecewiseExponential(breakpoints=[3.0], rates=[0.1, 0.2]),
        KaplanMeier(times=[0.0, 1.0, 2.0], survival_probs=[1.0, 0.8, 0.6]),
        ProportionalHazards(Weibull(shape=1.5, scale=5.0), hr=0.8),
        AcceleratedFailureTime(Weibull(shape=1.5, scale=5.0), af=1.2),
    ], ids=lambda curve: type(curve).__name__)
    def test_psm_survival_formulas(self, curve, tmp_path):
        path = tmp_path / "psm.xlsx"
        export_excel_model(psm_with(curve), str(path))
        assert_precedence_safe(path)

    def test_markov_workbook(self, tmp_path):
        model = MarkovModel(
            states=["Alive", "Dead"], strategies=["S1"], n_cycles=5,
            dr_cost=0.03, dr_qaly=0.03,
        )
        model.add_param("p_death", base=0.1)
        model.set_transitions("S1", lambda p, t: [[C, p["p_death"]], [0, 1]])
        model.set_state_cost("care", {"Alive": 1000, "Dead": 0})
        model.set_transition_cost("terminal", "Alive", "Dead", 5000)
        model.set_utility({"Alive": 0.8, "Dead": 0.0})

        path = tmp_path / "markov.xlsx"
        export_excel_model(model, str(path))
        assert_precedence_safe(path)
