"""Tests for pyheor/utils.py — C sentinel, matrix helpers, discounting."""

import numpy as np
import pytest
from pyheor.utils import C, _Complement, resolve_complement, validate_transition_matrix
from pyheor.utils import discount_factor, resolve_value


# =========================================================================
# C Sentinel
# =========================================================================

class TestComplement:
    def test_singleton(self):
        assert _Complement() is C

    def test_repr(self):
        assert repr(C) == "C"
        assert str(C) == "C"

    def test_equality(self):
        assert C == C
        assert C == _Complement()
        assert C != 0
        assert C != None  # noqa: E711

    def test_hashable(self):
        s = {C, C}
        assert len(s) == 1
        d = {C: "complement"}
        assert d[C] == "complement"


# =========================================================================
# resolve_complement
# =========================================================================

class TestResolveComplement:
    def test_basic(self):
        P = resolve_complement([
            [C, 0.3, 0.1],
            [0, C,   0.5],
            [0, 0,   1],
        ])
        np.testing.assert_allclose(P[0, 0], 0.6)
        np.testing.assert_allclose(P[1, 1], 0.5)
        np.testing.assert_allclose(P.sum(axis=1), [1, 1, 1])

    def test_no_complement(self):
        P = resolve_complement([
            [0.5, 0.3, 0.2],
            [0.1, 0.4, 0.5],
            [0,   0,   1],
        ])
        np.testing.assert_allclose(P[0], [0.5, 0.3, 0.2])

    def test_identity(self):
        P = resolve_complement([
            [1, 0],
            [0, 1],
        ])
        np.testing.assert_allclose(P, np.eye(2))

    def test_multiple_c_raises(self):
        with pytest.raises(ValueError, match="only one C"):
            resolve_complement([[C, C], [0, 1]])

    def test_negative_complement_raises(self):
        with pytest.raises(ValueError, match="negative"):
            resolve_complement([[C, 0.6, 0.5], [0, 0, 1], [0, 0, 1]])


# =========================================================================
# validate_transition_matrix
# =========================================================================

class TestValidateTransitionMatrix:
    def test_valid(self):
        P = np.array([[0.7, 0.2, 0.1], [0, 0.5, 0.5], [0, 0, 1]])
        assert validate_transition_matrix(P) is True

    def test_nonsquare_raises(self):
        with pytest.raises(ValueError, match="square"):
            validate_transition_matrix(np.array([[0.5, 0.5]]))

    def test_negative_raises(self):
        P = np.array([[1.1, -0.1], [0, 1]])
        with pytest.raises(ValueError, match="Negative"):
            validate_transition_matrix(P)

    def test_rowsum_raises(self):
        P = np.array([[0.5, 0.4], [0.3, 0.3]])
        with pytest.raises(ValueError, match="Row sums"):
            validate_transition_matrix(P)


# =========================================================================
# discount_factor
# =========================================================================

class TestDiscountFactor:
    def test_zero_rate_scalar(self):
        assert discount_factor(5, 0) == 1.0

    def test_zero_rate_array(self):
        result = discount_factor(np.array([0, 1, 2]), 0)
        np.testing.assert_allclose(result, [1, 1, 1])

    def test_known_values(self):
        np.testing.assert_allclose(
            discount_factor(1, 0.03, 1.0), 1 / 1.03, rtol=1e-10
        )
        np.testing.assert_allclose(
            discount_factor(2, 0.05, 0.5), 1 / 1.05, rtol=1e-10
        )

    def test_array_input(self):
        t = np.array([0, 1, 2, 3])
        result = discount_factor(t, 0.03)
        expected = (1.03) ** (-t.astype(float))
        np.testing.assert_allclose(result, expected)


# =========================================================================
# resolve_value
# =========================================================================

class TestResolveValue:
    def test_float(self):
        assert resolve_value(3.14, {}) == 3.14

    def test_int(self):
        assert resolve_value(42, {}) == 42.0

    def test_string(self):
        params = {"cost": 5000.0}
        assert resolve_value("cost", params) == 5000.0

    def test_string_missing_raises(self):
        with pytest.raises(KeyError, match="not_here"):
            resolve_value("not_here", {"x": 1})

    def test_callable(self):
        func = lambda params, t: params["x"] + t
        assert resolve_value(func, {"x": 10}, t=5) == 15.0
