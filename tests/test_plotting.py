"""Plot geometry, layout, and scenario-mapping regressions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyheor.plotting import plot_psm_trace, plot_tornado, plot_trace


class DenseOWSA:
    wtp = 50000

    def summary(self, **kwargs):
        return pd.DataFrame([
            {
                'Parameter': f'{i + 1}. 疾病进展后第二线治疗及随访检查年度总费用参数',
                'INMB (Low)': 100 + i * .1,
                'INMB (High)': 100 - i * .1,
                'INMB (Base)': 100,
                'Low Value': .01,
                'High Value': 10000000,
            }
            for i in reversed(range(20))
        ])


@pytest.mark.parametrize('show_values', [False, True])
def test_dense_tornado_labels_do_not_overlap(show_values):
    before = plt.rcParams.copy()
    fig = plot_tornado(DenseOWSA(), max_params=20, show_values=show_values)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [label.get_window_extent(renderer) for label in fig.axes[0].get_yticklabels()]
        for upper, lower in zip(boxes, boxes[1:]):
            assert upper.y0 > lower.y1
        for box in boxes:
            assert box.x0 >= 0
            assert box.y0 >= 0
        assert len(fig.axes[0].patches) == 20
        assert dict(plt.rcParams) == dict(before)
    finally:
        plt.close(fig)


def test_same_side_and_inverse_effects_preserve_endpoints():
    class SameSide:
        wtp = 100

        def summary(self, **kwargs):
            return pd.DataFrame([{
                'Parameter': 'Inverse effect', 'INMB (Low)': 120,
                'INMB (High)': 110, 'INMB (Base)': 100,
                'Low Value': 1, 'High Value': 2,
            }])

    fig = plot_tornado(SameSide())
    try:
        ax = fig.axes[0]
        assert len(ax.patches) == 1
        assert ax.patches[0].get_x() == 110
        assert ax.patches[0].get_width() == 10
        assert ax.lines[0].get_xdata()[0] == 120
        assert ax.lines[1].get_xdata()[0] == 110
        assert not ax.texts  # Inputs must not be squeezed against endpoints.
    finally:
        plt.close(fig)


def test_default_tornado_limits_parameter_count():
    fig = plot_tornado(DenseOWSA())
    try:
        assert len(fig.axes[0].get_yticklabels()) == 10
    finally:
        plt.close(fig)


def test_markov_trace_uses_one_shared_state_legend(simple_markov_model):
    before = plt.rcParams.copy()
    fig = plot_trace(simple_markov_model.run_base_case())
    try:
        assert len(fig.axes) == simple_markov_model.n_strategies
        assert len(fig.legends) == 1
        assert [text.get_text() for text in fig.legends[0].get_texts()] == simple_markov_model.states
        assert dict(plt.rcParams) == dict(before)
    finally:
        plt.close(fig)


def test_psm_trace_uses_one_shared_strategy_legend(simple_psm_model):
    fig = plot_psm_trace(simple_psm_model.run_base_case())
    try:
        assert len(fig.axes) == simple_psm_model.n_states
        assert len(fig.legends) == 1
        assert [text.get_text() for text in fig.legends[0].get_texts()] == [
            simple_psm_model.strategy_labels[strategy]
            for strategy in simple_psm_model.strategy_names
        ]
    finally:
        plt.close(fig)


def test_trace_rejects_unknown_style(simple_markov_model):
    with pytest.raises(ValueError, match="style"):
        plot_trace(simple_markov_model.run_base_case(), style="scatter")
