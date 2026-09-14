import os
import subprocess
import sys

import matplotlib
import pandas as pd
import pytest

from pyheor import MarkovModel
from pyheor.export.report import _build_base_case, generate_report

ALIVE_FOREVER = [[1, 0], [0, 1]]


class _BrokenIcerResult:
    def summary(self):
        return pd.DataFrame([{"Strategy": "A", "Cost": 1.0}])

    def icer(self):
        raise ValueError("incremental analysis failed")


def test_report_does_not_silently_omit_failed_icer():
    with pytest.raises(ValueError, match="incremental analysis failed"):
        _build_base_case(_BrokenIcerResult())


def reportable_model():
    model = MarkovModel(
        states=["Alive", "Dead"],
        strategies=["SOC", "TRT"],
        n_cycles=2,
    )
    model.add_param("u_trt", base=0.6, low=0.5, high=0.7)
    model.add_param("c_trt", base=100, low=90, high=110)
    for strategy in ("SOC", "TRT"):
        model.set_transitions(strategy, lambda p, t: ALIVE_FOREVER)
    model.set_utility({"SOC": {"Alive": 0.5}, "TRT": {"Alive": "u_trt"}})
    model.set_state_cost("care", {"SOC": {"Alive": 0}, "TRT": {"Alive": "c_trt"}})
    return model


class TestBackendIsolation:
    """Importing or using the library must not retarget the caller's plots."""

    def test_import_does_not_switch_backend(self):
        # A subprocess is required: pyheor is already imported here. "template"
        # loads anywhere and is not what the library would switch to.
        code = (
            "import matplotlib\n"
            "before = matplotlib.get_backend()\n"
            "import pyheor\n"
            "print(before, matplotlib.get_backend())\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
            env={**os.environ, "MPLBACKEND": "template"},
        )
        assert proc.returncode == 0, proc.stderr
        before, after = proc.stdout.split()
        assert before == after

    def test_generate_report_restores_the_backend(self, tmp_path):
        original = matplotlib.get_backend()
        matplotlib.use("template")
        try:
            generate_report(
                reportable_model(),
                str(tmp_path / "r.md"),
                run_psa=False,
            )
            assert matplotlib.get_backend() == "template"
        finally:
            matplotlib.use(original)

    def test_generate_report_still_writes_figures(self, tmp_path):
        original = matplotlib.get_backend()
        matplotlib.use("template")
        try:
            path = generate_report(
                reportable_model(),
                str(tmp_path / "r.md"),
                run_psa=False,
            )
        finally:
            matplotlib.use(original)

        assert os.path.exists(path)
        figures = list((tmp_path / "r_files").glob("*.png"))
        assert figures, "tornado figure was not rendered"
        assert all(figure.stat().st_size > 0 for figure in figures)
