import pandas as pd
import pytest

from pyheor.export.report import _build_base_case


class _BrokenIcerResult:
    def summary(self):
        return pd.DataFrame([{"Strategy": "A", "Cost": 1.0}])

    def icer(self):
        raise ValueError("incremental analysis failed")


def test_report_does_not_silently_omit_failed_icer():
    with pytest.raises(ValueError, match="incremental analysis failed"):
        _build_base_case(_BrokenIcerResult())
