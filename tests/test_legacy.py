"""The legacy (non-spatial) path must stay bit-identical to the reference.

The baseline in tests/data/legacy_baseline.json was produced with the code
before the spatial extension (see tests/legacy_regression.py); it depends on
library versions, so regenerate it on the reference commit when they change."""

import json
from pathlib import Path

import pytest

from legacy_regression import run

BASELINE = Path(__file__).parent / "data" / "legacy_baseline.json"


@pytest.mark.skipif(not BASELINE.exists(), reason="no legacy baseline")
def test_legacy_bit_identical(tmp_path):
    assert run(tmp_path) == json.loads(BASELINE.read_text())
