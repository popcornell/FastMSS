import numpy as np
import pytest

from fastmss.spatial.meeting import sensor_noise


@pytest.mark.parametrize("color", ["white", "pink"])
def test_unit_rms_independent(color):
    np.random.seed(0)
    x = sensor_noise((4, 16000 * 4), color)
    assert x.shape == (4, 64000)
    np.testing.assert_allclose(np.sqrt((x ** 2).mean(-1)), 1.0, rtol=1e-6)
    c = np.corrcoef(x)
    assert np.abs(c[~np.eye(4, dtype=bool)]).max() < 0.05


def test_pink_slope():
    np.random.seed(0)
    x = sensor_noise((1, 16000 * 8), "pink")[0]
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / 16000)
    lo = P[(f > 200) & (f < 400)].mean()
    hi = P[(f > 2000) & (f < 4000)].mean()
    assert 7 < lo / hi < 14          # 1/f: ~10x between the two band centres


def test_bad_color():
    with pytest.raises(ValueError):
        sensor_noise((1, 100), "brown")
