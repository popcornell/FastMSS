"""Aria glasses with the measured ATFs of the CHiME-8 MMCSG / MCAS release.

Needs aria_atfs.pkl (MCAS, CHiME_Aria_ATFs); set FASTMSS_ARIA_ATF, otherwise
the default MultichannelVAP data path is tried and the tests are skipped if it
is missing.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from conftest import load_scene, spatial_cfg
from fixtures import FS

from fastmss.spatial import SceneBankSimulator
from fastmss.spatial.atf import load_near, load_raw
from fastmss.spatial.devices import build_device

ATF = os.environ.get(
    "FASTMSS_ARIA_ATF",
    "/work/nvme/bjdh/cornell2/slocJEPA_data/Aria_resources/Aria_ATF/CHiME_Aria_ATFs/aria_atfs.pkl",
)
pytestmark = pytest.mark.skipif(not Path(ATF).exists(), reason="aria_atfs.pkl not available")

# CHiME-8 task 3 data page: mic positions (cm), x back to front, y right to left, z bottom to top
ARIA_CM = np.array([
    [9.95, -4.76, 0.68],   # 0 lower lens right
    [10.59, 0.74, 5.07],   # 1 nose bridge
    [9.95, 4.49, 0.76],    # 2 lower lens left
    [9.28, 6.41, 5.12],    # 3 front left
    [9.93, -5.66, 5.22],   # 4 front right
    [-0.42, -8.45, 3.35],  # 5 rear right
    [-0.48, 7.75, 3.49],   # 6 rear left
])


def test_load_raw_shapes():
    raw = load_raw(ATF)
    assert raw["ir"].shape == (7, 1674, 332)
    assert raw["fs"] == 48000.0
    assert raw["near"].shape == (7, 332)
    np.testing.assert_allclose(np.linalg.norm(raw["u"], axis=1), 1.0, atol=1e-9)


def test_fitted_geometry_matches_published_positions():
    """The plane-wave fit of the measured delays recovers the documented mic
    layout up to a translation: checks the azimuth (left = +y) and elevation
    (up = +z) conventions of the loader."""
    off = load_raw(ATF)["offsets"]
    pub = ARIA_CM / 100.0
    a, b = off - off.mean(0), pub - pub.mean(0)
    err = np.linalg.norm(a - b, axis=1)
    assert np.median(err) < 0.02, err          # metres
    # a mirrored convention would flip the sign of the y or z spread
    assert np.corrcoef(a[:, 1], b[:, 1])[0, 1] > 0.9
    assert np.corrcoef(a[:, 2], b[:, 2])[0, 1] > 0.5


def test_preset():
    d = build_device({"preset": "aria_atf", "atf": ATF})
    assert d.n_mics == 7 and d.worn and d.atf == ATF
    with pytest.raises(ValueError):
        build_device({"preset": "aria_atf"})


def _scene(work, **sp):
    np.random.seed(3)
    cfg = spatial_cfg(work, dict(devices=[{"preset": "aria_atf", "atf": ATF}], n_wearer_seats=1, n_yaw=2,
                                 source_directivity={"pattern": "omni", "n_orient": 1}, **sp),
                      min_max_spk=[2, 2])
    path = SceneBankSimulator(cfg).gen_scene("room_0")
    scene, banks = load_scene(path)
    return scene, banks["aria_atf"]


@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    far = _scene(tmp_path_factory.mktemp("far"), near_field=False)
    near = _scene(tmp_path_factory.mktemp("near"))
    loud = _scene(tmp_path_factory.mktemp("loud"), near_field_gain_db=6.0)
    return far, near, loud


def _own(scene):
    """(pose index, own mouth source index) of every worn pose."""
    poses = scene["devices"]["aria_atf"]["poses"]
    return [(pi, scene["mouth_entries"][p["seat"]][p["yaw_idx"]]) for pi, p in enumerate(poses)]


def test_own_voice_uses_near_field(scenes):
    (s_far, (rev_f, dir_f)), (s_near, (rev_n, dir_n)), (_, (_, dir_l)) = scenes
    h = load_near(ATF, FS)
    own = _own(s_near)
    assert own
    for pi, src in own:
        x = dir_n[pi, src]                                     # (ch, Ld)
        # the stored direct path is the measured near-field response
        xc = sum(np.correlate(x[m], h[m], mode="full") for m in range(7))
        assert xc.max() / np.sqrt(np.sum(x**2) * np.sum(h**2)) > 0.99
        # level of an omni point source at the mouth (omni talkers here: the far-field direct)
        e_far, e_near = np.sum(dir_f[pi, src] ** 2), np.sum(x**2)
        np.testing.assert_allclose(e_near, e_far, rtol=1e-3)
        np.testing.assert_allclose(np.sum(dir_l[pi, src] ** 2), 10 ** 0.6 * e_far, rtol=1e-3)
        # reflections unchanged
        n = x.shape[-1]
        np.testing.assert_allclose(rev_n[pi, src, :, :n] - x, rev_f[pi, src, :, :n] - dir_f[pi, src], atol=1e-6)
        np.testing.assert_allclose(rev_n[pi, src, :, n:], rev_f[pi, src, :, n:], atol=1e-7)


def test_other_sources_unchanged(scenes):
    (s_far, (rev_f, dir_f)), (s_near, (rev_n, dir_n)), _ = scenes
    mask = np.ones(rev_f.shape[:2], bool)
    for pi, src in _own(s_near):
        mask[pi, src] = False
    np.testing.assert_array_equal(rev_n[mask], rev_f[mask])
    np.testing.assert_array_equal(dir_n[mask], dir_f[mask])


def test_late_shared_over_yaw(tmp_path, scenes):
    """atf_late_shared_yaw: the first yaw bin is unchanged, the others reuse its
    late part (same direct paths, similar energy, different fine structure)."""
    _, (s_u, (rev_u, dir_u)), _ = scenes
    s_s, (rev_s, dir_s) = _scene(tmp_path, atf_late_shared_yaw=True)
    assert s_s["max_order"] > 3
    np.testing.assert_array_equal(rev_s[0], rev_u[0])
    np.testing.assert_array_equal(dir_s, dir_u)
    assert not np.array_equal(rev_s[1], rev_u[1])
    e_s, e_u = np.sum(rev_s[1].astype(np.float64) ** 2), np.sum(rev_u[1].astype(np.float64) ** 2)
    assert abs(10 * np.log10(e_s / e_u)) < 1.0
