"""spatial.near_field_render_gain_db scales only the wearer's own-voice direct path."""
import numpy as np
from fastmss.spatial.meeting import SpatialMeetingSimulator


def _fake(g_db):
    o = SpatialMeetingSimulator.__new__(SpatialMeetingSimulator)
    o.sp = {"near_field_render_gain_db": g_db}
    return o


def _scene():
    rng = np.random.default_rng(0)
    P, S, C, L, Ld = 4, 3, 2, 50, 20
    bank = rng.standard_normal((P, S, C, L)).astype(np.float32)
    dbank = rng.standard_normal((P, S, C, Ld)).astype(np.float32)
    poses = [{"kind": "seat", "seat": 0, "yaw_idx": y} for y in range(2)] + [{"kind": "traj"}, {"kind": "seat", "seat": 0, "yaw_idx": 1}]
    scene = {"mouth_entries": [[0, 1]], "devices": {"aria": {"placement": "head", "poses": poses},
                                                     "tbl": {"placement": "table", "poses": []}}}
    return scene, {"aria": (bank, dbank), "tbl": (bank.copy(), dbank.copy())}, bank.copy(), dbank.copy()


def test_zero_gain_untouched():
    scene, banks, b0, _ = _scene()
    ref = banks["aria"][0]
    _fake(0.0)._scale_own_direct(scene, banks)
    assert banks["aria"][0] is ref


def test_gain_only_on_own_direct():
    scene, banks, b0, d0 = _scene()
    g = 10 ** (6.0 / 20)
    _fake(6.0)._scale_own_direct(scene, banks)
    b, d = banks["aria"]
    own = {(0, 0), (1, 1), (3, 1)}
    for p in range(4):
        for s in range(3):
            if (p, s) in own:
                np.testing.assert_allclose(b[p, s, :, :20], b0[p, s, :, :20] + (g - 1) * d0[p, s], rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(b[p, s, :, 20:], b0[p, s, :, 20:])
                np.testing.assert_allclose(d[p, s], g * d0[p, s], rtol=1e-6)
            else:
                np.testing.assert_array_equal(b[p, s], b0[p, s])
                np.testing.assert_array_equal(d[p, s], d0[p, s])
    np.testing.assert_array_equal(banks["tbl"][0], b0)
