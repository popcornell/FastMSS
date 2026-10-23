import numpy as np
import pytest

from fastmss.spatial.devices import PRESETS, build_device, build_devices


@pytest.mark.parametrize(
    "preset,n_ch",
    [
        ("single", 1),
        ("ami", 8),
        ("circular", 8),
        ("notsofar1", 7),
        ("kinect", 4),
        ("binaural_worn", 2),
        ("glasses", 6),
        ("closetalk", 1),
    ],
)
def test_channel_counts(preset, n_ch):
    assert build_device({"preset": preset}).n_mics == n_ch


def test_ami_geometry():
    d = build_device({"preset": "ami"})
    r = np.linalg.norm(d.offsets[:, :2], axis=1)
    np.testing.assert_allclose(r, 0.1)
    adj = np.linalg.norm(d.offsets[0] - d.offsets[1])
    np.testing.assert_allclose(adj, 2 * 0.1 * np.sin(np.pi / 8))


def test_notsofar1_center_mic():
    d = build_device({"preset": "notsofar1"})
    np.testing.assert_allclose(d.offsets[0], 0.0)
    np.testing.assert_allclose(np.linalg.norm(d.offsets[1:, :2], axis=1), 0.0425)


def test_kinect_spacing():
    d = build_device({"preset": "kinect"})
    np.testing.assert_allclose(d.offsets[:, 1], [-0.113, 0.036, 0.076, 0.113])
    assert d.placement == "wall"


def test_pose_transform_preserves_geometry():
    for name in PRESETS:
        d = build_device({"preset": name})
        mp = d.mic_positions([1.0, 2.0, 1.5], 1.234)
        dist0 = np.linalg.norm(d.offsets[:, None] - d.offsets[None], axis=-1)
        dist1 = np.linalg.norm(mp[:, None] - mp[None], axis=-1)
        np.testing.assert_allclose(dist0, dist1, atol=1e-12)


def test_binaural_rotation():
    d = build_device({"preset": "binaural_worn"})
    mp = d.mic_positions([0, 0, 0], np.pi / 2)  # facing +y: left ear on -x
    np.testing.assert_allclose(mp[0], [-0.075, 0, 0], atol=1e-12)


def test_overrides_and_unique_names():
    d = build_device({"preset": "circular", "radius": 0.05, "n_mics": 4, "center_mic": True, "mic_pattern": "cardioid"})
    assert d.n_mics == 5 and d.mic_pattern == "cardioid"
    d = build_device({"preset": "single", "offsets": [[0, 0, 0], [0.1, 0, 0]]})
    assert d.n_mics == 2
    devs = build_devices(["kinect", "kinect"])
    assert len({x.name for x in devs}) == 2
    with pytest.raises(ValueError):
        build_device({"preset": "nope"})
    with pytest.raises(ValueError):
        build_devices(["closetalk", "closetalk"])
