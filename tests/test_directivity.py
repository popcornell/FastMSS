"""Directivity checks: analytic patterns, orientation bins in the bank, mic
directivity, DRR vs facing direction, and corner-aimed noise sources."""

import tempfile

import numpy as np
import pyroomacoustics as pra
import pytest
from conftest import load_scene, spatial_cfg

from fastmss.spatial import SceneBankSimulator, build_device
from fastmss.spatial.directivity import make_directivity, pattern_gain
from fastmss.spatial.metrics import drr_db
from fastmss.spatial.scene import direct_delay

FS = 16000
PATTERNS = ["cardioid", "subcardioid", "hypercardioid", "figure8"]


def _direct_amp(src, mic, src_dir=None, mic_dir=None):
    room = pra.ShoeBox([8, 8, 4], fs=FS, max_order=0)
    room.add_microphone_array(pra.MicrophoneArray(np.asarray(mic, float)[:, None], FS, directivity=None if mic_dir is None else [mic_dir]))
    room.add_source(src, directivity=src_dir)
    room.compute_rir()
    return np.max(np.abs(room.rir[0][0]))


@pytest.mark.parametrize("pattern", PATTERNS)
def test_source_pattern_sweep(pattern):
    src, mic = [4.0, 4.0, 1.5], [5.5, 4.0, 1.5]  # mic on +x
    ref = _direct_amp(src, mic)
    for deg in range(0, 360, 15):
        yaw = np.deg2rad(deg)
        amp = _direct_amp(src, mic, src_dir=make_directivity(pattern, yaw))
        expected = abs(pattern_gain(pattern, yaw))
        assert abs(amp / ref - expected) <= 0.02, (pattern, deg, amp / ref, expected)


@pytest.mark.parametrize("pattern", ["cardioid", "hypercardioid"])
def test_mic_pattern_sweep(pattern):
    """Device mic directivity with 'forward' orientation, device rotated."""
    dev = build_device({"preset": "single", "mic_pattern": pattern, "mic_orientation": "forward"})
    src, center = [5.5, 4.0, 1.5], np.array([4.0, 4.0, 1.5])
    ref = _direct_amp(src, center)
    for deg in range(0, 360, 30):
        yaw = np.deg2rad(deg)
        az, col = dev.mic_azimuths(yaw)
        amp = _direct_amp(src, dev.mic_positions(center, yaw)[0], mic_dir=make_directivity(pattern, az[0], col[0]))
        assert abs(amp / ref - abs(pattern_gain(pattern, yaw))) <= 0.02, (pattern, deg)


def test_bank_orientation_bins(scene_directional):
    """Direct-path energy across the precomputed yaw bins follows the cardioid
    pattern at the yaw stored in scene.json."""
    cfg, path = scene_directional
    scene, banks = load_scene(path)
    _, dbank = banks["single"]
    mic = np.asarray(scene["devices"]["single"]["poses"][0]["mic_pos"][0])
    src_pos = np.asarray(scene["sources"]["pos"])
    src_yaw = np.asarray(scene["sources"]["yaw"])
    for node in scene["nodes"]:
        ent = node["entries"]
        assert len(ent) == 8
        np.testing.assert_allclose(src_yaw[ent], scene["orient_yaws"])
        amps = np.array([np.max(np.abs(dbank[0, e, 0])) for e in ent])
        v = mic - src_pos[ent[0]]
        v /= np.linalg.norm(v)
        cos = np.array([np.cos(y) * v[0] + np.sin(y) * v[1] for y in src_yaw[ent]])
        expected = 0.5 + 0.5 * cos
        omni = amps.max() / expected.max()  # 1/r gain of an omni source
        np.testing.assert_allclose(amps / omni, expected, atol=0.03)


def test_drr_facing_vs_away(scene_directional):
    cfg, path = scene_directional
    scene, banks = load_scene(path)
    bank, _ = banks["single"]
    mic = np.asarray(scene["devices"]["single"]["poses"][0]["mic_pos"][0])
    src_pos = np.asarray(scene["sources"]["pos"])
    yaws = np.asarray(scene["orient_yaws"])
    for node in scene["nodes"]:
        p = np.asarray(node["pos"])
        to_mic = np.arctan2(*(mic - p)[[1, 0]])
        face = int(np.argmin(np.abs(np.angle(np.exp(1j * (yaws - to_mic))))))
        away = int(np.argmin(np.abs(np.angle(np.exp(1j * (yaws - to_mic - np.pi))))))
        idx = direct_delay(p, mic, FS)
        d_face = drr_db(bank[0, node["entries"][face], 0], idx, FS)
        d_away = drr_db(bank[0, node["entries"][away], 0], idx, FS)
        assert d_face > d_away + 6, (d_face, d_away)


def _noise_direct_share(pattern, seed=3):
    np.random.seed(seed)
    cfg = spatial_cfg(
        tempfile.mkdtemp(),
        dict(devices=[{"preset": "single", "placement": "table"}], n_trajectories=0, noise={"pattern": pattern}),
        n_pos_rirs=1,
    )
    scene, banks = load_scene(SceneBankSimulator(cfg).gen_scene("room_0"))
    bank, dbank = banks["single"]
    ent = scene["noise_entries"]
    assert len(ent) == 4
    d = sum(np.sum(dbank[0, e, 0] ** 2) for e in ent)
    t = sum(np.sum(bank[0, e, 0] ** 2) for e in ent)
    return d / t, scene


def test_corner_noise_sources_are_diffuse():
    share_dir, scene = _noise_direct_share("hypercardioid")
    share_omni, scene_o = _noise_direct_share("omni")
    # same geometry, only the noise directivity differs
    np.testing.assert_allclose(scene["sources"]["pos"], scene_o["sources"]["pos"])
    assert share_dir < 0.5 * share_omni, (share_dir, share_omni)
    # every noise source points into its corner
    l, w, h = scene["room_dim"]
    for e in scene["noise_entries"]:
        p = np.asarray(scene["sources"]["pos"][e])
        yaw = scene["sources"]["yaw"][e]
        corner = np.array([0.0 if p[0] < l / 2 else l, 0.0 if p[1] < w / 2 else w])
        to_corner = np.arctan2(*(corner - p[:2])[[1, 0]])
        assert abs(np.angle(np.exp(1j * (yaw - to_corner)))) < 1e-6
