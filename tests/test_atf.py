"""Measured ATF device (EasyCom glasses): checks against the measured data.

Needs Device_ATFs.h5 from the EasyCom repo; set FASTMSS_EASYCOM_ATF or put it
at /tmp/easycom_atf/Device_ATFs.h5, otherwise the tests are skipped.
"""

import os
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import pytest
from scipy.signal import detrend
from conftest import load_scene, spatial_cfg
from fixtures import FS

from fastmss.spatial import SceneBankSimulator, SpatialMeetingSimulator
from fastmss.spatial.atf import load_atf, load_raw, make_atf_directivities, phat_delay
from fastmss.spatial.devices import rot_z
from fastmss.spatial.metrics import gcc_phat, gcc_phat_track
from fastmss.spatial.scene import FRAC_DELAY_OFFSET

ATF = os.environ.get("FASTMSS_EASYCOM_ATF", "/tmp/easycom_atf/Device_ATFs.h5")
pytestmark = pytest.mark.skipif(not Path(ATF).exists(), reason="EasyCom Device_ATFs.h5 not available")
LEFT, RIGHT = 4, 5  # in-ear channels
BAND = (200.0, 0.45 * FS)  # band of atf.phat_delay


def _nearest_dir(u_meas, v):
    v = np.asarray(v, dtype=float)
    return int(np.argmax(u_meas @ (v / np.linalg.norm(v))))


@pytest.fixture(scope="module")
def raw_delays():
    """PHAT delays of the raw 48 kHz measurements [dir, ch], in samples at FS."""
    raw = load_raw(ATF)
    return phat_delay(raw["ir"], raw["fs"]).T * FS / raw["fs"]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    work = tmp_path_factory.mktemp("atf")
    np.random.seed(7)
    cfg = spatial_cfg(work, dict(devices=[{"preset": "easycom_atf", "atf": ATF}], n_wearer_seats=1, n_yaw=8), min_max_spk=[2, 2])
    path = SceneBankSimulator(cfg).gen_scene("room_0")
    scene, banks = load_scene(path)
    rd = np.load(Path(path).parent / scene["devices"]["easycom_atf"]["ref_delay"])
    sim = SpatialMeetingSimulator(cfg, work / "audio", {}, scenes=[path])
    sim._resolve_cfg()
    return sim, scene, banks, rd


def test_fitted_geometry():
    raw = load_raw(ATF)
    off = raw["offsets"]
    assert off.shape == (6, 3)
    assert off[LEFT, 1] > 0.05 and off[RIGHT, 1] < -0.05  # +y = left
    assert np.all(np.linalg.norm(off, axis=1) < 0.15)
    assert raw["fit_rms"] < 6.0  # samples at 48 kHz (head diffraction)


@pytest.mark.parametrize("yaw_deg", [0.0, 90.0, 215.0])
def test_rotation_selects_measured_direction(yaw_deg):
    """Free-field render through the rotated directivities = measured IR of the
    direction relative to the head."""
    atf = load_atf(ATF, FS)
    c = np.array([3.0, 2.5, 1.5])
    yaw = np.deg2rad(yaw_deg)
    for az in (0.0, 70.0, 180.0, 250.0):
        v_head = np.array([np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az)), -0.2])
        src = c + 1.5 * rot_z(yaw) @ (v_head / np.linalg.norm(v_head))
        room = pra.ShoeBox([6, 5, 3], fs=FS, max_order=0)
        room.add_microphone_array(pra.MicrophoneArray(np.repeat(c[:, None], 6, 1), FS, directivity=make_atf_directivities(ATF, FS, yaw)))
        room.add_source(src)
        room.compute_rir()
        k = int(1.5 / 343 * FS) + FRAC_DELAY_OFFSET
        seg = np.stack([room.rir[m][0][k - 8 : k - 8 + atf["ir"].shape[-1] + 16] for m in range(6)])
        # correlate with every measured direction
        score = np.zeros(atf["ir"].shape[1])
        for m in range(6):
            cc = np.array([np.max(np.correlate(seg[m], h, "full")) / (np.linalg.norm(h) + 1e-12) for h in atf["ir"][m]])
            score += cc / np.linalg.norm(seg[m])
        assert np.argmax(score) == _nearest_dir(atf["u"], v_head), (yaw_deg, az)


def test_ref_delays_match_measured_tdoas(env, raw_delays):
    """Inter-channel delays of the direct-path bank = measured ATF delays of the
    direction of each talker relative to the head (every head yaw)."""
    sim, scene, banks, rd = env
    meta = scene["devices"]["easycom_atf"]
    src_pos = np.asarray(scene["sources"]["pos"])
    u_meas = load_raw(ATF)["u"]
    errs = []
    for p, pose in enumerate(meta["poses"]):
        c, yaw = np.asarray(pose["center"]), pose["yaw"]
        for node in scene["nodes"]:
            s = node["entries"][0]
            d = _nearest_dir(u_meas, rot_z(-yaw) @ (src_pos[s] - c))
            errs.append((rd[p, s] - rd[p, s, 0]) - (raw_delays[d] - raw_delays[d, 0]))
    errs = np.abs(np.array(errs))
    assert np.median(errs) <= 0.25 and np.max(errs) <= 1.0, (np.median(errs), errs.max())


def test_direct_ref_aligned_with_dry(env):
    """Reference shift from the measured delays: dry source and the direct-path
    reference channel line up at lag 0, for external talkers and the wearer's
    own mouth (near field, where geometric delays are off)."""
    sim, scene, banks, rd = env
    bank, dbank = banks["easycom_atf"]
    meta = scene["devices"]["easycom_atf"]
    mic_pos = [np.asarray(p["mic_pos"]) for p in meta["poses"]]
    src_pos = np.asarray(scene["sources"]["pos"])
    x = np.random.default_rng(0).standard_normal(FS)
    ents = [(n["entries"][0], 0) for n in scene["nodes"][:4]] + [(scene["mouth_entries"][0][b], b) for b in (0, 3, 6)]
    for ref in (0, LEFT):
        for s, p in ents:
            _, dire, _, delay0 = sim._render_one(x, [(0.0, s)], [(0.0, p)], bank, dbank, mic_pos, src_pos, ref, False, rd)
            shift = int(round(delay0))
            # in-ear channels have no single dominant peak for the own mouth
            # (diffraction around the face): use the band of the delay table
            lag = gcc_phat(x, dire[ref, shift : shift + len(x)], max_lag=40, band=BAND, fs=FS)
            assert abs(lag) <= 0.5, (ref, s, lag)


def test_head_rotation_tracks_measured_tdoa(env):
    """Wearer turns the head: ear TDOA of an external talker follows the
    measured delays of the changing relative direction; the wearer's own voice
    rotates with the head, so its ear TDOA stays put."""
    sim, scene, banks, rd = env
    bank, dbank = banks["easycom_atf"]
    meta = scene["devices"]["easycom_atf"]
    mic_pos = [np.asarray(p["mic_pos"]) for p in meta["poses"]]
    src_pos = np.asarray(scene["sources"]["pos"])
    path = [0, 1, 2, 3, 4]
    t_turn = np.linspace(0.5, 3.5, len(path))
    mic_sched = [(float(t), p) for t, p in zip(t_turn, path)]
    x = np.random.default_rng(1).standard_normal(4 * FS)
    max_lag = 0.3 / 343 * FS + 3
    frame, hop = 2048, 1024

    s = scene["nodes"][0]["entries"][0]
    _, dire, _, _ = sim._render_one(x, [(0.0, s)], mic_sched, bank, dbank, mic_pos, src_pos, 0, False, rd)
    est, centers = gcc_phat_track(dire[LEFT, : len(x)], dire[RIGHT, : len(x)], frame, hop, max_lag)
    k = np.interp(centers / FS, t_turn, path)
    tdoa = rd[:, s, RIGHT] - rd[:, s, LEFT]
    lo = np.floor(k).astype(int)
    hi = np.minimum(lo + 1, len(path) - 1)
    a = k - lo
    # between yaw bins the render cross-fades two responses; compare where one dominates
    sel = (a < 0.25) | (a > 0.75)
    exp = np.where(a < 0.5, tdoa[np.array(path)[lo]], tdoa[np.array(path)[hi]])
    assert np.ptp(tdoa[path]) > 4  # the turn really changes the ear TDOA
    assert np.median(np.abs(est[sel] - exp[sel])) <= 1.0

    mouth = scene["mouth_entries"][0]
    src_sched = [(float(t), mouth[p]) for t, p in zip(t_turn, path)]
    _, dire, _, _ = sim._render_one(x, src_sched, mic_sched, bank, dbank, mic_pos, src_pos, 0, False, rd)
    est, _ = gcc_phat_track(dire[LEFT, : len(x)], dire[RIGHT, : len(x)], frame, hop, max_lag)
    assert np.ptp(est) <= 1.0, est


def test_reverberant_bank_starts_with_direct_path(env):
    """The late (omni + diffuse-field) part adds no energy around the direct
    sound: the low orders cancel exactly, so rev == direct there."""
    sim, scene, banks, rd = env
    bank, dbank = banks["easycom_atf"]
    n_win = int(0.5e-3 * FS)
    for node in scene["nodes"][:4]:
        s = node["entries"][0]
        for p in (0, 3):
            for m in range(6):
                k = int(round(rd[p, s, m]))
                a, b = max(0, k - n_win), k + n_win
                r, d = bank[p, s, m, a:b], dbank[p, s, m, a:b]
                # pyroomacoustics high-passes every RIR with a zero-phase filter
                # (rir_hpf_enable), which leaves a slow non-causal baseline from
                # the reflections: compare without the trend
                diff = detrend(r - d)
                assert np.sum(diff**2) <= 0.05 * np.sum(d**2), (s, p, m)
    # and there is a reverberant tail
    s = scene["nodes"][0]["entries"][0]
    k = int(round(rd[0, s, 0]))
    assert np.sum(bank[0, s, 0, k + int(0.02 * FS) :].astype(float) ** 2) > 0.05 * np.sum(dbank[0, s, 0].astype(float) ** 2)
