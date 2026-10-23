"""GCC-PHAT / SRP-PHAT checks of the rendered multi-channel signals against geometry."""

import numpy as np
import pytest
from conftest import load_scene

from fastmss.spatial import SpatialMeetingSimulator
from fastmss.spatial.metrics import gcc_phat, gcc_phat_track, srp_phat_azimuth
from fastmss.spatial.scene import direct_delay

FS = 16000
FIXED = ["ami", "kinect", "notsofar1", "single"]


@pytest.fixture(scope="module")
def env(scene_all, data_dir, tmp_path_factory):
    cfg, path = scene_all
    _, spk2cuts, noise = data_dir
    sim = SpatialMeetingSimulator(cfg, tmp_path_factory.mktemp("gcc") / "audio", spk2cuts, scenes=[path], noise_files=noise)
    sim._resolve_cfg()
    scene, banks = load_scene(path)
    return sim, scene, banks


def _white(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


def _render(sim, scene, banks, dev, src_sched, mic_sched, x, ref):
    bank, dbank = banks[dev]
    mic_pos = [np.asarray(p["mic_pos"]) for p in scene["devices"][dev]["poses"]]
    src_pos = np.asarray(scene["sources"]["pos"])
    return sim._render_one(x, src_sched, mic_sched, bank, dbank, mic_pos, src_pos, ref, True)


@pytest.mark.parametrize("dev", ["ami", "kinect", "notsofar1", "glasses", "binaural_worn"])
@pytest.mark.parametrize("kind", ["direct", "reverberant"])
def test_static_tdoa_matches_geometry(env, dev, kind):
    sim, scene, banks = env
    meta = scene["devices"][dev]
    ref = meta["ref_channel"]
    src_pos = np.asarray(scene["sources"]["pos"])
    x = _white(FS)
    tol = 1.0 if kind == "direct" else 2.0
    errs = []
    for node in scene["nodes"]:
        s = node["entries"][0]
        pose = 0
        mp = np.asarray(meta["poses"][pose]["mic_pos"])
        rev, dire, _, _ = _render(sim, scene, banks, dev, [(0.0, s)], [(0.0, pose)], x, ref)
        y = dire if kind == "direct" else rev
        geo = direct_delay(src_pos[s], mp, FS)
        max_lag = np.max(np.linalg.norm(mp - mp[ref], axis=1)) / 343 * FS + 3
        for m in range(meta["n_ch"]):
            if m == ref:
                continue
            est = gcc_phat(y[ref], y[m], max_lag=max_lag)
            errs.append(abs(est - (geo[m] - geo[ref])))
    errs = np.array(errs)
    if kind == "direct":
        assert np.all(errs <= tol), (dev, errs)
    else:
        # strong early reflections (e.g. a wall right behind a Kinect) can
        # legitimately win the GCC-PHAT peak for a few pairs
        assert np.mean(errs <= tol) >= 0.85, (dev, errs)
        assert np.median(errs) <= 0.5, (dev, errs)


@pytest.mark.parametrize("kind", ["direct", "reverberant"])
def test_ami_srp_doa(env, kind):
    sim, scene, banks = env
    meta = scene["devices"]["ami"]
    src_pos = np.asarray(scene["sources"]["pos"])
    mp = np.asarray(meta["poses"][0]["mic_pos"])
    x = _white(2 * FS, 1)
    tol = np.deg2rad(4 if kind == "direct" else 10)
    for node in scene["nodes"]:
        s = node["entries"][0]
        rev, dire, _, _ = _render(sim, scene, banks, "ami", [(0.0, s)], [(0.0, 0)], x, 0)
        y = dire if kind == "direct" else rev
        est, _, _ = srp_phat_azimuth(y, mp, FS)
        d = src_pos[s] - mp.mean(0)
        true = np.arctan2(d[1], d[0])
        err = abs((est - true + np.pi) % (2 * np.pi) - np.pi)
        assert err <= tol, (kind, np.rad2deg(est), np.rad2deg(true))


def test_reference_alignment_lag_zero(env):
    """Dry source vs direct and reverberant ref channels: lag 0 after the shift."""
    sim, scene, banks = env
    x = _white(FS, 2)
    for dev in FIXED:
        ref = scene["devices"][dev]["ref_channel"]
        for node in scene["nodes"][:3]:
            s = node["entries"][0]
            rev, dire, early, delay0 = _render(sim, scene, banks, dev, [(0.0, s)], [(0.0, 0)], x, ref)
            shift = int(round(delay0))
            for y in (dire, early, rev):
                lag = gcc_phat(x, y[ref, shift : shift + len(x)], max_lag=50)
                assert abs(lag) <= 0.5, (dev, lag)


def test_moving_source_tdoa_track(env):
    sim, scene, banks = env
    dev = "ami"
    meta = scene["devices"][dev]
    mp = np.asarray(meta["poses"][0]["mic_pos"])
    src_pos = np.asarray(scene["sources"]["pos"])
    traj = scene["trajectories"][0]
    speed = 0.5
    dt = traj["step"] / speed
    sched = [(k * dt, e) for k, e in enumerate(traj["entries"])]
    dur = sched[-1][0]
    x = _white(int(dur * FS), 3)
    # mic pair with the largest aperture
    m0, m1 = 0, 4
    max_lag = np.linalg.norm(mp[m1] - mp[m0]) / 343 * FS + 3
    for kind, med_tol, max_tol in (("direct", 0.5, 2.0), ("reverberant", 1.0, 4.0)):
        rev, dire, _, _ = _render(sim, scene, banks, dev, sched, [(0.0, 0)], x, 0)
        y = dire if kind == "direct" else rev
        frame, hop = 2048, 512
        est, centers = gcc_phat_track(y[m0, : len(x)], y[m1, : len(x)], frame, hop, max_lag)
        # geometric TDOA along the path at each frame center
        t = centers / FS
        k = np.clip(t / dt, 0, len(traj["entries"]) - 1)
        pos = np.array([np.interp(k, np.arange(len(traj["pos"])), np.asarray(traj["pos"])[:, i]) for i in range(3)]).T
        geo = np.array([np.diff(direct_delay(p, mp[[m0, m1]], FS))[0] for p in pos])
        err = np.abs(est - geo)
        assert np.median(err) <= med_tol and np.max(err) <= max_tol, (kind, np.median(err), np.max(err))
        # no jumps between consecutive frames beyond the geometric change
        jump = np.abs(np.diff(est)) - np.abs(np.diff(geo))
        assert np.max(jump) <= max_tol


def test_worn_head_rotation_tracks(env):
    sim, scene, banks = env
    dev = "binaural_worn"
    meta = scene["devices"][dev]
    src_pos = np.asarray(scene["sources"]["pos"])
    n_yaw = len(scene["yaws"])
    # half a turn over 4 s across the yaw bins of seat 0
    path = [0, 1, 2, 3, 4]
    t_turn = np.linspace(0.5, 3.5, len(path))
    mic_sched = [(float(t), p) for t, p in zip(t_turn, path)]
    x = _white(4 * FS, 4)
    max_lag = 0.15 / 343 * FS + 3
    frame, hop = 2048, 1024

    # external talker: ear TDOA follows the head yaw
    s = scene["nodes"][0]["entries"][0]
    _, dire, _, _ = _render(sim, scene, banks, dev, [(0.0, s)], mic_sched, x, 0)
    est, centers = gcc_phat_track(dire[0, : len(x)], dire[1, : len(x)], frame, hop, max_lag)
    t = centers / FS
    geo = []
    for tc in t:
        k = np.interp(tc, t_turn, path)
        lo, hi = int(np.floor(k)), int(np.ceil(k))
        a = k - lo
        mp = (1 - a) * np.asarray(meta["poses"][lo]["mic_pos"]) + a * np.asarray(meta["poses"][hi]["mic_pos"])
        geo.append(np.diff(direct_delay(src_pos[s], mp, FS))[0])
    geo = np.array(geo)
    assert np.median(np.abs(est - geo)) <= 1.0
    assert np.ptp(geo) > 4  # the rotation really changes the TDOA

    # self voice: the mouth rotates with the head -> ear TDOA stays ~constant
    mouth = scene["mouth_entries"][0]
    src_sched = [(float(t), mouth[p % n_yaw]) for t, p in zip(t_turn, path)]
    _, dire, _, _ = _render(sim, scene, banks, dev, src_sched, mic_sched, x, 0)
    est, _ = gcc_phat_track(dire[0, : len(x)], dire[1, : len(x)], frame, hop, max_lag)
    assert np.ptp(est) <= 1.0, est
