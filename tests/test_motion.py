import json

import numpy as np
import pytest

from fastmss.spatial.config import get_spatial_cfg
from fastmss.spatial.motion import (
    INTER,
    INTRA,
    MotionPlanner,
    SceneView,
    TalkerState,
    bin_path,
    nearest_bin,
)


class _Sup:
    def __init__(self, spk):
        self.speaker = spk


class _Cut:
    def __init__(self, spk, dur):
        self.supervisions = [_Sup(spk)]
        self.duration = dur


def _scene(scene_all):
    with open(scene_all[1]) as f:
        return json.load(f)


def _timeline(n=40, spks=("a", "b", "c"), seed=0):
    rng = np.random.default_rng(seed)
    utts, offs, t = [], [], 0.0
    for i in range(n):
        utts.append(_Cut(spks[i % len(spks)], float(rng.uniform(1.0, 3.0))))
        offs.append(t)
        t += utts[-1].duration + 0.1
    return list(spks), utts, offs


def _planner(scene, **motion):
    sp = get_spatial_cfg({"spatial": {"enabled": True, "motion": motion}})
    return MotionPlanner(scene, sp)


def test_bins():
    assert bin_path(0, 3, 8) == [0, 1, 2, 3]
    assert bin_path(1, 6, 8) == [1, 0, 7, 6]
    assert nearest_bin(np.pi - 0.01, np.linspace(0, 2 * np.pi, 4, endpoint=False)) == 2


def test_registry():
    assert {"random", "static", "random_walk"} <= set(INTER)
    assert {"none", "walk", "rotate", "walk_rotate"} <= set(INTRA)


@pytest.mark.parametrize("max_change", [0.5, 1.5])
def test_random_walk_step_bound(scene_all, max_change):
    scene = _scene(scene_all)
    view = SceneView(scene)
    strat = INTER["random_walk"](view, {"max_position_change": max_change})
    np.random.seed(0)
    for start in range(len(view.node_pos)):
        st = TalkerState(pos=view.node_pos[start], node=start)
        for _ in range(30):
            nxt = strat.next(st, occupied=set())
            d = np.linalg.norm(view.node_pos[nxt] - st.pos)
            reachable = np.linalg.norm(view.node_pos - st.pos, axis=1) <= max_change
            if np.any(reachable):
                assert d <= max_change + 1e-9
            st = TalkerState(pos=view.node_pos[nxt], node=nxt)


def test_static_keeps_positions(scene_all):
    scene = _scene(scene_all)
    np.random.seed(1)
    spks, utts, offs = _timeline()
    motions, _, _ = _planner(scene, inter="static").plan(spks, utts, offs, {})
    pos = {}
    for u, m in zip(utts, motions):
        spk = u.supervisions[0].speaker
        pos.setdefault(spk, m.pos_start)
        np.testing.assert_allclose(m.pos_start, pos[spk])
        assert not m.moving


def test_distinct_talker_nodes(scene_all):
    scene = _scene(scene_all)
    np.random.seed(2)
    spks, utts, offs = _timeline()
    _, _, tl = _planner(scene, inter="random").plan(spks, utts, offs, {})
    first = [tl[s][0][1].node for s in spks]
    assert len(set(first)) == len(first)


def test_walk_follows_trajectory(scene_all):
    scene = _scene(scene_all)
    view = SceneView(scene)
    np.random.seed(3)
    spks, utts, offs = _timeline()
    motions, _, _ = _planner(scene, inter="static", intra="walk", move_prob=1.0, speed_range=[0.5, 1.0]).plan(
        spks, utts, offs, {}
    )
    walked = [m for m in motions if m.moving]
    assert walked, "no walking utterance (trajectories missing?)"
    for m in walked:
        ts = [t for t, _ in m.schedule]
        assert ts == sorted(ts)
        pts = view.src_pos[[e for _, e in m.schedule]]
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        assert np.max(steps) <= 0.06  # traj_step = 0.05
        # implied speed never exceeds the configured maximum
        dt = np.diff(ts)
        moving = steps > 1e-9
        assert np.all(steps[moving] / dt[moving] <= 1.0 + 1e-6)


def test_wearer_head_schedule(scene_all):
    scene = _scene(scene_all)
    np.random.seed(4)
    spks, utts, offs = _timeline()
    wearers = {"glasses": ("a", 0)}
    motions, head, _ = _planner(scene, head_turn_prob=1.0).plan(spks, utts, offs, wearers)
    n_yaw = len(scene["yaws"])
    sched = head["glasses"]
    assert len(sched) > 1
    ts = [t for t, _ in sched]
    assert ts == sorted(ts)
    poses = [p for _, p in sched]
    assert all(0 <= p < n_yaw for p in poses)  # seat 0
    # consecutive head poses are adjacent yaw bins (smooth turns)
    for p0, p1 in zip(poses[:-1], poses[1:]):
        assert (p1 - p0) % n_yaw in (0, 1, n_yaw - 1)
    # the wearer's own speech comes from the mouth entries of seat 0
    mouth = set(scene["mouth_entries"][0])
    for u, m in zip(utts, motions):
        if u.supervisions[0].speaker == "a":
            assert all(e in mouth for _, e in m.schedule)


def test_walkers_face_walking_direction(scene_directional):
    """Directional talkers face where they walk, also on paths walked backwards."""
    scene = _scene(scene_directional)
    view = SceneView(scene)
    np.random.seed(5)
    spks, utts, offs = _timeline(n=80)
    motions, _, _ = _planner(scene, inter="random", intra="walk", move_prob=1.0).plan(spks, utts, offs, {})
    n_checked = {+1: 0, -1: 0}
    for m in motions:
        if not m.moving:
            continue
        ents = [e for _, e in m.schedule]
        pts = view.src_pos[ents]
        for k in range(1, len(ents)):
            d = pts[k] - pts[k - 1]
            if np.linalg.norm(d[:2]) < 1e-9:
                continue
            walk_yaw = np.arctan2(d[1], d[0])
            assert abs(np.angle(np.exp(1j * (view.src_yaw[ents[k]] - walk_yaw)))) < 1e-6
            t = next(t for t in scene["trajectories"] if ents[k] in t["entries"] or ents[k] in t["entries_rev"])
            n_checked[+1 if ents[k] in t["entries"] else -1] += 1
    assert n_checked[+1] > 0 and n_checked[-1] > 0, n_checked
