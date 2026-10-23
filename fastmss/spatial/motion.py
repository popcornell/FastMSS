"""Talker and wearer motion strategies over a precomputed scene.

Talkers live on the scene's source entries: static nodes (with yaw bins),
trajectory points (walking paths between nodes) or wearer mouths. For each
utterance a strategy produces a source schedule ``[(t_seconds, entry), ...]``
relative to the utterance start; the renderer interpolates between entries.

Between utterances (``inter``):
    random       -- new random free node every utterance (legacy behaviour)
    static       -- one node per talker for the whole meeting
    random_walk  -- move to a node within max_position_change, favouring
                    closer ones (inverse distance); nearest node if none.
Within an utterance (``intra``, applied with probability ``move_prob``):
    none | walk (along a precomputed path) | rotate (yaw sweep) | walk_rotate

Register new strategies with ``@register_inter`` / ``@register_intra``.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

INTER = {}
INTRA = {}


def register_inter(name):
    def deco(cls):
        INTER[name] = cls
        return cls

    return deco


def register_intra(name):
    def deco(cls):
        INTRA[name] = cls
        return cls

    return deco


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def nearest_bin(yaw, bins):
    return int(np.argmin(np.abs(wrap(np.asarray(bins) - yaw))))


def bin_path(b0, b1, n_bins):
    """Yaw bins visited when turning from b0 to b1 along the shortest way."""
    if b0 == b1:
        return [b0]
    d = (b1 - b0) % n_bins
    step = 1 if d <= n_bins / 2 else -1
    path = [b0]
    while path[-1] != b1:
        path.append((path[-1] + step) % n_bins)
    return path


@dataclass
class TalkerState:
    pos: np.ndarray
    node: Optional[int] = None  # static node index
    traj: Optional[int] = None  # trajectory index when standing on a path
    traj_pt: Optional[int] = None
    traj_dir: int = 1  # +1 walking from_node -> to_node, -1 backwards
    orient: int = 0  # yaw bin for nodes


@dataclass
class UttMotion:
    schedule: List  # [(t_seconds, entry)]
    pos_start: list
    pos_end: list
    moving: bool = False
    rotating: bool = False
    yaw_start: float = 0.0
    extra: dict = field(default_factory=dict)


class SceneView:
    """Convenience accessors over a loaded scene.json dict."""

    def __init__(self, scene):
        self.scene = scene
        self.src_pos = np.asarray(scene["sources"]["pos"])
        self.src_yaw = np.asarray(scene["sources"]["yaw"])
        self.node_pos = np.asarray([n["pos"] for n in scene["nodes"]])
        self.node_entries = [n["entries"] for n in scene["nodes"]]
        self.orient_yaws = np.asarray(scene["orient_yaws"])
        self.trajs = scene["trajectories"]
        self.node2traj = {}
        for ti, t in enumerate(self.trajs):
            self.node2traj.setdefault(t["from_node"], []).append((ti, +1))
            self.node2traj.setdefault(t["to_node"], []).append((ti, -1))

    def traj_entries(self, ti, direction):
        t = self.trajs[ti]
        return t["entries"] if direction > 0 else t.get("entries_rev", t["entries"])

    def entry_for(self, st: TalkerState):
        if st.traj is not None:
            return self.traj_entries(st.traj, st.traj_dir)[st.traj_pt]
        return self.node_entries[st.node][st.orient]

    def nearest_node(self, pos, exclude=()):
        d = np.linalg.norm(self.node_pos - np.asarray(pos)[None, :], axis=1)
        for i in exclude:
            d[i] = np.inf
        return int(np.argmin(d))


# --------------------------------------------------------------------------- #
# Between-utterance strategies
# --------------------------------------------------------------------------- #
class InterStrategy:
    def __init__(self, view: SceneView, mcfg):
        self.view = view
        self.mcfg = mcfg

    def free_nodes(self, occupied):
        free = [i for i in range(len(self.view.node_pos)) if i not in occupied]
        return free or list(range(len(self.view.node_pos)))

    def initial(self, occupied):
        return int(np.random.choice(self.free_nodes(occupied)))

    def next(self, st: TalkerState, occupied):
        raise NotImplementedError


@register_inter("random")
class RandomInter(InterStrategy):
    def next(self, st, occupied):
        return int(np.random.choice(self.free_nodes(occupied)))


@register_inter("static")
class StaticInter(InterStrategy):
    def next(self, st, occupied):
        if st.node is not None:
            return st.node
        return self.view.nearest_node(st.pos, exclude=occupied)


@register_inter("random_walk")
class RandomWalkInter(InterStrategy):
    def next(self, st, occupied):
        free = self.free_nodes(occupied)
        d = np.linalg.norm(self.view.node_pos[free] - st.pos[None, :], axis=1)
        cand = d <= self.mcfg["max_position_change"]
        if np.any(cand):
            idx = np.asarray(free)[cand]
            w = 1.0 / (d[cand] + 0.01)
            return int(np.random.choice(idx, p=w / w.sum()))
        return int(np.asarray(free)[np.argmin(d)])


# --------------------------------------------------------------------------- #
# Within-utterance strategies
# --------------------------------------------------------------------------- #
class IntraStrategy:
    def __init__(self, view: SceneView, mcfg):
        self.view = view
        self.mcfg = mcfg

    def __call__(self, st: TalkerState, duration):
        """Return (schedule, new_state) or None when not applicable."""
        raise NotImplementedError


@register_intra("none")
class NoMotion(IntraStrategy):
    def __call__(self, st, duration):
        return None


@register_intra("walk")
class Walk(IntraStrategy):
    def __call__(self, st, duration):
        v = self.view
        if st.traj is not None:
            ti, pt = st.traj, st.traj_pt
            n = len(v.trajs[ti]["entries"])
            direction = +1 if pt == 0 else (-1 if pt == n - 1 else np.random.choice([-1, 1]))
        else:
            options = v.node2traj.get(st.node, [])
            if not options:
                return None
            ti, direction = options[np.random.randint(len(options))]
            n = len(v.trajs[ti]["entries"])
            pt = 0 if direction > 0 else n - 1
        traj = v.trajs[ti]
        ent = v.traj_entries(ti, direction)
        speed = np.random.uniform(*self.mcfg["speed_range"])
        dt = traj["step"] / speed
        # optional pause before starting to walk
        t0 = np.random.uniform(0, 0.3 * duration)
        sched = [(0.0, ent[pt])]
        t = t0
        while 0 <= pt + direction < n and t + dt <= duration:
            pt += direction
            t += dt
            sched.append((t, ent[pt]))
        if len(sched) == 1:
            return None
        if t0 > 0:
            sched.insert(1, (t0, sched[0][1]))
        new = TalkerState(pos=np.asarray(traj["pos"][pt]), traj=ti, traj_pt=pt, traj_dir=direction)
        # snap to the node when the path end is reached
        if pt in (0, n - 1):
            node = traj["from_node"] if pt == 0 else traj["to_node"]
            heading = traj["heading"] + (0.0 if direction > 0 else np.pi)
            new = TalkerState(pos=v.node_pos[node], node=node, orient=nearest_bin(heading, v.orient_yaws))
        return sched, new


@register_intra("rotate")
class Rotate(IntraStrategy):
    def __call__(self, st, duration):
        v = self.view
        K = len(v.orient_yaws)
        if st.traj is not None or K < 2:
            return None
        delta = np.deg2rad(np.random.uniform(*self.mcfg["rotate_range"])) * np.random.choice([-1, 1])
        b1 = nearest_bin(v.orient_yaws[st.orient] + delta, v.orient_yaws)
        path = bin_path(st.orient, b1, K)
        if len(path) == 1:
            return None
        t0 = np.random.uniform(0, 0.5 * duration)
        t1 = np.random.uniform(t0, duration)
        ts = np.linspace(t0, t1, len(path))
        sched = [(0.0, v.node_entries[st.node][path[0]])]
        sched += [(float(t), v.node_entries[st.node][b]) for t, b in zip(ts, path)]
        new = TalkerState(pos=st.pos, node=st.node, orient=path[-1])
        return sched, new


@register_intra("walk_rotate")
class WalkRotate(IntraStrategy):
    def __call__(self, st, duration):
        first, second = (Walk, Rotate) if np.random.rand() < 0.5 else (Rotate, Walk)
        return first(self.view, self.mcfg)(st, duration) or second(self.view, self.mcfg)(st, duration)


# --------------------------------------------------------------------------- #
# Meeting-level planner
# --------------------------------------------------------------------------- #
class MotionPlanner:
    """Plans talker positions/orientations for every utterance of a meeting and
    the head-yaw schedule of every wearer."""

    def __init__(self, scene, sp):
        self.view = SceneView(scene)
        self.scene = scene
        self.sp = sp
        self.mcfg = sp["motion"]
        if self.mcfg["inter"] not in INTER:
            raise ValueError(f"Unknown inter strategy {self.mcfg['inter']}; choose from {list(INTER)}")
        if self.mcfg["intra"] not in INTRA:
            raise ValueError(f"Unknown intra strategy {self.mcfg['intra']}; choose from {list(INTRA)}")
        self.inter = INTER[self.mcfg["inter"]](self.view, self.mcfg)
        self.intra = INTRA[self.mcfg["intra"]](self.view, self.mcfg)

    def _face_bin(self, spk, states, fixed_center):
        v = self.view
        face = self.sp["face"]
        st = states[spk]
        if face == "random" or len(v.orient_yaws) == 1:
            return int(np.random.randint(len(v.orient_yaws)))
        if face == "device" and fixed_center is not None:
            target = fixed_center
        else:
            others = [s.pos for k, s in states.items() if k != spk]
            if not others:
                return int(np.random.randint(len(v.orient_yaws)))
            target = np.mean(others, axis=0)
        d = np.asarray(target) - st.pos
        return nearest_bin(np.arctan2(d[1], d[0]), v.orient_yaws)

    def plan(self, speakers, utterances, offsets, wearers):
        """
        Args:
            speakers: list of talkers in the meeting.
            utterances / offsets: timeline (cuts and start times in seconds).
            wearers: {device_name: (speaker or None, seat index)}.

        Returns:
            motions: list of UttMotion (one per utterance).
            head_sched: {device_name: [(t_seconds, pose_index)]} for worn devices.
            spk_pos_timeline: {spk: [(t_seconds, TalkerState)]} positions over time.
        """
        v = self.view
        scene = self.scene
        n_yaw = len(scene["yaws"])
        wearer_spk = {s: (d, seat) for d, (s, seat) in wearers.items() if s is not None}
        fixed_center = None
        for meta in scene["devices"].values():
            if meta["placement"] in ("table", "wall", "free"):
                fixed_center = np.asarray(meta["poses"][0]["center"])
                break

        # initial talker states
        states = {}
        occupied = set()
        for spk in speakers:
            if spk in wearer_spk:
                seat = wearer_spk[spk][1]
                states[spk] = TalkerState(pos=np.asarray(scene["seats"][seat]))
                continue
            node = self.inter.initial(occupied)
            occupied.add(node)
            states[spk] = TalkerState(pos=v.node_pos[node], node=node)
        for spk in speakers:
            if spk not in wearer_spk and states[spk].node is not None:
                states[spk].orient = self._face_bin(spk, states, fixed_center)

        # head yaw of each wearer: (t, yaw_bin); starts facing the group
        head = {}
        for dname, (spk, seat) in wearers.items():
            seat_pos = np.asarray(scene["seats"][seat])
            others = [s.pos for k, s in states.items() if k != spk]
            tgt = np.mean(others, axis=0) if others else seat_pos + np.array([1.0, 0, 0])
            yb = nearest_bin(np.arctan2(*(tgt - seat_pos)[[1, 0]]), scene["yaws"])
            head[dname] = [(0.0, yb)]

        order = np.argsort(offsets, kind="stable")
        motions = [None] * len(utterances)
        spk_pos_timeline = {s: [(0.0, states[s])] for s in speakers}
        first_utt = set()
        for i in order:
            cut, off = utterances[i], offsets[i]
            spk = cut.supervisions[0].speaker
            dur = cut.duration

            # wearers turn their head towards the active talker
            for dname, (wspk, seat) in wearers.items():
                if wspk == spk or np.random.rand() >= self.mcfg["head_turn_prob"]:
                    continue
                seat_pos = np.asarray(scene["seats"][seat])
                d = states[spk].pos - seat_pos
                yaw = np.arctan2(d[1], d[0]) + np.deg2rad(np.random.uniform(-1, 1) * self.mcfg["head_jitter"])
                b1 = nearest_bin(yaw, scene["yaws"])
                b0 = head[dname][-1][1]
                path = bin_path(b0, b1, n_yaw)
                if len(path) > 1:
                    t_start = max(off - 0.2, head[dname][-1][0])
                    t_end = t_start + 0.15 * (len(path) - 1)
                    for t, b in zip(np.linspace(t_start, t_end, len(path)), path):
                        head[dname].append((float(t), b))

            if spk in wearer_spk:
                dname, seat = wearer_spk[spk]
                # wearer mouth follows the head yaw; relative schedule over the utterance
                hs = head[dname]
                mouth = scene["mouth_entries"][seat]
                sched = [(t - off, mouth[b]) for t, b in hs if off < t < off + dur]
                b_start = _state_at(hs, off)
                sched = [(0.0, mouth[b_start])] + sched
                pos = list(v.src_pos[mouth[b_start]])
                motions[i] = UttMotion(sched, pos, list(v.src_pos[sched[-1][1]]), yaw_start=float(scene["yaws"][b_start]))
                continue

            st = states[spk]
            if spk in first_utt:
                occ = {s.node for k, s in states.items() if k != spk and s.node is not None}
                node = self.inter.next(st, occ)
                if node != st.node or st.traj is not None:
                    st = TalkerState(pos=v.node_pos[node], node=node)
                    states[spk] = st
                    st.orient = self._face_bin(spk, states, fixed_center)
            first_utt.add(spk)

            spk_pos_timeline[spk].append((off, st))
            res = None
            if np.random.rand() < self.mcfg["move_prob"]:
                res = self.intra(st, dur)
            if res is None:
                e = v.entry_for(st)
                motions[i] = UttMotion([(0.0, e)], list(st.pos), list(st.pos), yaw_start=float(v.src_yaw[e]))
            else:
                sched, new = res
                e0, e1 = sched[0][1], sched[-1][1]
                moving = not np.allclose(v.src_pos[e0], v.src_pos[e1])
                motions[i] = UttMotion(
                    sched,
                    list(v.src_pos[e0]),
                    list(v.src_pos[e1]),
                    moving=moving,
                    rotating=not moving,
                    yaw_start=float(v.src_yaw[e0]),
                )
                states[spk] = new
                spk_pos_timeline[spk].append((off + dur, new))

        head_sched = {}
        for dname, (spk, seat) in wearers.items():
            head_sched[dname] = [(t, seat * n_yaw + b) for t, b in head[dname]]
        return motions, head_sched, spk_pos_timeline


def _state_at(sched, t):
    """Last key of a (time, key) schedule at or before t."""
    key = sched[0][1]
    for ts, k in sched:
        if ts <= t:
            key = k
        else:
            break
    return key
