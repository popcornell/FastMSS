"""Stage 3 (spatial mode): precompute per-room RIR banks for every device.

For each room we sample devices, talker positions ("nodes", each with
``n_orient`` yaw bins for directional talkers), walking trajectories between
nodes, wearer seats (worn devices) with ``n_yaw`` head-yaw bins and the mouth
positions that go with them, and diffuse-noise sources aimed at the room
corners. Then for every device and every microphone pose we compute the
reverberant and the direct-path (max_order=0) RIRs from every source entry.

Layout of ``<rir_dir>/rirs/room_<id>/``:
    scene.json            -- geometry + index maps (see SceneBankSimulator.gen_scene)
    <device>.npy          -- reverberant bank, shape [n_poses, n_sources, n_ch, L]
    <device>-direct.npy   -- direct-path bank, shape [n_poses, n_sources, n_ch, Ld]
"""

import json
import logging
from pathlib import Path

import numpy as np
import pyroomacoustics

from .config import get_spatial_cfg
from .devices import build_devices, rot_z
from .directivity import is_omni, make_directivity

logger = logging.getLogger(__name__)

SPEED_OF_SOUND = pyroomacoustics.constants.get("c")
# pyroomacoustics centers its fractional-delay filter: direct sound of a source
# at distance d lands at sample d / c * fs + FRAC_DELAY_OFFSET.
FRAC_DELAY_OFFSET = pyroomacoustics.constants.get("frac_delay_length") // 2

MIC_CHUNK = 8
SRC_CHUNK = 64


def direct_delay(src_pos, mic_pos, fs):
    """Direct-path arrival (fractional samples) in pyroomacoustics RIRs."""
    d = np.linalg.norm(np.asarray(src_pos) - np.asarray(mic_pos), axis=-1)
    return d / SPEED_OF_SOUND * fs + FRAC_DELAY_OFFSET


def yaw_towards(src, dst):
    v = np.asarray(dst) - np.asarray(src)
    return float(np.arctan2(v[1], v[0]))


class SceneBankSimulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sp = get_spatial_cfg(cfg)
        if self.cfg.save_anechoic:
            if self.cfg.use_rand_ism is not False or self.cfg.rand_disp != 0.0:
                raise ValueError(
                    "If you want to save the anechoic signal, pyroomacoustics "
                    "use_rand_ism should be False and rand_disp should be 0.0."
                )

    # ------------------------------------------------------------------ #
    # sampling helpers
    # ------------------------------------------------------------------ #
    def _sample_room(self):
        while True:
            l = np.random.uniform(*self.cfg.room_sz)
            w = np.random.uniform(l / 2, self.cfg.room_sz[-1])
            z = np.random.uniform(*self.cfg.room_ceiling)
            room_dim = [l, w, z]
            rt60 = np.random.uniform(*self.cfg.rt60)
            try:
                e_absorption, max_order = pyroomacoustics.inverse_sabine(rt60, room_dim)
            except ValueError:
                continue
            return room_dim, rt60, e_absorption, max_order

    def _rand_point(self, room_dim, z_range, avoid=(), min_dist=(), tries=2000):
        """Uniform point at delta_dist from the walls and at least min_dist[i]
        (horizontally) from every point of avoid[i]."""
        l, w, h = room_dim
        dd = self.cfg.delta_dist
        z_lo, z_hi = z_range[0], min(z_range[1], h - dd)
        for t in range(tries):
            p = np.array(
                [
                    np.random.uniform(dd, l - dd),
                    np.random.uniform(dd, w - dd),
                    np.random.uniform(z_lo, max(z_lo, z_hi)),
                ]
            )
            # relax the constraints progressively if the room is crowded
            relax = 1.0 if t < tries // 2 else 0.5
            ok = True
            for pts, md in zip(avoid, min_dist):
                if len(pts) and np.min(np.linalg.norm(np.asarray(pts)[:, :2] - p[:2], axis=1)) < md * relax:
                    ok = False
                    break
            if ok:
                return p
        return p

    def _place_fixed(self, dev, room_dim):
        l, w, h = room_dim
        dd = self.cfg.delta_dist
        z_lo, z_hi = dev.height
        z_hi = min(z_hi, h - dd)
        if dev.placement == "table":
            center = np.array(
                [
                    np.random.uniform(0.3 * l, 0.7 * l),
                    np.random.uniform(0.3 * w, 0.7 * w),
                    np.random.uniform(z_lo, z_hi),
                ]
            )
            yaw = np.random.uniform(0, 2 * np.pi)
        elif dev.placement == "wall":
            wall = np.random.randint(4)
            margin = min(0.5, l / 4, w / 4)
            off = np.random.uniform(*dev.wall_dist) + np.max(np.abs(dev.offsets[:, 0]))
            z_c = np.random.uniform(z_lo, z_hi)
            if wall == 0:
                center, yaw = [np.random.uniform(margin, l - margin), off, z_c], np.pi / 2
            elif wall == 1:
                center, yaw = [np.random.uniform(margin, l - margin), w - off, z_c], -np.pi / 2
            elif wall == 2:
                center, yaw = [off, np.random.uniform(margin, w - margin), z_c], 0.0
            else:
                center, yaw = [l - off, np.random.uniform(margin, w - margin), z_c], np.pi
            center = np.array(center)
        else:  # free (legacy single mic)
            center = np.array(
                [
                    np.random.uniform(dd, l - dd),
                    np.random.uniform(dd, w - dd),
                    np.random.uniform(0.5, h - dd),
                ]
            )
            yaw = np.random.uniform(0, 2 * np.pi)
        return center, float(yaw)

    # ------------------------------------------------------------------ #
    # RIR computation
    # ------------------------------------------------------------------ #
    def _compute(self, room_dim, e_absorption, max_order, src_pos, src_dir, mic_pos, mic_dir, length):
        """RIRs for all (mic, source) pairs -> float32 [n_mics, n_src, length]."""
        n_mic, n_src = len(mic_pos), len(src_pos)
        out = np.zeros((n_mic, n_src, length), dtype=np.float32)
        for m0 in range(0, n_mic, MIC_CHUNK):
            m1 = min(m0 + MIC_CHUNK, n_mic)
            for s0 in range(0, n_src, SRC_CHUNK):
                s1 = min(s0 + SRC_CHUNK, n_src)
                room = pyroomacoustics.ShoeBox(
                    room_dim,
                    fs=self.cfg.samplerate,
                    max_order=max_order,
                    materials=pyroomacoustics.Material(e_absorption),
                    use_rand_ism=self.cfg.use_rand_ism if max_order > 0 else False,
                    max_rand_disp=self.cfg.rand_disp,
                )
                dirs = mic_dir[m0:m1]
                room.add_microphone_array(
                    pyroomacoustics.MicrophoneArray(
                        np.asarray(mic_pos[m0:m1]).T,
                        self.cfg.samplerate,
                        directivity=None if all(d is None for d in dirs) else dirs,
                    )
                )
                for s in range(s0, s1):
                    room.add_source(src_pos[s], directivity=src_dir[s])
                room.compute_rir()
                for mi in range(m1 - m0):
                    for si in range(s1 - s0):
                        r = room.rir[mi][si][:length]
                        out[m0 + mi, s0 + si, : len(r)] = r
        return out

    # ------------------------------------------------------------------ #
    def gen_scene(self, room_id):
        sp = self.sp
        fs = self.cfg.samplerate
        room_dim, rt60, e_absorption, max_order = self._sample_room()
        l, w, h = room_dim
        devices = build_devices(sp["devices"])
        src_pattern = sp["source_directivity"]["pattern"]
        n_orient = 1 if is_omni(src_pattern) else int(sp["source_directivity"]["n_orient"])

        # fixed devices
        fixed_pts = []
        dev_pose = {}
        for d in devices:
            if d.placement in ("table", "wall", "free"):
                c, y = self._place_fixed(d, room_dim)
                dev_pose[d.name] = (c, y)
                fixed_pts.extend(d.mic_positions(c, y).tolist())
        md_dev = sp["min_dist_device"]

        # wearer seats (head centers)
        worn = [d for d in devices if d.worn]
        seats = []
        if worn:
            # every worn device has its own wearer, so at least one seat each
            for _ in range(max(sp["n_wearer_seats"], len(worn))):
                seats.append(
                    self._rand_point(room_dim, sp["head_height"], [fixed_pts, seats], [md_dev, 1.0])
                )
        yaws = 2 * np.pi * np.arange(sp["n_yaw"]) / sp["n_yaw"]

        # talker nodes
        nodes = []
        for _ in range(self.cfg.n_pos_rirs):
            nodes.append(
                self._rand_point(room_dim, sp["node_height"], [fixed_pts, seats], [md_dev, 0.7])
            )
        nodes = np.array(nodes)

        # source entries
        src_pos, src_yaw, src_kind, src_dir = [], [], [], []

        def add_src(p, yaw, kind, pattern, colatitude=np.pi / 2):
            src_pos.append(np.asarray(p, dtype=float))
            src_yaw.append(float(yaw))
            src_kind.append(kind)
            src_dir.append(make_directivity(pattern, yaw, colatitude))
            return len(src_pos) - 1

        orient_offset = np.random.uniform(0, 2 * np.pi)
        orient_yaws = (orient_offset + 2 * np.pi * np.arange(n_orient) / n_orient) % (2 * np.pi)
        node_entries = [[add_src(p, y, "node", src_pattern) for y in orient_yaws] for p in nodes]

        # trajectories between node pairs
        trajectories = []
        lo, hi = sp["traj_len"]
        step = sp["traj_step"]
        attempts = 0
        while len(trajectories) < sp["n_trajectories"] and attempts < 200 * max(1, sp["n_trajectories"]):
            attempts += 1
            i, j = np.random.choice(len(nodes), 2, replace=False)
            dist = np.linalg.norm(nodes[j] - nodes[i])
            if not lo <= dist <= hi:
                continue
            n_pts = int(np.ceil(dist / step)) + 1
            pts = np.linspace(nodes[i], nodes[j], n_pts)
            if fixed_pts and np.min(
                np.linalg.norm(pts[:, None, :2] - np.asarray(fixed_pts)[None, :, :2], axis=-1)
            ) < 0.5:
                continue
            if seats and np.min(
                np.linalg.norm(pts[:, None, :2] - np.asarray(seats)[None, :, :2], axis=-1)
            ) < 0.5:
                continue
            heading = yaw_towards(nodes[i], nodes[j])
            ent = [add_src(p, heading, "traj", src_pattern) for p in pts]
            # directional talkers face the walking direction: walking the path
            # backwards (to_node -> from_node) needs the opposite yaw
            ent_rev = ent if is_omni(src_pattern) else [add_src(p, heading + np.pi, "traj", src_pattern) for p in pts]
            trajectories.append(
                {
                    "from_node": int(i),
                    "to_node": int(j),
                    "entries": ent,
                    "entries_rev": ent_rev,
                    "pos": pts.tolist(),
                    "heading": heading,
                    "step": float(dist / (n_pts - 1)),
                }
            )

        if len(trajectories) < sp["n_trajectories"]:
            logger.warning(
                f"{room_id}: only {len(trajectories)}/{sp['n_trajectories']} walking trajectories "
                "found (paths must avoid devices and seats by 0.5 m); increase n_pos_rirs or traj_len."
            )

        # wearer mouths (self-voice + wearer heard by the other devices)
        mouth_entries = []
        if worn and sp["wearer"] == "speaker":
            mo = np.asarray(sp["mouth_offset"], dtype=float)
            for s in seats:
                mouth_entries.append(
                    [add_src(s + rot_z(y) @ mo, y, "mouth", src_pattern) for y in yaws]
                )

        # diffuse noise: highly directional sources aimed at the room corners
        noise_entries = []
        ncfg = sp["noise"]
        if self.cfg.add_noise and ncfg["type"] == "diffuse":
            corners = [(0.0, 0.0), (l, 0.0), (0.0, w), (l, w)]
            for k in range(ncfg["n_sources"]):
                cx, cy = corners[k % 4]
                center = np.array([l / 2, w / 2])
                v = center - np.array([cx, cy])
                v = v / np.linalg.norm(v)
                dist = min(ncfg["corner_dist"], 0.4 * min(l, w))
                p_xy = np.array([cx, cy]) + v * dist
                pz = np.random.uniform(0.5, h - 0.5)
                corner_z = 0.0 if np.random.rand() < 0.5 else h
                col = np.arctan2(dist, corner_z - pz)  # colatitude towards the corner
                yaw = float(np.arctan2(-v[1], -v[0]))
                noise_entries.append(add_src([p_xy[0], p_xy[1], pz], yaw, "noise", ncfg["pattern"], col))

        src_pos = np.array(src_pos)
        n_src = len(src_pos)

        # microphone poses per device
        out_dir = Path(self.cfg.output_dir, "rirs", room_id).absolute()
        out_dir.mkdir(parents=True, exist_ok=True)
        L = int(sp["rir_max_len"] * fs)
        diag = np.linalg.norm(room_dim)
        Ld = int(np.ceil(diag / SPEED_OF_SOUND * fs)) + 2 * FRAC_DELAY_OFFSET + 2
        dtype = np.dtype(sp["rir_dtype"])

        # closetalk headset mics follow the talker; they face the room center
        center_xy = np.array([l / 2, w / 2])
        for d in devices:
            if d.placement == "table":
                center_xy = dev_pose[d.name][0][:2]
                break

        dev_meta = {}
        for d in devices:
            poses = []
            if d.placement in ("table", "wall", "free"):
                c, y = dev_pose[d.name]
                poses.append({"center": c.tolist(), "yaw": y, "kind": "fixed"})
            elif d.worn:
                for si, s in enumerate(seats):
                    for yi, y in enumerate(yaws):
                        poses.append({"center": s.tolist(), "yaw": float(y), "kind": "seat", "seat": si, "yaw_idx": yi})
            else:
                # closetalk: one headset pose in front of every talker mouth
                # orientation: per node and yaw bin (omni talkers: facing the
                # table), per wearer seat and head-yaw bin (matches mouth_entries)
                off = np.asarray(sp["closetalk_offset"], dtype=float)
                mo = np.asarray(sp["mouth_offset"], dtype=float)
                for ni, p in enumerate(nodes):
                    ys = orient_yaws if n_orient > 1 else [yaw_towards(p, np.r_[center_xy, p[2]])]
                    for k, y in enumerate(ys):
                        y = float(y)
                        poses.append({"center": (p + rot_z(y) @ off).tolist(), "yaw": y, "kind": "node", "node": ni, "orient": k})
                if mouth_entries:
                    for si, s in enumerate(seats):
                        for yi, y in enumerate(yaws):
                            y = float(y)
                            m = s + rot_z(y) @ mo
                            poses.append({"center": (m + rot_z(y) @ off).tolist(), "yaw": y, "kind": "seat", "seat": si, "yaw_idx": yi})
            mic_pos, mic_dir = [], []
            for pose in poses:
                if d.per_speaker:
                    # headset capsule in front of the mouth, pointing back at it
                    mp = np.asarray(pose["center"])[None, :]
                    az, col = np.array([pose["yaw"] + np.pi]), np.array([np.pi / 2])
                else:
                    mp = d.mic_positions(pose["center"], pose["yaw"])
                    az, col = d.mic_azimuths(pose["yaw"])
                pose["mic_pos"] = mp.tolist()
                mic_pos.extend(mp)
                mic_dir.extend(make_directivity(d.mic_pattern, a, c) for a, c in zip(az, col))
            n_ch = d.n_mics
            P = len(poses)
            rev = self._compute(room_dim, e_absorption, max_order, src_pos, src_dir, mic_pos, mic_dir, L)
            dirp = self._compute(room_dim, e_absorption, 0, src_pos, src_dir, mic_pos, mic_dir, Ld)
            rev = rev.reshape(P, n_ch, n_src, L).transpose(0, 2, 1, 3)
            dirp = dirp.reshape(P, n_ch, n_src, Ld).transpose(0, 2, 1, 3)
            np.save(out_dir / f"{d.name}.npy", np.ascontiguousarray(rev.astype(dtype)))
            np.save(out_dir / f"{d.name}-direct.npy", np.ascontiguousarray(dirp.astype(dtype)))
            dev_meta[d.name] = {
                "preset": d.name,
                "placement": d.placement,
                "n_ch": n_ch,
                "ref_channel": d.ref_channel,
                "mic_pattern": d.mic_pattern,
                "offsets": d.offsets.tolist(),
                "poses": poses,
                "bank": f"{d.name}.npy",
                "direct_bank": f"{d.name}-direct.npy",
            }

        scene = {
            "room_id": room_id,
            "room_dim": [float(x) for x in room_dim],
            "rt60": float(rt60),
            "max_order": int(max_order),
            "fs": fs,
            "rir_len": L,
            "direct_len": Ld,
            "source_pattern": src_pattern,
            "orient_yaws": orient_yaws.tolist(),
            "sources": {
                "pos": src_pos.tolist(),
                "yaw": src_yaw,
                "kind": src_kind,
            },
            "nodes": [{"pos": p.tolist(), "entries": e} for p, e in zip(nodes, node_entries)],
            "trajectories": trajectories,
            "seats": [s.tolist() for s in seats],
            "yaws": yaws.tolist(),
            "mouth_entries": mouth_entries,
            "noise_entries": noise_entries,
            "devices": dev_meta,
        }
        with open(out_dir / "scene.json", "w") as f:
            json.dump(scene, f)
        return str(out_dir / "scene.json")

    def estimate_storage(self):
        """Rough upper bound of the bank size per room (bytes)."""
        sp = self.sp
        devices = build_devices(sp["devices"])
        n_orient = 1 if is_omni(sp["source_directivity"]["pattern"]) else sp["source_directivity"]["n_orient"]
        n_traj_pts = sp["n_trajectories"] * (int(sp["traj_len"][1] / sp["traj_step"]) + 1)
        if n_orient > 1:
            n_traj_pts *= 2  # forward + reverse walking direction
        n_src = self.cfg.n_pos_rirs * n_orient + n_traj_pts + 4
        n_seats = max(sp["n_wearer_seats"], sum(d.worn for d in devices))
        if n_seats and any(d.worn for d in devices):
            n_src += n_seats * sp["n_yaw"]
        L = int(sp["rir_max_len"] * self.cfg.samplerate)
        total = 0
        for d in devices:
            if d.worn:
                P = n_seats * sp["n_yaw"]
            elif d.per_speaker:
                wearer_talks = any(x.worn for x in devices) and sp["wearer"] == "speaker"
                P = self.cfg.n_pos_rirs * n_orient + (n_seats * sp["n_yaw"] if wearer_talks else 0)
            else:
                P = 1
            total += P * n_src * d.n_mics * L
        return total * np.dtype(sp["rir_dtype"]).itemsize
