"""Stage 4 (spatial mode): multi-channel meeting rendering from precomputed banks.

Reuses the turn-taking timeline and manifest code of
ConversationalMeetingSimulator, and replaces only the rendering:

* every utterance is rendered through the time-varying RIR interpolation of
  ``render.py`` (moving talkers, rotating talkers, rotating wearer heads);
* each rendered utterance is shifted so that the direct sound at the
  reference channel lands exactly at the supervision start (all channels share
  the shift, so inter-channel delays are preserved);
* reverberant, direct-path (anechoic) and early (direct + early_ms) speaker
  images are rendered with the same waypoints, gains and shifts as the mixture,
  so the dumped references align with the mixture's reference channel;
* noise is either diffuse (one noise clip played by the corner sources) or
  mono (the same clip on every channel, as in the legacy path).
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.signal import convolve

from ..simulator import ConversationalMeetingSimulator, cached_load_audio
from .config import get_spatial_cfg
from .motion import MotionPlanner, SceneView, _state_at, nearest_bin
from .render import combine_schedules, render
from .scene import direct_delay

from lhotse.utils import uuid4

logger = logging.getLogger(__name__)


def _f(x):
    return [float(v) for v in x]


class SpatialMeetingSimulator(ConversationalMeetingSimulator):
    def __init__(self, cfg, output_dir, spk2cuts, scenes, noise_files=None):
        super().__init__(cfg, output_dir, spk2cuts, rirs=None, noise_files=noise_files)
        self.scenes = scenes  # list of scene.json paths
        self.sp = get_spatial_cfg(cfg)
        self._loaded = {}
        # worn devices of the current meeting: {device: (speaker or None, seat)}
        # and {device: [(t, head pose)]}
        self._wearers, self._head_sched = {}, {}

    # ------------------------------------------------------------------ #
    def _load_scene(self, path):
        if path not in self._loaded:
            if len(self._loaded) > 8:
                self._loaded.clear()
            with open(path) as f:
                scene = json.load(f)
            d = Path(path).parent
            banks = {
                name: (
                    np.load(d / meta["bank"], mmap_mode="r"),
                    np.load(d / meta["direct_bank"], mmap_mode="r"),
                )
                for name, meta in scene["devices"].items()
            }
            self._loaded[path] = (scene, banks)
        return self._loaded[path]

    def _load_dry(self, cut, fir_highpass):
        c_audio = cached_load_audio(cut)
        assert c_audio.shape[0] == 1
        c_audio = c_audio - np.mean(c_audio, -1, keepdims=True)
        if fir_highpass is not None:
            c_audio = convolve(c_audio, fir_highpass[None, :], mode="full")
        return c_audio[0]

    def _headset_poser(self, meta, scene, pos_tl):
        """Closetalk headset pose lookup.

        Returns (pose_for_entry, headset_pose): the headset pose in front of the
        mouth of talker source entry e, and the pose of speaker spk's headset at
        time t (wearers follow their head yaw, walkers snap to the nearest node
        and orientation)."""
        poses = meta["poses"]
        view = SceneView(scene)
        n_yaw = len(scene["yaws"])
        src_pos = np.asarray(scene["sources"]["pos"])
        src_yaw = np.asarray(scene["sources"]["yaw"])
        orient_yaws = np.asarray(scene["orient_yaws"])
        node_pos = np.asarray([n["pos"] for n in scene["nodes"]])
        node_pose = {(p["node"], p.get("orient", 0)): k for k, p in enumerate(poses) if p["kind"] == "node"}
        seat_pose = {(p["seat"], p.get("yaw_idx", 0)): k for k, p in enumerate(poses) if p["kind"] == "seat"}
        n_orient = 1 + max(o for _, o in node_pose)
        e2pose = {}
        for ni, n in enumerate(scene["nodes"]):
            for k, e in enumerate(n["entries"]):
                e2pose[e] = node_pose[(ni, k if n_orient > 1 else 0)]
        for si, ents in enumerate(scene["mouth_entries"]):
            for yi, e in enumerate(ents):
                e2pose[e] = seat_pose[(si, yi)]
        wearer_dev = {s: (d, seat) for d, (s, seat) in self._wearers.items() if s is not None}

        def pose_for_entry(e):
            if e in e2pose:
                return e2pose[e]
            # walking: nearest node, orientation closest to the walking direction
            ni = int(np.argmin(np.linalg.norm(node_pos - src_pos[e][None, :], axis=1)))
            k = nearest_bin(src_yaw[e], orient_yaws) if n_orient > 1 else 0
            return node_pose[(ni, k)]

        def headset_pose(spk, t):
            if spk in wearer_dev:
                dname, seat = wearer_dev[spk]
                b = _state_at(self._head_sched[dname], t) - seat * n_yaw
                return pose_for_entry(scene["mouth_entries"][seat][b])
            return pose_for_entry(view.entry_for(_state_at(pos_tl[spk], t)))

        return pose_for_entry, headset_pose

    # ------------------------------------------------------------------ #
    def _render_one(self, x, src_sched, mic_sched, bank, dbank, mic_pos, src_pos, ref_ch, want_early):
        """Render x for one device (or one closetalk channel).

        Schedules are [(t_seconds, key)] relative to the utterance start.
        Returns (rev, direct, early or None, ref_delay_at_start)."""
        fs = self.cfg.samplerate
        n = len(x)
        ss = [(t * fs, e) for t, e in src_sched]
        ms = [(t * fs, p) for t, p in mic_sched]
        times, combos = combine_schedules(ss, ms, n)
        H = np.stack([sum(w * bank[p, s].astype(np.float64) for w, p, s in c) for c in combos])
        Hd = np.stack([sum(w * dbank[p, s].astype(np.float64) for w, p, s in c) for c in combos])
        # direct-path arrival at the reference channel per waypoint (dominant term)
        delays = []
        for c in combos:
            w, p, s = max(c, key=lambda z: z[0])
            delays.append(direct_delay(src_pos[s], mic_pos[p][ref_ch], fs))
        rev = render(x, times, H)
        dire = render(x, times, Hd)
        early = None
        if want_early:
            cut = int(self.sp["reference"]["early_ms"] * 1e-3 * fs)
            He = H.copy()
            for m, dl in enumerate(delays):
                He[m, :, int(round(dl)) + cut :] = 0.0
            early = render(x, times, He)
        return rev, dire, early, delays[0]

    def _render_device(self, dname, meta, bank, dbank, scene, timeline, motions, head_sched, pos_tl, dry, speakers, noise):
        sp = self.sp
        fs = self.cfg.samplerate
        ref_cfg = sp["reference"]
        src_pos = np.asarray(scene["sources"]["pos"])
        poses = meta["poses"]
        mic_pos = [np.asarray(p["mic_pos"]) for p in poses]
        closetalk = meta["placement"] == "mouth"
        worn = meta["placement"] == "head"
        want_dumps = self.cfg.save_spk
        want_early = want_dumps and self.cfg.save_anechoic and ref_cfg["type"] in ("early", "both")
        want_direct = want_dumps and self.cfg.save_anechoic and ref_cfg["type"] in ("direct", "both")
        dump_all = ref_cfg["dump"] == "all"

        if closetalk:
            n_ch = len(speakers)
            pose_for_entry, headset_pose = self._headset_poser(meta, scene, pos_tl)
        else:
            n_ch = meta["n_ch"]
            dev_ref = meta["ref_channel"] if ref_cfg["channel"] is None else int(ref_cfg["channel"])
            if not 0 <= dev_ref < n_ch:
                raise ValueError(f"reference.channel={dev_ref} out of range for device {dname} ({n_ch} channels)")

        placed = []
        for i, (cut, off, lvl, x, mot) in enumerate(
            zip(timeline.utterances, timeline.offsets, timeline.speech_lvls, dry, motions)
        ):
            spk = cut.supervisions[0].speaker
            if closetalk:
                own = speakers.index(spk)
                outs = [None] * n_ch
                # own headset in front of the mouth at the utterance start; it
                # moves rigidly with the head: static self RIR
                own_pose = pose_for_entry(mot.schedule[0][1])
                outs[own] = self._render_one(
                    x, [(0.0, mot.schedule[0][1])], [(0.0, own_pose)], bank, dbank, mic_pos, src_pos, 0, want_early
                )
                for c, cspk in enumerate(speakers):
                    if c != own:
                        outs[c] = self._render_one(
                            x, mot.schedule, [(0.0, headset_pose(cspk, off))], bank, dbank, mic_pos, src_pos, 0, want_early
                        )
                L = max(o[0].shape[-1] for o in outs)
                Ld = max(o[1].shape[-1] for o in outs)

                def cat(k, length):
                    return np.concatenate(
                        [np.pad(o[k], ((0, 0), (0, length - o[k].shape[-1]))) for o in outs], 0
                    )

                rev, dire = cat(0, L), cat(1, Ld)
                early = cat(2, L) if want_early else None
                delay0 = outs[own][3]
                ref = own
            else:
                if worn:
                    hs = head_sched[dname]
                    p0 = _state_at(hs, off)
                    mic_sched = [(0.0, p0)] + [(t - off, p) for t, p in hs if off < t < off + cut.duration]
                else:
                    mic_sched = [(0.0, 0)]
                rev, dire, early, delay0 = self._render_one(
                    x, mot.schedule, mic_sched, bank, dbank, mic_pos, src_pos, dev_ref, want_early
                )
                ref = dev_ref

            shift = int(round(delay0))
            # gain from the reverberant reference channel (legacy: RMS normalization)
            rms = np.sqrt(np.mean(rev[ref] ** 2))
            gain = 10 ** (lvl / 20.0) / rms if rms > 0 else 1.0
            start = int(off * fs) - shift
            placed.append((spk, ref, start, rev * gain, dire * gain, None if early is None else early * gain))

        T = max(int(timeline.current_time * fs), max(max(p[2] + p[3].shape[-1], p[2] + p[4].shape[-1]) for p in placed))
        mix = np.zeros((n_ch, T))
        spk_rev, spk_dir, spk_early = {}, {}, {}

        def add(buf, sig, start):
            if start < 0:
                sig = sig[:, -start:]
                start = 0
            end = min(start + sig.shape[-1], buf.shape[-1])
            buf[:, start:end] += sig[:, : end - start]

        spk_ref = {}
        for spk, ref, start, rev, dire, early in placed:
            add(mix, rev, start)
            if not want_dumps:
                continue
            spk_ref.setdefault(spk, ref)
            sel = slice(None) if dump_all else slice(spk_ref[spk], spk_ref[spk] + 1)
            n_out = n_ch if dump_all else 1
            for buf, sig, on in ((spk_rev, rev, True), (spk_dir, dire, want_direct), (spk_early, early, want_early)):
                if not on:
                    continue
                if spk not in buf:
                    buf[spk] = np.zeros((n_out, T))
                add(buf[spk], sig[sel], start)

        if noise is not None:
            mix = mix + self._render_noise(noise, meta, bank, scene, head_sched.get(dname), pos_tl, speakers, T, dev_ref if not closetalk else None)

        maxval = np.amax(np.abs(mix))
        if maxval > 0.99:
            gain_f = 0.99 / maxval
            mix = mix * gain_f
            for buf in (spk_rev, spk_dir, spk_early):
                for k in buf:
                    buf[k] = buf[k] * gain_f
        return mix, spk_rev, spk_dir, spk_early, n_ch

    def _render_noise(self, noise, meta, bank, scene, head_sched, pos_tl, speakers, T, ref):
        """noise: dict(signal=(T0,) unit RMS, level_db=float)."""
        x = noise["signal"]
        if len(x) < T:
            x = np.pad(x, (0, T - len(x)), mode="wrap")
        x = x[:T]
        n_ch = len(speakers) if meta["placement"] == "mouth" else meta["n_ch"]
        if self.sp["noise"]["type"] == "mono" or not scene["noise_entries"]:
            out = np.tile(x[None, :], (n_ch, 1))
        else:
            ent = scene["noise_entries"]
            if self.sp["noise"]["signal"] == "shifted":
                groups = [(np.roll(x, o), [e]) for o, e in zip(noise["shifts"], ent)]
            else:
                # same clip on every source: sum the RIRs, one convolution
                groups = [(x, ent)]
            out = 0.0
            for xs, ents in groups:
                out = out + self._render_noise_group(xs, ents, meta, bank, scene, head_sched, pos_tl, speakers, T)
        r = 0 if ref is None else ref
        rms = np.sqrt(np.mean(out[r] ** 2))
        return out * (10 ** (noise["level_db"] / 20.0) / (rms + 1e-8))

    def _render_noise_group(self, x, ent, meta, bank, scene, head_sched, pos_tl, speakers, T):
        """Render signal x played by the noise source entries ent."""
        fs = self.cfg.samplerate

        def h(p):
            return np.sum(bank[p, ent].astype(np.float64), axis=0)  # (C, L)

        if meta["placement"] == "head" and head_sched:
            ms = [(t * fs, p) for t, p in head_sched]
            times, combos = combine_schedules([(0.0, -1)], ms, T)
            H = np.stack([sum(w * h(p) for w, p, _ in c) for c in combos])
            return render(x, times, H)[:, :T]
        if meta["placement"] == "mouth":
            _, headset_pose = self._headset_poser(meta, scene, pos_tl)
            chans = []
            for spk in speakers:
                chans.append(render(x, [0], h(headset_pose(spk, 0.0))[None])[:, :T])
            return np.concatenate(chans, 0)
        return render(x, [0], h(0)[None])[:, :T]

    # ------------------------------------------------------------------ #
    def gen_audio(self, indx):
        self._resolve_cfg()
        sp = self.sp
        fir_highpass = self.create_fir_highpass(40, 63) if self.cfg.use_fir else None
        picked = {}

        def pick_scene(sampled_spk):
            picked["i"] = np.random.randint(0, len(self.scenes))

        timeline = self._sample_timeline(pre_loop_hook=pick_scene)
        scene, banks = self._load_scene(self.scenes[picked["i"]])
        speakers = sorted({u.supervisions[0].speaker for u in timeline.utterances})

        # worn devices: who wears them and on which seat
        worn = [n for n, m in scene["devices"].items() if m["placement"] == "head"]
        if len(worn) > len(scene["seats"]):
            raise ValueError("n_wearer_seats must be >= the number of worn devices")
        seats = np.random.permutation(len(scene["seats"]))[: len(worn)]
        wearer_spk = list(np.random.permutation(speakers))
        wearers = {}
        for k, dname in enumerate(worn):
            spk = None
            if sp["wearer"] == "speaker" and k < len(wearer_spk):
                if not scene["mouth_entries"]:
                    raise ValueError("Scene bank was generated with wearer=listener")
                spk = wearer_spk[k]
            wearers[dname] = (spk, int(seats[k]))
        self._wearers = wearers

        planner = MotionPlanner(scene, sp)
        motions, head_sched, pos_tl = planner.plan(speakers, timeline.utterances, timeline.offsets, wearers)
        self._head_sched = head_sched

        do_reverb = self.cfg.reverberate and np.random.random() < getattr(self.cfg, "reverb_prob", 1.0)
        dry = [self._load_dry(cut, fir_highpass) for cut in timeline.utterances]

        noise = None
        noise_prob = getattr(self.cfg, "noise_probability_global", 1.0)
        if self.cfg.add_noise and np.random.random() < noise_prob:
            n_len = int(timeline.current_time * self.cfg.samplerate) + int(self.sp["rir_max_len"] * self.cfg.samplerate)
            min_lvl = min(timeline.speech_lvls)
            if self.noise_files is None:
                sig = np.random.normal(0, 1, n_len)
                lvl = min_lvl + np.random.uniform(-30, 3)
            else:
                sig = self.sample_noise_signal(n_len)
                lvl = min_lvl + np.random.uniform(*self.cfg.noise_rel_gain)
            n_src = len(scene["noise_entries"])
            # time shifts for noise.signal == "shifted": spread over the clip
            shifts = [k * n_len // max(n_src, 1) + np.random.randint(0, max(1, n_len // (4 * max(n_src, 1)))) for k in range(n_src)]
            noise = {"signal": sig, "level_db": lvl, "shifts": shifts}

        recording_id = str(uuid4())
        multi = len(scene["devices"]) > 1
        recordings, supervisions = [], []
        for dname, meta in scene["devices"].items():
            bank, dbank = banks[dname]
            if not do_reverb:
                bank = dbank
            mix, spk_rev, spk_dir, spk_early, n_ch = self._render_device(
                dname, meta, bank, dbank, scene, timeline, motions, head_sched, pos_tl, dry, speakers, noise
            )
            rid = f"{recording_id}_{dname}" if multi else recording_id
            extra = {"early": spk_early} if spk_early else None
            recordings.append(self._write_outputs(rid, mix, spk_rev, spk_dir, extra))
            custom = []
            for cut, mot in zip(timeline.utterances, motions):
                c = {
                    "device": dname,
                    "room_id": scene["room_id"],
                    "reverberant": bool(do_reverb),
                    "position_start": _f(mot.pos_start),
                    "position_end": _f(mot.pos_end),
                    "moving": bool(mot.moving),
                    "rotating": bool(mot.rotating),
                    "yaw_start": float(mot.yaw_start),
                }
                if meta["placement"] == "mouth":
                    c["closetalk_channel"] = speakers.index(cut.supervisions[0].speaker)
                custom.append({"spatial": c})
            channel = list(range(n_ch)) if n_ch > 1 else 0
            supervisions.extend(self._build_supervisions(rid, timeline, channel=channel, extra_custom=custom))
        if not multi:
            return recordings[0], supervisions
        return recordings, supervisions
