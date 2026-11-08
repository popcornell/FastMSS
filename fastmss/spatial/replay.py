"""Replay mode: spatialize real conversations with their original timing.

Instead of the HMM turn-taking timeline, each item is a real conversation (or
a fixed-length chunk of it) where every talker has a separate recording or
channel, e.g. the two sides of a Fisher telephone call:

* the whole channel of every talker is rendered (speech, breaths, laughter,
  unannotated backchannels and silence), not only the annotated segments;
* each channel is cut into contiguous pieces at the talker's pauses
  (``min_gap``); the pieces are the "utterances" of the motion planner, so
  talkers move, turn and the wearer's head turns between turns. Consecutive
  pieces overlap by ``xfade`` seconds with complementary sin^2 fades (summed
  dry pieces = the original channel), and every piece keeps its full
  reverberant tail;
* one gain per talker for the whole item (``level``): ``mic`` normalizes the
  talker's whole reverberant image at the reference channel to its level;
  ``source`` normalizes the dry speech and keeps the level differences
  produced by the acoustics (e.g. the glasses wearer's own voice is louder),
  with one common scale for all talkers;
* the output supervisions are the original ones (and their word alignments)
  shifted to the item start: the rendered speech is aligned to the original
  time line (reference channel direct path, as in the HMM mode).

Configured by ``spatial.replay`` (see config.py), used by stage 4 of
recipes/sim.py when ``spatial.replay.enabled`` is true.
"""

import logging
import re
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from lhotse import MonoCut
from lhotse.supervision import AlignmentItem, SupervisionSegment
from scipy.signal import convolve

from ..simulator import cached_load_audio
from .meeting import SpatialMeetingSimulator

logger = logging.getLogger(__name__)


def _words_abs(cut, sup):
    """Word alignment of a supervision in recording time.

    Word times are relative to the cut in some manifests (e.g. after stage-1
    splitting) and relative to the recording in others: keep the reading
    that puts the words inside the supervision."""
    if not (sup.alignment and "word" in sup.alignment):
        return None
    w = sup.alignment["word"]
    if not w:
        return []
    s0 = cut.start + sup.start
    s1 = s0 + sup.duration

    def outside(shift):
        return sum(max(0.0, s0 - (a.start + shift)) + max(0.0, a.start + a.duration + shift - s1) for a in w)

    shift = cut.start if outside(cut.start) <= outside(0.0) else 0.0
    return [(a.symbol, a.start + shift, a.duration, getattr(a, "score", None)) for a in w]


def build_conversations(cuts, conv_regex=r"^(.*)-[AB]$", min_speakers=2, max_speakers=None):
    """Group source cuts into conversations.

    A *side* is one (recording, channel) with one talker. Sides whose
    recording ids match ``conv_regex`` with the same first group form one
    conversation (Fisher: ``fe_03_00001-A`` and ``-B``); a multi-channel
    recording with one talker per channel is a conversation by itself.

    Returns a list (sorted by id) of dicts: id, duration, sides = [dict(spk,
    recording, channel, sups=[dict(start, end, text, words, id, ...)])]."""
    sides = {}
    for cut in cuts:
        rec = cut.recording
        ch = cut.channel if isinstance(cut.channel, int) else cut.channel[0]
        key = (rec.id, ch)
        side = sides.setdefault(key, {"recording": rec, "channel": ch, "sups": [], "spk": set()})
        for sup in cut.supervisions:
            start = max(0.0, cut.start + sup.start)
            end = min(rec.duration, cut.start + sup.start + sup.duration)
            if end <= start:
                continue
            side["spk"].add(sup.speaker)
            side["sups"].append(
                dict(
                    id=sup.id,
                    start=start,
                    end=end,
                    speaker=sup.speaker,
                    text=getattr(sup, "text", "") or "",
                    language=getattr(sup, "language", None),
                    words=_words_abs(cut, sup),
                )
            )

    groups = defaultdict(list)
    for (rid, ch), side in sides.items():
        if len(side["spk"]) != 1:
            logger.warning(f"replay: skipping {rid} channel {ch}: {len(side['spk'])} speakers on one channel")
            continue
        side["spk"] = next(iter(side["spk"]))
        side["sups"].sort(key=lambda s: s["start"])
        m = re.match(conv_regex, rid) if conv_regex else None
        groups[m.group(1) if m else rid].append(side)

    convs = []
    for cid in sorted(groups):
        ss = sorted(groups[cid], key=lambda s: (s["recording"].id, s["channel"]))
        spks = [s["spk"] for s in ss]
        if len(set(spks)) != len(spks):
            logger.warning(f"replay: skipping {cid}: the same speaker on several sides")
            continue
        if len(ss) < min_speakers or (max_speakers is not None and len(ss) > max_speakers):
            continue
        dur = min(s["recording"].duration for s in ss)
        convs.append({"id": cid, "duration": dur, "sides": ss})
    return convs


def make_items(convs, chunk=None, min_chunk=None):
    """(conversation index, start, end) windows: whole conversations, or
    consecutive non-overlapping chunks of ``chunk`` seconds (a last chunk
    shorter than ``min_chunk``, default chunk / 2, is dropped)."""
    items = []
    for i, c in enumerate(convs):
        if not chunk:
            items.append((i, 0.0, c["duration"]))
            continue
        lo = chunk / 2 if min_chunk is None else min_chunk
        t = 0.0
        while t < c["duration"]:
            e = min(t + chunk, c["duration"])
            if e - t >= lo:
                items.append((i, t, e))
            t += chunk
    return items


def piece_bounds(sups, w0, w1, min_gap):
    """Contiguous pieces [(a, b)] covering [w0, w1], cut at the middle of the
    pauses >= min_gap between the talker's supervisions."""
    iv = sorted((max(s["start"], w0), min(s["end"], w1)) for s in sups if s["end"] > w0 and s["start"] < w1)
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    cuts = [0.5 * (merged[k][1] + merged[k + 1][0]) for k in range(len(merged) - 1) if merged[k + 1][0] - merged[k][1] >= min_gap]
    edges = [w0] + cuts + [w1]
    return [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)], merged


class ReplayMeetingSimulator(SpatialMeetingSimulator):
    """Spatial rendering of real conversations (see module docstring)."""

    def __init__(self, cfg, output_dir, conversations, items, scenes, noise_files=None):
        super().__init__(cfg, output_dir, {}, scenes=scenes, noise_files=noise_files)
        self.rp = self.sp["replay"]
        if self.rp["level"] not in ("mic", "source"):
            raise ValueError(f"spatial.replay.level must be mic or source, got {self.rp['level']}")
        self.convs = conversations
        self.items = items
        self._item = None
        self._fades, self._dry_ms = {}, {}

    # one item per gen_audio call
    def gen_audio(self, indx):
        self._item = self.items[int(indx)]
        self._fades, self._dry_ms = {}, {}
        return super().gen_audio(indx)

    def _new_recording_id(self):
        ci, w0, w1 = self._item
        return f"{self.convs[ci]['id']}_{int(round(w0 * 100)):07d}_{int(round(w1 * 100)):07d}"

    def _sample_timeline(self, pre_loop_hook=None):
        ci, w0, w1 = self._item
        conv = self.convs[ci]
        fs = self.cfg.samplerate
        xf = float(self.rp["xfade"])
        speakers = [s["spk"] for s in conv["sides"]]
        if pre_loop_hook is not None:
            pre_loop_hook(np.array(speakers))

        base_gain = np.random.uniform(*self.cfg.base_gain)
        spk_lvl = {s: base_gain + np.random.uniform(*self.cfg.rel_gain) for s in speakers}

        utterances, offsets, lvls, active = [], [], [], []
        for side in conv["sides"]:
            rec, spk = side["recording"], side["spk"]
            if rec.sampling_rate != fs:
                raise ValueError(f"replay: {rec.id} is at {rec.sampling_rate} Hz, samplerate is {fs}")
            bounds, merged = piece_bounds(side["sups"], w0, w1, float(self.rp["min_gap"]))
            for k, (a, b) in enumerate(bounds):
                # overlap the neighbours by xfade (inner boundaries only)
                a0 = a - xf / 2 if k > 0 else a
                b0 = b + xf / 2 if k < len(bounds) - 1 else b
                n = int(round((b0 - a0) * fs))
                n_in = int(round(xf * fs)) if k > 0 else 0
                n_out = int(round(xf * fs)) if k < len(bounds) - 1 else 0
                cid = f"{rec.id}-{side['channel']}-{int(round(a0 * fs))}"
                sup = SupervisionSegment(id=cid, recording_id=rec.id, start=0.0, duration=n / fs, channel=side["channel"], speaker=spk)
                cut = MonoCut(id=cid, start=a0, duration=n / fs, channel=side["channel"], recording=rec, supervisions=[sup])
                self._fades[cid] = (n_in, n_out)
                utterances.append(cut)
                offsets.append(a0 - w0)
                lvls.append(spk_lvl[spk])
                active.append([(max(s, a0) - a0, min(e, b0) - a0) for s, e in merged if e > a0 and s < b0])

        return SimpleNamespace(
            sampled_spk=np.array(speakers),
            utterances=utterances,
            offsets=offsets,
            transition_types=[None] * len(utterances),
            speech_lvls=lvls,
            current_time=w1 - w0,
            active=active,
            spk_lvl=spk_lvl,
        )

    def _load_dry(self, cut, fir_highpass):
        x = cached_load_audio(cut)
        assert x.shape[0] == 1
        x = x[0].astype(np.float64)
        n = int(round(cut.duration * self.cfg.samplerate))
        x = np.pad(x[:n], (0, max(0, n - len(x))))
        n_in, n_out = self._fades.get(cut.id, (0, 0))
        # complementary sin^2 / cos^2 fades: overlapping pieces sum to the channel
        if n_in:
            x[:n_in] *= np.sin(0.5 * np.pi * (np.arange(n_in) + 0.5) / n_in) ** 2
        if n_out:
            x[-n_out:] *= np.cos(0.5 * np.pi * (np.arange(n_out) + 0.5) / n_out) ** 2
        if fir_highpass is not None:
            x = convolve(x, fir_highpass, mode="full")
        self._dry_ms[cut.id] = x
        return x

    def _utt_gains(self, timeline, placed):
        fs = self.cfg.samplerate
        utts = timeline.utterances
        e_rev, e_dry, cnt = defaultdict(float), defaultdict(float), defaultdict(int)
        # level measured over the talker's speech; whole channel if it never speaks
        talks = {p[0] for p, act in zip(placed, timeline.active) if act}
        for i, (spk, ref, _, rev, _, _, shift) in enumerate(placed):
            x = self._dry_ms[utts[i].id]
            iv = timeline.active[i] if spk in talks else [(0.0, utts[i].duration)]
            for a, b in iv:
                ia, ib = int(a * fs), int(b * fs)
                e_rev[spk] += float(np.sum(rev[ref, ia + shift : ib + shift] ** 2))
                e_dry[spk] += float(np.sum(x[ia:ib] ** 2))
                cnt[spk] += max(0, ib - ia)
        rms_rev = {s: np.sqrt(e_rev[s] / cnt[s]) if cnt[s] else 0.0 for s in e_rev}
        rms_dry = {s: np.sqrt(e_dry[s] / cnt[s]) if cnt[s] else 0.0 for s in e_dry}
        lvl = timeline.spk_lvl
        g = {}
        if self.rp["level"] == "mic":
            for s in rms_rev:
                g[s] = 10 ** (lvl[s] / 20.0) / rms_rev[s] if rms_rev[s] > 0 else 1.0
        else:
            ok = [s for s in rms_rev if rms_rev[s] > 0 and rms_dry[s] > 0]
            g0 = {s: 1.0 / rms_dry[s] for s in ok}
            # one common scale: mean level at the reference channel = mean target level
            c_db = np.mean([lvl[s] for s in ok]) - np.mean([20 * np.log10(g0[s] * rms_rev[s]) for s in ok]) if ok else 0.0
            for s in rms_rev:
                g[s] = g0[s] * 10 ** (c_db / 20.0) if s in g0 else 1.0
        self._spk_gain_db = {s: float(20 * np.log10(g[s] * rms_rev[s])) if rms_rev[s] > 0 else None for s in g}
        return [g[p[0]] for p in placed]

    def _build_supervisions(self, recording_id, timeline, channel=0, extra_custom=None):
        ci, w0, w1 = self._item
        conv = self.convs[ci]
        wearer = {s for s, _ in self._wearers.values() if s is not None}
        # piece of each talker containing a time (for the spatial metadata)
        pieces = defaultdict(list)
        for i, (cut, off) in enumerate(zip(timeline.utterances, timeline.offsets)):
            pieces[cut.supervisions[0].speaker].append((off, i))
        out = []
        for side in conv["sides"]:
            spk = side["spk"]
            for k, s in enumerate(side["sups"]):
                a, b = max(s["start"], w0), min(s["end"], w1)
                if b <= a:
                    continue
                words = None
                if s["words"] is not None:
                    words = [
                        AlignmentItem(symbol=sym, start=t - w0, duration=d, score=sc)
                        for sym, t, d, sc in s["words"]
                        if t >= w0 and t + d <= w1
                    ]
                piece = max((i for off, i in pieces[spk] if off <= a - w0 + 1e-9), default=pieces[spk][0][1], key=lambda i: timeline.offsets[i])
                custom = {
                    "original_supervision_id": s["id"],
                    "original_recording_id": side["recording"].id,
                    "conversation_id": conv["id"],
                    "window": [float(w0), float(w1)],
                    "speech_level_db": self._spk_gain_db.get(spk),
                    "replay_level": self.rp["level"],
                    "is_wearer": spk in wearer,
                    **(extra_custom[piece] if extra_custom is not None else {}),
                }
                out.append(
                    SupervisionSegment(
                        id=f"{recording_id}_{len(out):05d}",
                        recording_id=recording_id,
                        start=round(a - w0, 4),
                        duration=round(b - a, 4),
                        channel=channel,
                        speaker=spk,
                        text=s["text"],
                        language=s["language"],
                        alignment={"word": words} if words is not None else None,
                        custom=custom,
                    )
                )
        out.sort(key=lambda x: x.start)
        return out
