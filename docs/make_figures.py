"""Figures for docs/multichannel.md.

    python docs/make_figures.py [--out docs/figures]

Builds two small rooms with synthetic speech-shaped signals (no corpus is
needed), renders a short meeting and writes the PNG files used by the
documentation, plus numbers.json with the values quoted in the text.

scene_figures() also works on an existing bank (see tests/plot_scene.py).
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyroomacoustics as pra  # noqa: E402
import soundfile as sf  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from fixtures import base_cfg, make_noise_files, make_spk2cuts  # noqa: E402

from fastmss.spatial import SceneBankSimulator, SpatialMeetingSimulator, build_device  # noqa: E402
from fastmss.spatial.directivity import is_omni, make_directivity, pattern_gain  # noqa: E402
from fastmss.spatial.metrics import diffuse_coherence, gcc_phat, gcc_phat_track, msc, schroeder_rt60  # noqa: E402
from fastmss.spatial.render import interp_weights, render  # noqa: E402
from fastmss.spatial.scene import direct_delay  # noqa: E402

FS = 16000
C = 343.0

# colors (fixed order) and ink
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "lines.linewidth": 1.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _save(fig, out, name):
    fig.savefig(out / name)
    plt.close(fig)
    print(f"wrote {out / name}")


def _load(scene_json):
    scene = json.loads(Path(scene_json).read_text())
    d = Path(scene_json).parent
    banks = {
        n: (np.load(d / m["bank"], mmap_mode="r"), np.load(d / m["direct_bank"], mmap_mode="r"))
        for n, m in scene["devices"].items()
    }
    return scene, banks


def _fixed_devices(scene, multi=False):
    return [
        n
        for n, m in scene["devices"].items()
        if m["placement"] in ("table", "wall", "free") and (m["n_ch"] > 1 or not multi)
    ]


# --------------------------------------------------------------------------- #
# Figures that only need a scene bank
# --------------------------------------------------------------------------- #
def fig_scene(scene, out):
    l, w, _ = scene["room_dim"]
    pos = np.asarray(scene["sources"]["pos"])
    yaw = np.asarray(scene["sources"]["yaw"])
    fig, ax = plt.subplots(figsize=(6.0, 6.0 * w / l))
    ax.grid(False)
    ax.add_patch(plt.Rectangle((0, 0), l, w, fill=False, lw=1.5, ec=INK))
    for t in scene["trajectories"]:
        tp = np.asarray(t["pos"])
        ax.plot(tp[:, 0], tp[:, 1], "-", color=S3, lw=1.5, label="walking path")
    for k, n in enumerate(scene["nodes"]):
        p = np.asarray(n["pos"])
        ax.plot(*p[:2], "o", color=INK, ms=5, label="talker node")
        ax.annotate(str(k), p[:2], fontsize=7, color=INK2, xytext=(4, 4), textcoords="offset points")
        if not is_omni(scene["source_pattern"]):
            for e in n["entries"]:
                ax.annotate("", p[:2] + 0.18 * np.array([np.cos(yaw[e]), np.sin(yaw[e])]), p[:2],
                            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    for s in scene["seats"]:
        ax.plot(s[0], s[1], "s", color=S4, ms=8, label="wearer seat")
    for e in scene["noise_entries"]:
        ax.plot(*pos[e][:2], "x", color=S2, ms=8, mew=2, label="noise source")
        ax.annotate("", pos[e][:2] + 0.45 * np.array([np.cos(yaw[e]), np.sin(yaw[e])]), pos[e][:2],
                    arrowprops=dict(arrowstyle="-|>", color=S2, lw=1.2))
    for name in _fixed_devices(scene):
        mp = np.asarray(scene["devices"][name]["poses"][0]["mic_pos"])
        ax.plot(mp[:, 0], mp[:, 1], ".", color=S1, ms=5, label="fixed array mics")
        ax.annotate(name, mp.mean(0)[:2], fontsize=7, color=S1, xytext=(6, -10), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, l + 0.3)
    ax.set_ylim(-0.3, w + 0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    h, lab = ax.get_legend_handles_labels()
    uniq = dict(zip(lab, h))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title(f"Room {l:.1f} x {w:.1f} x {scene['room_dim'][2]:.1f} m, RT60 {scene['rt60']:.2f} s (top view)")
    _save(fig, out, "scene.png")


def fig_directivity_bank(scene, banks, out):
    """Direct-path gain over the yaw bins of every node vs the analytic pattern."""
    if is_omni(scene["source_pattern"]):
        return
    name = _fixed_devices(scene)[0]
    m = scene["devices"][name]
    _, dbank = banks[name]
    ref = m["ref_channel"]
    mic = np.asarray(m["poses"][0]["mic_pos"][ref])
    pos = np.asarray(scene["sources"]["pos"])
    yaw = np.asarray(scene["sources"]["yaw"])
    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_subplot(projection="polar")
    th = np.linspace(-np.pi, np.pi, 361)
    ax.plot(th, np.abs(pattern_gain(scene["source_pattern"], th)), "-", color=INK2, lw=1.2, label="analytic")
    for i, n in enumerate(scene["nodes"]):
        ent = n["entries"]
        v = mic - pos[ent[0]]
        az_mic = np.arctan2(v[1], v[0])
        amp = np.array([np.max(np.abs(dbank[0, e, ref])) for e in ent], dtype=float)
        off = np.angle(np.exp(1j * (az_mic - yaw[ent])))
        g = np.abs(pattern_gain(scene["source_pattern"], off))
        ax.plot(off, amp / (amp.max() / g.max()), "o", color=S1, ms=4, alpha=0.8,
                label="measured (bank)" if i == 0 else None)
    ax.set_rlim(0, 1.05)
    ax.set_title(f"{scene['source_pattern']} talker, direct-path gain vs angle off-axis", pad=14)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.15))
    _save(fig, out, "directivity_bank.png")


def fig_tdoa_walk(scene, banks, out, sim=None):
    """GCC-PHAT TDOA of white noise moved along trajectory 0 vs geometry."""
    if not scene["trajectories"]:
        return None
    name = _fixed_devices(scene, multi=True)[0]
    m = scene["devices"][name]
    bank, dbank = banks[name]
    mp = np.asarray(m["poses"][0]["mic_pos"])
    m1 = int(np.argmax(np.linalg.norm(mp - mp[0], axis=1)))
    traj = scene["trajectories"][0]
    dt = traj["step"] / 0.5  # 0.5 m/s
    times = np.arange(len(traj["entries"])) * dt * FS
    x = np.random.default_rng(0).standard_normal(int(times[-1]) + 1)
    max_lag = np.linalg.norm(mp[m1] - mp[0]) / C * FS + 3
    tp = np.asarray(traj["pos"])
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    res = {}
    for kind, b, col in (("direct path", dbank, S1), ("reverberant", bank, S2)):
        H = np.stack([np.asarray(b[0, e][[0, m1]], dtype=float) for e in traj["entries"]])
        y = render(x, times, H)
        est, c = gcc_phat_track(y[0, : len(x)], y[1, : len(x)], 2048, 512, max_lag)
        k = np.clip(c / (dt * FS), 0, len(tp) - 1)
        p = np.stack([np.interp(k, np.arange(len(tp)), tp[:, i]) for i in range(3)], 1)
        geo = np.array([np.diff(direct_delay(q, mp[[0, m1]], FS))[0] for q in p])
        ax.plot(c / FS, est, "o", color=col, ms=3, label=f"GCC-PHAT, {kind}")
        res[kind] = float(np.median(np.abs(est - geo)))
    ax.plot(c / FS, geo, "-", color=INK, lw=1.2, label="geometric")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(f"TDOA mic {m1} - mic 0 [samples]")
    ax.set_title(f"Talker walking at 0.5 m/s along a {traj['step'] * (len(tp) - 1):.1f} m path ({name})")
    ax.legend(loc="best")
    _save(fig, out, "tdoa_walk.png")
    return res


def scene_figures(scene_json, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scene, banks = _load(scene_json)
    fig_scene(scene, out)
    fig_directivity_bank(scene, banks, out)
    fig_tdoa_walk(scene, banks, out)


# --------------------------------------------------------------------------- #
# Figures that need no room
# --------------------------------------------------------------------------- #
def fig_devices(out):
    specs = [
        ("ami", "ami", (0, 1)),
        ("notsofar1", "notsofar1", (0, 1)),
        ("kinect", "kinect", (0, 1)),
        ("binaural_worn", "binaural_worn", (0, 1)),
        ("glasses (top)", "glasses", (0, 1)),
        ("glasses (side)", "glasses", (0, 2)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    for ax, (title, preset, (i, j)) in zip(axes.flat, specs):
        d = build_device({"preset": preset})
        o = d.offsets * 100
        ax.plot(o[:, i], o[:, j], "o", color=S1, ms=6)
        ax.plot(o[d.ref_channel, i], o[d.ref_channel, j], "o", mfc="none", mec=S2, ms=11, mew=1.5)
        for c, q in enumerate(o):
            ax.annotate(str(c), (q[i], q[j]), fontsize=7, color=INK2, xytext=(5, 3), textcoords="offset points")
        lim = max(np.abs(o[:, [i, j]]).max() * 1.35, 3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"{title}, {d.n_mics} ch")
        ax.set_xlabel("x [cm]" if i == 0 else "")
        ax.set_ylabel("y [cm]" if j == 1 else "z [cm]")
    fig.suptitle("Device frames (x forward, y left, z up); open circle: reference channel", fontsize=9, color=INK)
    fig.tight_layout()
    _save(fig, out, "devices.png")


def fig_interp(out):
    n = FS
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * (n - 1)
    w = np.zeros((len(times), n))
    for k, (a, b, wk) in enumerate(interp_weights(n, times)):
        w[k, a:b] = wk
    t = np.arange(n) / FS
    cols = [S1, S2, S3, S4, INK2]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    for k in range(len(times)):
        ax.plot(t, w[k], color=cols[k], label=f"$w_{k}$ (RIR $h_{k}$)")
    ax.plot(t, w.sum(0), "--", color=INK, lw=1.0, label=r"$\sum_k w_k$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("weight")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Triangular input weights for 5 waypoints")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, out, "interp_weights.png")


def _direct_amp(src, mic, src_dir=None):
    room = pra.ShoeBox([8, 8, 4], fs=FS, max_order=0)
    room.add_microphone_array(pra.MicrophoneArray(np.asarray(mic, float)[:, None], FS))
    room.add_source(src, directivity=src_dir)
    room.compute_rir()
    return np.max(np.abs(room.rir[0][0]))


def fig_patterns(out):
    """Anechoic sweep of every source pattern against the analytic gain."""
    src, mic = [4.0, 4.0, 1.5], [5.5, 4.0, 1.5]
    ref = _direct_amp(src, mic)
    th = np.linspace(-np.pi, np.pi, 361)
    deg = np.arange(0, 360, 15)
    pats = ["subcardioid", "cardioid", "hypercardioid", "figure8"]
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.6), subplot_kw={"projection": "polar"})
    err = {}
    for ax, p in zip(axes, pats):
        amp = np.array([_direct_amp(src, mic, make_directivity(p, np.deg2rad(d))) / ref for d in deg])
        ax.plot(th, np.abs(pattern_gain(p, th)), "-", color=INK2, lw=1.2)
        ax.plot(-np.deg2rad(deg), amp, "o", color=S1, ms=3)
        ax.set_rlim(0, 1.05)
        ax.set_yticklabels([])
        ax.set_title(p, pad=10)
        err[p] = float(np.max(np.abs(amp - np.abs(pattern_gain(p, np.deg2rad(deg))))))
    fig.suptitle("Source directivity: analytic gain (line), pyroomacoustics direct path (dots)", fontsize=9, color=INK)
    _save(fig, out, "patterns.png")
    return err


# --------------------------------------------------------------------------- #
# Main: build rooms, render a meeting, write all figures
# --------------------------------------------------------------------------- #
def _cfg(work, spatial, **kw):
    base = dict(
        n_pos_rirs=6,
        rt60=[0.4, 0.5],
        room_sz=[6, 8],
        room_ceiling=[2.8, 3.2],
        reverberate=True,
        reverb_prob=1.0,
        add_noise=False,
        noise_probability_global=1.0,
        duration=8,
        min_max_spk=[3, 3],
        save_spk=True,
        save_anechoic=True,
    )
    base.update(kw)
    sp = dict(enabled=True, n_trajectories=3, traj_len=[1.5, 2.5], rir_max_len=0.5, rir_dtype="float32")
    sp.update(spatial)
    return base_cfg(work, spatial=sp, **base)


def fig_rirs(scene, banks, out):
    name = "ami"
    m = scene["devices"][name]
    bank, dbank = banks[name]
    ref = m["ref_channel"]
    node = scene["nodes"][0]
    e = node["entries"][0]
    h = np.asarray(bank[0, e, ref], dtype=float)
    hd = np.asarray(dbank[0, e, ref], dtype=float)
    dl = direct_delay(np.asarray(scene["sources"]["pos"][e]), np.asarray(m["poses"][0]["mic_pos"])[ref], FS)
    cut = int(0.05 * FS)
    he = h.copy()
    he[int(round(dl)) + cut :] = 0
    t = (np.arange(len(h)) - dl) / FS * 1000
    td = (np.arange(len(hd)) - dl) / FS * 1000
    fig, axes = plt.subplots(3, 1, figsize=(6.0, 4.2), sharex=True)
    for ax, (lab, x, tt, col) in zip(
        axes, (("reverberant", h, t, S1), ("early (direct + 50 ms)", he, t, S3), ("direct path (max_order=0)", hd, td, S2))
    ):
        ax.plot(tt, x / np.max(np.abs(h)), color=col, lw=0.8)
        ax.axvline(0, color=INK2, lw=0.8, ls="--")
        ax.set_ylabel("amplitude")
        ax.set_title(lab, loc="left")
    axes[-1].set_xlim(-5, 150)
    axes[-1].set_xlabel("time relative to the geometric direct-path arrival at the reference channel [ms]")
    fig.tight_layout()
    _save(fig, out, "rirs.png")
    est = [schroeder_rt60(np.asarray(bank[0, n["entries"][0], ref], dtype=float), FS) for n in scene["nodes"]]
    return float(np.nanmedian(est))


def fig_head_rotation(sim, scene, banks, out):
    """Glasses in-ear pair: TDOA of an external talker follows the head yaw;
    the wearer's own voice rotates with the head and keeps its TDOA."""
    dev = "glasses"
    meta = scene["devices"][dev]
    bank, dbank = banks[dev]
    mic_pos = [np.asarray(p["mic_pos"]) for p in meta["poses"]]
    src_pos = np.asarray(scene["sources"]["pos"])
    n_yaw = len(scene["yaws"])
    path = [0, 1, 2, 3, 4]
    t_turn = np.linspace(0.5, 3.5, len(path))
    mic_sched = [(float(t), p) for t, p in zip(t_turn, path)]
    x = np.random.default_rng(4).standard_normal(4 * FS)
    a, b = 4, 5  # in-ear pair
    max_lag = 0.16 / C * FS + 3
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    s = scene["nodes"][0]["entries"][0]
    _, dire, _, _ = sim._render_one(x, [(0.0, s)], mic_sched, bank, dbank, mic_pos, src_pos, a, False)
    est, c = gcc_phat_track(dire[a, : len(x)], dire[b, : len(x)], 2048, 1024, max_lag)
    geo = []
    for tc in c / FS:
        k = np.interp(tc, t_turn, path)
        lo, hi = int(np.floor(k)), int(np.ceil(k))
        f = k - lo
        mp = (1 - f) * mic_pos[lo] + f * mic_pos[hi]
        geo.append(direct_delay(src_pos[s], mp[[a, b]], FS) @ [-1, 1])
    ax.plot(c / FS, geo, "-", color=INK, lw=1.2, label="geometric, external talker")
    ax.plot(c / FS, est, "o", color=S1, ms=3, label="GCC-PHAT, external talker")
    mouth = scene["mouth_entries"][0]
    src_sched = [(float(t), mouth[p % n_yaw]) for t, p in zip(t_turn, path)]
    _, dire, _, _ = sim._render_one(x, src_sched, mic_sched, bank, dbank, mic_pos, src_pos, a, False)
    est2, _ = gcc_phat_track(dire[a, : len(x)], dire[b, : len(x)], 2048, 1024, max_lag)
    ax.plot(c / FS, est2, "s", color=S2, ms=3, label="GCC-PHAT, wearer's own voice")
    for t in t_turn:
        ax.axvline(t, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("time [s] (vertical lines: head-yaw waypoints, 45 degrees apart)")
    ax.set_ylabel("TDOA right - left ear [samples]")
    ax.set_title("Glasses wearer turning the head by 180 degrees (direct path)")
    ax.legend(loc="best")
    _save(fig, out, "head_rotation.png")
    return float(np.median(np.abs(est - np.array(geo)))), float(np.ptp(est2))


def fig_coherence(sim, scene, banks, out):
    meta = scene["devices"]["ami"]
    bank, _ = banks["ami"]
    mp = np.asarray(meta["poses"][0]["mic_pos"])
    m0, m1 = 0, 4
    d = np.linalg.norm(mp[m0] - mp[m1])
    T = 20 * FS
    x = np.random.default_rng(0).standard_normal(T)
    noise = {"signal": x, "level_db": -20.0, "shifts": [k * T // 4 + 1234 * k for k in range(4)]}
    curves = {}
    for mode in ("same", "shifted"):
        sim.sp["noise"]["signal"] = mode
        y = sim._render_noise(noise, meta, bank, scene, None, None, [], T, 0)
        f, curves[mode] = msc(y[m0], y[m1], FS, nperseg=1024)
    sim.sp["noise"]["signal"] = "same"
    point = render(x, [0], np.asarray(bank[0, scene["noise_entries"][0]], dtype=float)[None])
    f, curves["point"] = msc(point[m0], point[m1], FS, nperseg=1024)
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.plot(f, curves["point"], color=INK2, lw=1.0, label="one corner source")
    ax.plot(f, curves["same"], color=S1, label="4 corner sources, signal: same")
    ax.plot(f, curves["shifted"], color=S2, label="4 corner sources, signal: shifted")
    ax.plot(f, diffuse_coherence(f, d), "--", color=INK, lw=1.0, label=r"spherically diffuse, sinc$^2(2\pi f d/c)$")
    ax.set_xlim(0, 8000)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("magnitude-squared coherence")
    ax.set_title(f"Noise coherence between two AMI mics {d * 100:.0f} cm apart")
    ax.legend(loc="upper right")
    _save(fig, out, "noise_coherence.png")
    band = (f > 300) & (f < 4000)
    return {k: float(np.mean(v[band])) for k, v in curves.items()} | {
        "diffuse": float(np.mean(diffuse_coherence(f[band], d)))
    }


def _read(p):
    x, _ = sf.read(p, always_2d=True)
    return x.T


def fig_alignment(recs, sups, spk2cuts, scene, out):
    rec = next(r for r in recs if r.id.endswith("_ami"))
    meta = scene["devices"]["ami"]
    ref = meta["ref_channel"]
    rs = [s for s in sups if s.recording_id == rec.id]
    solo = [s for s in rs if not any(o.speaker != s.speaker and o.start < s.end and o.end > s.start for o in rs)]
    s = solo[0]
    d = Path(rec.sources[0].source).parent
    mix = _read(rec.sources[0].source)
    rev = _read(d / f"{rec.id}-spk-{s.speaker}.wav")
    dire = _read(d / f"{rec.id}-spk-{s.speaker}-anechoic.wav")
    early = _read(d / f"{rec.id}-spk-{s.speaker}-early.wav")
    cuts = {c.id: c for cl in spk2cuts.values() for c in cl}
    dry = cuts[s.custom["original_cut_id"]].load_audio()[0]
    a = int(s.start * FS)
    w0, w1 = a - int(0.004 * FS), a + int(0.020 * FS)
    t = (np.arange(w0, w1) - a) / FS * 1000
    mp = np.asarray(meta["poses"][0]["mic_pos"])
    geo = direct_delay(s.custom["spatial"]["position_start"], mp, FS)
    far = int(np.argmax(np.abs(geo - geo[ref])))
    drys = np.zeros(w1 - w0)
    drys[a - w0 :] = dry[: w1 - a]
    traces = [
        ("dry source at the supervision start", drys, INK),
        (f"direct-path image, ch {ref} (reference)", dire[ref, w0:w1], S2),
        (f"direct-path image, ch {far}", dire[far, w0:w1], S4),
        (f"early image, ch {ref}", early[ref, w0:w1], S3),
        (f"reverberant image, ch {ref}", rev[ref, w0:w1], S1),
        (f"mixture, ch {ref}", mix[ref, w0:w1], INK2),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    lags = {}
    for k, (lab, x, col) in enumerate(traces):
        ax.plot(t, x / (np.max(np.abs(x)) + 1e-12) * 0.45 - k, color=col, lw=1.0)
        ax.text(t[-1] + 0.4, -k, lab, va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK2, lw=0.8, ls="--")
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time relative to the supervision start [ms]")
    ax.set_title(f"Onset of one utterance (AMI array, {s.speaker}); channel {far} lags by {geo[far] - geo[ref]:.1f} samples")
    _save(fig, out, "alignment.png")
    band = (300, 3400)
    seg = slice(a, a + len(dry))
    for lab, img in (("direct", dire), ("early", early), ("reverberant", rev)):
        lags[lab] = float(gcc_phat(dry, img[ref, seg][: len(dry)], max_lag=40, band=band))
    lags["ch_far_minus_ref_est"] = float(gcc_phat(dire[ref, seg], dire[far, seg], max_lag=40, band=band))
    lags["ch_far_minus_ref_geo"] = float(geo[far] - geo[ref])
    return lags


def fig_crosstalk(recs, sups, out):
    rec = next(r for r in recs if r.id.endswith("_closetalk"))
    rs = [s for s in sups if s.recording_id == rec.id]
    d = Path(rec.sources[0].source).parent
    spks = sorted({s.speaker for s in rs})
    own = {s.speaker: s.custom["spatial"]["closetalk_channel"] for s in rs}
    E = np.array([np.sum(_read(d / f"{rec.id}-spk-{spk}.wav") ** 2, axis=1) for spk in spks])
    D = 10 * np.log10(E / E[np.arange(len(spks)), [own[s] for s in spks]][:, None])
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    ax.grid(False)
    im = ax.imshow(D, cmap="Blues", vmin=-40, vmax=0)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            ax.text(j, i, f"{D[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if D[i, j] > -15 else INK)
    ax.set_xticks(range(len(spks)), [f"ch {j}" for j in range(len(spks))])
    ax.set_yticks(range(len(spks)), spks)
    ax.set_xlabel("headset channel")
    ax.set_ylabel("talker image")
    fig.colorbar(im, ax=ax, label="energy rel. own headset [dB]", shrink=0.85)
    ax.set_title("Close-talk crosstalk")
    _save(fig, out, "crosstalk.png")
    off = D[~np.eye(len(spks), dtype=bool)]
    return float(off.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "docs" / "figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="fastmss_docs_"))
    spk2cuts = make_spk2cuts(work)
    noise_files = make_noise_files(work)
    numbers = {}

    fig_devices(out)
    fig_interp(out)
    numbers["pattern_max_abs_error"] = fig_patterns(out)

    # room 1: omni talkers, several devices, a talking glasses wearer
    np.random.seed(0)
    cfg = _cfg(work / "omni", dict(
        devices=[{"preset": "ami"}, {"preset": "kinect"}, {"preset": "glasses"}, {"preset": "closetalk"}],
        n_wearer_seats=1, n_yaw=8,
        reference={"dump": "all", "type": "both"},
        motion={"inter": "static", "intra": "none", "move_prob": 0.0, "head_turn_prob": 0.0},
    ), add_noise=True)
    path = SceneBankSimulator(cfg).gen_scene("room_0")
    scene, banks = _load(path)
    numbers["rt60_target"] = scene["rt60"]
    fig_scene(scene, out)
    numbers["tdoa_walk_median_abs_err"] = fig_tdoa_walk(scene, banks, out)
    numbers["rt60_schroeder_median"] = fig_rirs(scene, banks, out)

    sim = SpatialMeetingSimulator(cfg, work / "omni" / "audio", spk2cuts, scenes=[path], noise_files=noise_files)
    sim._resolve_cfg()
    err, ptp_self = fig_head_rotation(sim, scene, banks, out)
    numbers["head_rotation"] = {"external_median_abs_err": err, "self_voice_ptp": ptp_self}
    numbers["noise_coherence_300_4000"] = fig_coherence(sim, scene, banks, out)

    # the meeting is rendered without noise so that the onsets stay visible
    sim.cfg.add_noise = False
    np.random.seed(11)
    recs, sups = sim.gen_audio(0)
    numbers["alignment_lags"] = fig_alignment(recs, sups, spk2cuts, scene, out)
    numbers["crosstalk_max_db"] = fig_crosstalk(recs, sups, out)

    # room 2: cardioid talkers with 8 yaw bins
    np.random.seed(1)
    cfg2 = _cfg(work / "dir", dict(devices=[{"preset": "single"}], source_directivity={"pattern": "cardioid", "n_orient": 8}))
    path2 = SceneBankSimulator(cfg2).gen_scene("room_0")
    scene2, banks2 = _load(path2)
    fig_directivity_bank(scene2, banks2, out)

    (out / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
