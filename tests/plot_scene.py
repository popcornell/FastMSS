"""Plots for manual inspection of a spatial scene bank.

    python tests/plot_scene.py <rir_dir>/rirs/room_0/scene.json --out plots/

Writes:
    scene.png        top view: room, devices, talker nodes + yaw bins,
                     trajectories, wearer seats, noise sources (+ aim)
    directivity.png  measured direct-path gain over the yaw bins of every node
                     vs the analytic pattern (polar)
    tdoa_track.png   GCC-PHAT TDOA track of white noise moved along the first
                     trajectory vs the geometric TDOA (first fixed array)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastmss.spatial.directivity import is_omni, pattern_gain  # noqa: E402
from fastmss.spatial.metrics import gcc_phat_track  # noqa: E402
from fastmss.spatial.render import render  # noqa: E402
from fastmss.spatial.scene import direct_delay  # noqa: E402


def plot_scene(scene, out):
    l, w, _ = scene["room_dim"]
    pos = np.asarray(scene["sources"]["pos"])
    yaw = np.asarray(scene["sources"]["yaw"])
    fig, ax = plt.subplots(figsize=(7, 7 * w / l))
    ax.add_patch(plt.Rectangle((0, 0), l, w, fill=False, lw=2))
    for k, n in enumerate(scene["nodes"]):
        p = np.asarray(n["pos"])
        ax.plot(*p[:2], "ko", ms=4)
        ax.annotate(str(k), p[:2], fontsize=7, xytext=(3, 3), textcoords="offset points")
        if not is_omni(scene["source_pattern"]):
            for e in n["entries"]:
                ax.arrow(*p[:2], 0.15 * np.cos(yaw[e]), 0.15 * np.sin(yaw[e]), width=0.003, color="0.6")
    for t in scene["trajectories"]:
        tp = np.asarray(t["pos"])
        ax.plot(tp[:, 0], tp[:, 1], "-", color="tab:green", alpha=0.6)
    for s in scene["seats"]:
        ax.plot(s[0], s[1], "s", color="tab:purple", ms=8, label="wearer seat")
    for e in scene["noise_entries"]:
        ax.plot(*pos[e][:2], "x", color="tab:red", ms=8)
        ax.arrow(*pos[e][:2], 0.4 * np.cos(yaw[e]), 0.4 * np.sin(yaw[e]), width=0.01, color="tab:red")
    for name, m in scene["devices"].items():
        if m["placement"] in ("table", "wall", "free"):
            mp = np.asarray(m["poses"][0]["mic_pos"])
            ax.plot(mp[:, 0], mp[:, 1], ".", ms=6, label=name)
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, l + 0.2)
    ax.set_ylim(-0.2, w + 0.2)
    ax.set_title(f"{scene['room_id']}  RT60={scene['rt60']:.2f}s  (nodes, paths, seats, noise)")
    h, lab = ax.get_legend_handles_labels()
    uniq = dict(zip(lab, h))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper right")
    fig.savefig(out / "scene.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_directivity(scene, d, out):
    if is_omni(scene["source_pattern"]):
        return
    name, m = next((n, m) for n, m in scene["devices"].items() if m["placement"] in ("table", "wall", "free"))
    dbank = np.load(d / m["direct_bank"], mmap_mode="r")
    mic = np.asarray(m["poses"][0]["mic_pos"][m["ref_channel"]])
    pos = np.asarray(scene["sources"]["pos"])
    yaw = np.asarray(scene["sources"]["yaw"])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="polar")
    th = np.linspace(-np.pi, np.pi, 361)
    ax.plot(th, np.abs(pattern_gain(scene["source_pattern"], th)), "k-", label=scene["source_pattern"])
    for n in scene["nodes"]:
        ent = n["entries"]
        v = mic - pos[ent[0]]
        v /= np.linalg.norm(v)
        amp = np.array([np.max(np.abs(dbank[0, e, m["ref_channel"]])) for e in ent], dtype=float)
        ang = np.array([np.arccos(np.clip(np.cos(yaw[e]) * v[0] + np.sin(yaw[e]) * v[1], -1, 1)) for e in ent])
        g = np.abs(pattern_gain(scene["source_pattern"], ang))
        omni = amp.max() / g.max()
        sgn = np.sign(np.sin(yaw[ent] - np.arctan2(v[1], v[0])))
        ax.plot(ang * sgn, amp / omni, ".", alpha=0.7)
    ax.set_title(f"measured direct-path gain vs angle off-axis ({name})")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(out / "directivity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_tdoa(scene, d, out, fs):
    if not scene["trajectories"]:
        return
    name, m = next((n, m) for n, m in scene["devices"].items() if m["placement"] in ("table", "wall") and m["n_ch"] > 1)
    bank = np.load(d / m["direct_bank"], mmap_mode="r")
    mp = np.asarray(m["poses"][0]["mic_pos"])
    m1 = int(np.argmax(np.linalg.norm(mp - mp[0], axis=1)))
    traj = scene["trajectories"][0]
    dt = traj["step"] / 0.5  # 0.5 m/s
    times = np.arange(len(traj["entries"])) * dt * fs
    x = np.random.default_rng(0).standard_normal(int(times[-1]) + 1)
    H = np.stack([bank[0, e][[0, m1]].astype(float) for e in traj["entries"]])
    y = render(x, times, H)
    max_lag = np.linalg.norm(mp[m1] - mp[0]) / 343 * fs + 3
    est, c = gcc_phat_track(y[0, : len(x)], y[1, : len(x)], 2048, 512, max_lag)
    k = np.clip(c / (dt * fs), 0, len(traj["pos"]) - 1)
    tp = np.asarray(traj["pos"])
    p = np.stack([np.interp(k, np.arange(len(tp)), tp[:, i]) for i in range(3)], 1)
    geo = [np.diff(direct_delay(q, mp[[0, m1]], fs))[0] for q in p]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(c / fs, geo, "k-", label="geometric")
    ax.plot(c / fs, est, ".", label="GCC-PHAT")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(f"TDOA mic{m1}-mic0 [samples]")
    ax.set_title(f"moving source along trajectory 0 ({name}, direct path)")
    ax.legend()
    fig.savefig(out / "tdoa_track.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_json")
    ap.add_argument("--out", default="plots")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = json.loads(Path(args.scene_json).read_text())
    d = Path(args.scene_json).parent
    plot_scene(scene, out)
    plot_directivity(scene, d, out)
    plot_tdoa(scene, d, out, scene["fs"])
    print(f"plots written to {out}")


if __name__ == "__main__":
    main()
