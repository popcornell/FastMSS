"""Time-varying convolution by interpolation between precomputed RIRs.

A trajectory is a list of waypoints (sample index, RIR). Between waypoints the
input is split with triangular weights that sum to one (piecewise-linear
interpolation in time), each part is convolved with its waypoint RIR and the
results are overlap-added:

    y = sum_k h_k * (w_k . x)

Sound emitted near waypoint k reverberates with h_k, so the reverb tail of a
moving source stays physically consistent. With a single waypoint this is
exactly a plain convolution.
"""

import numpy as np
from scipy.signal import fftconvolve


def interp_weights(n, times):
    """Triangular interpolation weights.

    Args:
        n: signal length.
        times: strictly increasing waypoint sample indices (M,).

    Returns:
        list of (start, stop, weights) with weights.shape == (stop - start,),
        one per waypoint; the weights of all waypoints sum to one at every sample.
    """
    times = np.asarray(times, dtype=float)
    M = len(times)
    if M == 1:
        return [(0, n, np.ones(n))]
    out = []
    for m in range(M):
        start = 0 if m == 0 else int(np.floor(times[m - 1]))
        stop = n if m == M - 1 else int(np.ceil(times[m + 1])) + 1
        start, stop = max(0, start), min(n, stop)
        if stop <= start:
            out.append((0, 0, np.zeros(0)))
            continue
        idx = np.arange(start, stop)
        onehot = np.zeros(M)
        onehot[m] = 1.0
        out.append((start, stop, np.interp(idx, times, onehot)))
    return out


def render(x, times, rirs):
    """Render a mono signal through a sequence of multi-channel RIRs.

    Args:
        x: (N,) dry signal.
        times: (M,) strictly increasing waypoint sample indices.
        rirs: (M, C, L) RIR per waypoint.

    Returns:
        (C, N + L - 1) output.
    """
    x = np.asarray(x, dtype=np.float64)
    rirs = np.asarray(rirs, dtype=np.float64)
    M, C, L = rirs.shape
    N = len(x)
    if M == 1:
        return fftconvolve(x[None, :], rirs[0], axes=-1)
    y = np.zeros((C, N + L - 1))
    for (start, stop, w), h in zip(interp_weights(N, times), rirs):
        if stop <= start or not np.any(w):
            continue
        seg = x[start:stop] * w
        y[:, start : stop + L - 1] += fftconvolve(seg[None, :], h, axes=-1)
    return y


def eval_schedule(schedule, t):
    """Linear interpolation state of a schedule at time t.

    schedule: list of (time, key) with non-decreasing times.
    Returns list of (weight, key) (one or two entries)."""
    ts = [s[0] for s in schedule]
    if t <= ts[0]:
        return [(1.0, schedule[0][1])]
    if t >= ts[-1]:
        return [(1.0, schedule[-1][1])]
    k = int(np.searchsorted(ts, t, side="right")) - 1
    t0, t1 = ts[k], ts[k + 1]
    if t1 <= t0:
        return [(1.0, schedule[k + 1][1])]
    a = (t - t0) / (t1 - t0)
    out = []
    if a < 1.0:
        out.append((1.0 - a, schedule[k][1]))
    if a > 0.0:
        out.append((a, schedule[k + 1][1]))
    return out


def combine_schedules(src_sched, mic_sched, n):
    """Merge a source schedule and a mic-pose schedule (times in samples).

    Returns (times, combos) where times are strictly increasing sample indices
    in [0, n) and combos[m] is a list of (weight, pose, src) whose weighted RIR
    sum is the RIR at times[m]."""
    ts = {float(t) for t, _ in src_sched} | {float(t) for t, _ in mic_sched}
    ts = sorted({t for t in ts if 0 <= t < n} | {0.0})
    # keep the state at the end of the signal when a change crosses the end
    if len(src_sched) > 1 or len(mic_sched) > 1:
        ts = sorted(set(ts) | {float(max(n - 1, 0))})
    # merge waypoints closer than one sample
    merged = [ts[0]]
    for t in ts[1:]:
        if t - merged[-1] >= 1.0:
            merged.append(t)
    combos = []
    for t in merged:
        c = []
        for ws, s in eval_schedule(src_sched, t):
            for wp, p in eval_schedule(mic_sched, t):
                c.append((ws * wp, p, s))
        combos.append(c)
    return np.array(merged), combos
