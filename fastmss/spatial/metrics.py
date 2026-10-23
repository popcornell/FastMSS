"""Signal metrics used to verify the spatial simulation (GCC-PHAT, SRP-PHAT,
Schroeder RT60, DRR, magnitude-squared coherence)."""

import numpy as np
from scipy.signal import csd, welch

SPEED_OF_SOUND = 343.0


def gcc_phat(x, y, max_lag=None, interp=16, band=None, fs=16000):
    """Delay of y relative to x in (fractional) samples; positive if y lags x.

    band: optional (fmin, fmax) in Hz; PHAT whitens every bin, so for
    band-limited signals restrict it to the band that carries signal."""
    n = len(x) + len(y)
    X = np.fft.rfft(x, n)
    Y = np.fft.rfft(y, n)
    R = Y * np.conj(X)
    R /= np.abs(R) + 1e-12
    if band is not None:
        f = np.fft.rfftfreq(n, 1 / fs)
        R[(f < band[0]) | (f > band[1])] = 0.0
    cc = np.fft.irfft(R, n * interp)
    max_shift = n * interp // 2
    if max_lag is not None:
        max_shift = min(int(interp * max_lag), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    return (np.argmax(np.abs(cc)) - max_shift) / interp


def gcc_phat_track(x, y, frame, hop, max_lag=None, interp=16):
    """Frame-wise GCC-PHAT delays (samples) and frame centers (samples)."""
    out, centers = [], []
    for s in range(0, len(x) - frame + 1, hop):
        out.append(gcc_phat(x[s : s + frame], y[s : s + frame], max_lag, interp))
        centers.append(s + frame // 2)
    return np.array(out), np.array(centers)


def srp_phat_azimuth(signals, mic_pos, fs, n_az=360, nfft=1024):
    """Far-field SRP-PHAT azimuth (radians, world frame) for a horizontal array.

    signals: (M, N); mic_pos: (M, 3)."""
    M = signals.shape[0]
    hop = nfft // 2
    win = np.hanning(nfft)
    frames = [
        np.fft.rfft(signals[:, s : s + nfft] * win, axis=-1)
        for s in range(0, signals.shape[1] - nfft + 1, hop)
    ]
    S = np.stack(frames, 0)  # (T, M, F)
    S = S / (np.abs(S) + 1e-12)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    az = np.linspace(-np.pi, np.pi, n_az, endpoint=False)
    u = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], 1)  # (A, 3)
    center = mic_pos.mean(0)
    # arrival advance of each mic for a plane wave from direction u
    tau = (mic_pos - center) @ u.T / SPEED_OF_SOUND  # (M, A)
    power = np.zeros(n_az)
    for a in range(n_az):
        steer = np.exp(-2j * np.pi * f[None, :] * tau[:, a][:, None])  # (M, F)
        beam = np.sum(S * steer[None], axis=1)
        power[a] = np.sum(np.abs(beam) ** 2)
    return az[np.argmax(power)], az, power


def schroeder_rt60(rir, fs, lo=-5.0, hi=-25.0):
    """RT60 from the Schroeder decay curve (T20-style fit, extrapolated)."""
    e = np.cumsum(rir[::-1] ** 2)[::-1]
    edc = 10 * np.log10(e / e[0] + 1e-30)
    i0 = np.argmax(edc <= lo)
    i1 = np.argmax(edc <= hi)
    if i1 <= i0:
        return np.nan
    t = np.arange(i0, i1) / fs
    slope, _ = np.polyfit(t, edc[i0:i1], 1)
    return -60.0 / slope


def drr_db(rir, direct_idx, fs, window_ms=2.5):
    """Direct-to-reverberant ratio around direct_idx."""
    w = int(window_ms * 1e-3 * fs)
    a, b = max(0, int(direct_idx) - w), int(direct_idx) + w + 1
    d = np.sum(rir[a:b] ** 2)
    r = np.sum(rir[b:] ** 2) + np.sum(rir[:a] ** 2)
    return 10 * np.log10(d / (r + 1e-30) + 1e-30)


def msc(x, y, fs, nperseg=512):
    """Magnitude-squared coherence (frequencies, values)."""
    f, pxy = csd(x, y, fs, nperseg=nperseg)
    _, pxx = welch(x, fs, nperseg=nperseg)
    _, pyy = welch(y, fs, nperseg=nperseg)
    return f, np.abs(pxy) ** 2 / (pxx * pyy + 1e-30)


def diffuse_coherence(f, d):
    """Theoretical spherically-isotropic coherence sinc^2(2 pi f d / c)."""
    return np.sinc(2 * f * d / SPEED_OF_SOUND) ** 2
