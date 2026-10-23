import numpy as np
from scipy.signal import convolve

from fastmss.spatial.render import combine_schedules, eval_schedule, interp_weights, render


def test_static_equals_convolution():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    h = rng.standard_normal((3, 700))
    y = render(x, [0], h[None])
    ref = convolve(x[None, :], h, mode="full")
    assert y.shape == ref.shape
    np.testing.assert_allclose(y, ref, atol=1e-9)


def test_weights_partition_of_unity():
    n = 4000
    times = [0, 333, 1000, 1001, 2500, 3999]
    acc = np.zeros(n)
    for start, stop, w in interp_weights(n, times):
        acc[start:stop] += w
    np.testing.assert_allclose(acc, 1.0, atol=1e-12)
    # starts before the first waypoint / ends after the last one
    acc = np.zeros(n)
    for start, stop, w in interp_weights(n, [500, 1200]):
        acc[start:stop] += w
    np.testing.assert_allclose(acc, 1.0, atol=1e-12)


def test_identical_rirs_crossfade_equals_convolution():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(6000)
    h = rng.standard_normal((2, 300))
    times = [0, 1000, 2500, 5999]
    y = render(x, times, np.repeat(h[None], len(times), 0))
    np.testing.assert_allclose(y, convolve(x[None, :], h, mode="full"), atol=1e-9)


def test_crossfade_is_linear_in_rirs():
    # with two waypoints, the output is the sum of the two weighted renders
    rng = np.random.default_rng(2)
    x = rng.standard_normal(3000)
    h0, h1 = rng.standard_normal((2, 1, 200))
    y = render(x, [0, 2999], np.stack([h0, h1]))
    w1 = np.linspace(0, 1, 3000)
    ref = convolve((x * (1 - w1))[None], h0) + convolve((x * w1)[None], h1)
    np.testing.assert_allclose(y, ref, atol=1e-9)


def test_schedules():
    s = [(0.0, "a"), (10.0, "b")]
    assert eval_schedule(s, -1) == [(1.0, "a")]
    assert eval_schedule(s, 20) == [(1.0, "b")]
    (wa, a), (wb, b) = eval_schedule(s, 2.5)
    assert (a, b) == ("a", "b") and np.isclose(wa, 0.75) and np.isclose(wb, 0.25)
    times, combos = combine_schedules([(0, 1), (100, 2)], [(50, 7)], 200)
    assert list(times) == [0, 50, 100, 199]
    for c in combos:
        assert np.isclose(sum(w for w, _, _ in c), 1.0)
    # mic pose change that started before the utterance: state at 0 is interpolated
    times, combos = combine_schedules([(0, 1)], [(-50, 3), (50, 4)], 200)
    assert times[0] == 0
    assert sorted((p, round(w, 3)) for w, p, _ in combos[0]) == [(3, 0.5), (4, 0.5)]
