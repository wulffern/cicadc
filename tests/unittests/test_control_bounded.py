#!/usr/bin/env python3
"""Unit tests for cicadc.control_bounded (numpy + scipy, Qt-free)."""

import math
import unittest

import numpy as np
import scipy.linalg

from cicadc.control_bounded import LeapfrogCBADC


def _snr_db(u, fsig, T, BW):
    u = u[np.isfinite(u)]
    w = np.blackman(len(u))
    P = np.abs(np.fft.rfft(u * w)) ** 2
    fr = np.fft.rfftfreq(len(u), T)
    sb = int(np.argmin(np.abs(fr - fsig)))
    band = fr <= BW
    sig = P[max(0, sb - 3): sb + 4].sum()
    noise = P[band].copy()
    noise[max(0, sb - 3): sb + 4] = 0
    return 10.0 * np.log10(sig / max(noise.sum(), 1e-30))


class TestLeapfrogCBADC(unittest.TestCase):
    def setUp(self):
        # Order-3 leapfrog, BW=1 (normalised). OSR ~30.
        self.adc = LeapfrogCBADC(N=3, BW=1.0, ENOB=12)

    def test_system_shapes_and_leapfrog_structure(self):
        a = self.adc
        self.assertEqual(a.A.shape, (3, 3))
        self.assertEqual(a.B.shape, (3, 1))
        self.assertEqual(a.Gamma.shape, (3, 3))
        # Leapfrog: zero diagonal, forward beta on the sub-diagonal, backward
        # alpha on the super-diagonal.
        self.assertTrue(np.allclose(np.diag(a.A), 0.0))
        self.assertAlmostEqual(a.A[1, 0], a.beta)
        self.assertAlmostEqual(a.A[0, 1], a.alpha)
        self.assertGreater(a.OSR, 10.0)

    def test_control_is_one_bit(self):
        s = self.adc.simulate(lambda t: 0.3 * math.sin(2 * math.pi * (self.adc.BW / 7) * t), 2000)
        self.assertEqual(s.shape, (2000, 3))
        self.assertTrue(set(np.unique(s)).issubset({-1.0, 1.0}))

    def test_states_stay_bounded(self):
        # Control-bounded: the local feedback must keep the integrators bounded.
        a = self.adc
        Ad = scipy.linalg.expm(a.A * a.T)
        aug = np.zeros((6, 6)); aug[:3, :3] = a.A; aug[:3, 3:] = np.eye(3)
        Md = scipy.linalg.expm(aug * a.T)[:3, 3:]
        x = np.zeros(3)
        worst = 0.0
        for k in range(4000):
            sk = np.where(x >= 0, 1.0, -1.0)
            x = Ad @ x + Md @ (a.B.flatten() * 0.5 * math.sin(2 * math.pi * (a.BW / 7) * k * a.T) + a.Gamma @ sk)
            worst = max(worst, np.abs(x).max())
        # Bounded relative to the control gain |kappa| (states ~ a fraction of it).
        self.assertLess(worst, abs(a.kappa))

    def test_reconstructs_sine_with_reasonable_snr(self):
        a = self.adc
        fsig = a.BW / 7.0
        amp = 0.5
        n = 1 << 14
        s = a.simulate(lambda t: amp * math.sin(2 * math.pi * fsig * t), n)
        K1 = K2 = 256
        taps = a.estimator_taps(K1=K1, K2=K2)
        self.assertEqual(taps.shape, (K1 + K2, 3))
        u = a.reconstruct(s, taps, K1, K2)
        v = u[K1 + 4: n - K2 - 4]
        # Amplitude recovered and a clearly band-limited reconstruction.
        self.assertLess(abs(np.nanmax(np.abs(v)) - amp), 0.2)
        self.assertGreater(_snr_db(v, fsig, a.T, a.BW), 35.0)

    def test_default_eta2_positive(self):
        self.assertGreater(self.adc.default_eta2(), 0.0)


if __name__ == "__main__":
    unittest.main()
