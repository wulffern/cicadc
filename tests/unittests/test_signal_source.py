#!/usr/bin/env python3
"""Unit tests for cicadc.signal_source (Qt-free)."""

import math
import unittest

from cicadc.signal_source import SignalSource


class TestSignalSource(unittest.TestCase):
    def test_sinusoid_value(self):
        s = SignalSource(frequency=1.0, amplitude=0.5, sample_period=0.25)
        # At t=0 the sine is 0; at quarter period it is the amplitude.
        self.assertAlmostEqual(s.value_continuous(0.0), 0.0, places=6)
        self.assertAlmostEqual(s.value_continuous(0.25), 0.5, places=6)

    def test_advance_scaled_by_speed(self):
        s = SignalSource(speed=2.0)
        s.advance(0.5)
        self.assertAlmostEqual(s.t_now, 1.0)

    def test_noise_is_deterministic_in_time(self):
        s = SignalSource(frequency=0.0, amplitude=0.0, noise_amp=0.3)
        a = s.value_continuous(1.234)
        b = s.value_continuous(1.234)
        self.assertEqual(a, b)  # same time -> same noise sample
        self.assertNotEqual(a, s.value_continuous(2.5))

    def test_no_noise_when_amp_zero(self):
        s = SignalSource(frequency=0.0, amplitude=0.0, noise_amp=0.0)
        self.assertEqual(s.value_continuous(3.14), 0.0)

    def test_sample_indices_span_now(self):
        s = SignalSource(sample_period=0.25, window=4.0)
        s.t_now = 1.0
        first_k, last_k = s.sample_indices_in(-2.0, 2.0)
        self.assertEqual(first_k * 0.25, -1.0)
        self.assertEqual(last_k * 0.25, 3.0)

    def test_derivative_matches_finite_difference(self):
        s = SignalSource(frequency=0.7, amplitude=0.8)
        h = 1e-5
        approx = (s.value_continuous(h) - s.value_continuous(-h)) / (2 * h)
        self.assertTrue(math.isclose(approx, s.derivative(0.0), rel_tol=1e-3, abs_tol=1e-4))


if __name__ == "__main__":
    unittest.main()
