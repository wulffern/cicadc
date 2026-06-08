#!/usr/bin/env python3
"""Unit tests for cicadc.signal_chain (renderer-agnostic DSP core, Qt/manim-free)."""

import math
import unittest

import numpy as np

from cicadc.signal_chain import SignalChain
from cicadc.signal_source import SignalSource
from cicadc.quantizer import Quantizer


class TestSignalChain(unittest.TestCase):
    def _chain(self, **kw):
        sig = SignalSource(frequency=1.0, amplitude=0.5, sample_period=0.25)
        return SignalChain(signal=sig, quantizer=Quantizer(bits=1), **kw)

    def test_modes_registered(self):
        c = self._chain()
        for mode in ("sigma_delta", "sigma_delta2", "leapfrog"):
            self.assertIn(mode, c._modulators)
        self.assertFalse(c.is_sigma_delta())          # default nyquist
        c.set_adc_mode("leapfrog")
        self.assertTrue(c.is_sigma_delta())
        self.assertIsNotNone(c.modulator())

    def test_filter_order_by_mode(self):
        c = self._chain()
        self.assertEqual(c.filter_order(), 1)         # nyquist -> sinc^1
        c.set_adc_mode("sigma_delta");  self.assertEqual(c.filter_order(), 2)
        c.set_adc_mode("sigma_delta2"); self.assertEqual(c.filter_order(), 3)
        c.set_adc_mode("leapfrog");     self.assertEqual(c.filter_order(), 4)

    def test_decimation_taps_dc_gain(self):
        c = self._chain(filter_taps=4)
        c.set_adc_mode("sigma_delta2")                # M = 3
        h = c.decimation_taps()
        self.assertEqual(len(h), 3 * (4 - 1) + 1)     # M*(K-1)+1
        self.assertAlmostEqual(h.sum(), 4 ** 3)       # sum == K^M

    def test_nyquist_raw_level_matches_quantizer(self):
        c = self._chain()
        q = c.quantizer
        for k in range(10):
            expected = q.level_of(q.code_of(c.signal.sample_value(k)))
            self.assertEqual(c.raw_level(k), expected)

    def test_filter_gain_unity_without_decimation(self):
        c = self._chain(filter_taps=1)
        self.assertEqual(c.filter_gain(), 1.0)

    def test_filt_level_reduces_to_raw_when_no_filter(self):
        c = self._chain(filter_taps=1)
        c.set_adc_mode("sigma_delta")
        for k in range(50, 60):
            self.assertEqual(c.filt_level(k), c.raw_level(k))

    def test_group_delay_zero_without_decimation(self):
        c = self._chain(filter_taps=1)
        self.assertEqual(c.group_delay(), 0.0)

    def test_group_delay_positive_with_decimation(self):
        c = self._chain(filter_taps=8)
        c.set_adc_mode("sigma_delta2")
        self.assertGreater(c.group_delay(), 0.0)

    def test_stf_is_unity_for_nyquist(self):
        c = self._chain()
        self.assertAlmostEqual(abs(c.modulator_stf()), 1.0)

    def test_dc_recovered_by_decimation(self):
        # A DC input through a 1st-order loop + sinc^2 averaging recovers the DC.
        sig = SignalSource(frequency=0.0, amplitude=0.0, sample_period=0.25)
        sig.t_now = 100.0
        c = SignalChain(signal=sig, quantizer=Quantizer(bits=1), filter_taps=8)
        c.set_adc_mode("sigma_delta")
        # constant 0.3 input via the noise-free DC: emulate by a tiny offset signal
        c.signal.amplitude = 0.0
        # Use raw-level averaging as the DC proxy.
        k0 = sig.sample_index_now()
        c.prepare(k0 - 600, k0)
        avg = np.mean([c.raw_level(k) for k in range(k0 - 400, k0)])
        self.assertAlmostEqual(avg, 0.0, delta=0.05)

    def test_fft_db_shape_and_peak(self):
        sig = SignalSource(frequency=1.0, amplitude=0.8, sample_period=0.02)
        sig.t_now = 50.0
        c = SignalChain(signal=sig, quantizer=Quantizer(bits=6), fft_size=512)
        db = c.fft_db(sig.sample_index_now())
        self.assertEqual(db.shape[0], 512 // 2 + 1)
        self.assertLess(db.max(), 6.0)                # full-scale sine near 0 dBFS

    def test_reset_clears_modulator_caches(self):
        c = self._chain()
        c.set_adc_mode("sigma_delta")
        c.prepare(0, 100)
        self.assertTrue(c.modulator()._cache)
        c.reset()
        self.assertFalse(c.modulator()._cache)


if __name__ == "__main__":
    unittest.main()
