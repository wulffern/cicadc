#!/usr/bin/env python3
"""Unit tests for cicadc.sigma_delta (Qt-free)."""

import math
import unittest

from cicadc.sigma_delta import SigmaDelta1, SigmaDelta2, Leapfrog


class TestSigmaDelta1(unittest.TestCase):
    def test_one_bit_has_exactly_two_levels(self):
        # A true 1-bit quantizer outputs only +/- vref (two distinct levels).
        sd = SigmaDelta1(input_fn=lambda k: 0.4, bits=1, vref=1.0)
        sd.prepare(0, 400)
        levels = {sd.output(k) for k in range(0, 401)}
        self.assertTrue(levels.issubset({-1.0, 1.0}))
        self.assertLessEqual(len(levels), 2)

    def test_multibit_levels_span_full_scale(self):
        # bits=2 -> 4 levels symmetric across [-1, 1] including the extremes.
        sd = SigmaDelta1(input_fn=lambda k: 0.37, bits=2, vref=1.0)
        sd.prepare(0, 400)
        allowed = [-1.0, -1 / 3, 1 / 3, 1.0]
        outputs = {sd.output(k) for k in range(0, 401)}
        self.assertLessEqual(len(outputs), 4)
        for y in outputs:
            self.assertTrue(min(abs(y - a) for a in allowed) < 1e-9)

    def test_dc_is_recovered_by_averaging(self):
        # The average of the coarse 1-bit stream tracks the DC input.
        dc = 0.3
        sd = SigmaDelta1(input_fn=lambda k: dc, bits=1)
        sd.prepare(0, 4000)
        ys = [sd.output(k) for k in range(200, 4001)]
        self.assertAlmostEqual(sum(ys) / len(ys), dc, delta=0.02)

    def test_deterministic_without_dither(self):
        def sig(k):
            return 0.5  # constant; exact value irrelevant

        a = SigmaDelta1(input_fn=sig, bits=1)
        b = SigmaDelta1(input_fn=sig, bits=1)
        a.prepare(0, 300)
        b.prepare(0, 300)
        for k in range(0, 301):
            self.assertEqual(a.output(k), b.output(k))

    def test_cache_extends_forward_consistently(self):
        # Reading a cached index, then extending, must not change earlier values.
        sd = SigmaDelta1(input_fn=lambda k: 0.2, bits=1)
        sd.prepare(0, 100)
        snapshot = [sd.output(k) for k in range(0, 101)]
        sd.prepare(101, 200)  # extend the chain
        for k in range(0, 101):
            self.assertEqual(sd.output(k), snapshot[k])

    def test_reset_clears_cache(self):
        sd = SigmaDelta1(input_fn=lambda k: 0.2, bits=1)
        sd.prepare(0, 50)
        self.assertTrue(sd._cache)
        sd.reset()
        self.assertFalse(sd._cache)


class TestSigmaDelta2(unittest.TestCase):
    def test_order_attribute(self):
        self.assertEqual(SigmaDelta2(input_fn=lambda k: 0.0).order, 2)

    def test_one_bit_has_exactly_two_levels(self):
        sd = SigmaDelta2(input_fn=lambda k: 0.3, bits=1, vref=1.0)
        sd.prepare(0, 400)
        levels = {sd.output(k) for k in range(0, 401)}
        self.assertTrue(levels.issubset({-1.0, 1.0}))
        self.assertLessEqual(len(levels), 2)

    def test_default_coefficients_are_standard(self):
        sd = SigmaDelta2(input_fn=lambda k: 0.0)
        self.assertEqual((sd.a1, sd.a2, sd.b1, sd.g1), (1.0, 2.0, 1.0, 1.0))

    def test_dc_is_recovered_by_averaging(self):
        dc = 0.3
        sd = SigmaDelta2(input_fn=lambda k: dc, bits=1)
        sd.prepare(0, 4000)
        ys = [sd.output(k) for k in range(400, 4001)]
        self.assertAlmostEqual(sum(ys) / len(ys), dc, delta=0.02)

    def test_deterministic_and_cache_extends(self):
        sd = SigmaDelta2(input_fn=lambda k: 0.25, bits=1)
        sd.prepare(0, 100)
        snapshot = [sd.output(k) for k in range(0, 101)]
        sd.prepare(101, 200)
        for k in range(0, 101):
            self.assertEqual(sd.output(k), snapshot[k])

    def test_loop_is_stable_for_moderate_input(self):
        # A sine within the stable input range must not let the integrators run
        # away: the output stays in range and the integrator states stay bounded.
        sd = SigmaDelta2(input_fn=lambda k: 0.5 * math.sin(2 * math.pi * k / 64.0), bits=1)
        sd.prepare(0, 4000)
        for k in range(0, 4001):
            x1, x2, y = sd._cache[k]
            self.assertLessEqual(abs(y), 1.0 + 1e-9)
            self.assertLess(abs(x1), 10.0)
            self.assertLess(abs(x2), 10.0)


class TestLeapfrog(unittest.TestCase):
    def test_order_is_three(self):
        self.assertEqual(Leapfrog(input_fn=lambda k: 0.0).order, 3)

    def test_one_bit_has_two_levels(self):
        lf = Leapfrog(input_fn=lambda k: 0.3, bits=1, vref=1.0)
        lf.prepare(0, 400)
        levels = {lf.output(k) for k in range(0, 401)}
        self.assertTrue(levels.issubset({-1.0, 1.0}))

    def test_states_bounded_to_full_scale(self):
        # The tuned coefficients keep all three integrators bounded even for a
        # full-scale sine - the design's max stable amplitude is ~1.0 FS.
        lf = Leapfrog(input_fn=lambda k: 1.0 * math.sin(2 * math.pi * k / 128.0), bits=1)
        lf.prepare(0, 6000)
        for k in range(200, 6001):
            x1, x2, x3, y = lf._cache[k]
            self.assertLessEqual(abs(y), 1.0 + 1e-9)
            self.assertLess(abs(x1), 5.0)
            self.assertLess(abs(x2), 8.0)
            self.assertLess(abs(x3), 12.0)

    def test_dc_recovered_with_half_scale_stf(self):
        # Input scaling b1 = a1/2 makes the signal-transfer gain 1/2, so the
        # coarse stream averages to half the DC input.
        dc = 0.4
        lf = Leapfrog(input_fn=lambda k: dc, bits=1)
        lf.prepare(0, 6000)
        ys = [lf.output(k) for k in range(400, 6001)]
        self.assertAlmostEqual(sum(ys) / len(ys), 0.5 * dc, delta=0.02)

    def test_noise_shaping_beats_second_order(self):
        # In-band quantization noise (sum of |y - mean| spectrum below f_s/2/OSR)
        # should be much lower than the 2nd-order loop for the same input.
        import cmath

        def insig(k):
            return 0.5 * math.sin(2 * math.pi * k * (1.0 / 256.0))

        def inband_noise(mod):
            mod.prepare(0, 8192)
            ys = [mod.output(k) for k in range(0, 8192)]
            # crude in-band power: DFT magnitude over the lowest OSR=32 band,
            # excluding the signal bin at k=8192/256=32.
            N = len(ys)
            osr_bins = N // (2 * 32)
            pw = 0.0
            for b in range(1, osr_bins):
                if abs(b - N // 256) <= 1:
                    continue
                acc = sum(ys[n] * cmath.exp(-2j * math.pi * b * n / N) for n in range(N))
                pw += abs(acc) ** 2
            return pw

        lf = inband_noise(Leapfrog(input_fn=insig, bits=1))
        sd2 = inband_noise(SigmaDelta2(input_fn=insig, bits=1))
        self.assertLess(lf, sd2)

    def test_deterministic_and_cache_extends(self):
        lf = Leapfrog(input_fn=lambda k: 0.25, bits=1)
        lf.prepare(0, 100)
        snapshot = [lf.output(k) for k in range(0, 101)]
        lf.prepare(101, 200)
        for k in range(0, 101):
            self.assertEqual(lf.output(k), snapshot[k])


if __name__ == "__main__":
    unittest.main()
