#!/usr/bin/env python3
"""Unit tests for cicadc.sigma_delta (Qt-free)."""

import unittest

from cicadc.sigma_delta import SigmaDelta1


class TestSigmaDelta1(unittest.TestCase):
    def test_one_bit_levels_in_range(self):
        # bits=1 quantizes to multiples of 0.5, saturated to [-1, 1].
        sd = SigmaDelta1(input_fn=lambda k: 0.4, bits=1)
        sd.prepare(0, 200)
        for k in range(0, 201):
            y = sd.output(k)
            self.assertLessEqual(abs(y), 1.0 + 1e-9)
            self.assertAlmostEqual(y * 2.0, round(y * 2.0), places=9)

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


if __name__ == "__main__":
    unittest.main()
