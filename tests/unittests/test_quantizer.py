#!/usr/bin/env python3
"""Unit tests for cicadc.quantizer (Qt-free)."""

import unittest

from cicadc.quantizer import Quantizer


class TestQuantizer(unittest.TestCase):
    def test_levels_and_step(self):
        q = Quantizer(bits=3, vref=1.0)
        self.assertEqual(q.num_levels, 8)
        self.assertAlmostEqual(q.step, 0.25)
        levels = q.levels()
        self.assertEqual(len(levels), 8)
        self.assertAlmostEqual(levels[0], -0.875)
        self.assertAlmostEqual(levels[-1], 0.875)

    def test_code_string_padding(self):
        q = Quantizer(bits=4)
        self.assertEqual(q.code_string(6), "0110")
        self.assertEqual(q.code_string(0), "0000")

    def test_saturation(self):
        q = Quantizer(bits=3, vref=1.0)
        self.assertEqual(q.code_of(5.0), q.num_levels - 1)
        self.assertEqual(q.code_of(-5.0), 0)

    def test_quantize_error_within_half_lsb(self):
        q = Quantizer(bits=5, vref=1.0)
        for v in (-0.9, -0.3, 0.0, 0.2, 0.77, 0.95):
            r = q.quantize(v)
            self.assertLessEqual(abs(r.error), q.step / 2 + 1e-9)
            self.assertEqual(r.level, q.level_of(r.code))


if __name__ == "__main__":
    unittest.main()
