"""First-order delta-sigma (sigma-delta) modulator.

Mirrors ``ex/sd_1st.py``: a single integrator accumulates the error between the
input and the fed-back quantizer output, and a coarse quantizer (1-bit by
default) closes the loop::

    x[n] = x[n-1] + (u[n] - y[n-1])          # integrator
    y[n] = Q(x[n] + dither[n])               # coarse quantizer + feedback

where ``Q(v) = round(v * 2**bits) / 2**bits`` (saturated to +/- vref). Averaging
or decimating the coarse output stream ``y`` recovers the input ``u`` with the
quantization noise shaped towards high frequencies - the whole point of a
sigma-delta converter.

The modulator is evaluated on the app's fixed, absolute-time sample grid (index
``k`` corresponds to time ``k * Ts``). Because the loop has memory, outputs are
produced by a causal recursion and cached per index, so a fixed ``k`` always
yields the same value across animation frames (no flicker). When evaluation
starts "cold" at some index a short warm-up settles the integrator first.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


class SigmaDelta1:
    """A 1st-order sigma-delta modulator over an absolute-time sample grid.

    ``input_fn(k)`` returns the (noisy) ADC input ``u[k]`` at sample index ``k``.
    ``dither`` is the dither amplitude in LSBs (0 disables it); the dither is a
    deterministic function of ``k`` so the animation does not flicker.
    """

    def __init__(
        self,
        input_fn: Callable[[int], float],
        bits: int = 1,
        vref: float = 1.0,
        dither: float = 0.0,
        warmup: int = 128,
    ) -> None:
        self.input_fn = input_fn
        self.bits = bits
        self.vref = vref
        self.dither = dither
        self.warmup = warmup
        # k -> (integrator state x[k], output y[k])
        self._cache: Dict[int, Tuple[float, float]] = {}

    def reset(self) -> None:
        """Drop all cached state (call when input/quantizer parameters change)."""
        self._cache.clear()

    @staticmethod
    def _dither_value(k: int) -> float:
        """Deterministic pseudo-random dither in ``[-1, 1]`` for sample index k."""
        x = np.sin(k * 78.233 + 1.0) * 43758.5453
        return 2.0 * (x - np.floor(x)) - 1.0

    def _quantize(self, v: float, k: int) -> float:
        scale = float(1 << self.bits)  # 2**bits
        code = v * scale
        if self.dither:
            code += self.dither * 0.5 * self._dither_value(k)
        y = np.round(code) / scale
        return float(np.clip(y, -self.vref, self.vref))

    def _step(self, k: int, x_prev: float, y_prev: float) -> Tuple[float, float]:
        x = x_prev + (self.input_fn(k) - y_prev)
        y = self._quantize(x, k)
        self._cache[k] = (x, y)
        return x, y

    def output(self, k: int) -> float:
        """Coarse modulator output ``y[k]`` (a reconstructed quantizer level)."""
        hit = self._cache.get(k)
        if hit is not None:
            return hit[1]
        prev = self._cache.get(k - 1)
        if prev is not None:
            return self._step(k, prev[0], prev[1])[1]
        # Cold start: warm up the integrator from a settled-enough past.
        x_prev, y_prev = 0.0, 0.0
        for n in range(k - self.warmup, k + 1):
            x_prev, y_prev = self._step(n, x_prev, y_prev)
        return y_prev

    def prepare(self, k0: int, k1: int) -> None:
        """Ensure outputs for the contiguous range ``[k0, k1]`` are cached.

        Evaluating ascending lets each index extend the previous one in O(1)
        after a single cold warm-up at ``k0``.
        """
        for k in range(k0, k1 + 1):
            self.output(k)
