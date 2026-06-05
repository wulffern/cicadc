"""Delta-sigma (sigma-delta) modulators, 1st and 2nd order.

The 1st-order modulator mirrors ``ex/sd_1st.py``: a single integrator
accumulates the error between the input and the fed-back quantizer output, and a
coarse quantizer (1-bit by default) closes the loop::

    x[n] = x[n-1] + (u[n] - y[n-1])          # integrator
    y[n] = Q(x[n] + dither[n])               # coarse quantizer + feedback

The 2nd-order modulator is the standard CIFB (cascade-of-integrators,
feedback) structure with the quantized output fed back to both integrator
inputs::

    x1[n] = x1[n-1] + g1 * (b1 * u[n]  - a1 * y[n-1])
    x2[n] = x2[n-1] + g1 * (     x1[n] - a2 * y[n-1])
    y[n]  = Q(x2[n] + dither[n])

For delaying integrators the noise transfer function is ``(1 - z^-1)**2`` when
``a1 = 1, a2 = 2`` (with ``b1 = 1, g1 = 1``) - the standard stable single-bit
design - which pushes more quantization noise out of band than 1st order.

``Q`` is a true ``bits``-bit mid-rise quantizer: it has exactly ``2**bits``
output levels spaced symmetrically across ``[-vref, vref]`` (so a 1-bit
quantizer outputs only ``+/- vref``). Averaging or decimating the coarse output
stream ``y`` recovers the input ``u`` with the quantization noise shaped
towards high frequencies.

The modulators are evaluated on the app's fixed, absolute-time sample grid
(index ``k`` corresponds to time ``k * Ts``). Because the loop has memory,
outputs are produced by a causal recursion and cached per index, so a fixed
``k`` always yields the same value across animation frames (no flicker). When
evaluation starts "cold" at some index a short warm-up settles the integrators.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


class _SigmaDeltaBase:
    """Shared quantizer, dither, caching and warm-up for the modulators.

    Subclasses define the loop's state and its one-step recursion via
    :meth:`_zero_state` and :meth:`_advance`. The per-index cache stores the full
    state tuple whose last element is the coarse output ``y[k]``.
    """

    order: int = 0

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
        # k -> state tuple; the last element is the output y[k].
        self._cache: Dict[int, Tuple[float, ...]] = {}

    def reset(self) -> None:
        """Drop all cached state (call when input/quantizer parameters change)."""
        self._cache.clear()

    @staticmethod
    def _dither_value(k: int) -> float:
        """Deterministic pseudo-random dither in ``[-1, 1]`` for sample index k."""
        x = np.sin(k * 78.233 + 1.0) * 43758.5453
        return 2.0 * (x - np.floor(x)) - 1.0

    def _quantize(self, v: float, k: int) -> float:
        """True ``bits``-bit mid-rise quantizer: ``2**bits`` levels symmetric in
        ``[-vref, vref]``. A 1-bit quantizer therefore returns only ``+/- vref``."""
        if self.dither:
            v = v + self.dither * 0.5 * self._dither_value(k)
        levels = 1 << self.bits
        if levels <= 2:
            return self.vref if v >= 0.0 else -self.vref
        step = 2.0 * self.vref / (levels - 1)
        idx = int(round((v + self.vref) / step))
        idx = max(0, min(levels - 1, idx))
        return -self.vref + idx * step

    def _zero_state(self) -> Tuple[float, ...]:
        raise NotImplementedError

    def _advance(self, prev: Tuple[float, ...], k: int) -> Tuple[float, ...]:
        raise NotImplementedError

    def output(self, k: int) -> float:
        """Coarse modulator output ``y[k]`` (a reconstructed quantizer level)."""
        hit = self._cache.get(k)
        if hit is not None:
            return hit[-1]
        prev = self._cache.get(k - 1)
        if prev is not None:
            state = self._advance(prev, k)
            self._cache[k] = state
            return state[-1]
        # Cold start: warm up the integrators from a settled-enough past.
        state = self._zero_state()
        for n in range(k - self.warmup, k + 1):
            state = self._advance(state, n)
            self._cache[n] = state
        return state[-1]

    def prepare(self, k0: int, k1: int) -> None:
        """Ensure outputs for the contiguous range ``[k0, k1]`` are cached.

        Evaluating ascending lets each index extend the previous one in O(1)
        after a single cold warm-up at ``k0``.
        """
        for k in range(k0, k1 + 1):
            self.output(k)


class SigmaDelta1(_SigmaDeltaBase):
    """First-order sigma-delta modulator (single integrator)."""

    order = 1

    def _zero_state(self) -> Tuple[float, float]:
        return (0.0, 0.0)  # (x, y)

    def _advance(self, prev: Tuple[float, ...], k: int) -> Tuple[float, float]:
        x_prev, y_prev = prev
        x = x_prev + (self.input_fn(k) - y_prev)
        y = self._quantize(x, k)
        return (x, y)


class SigmaDelta2(_SigmaDeltaBase):
    """Second-order CIFB sigma-delta modulator (two integrators, feedback).

    The loop coefficients default to the standard stable single-bit design
    ``a1 = 1, a2 = 2, b1 = 1, g1 = 1`` (delaying integrators), which realises a
    ``(1 - z^-1)**2`` noise transfer function.
    """

    order = 2

    def __init__(
        self,
        input_fn: Callable[[int], float],
        bits: int = 1,
        vref: float = 1.0,
        dither: float = 0.0,
        warmup: int = 128,
        a1: float = 1.0,
        a2: float = 2.0,
        b1: float = 1.0,
        g1: float = 1.0,
    ) -> None:
        super().__init__(input_fn, bits=bits, vref=vref, dither=dither, warmup=warmup)
        self.a1 = a1  # feedback to the 1st integrator
        self.a2 = a2  # feedback to the 2nd integrator
        self.b1 = b1  # input scaling
        self.g1 = g1  # integrator gain

    def _zero_state(self) -> Tuple[float, float, float]:
        return (0.0, 0.0, 0.0)  # (x1, x2, y)

    def _advance(self, prev: Tuple[float, ...], k: int) -> Tuple[float, float, float]:
        x1_prev, x2_prev, y_prev = prev
        x1 = x1_prev + self.g1 * (self.b1 * self.input_fn(k) - self.a1 * y_prev)
        x2 = x2_prev + self.g1 * (x1 - self.a2 * y_prev)
        y = self._quantize(x2, k)
        return (x1, x2, y)
