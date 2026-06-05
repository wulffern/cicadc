"""A simple uniform mid-tread N-bit quantizer.

The input range is the bipolar full-scale interval ``[-vref, vref]`` which is
divided into ``2**bits`` equal levels. Values outside the range are clamped
(saturation), as a real ADC would.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantizeResult:
    code: int  # integer output code in [0, 2**bits - 1]
    level: float  # reconstructed analog value for that code
    error: float  # quantization error: input - level


@dataclass
class Quantizer:
    bits: int = 3
    vref: float = 1.0  # full-scale magnitude; range is [-vref, vref]

    @property
    def num_levels(self) -> int:
        return 1 << self.bits

    @property
    def step(self) -> float:
        """Width of one quantization interval (LSB size)."""
        return (2.0 * self.vref) / self.num_levels

    def code_of(self, value: float) -> int:
        """Return the integer code for ``value``, clamped to the valid range."""
        vmin = -self.vref
        idx = int((value - vmin) // self.step)
        return max(0, min(self.num_levels - 1, idx))

    def level_of(self, code: int) -> float:
        """Reconstructed analog value at the centre of interval ``code``."""
        return -self.vref + (code + 0.5) * self.step

    def quantize(self, value: float) -> QuantizeResult:
        code = self.code_of(value)
        level = self.level_of(code)
        return QuantizeResult(code=code, level=level, error=value - level)

    def code_string(self, code: int) -> str:
        """Binary representation of ``code`` padded to ``bits`` digits."""
        return format(code, f"0{self.bits}b")

    def levels(self) -> list[float]:
        """All reconstructed levels, lowest to highest."""
        return [self.level_of(c) for c in range(self.num_levels)]
