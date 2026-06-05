"""Time-varying analog signal model for the ADC demonstration.

The signal is modelled as a continuous function of *absolute* time. The view
window always spans ``[t_now, t_now + window]`` where ``t_now`` is "now" (the
bottom x-axis of the analog panel) and ``t_now + window`` is the furthest
future (top of the panel). Advancing time makes the curve scroll downwards.

For the MVP the only input shape is a single sinusoid, but the design keeps the
amplitude/frequency as plain parameters and isolates the waveform in
``_waveform`` so that random and multi-sinusoid inputs can be added later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class SignalSource:
    """A scrolling analog signal sampled by a fixed-rate sample clock.

    Attributes are normalised so that the full-scale signal range is ``[-1, 1]``
    (matching :class:`~cicadc.quantizer.Quantizer` defaults). ``amplitude`` is
    therefore a fraction of full scale.
    """

    frequency: float = 1.0  # Hz
    amplitude: float = 0.8  # fraction of full scale (0..1)
    speed: float = 1.0  # time units of signal per second of wall clock
    window: float = 4.0  # seconds of signal visible from "now" to "future"
    sample_period: float = 0.25  # seconds between ADC samples (sample clock)
    noise_amp: float = 0.0  # additive analog noise amplitude (fraction of FS)
    noise_dt: float = 0.06  # correlation time of the value noise

    t_now: float = 0.0
    _phase0: float = field(default=0.0, repr=False)

    def _clean(self, t: np.ndarray | float):
        """Noise-free sinusoid value(s) at absolute time ``t``."""
        return self.amplitude * np.sin(2.0 * np.pi * self.frequency * t + self._phase0)

    @staticmethod
    def _hash(n: np.ndarray):
        """Deterministic pseudo-random value in ``[-1, 1]`` for integer grid n."""
        x = np.sin(n * 12.9898) * 43758.5453
        return 2.0 * (x - np.floor(x)) - 1.0

    def _noise(self, t: np.ndarray | float):
        """Smooth value-noise that is a fixed function of absolute time, so the
        noise pattern scrolls with the signal instead of flickering per frame."""
        if not self.noise_amp:
            return np.zeros_like(np.asarray(t, dtype=float))
        i = np.asarray(t, dtype=float) / self.noise_dt
        i0 = np.floor(i)
        f = i - i0
        f = f * f * (3.0 - 2.0 * f)  # smoothstep
        r0, r1 = self._hash(i0), self._hash(i0 + 1.0)
        return self.noise_amp * (r0 * (1.0 - f) + r1 * f)

    def _waveform(self, t: np.ndarray | float):
        """Analog value(s) at absolute time ``t`` (sinusoid plus optional noise)."""
        return self._clean(t) + self._noise(t)

    def advance(self, dt: float) -> None:
        """Advance "now" by ``dt`` seconds of wall clock, scaled by ``speed``."""
        self.t_now += dt * self.speed

    def value_now(self) -> float:
        """Analog value at the most recent sample instant at or before now."""
        ts = self.latest_sample_time()
        return float(self._waveform(ts))

    def value_continuous(self, t_rel: float = 0.0) -> float:
        """Continuous analog value at ``t_now + t_rel`` (no sample-and-hold)."""
        return float(self._waveform(self.t_now + t_rel))

    def derivative(self, t_rel: float = 0.0) -> float:
        """d(value)/d(time) of the continuous signal at ``t_now + t_rel``."""
        t = self.t_now + t_rel
        return float(
            self.amplitude
            * 2.0
            * np.pi
            * self.frequency
            * np.cos(2.0 * np.pi * self.frequency * t + self._phase0)
        )

    def latest_sample_time(self) -> float:
        """Absolute time of the most recent sample instant at or before now."""
        return np.floor(self.t_now / self.sample_period) * self.sample_period

    def sample_index_now(self) -> int:
        """Index ``k`` of the most recent sample instant at or before now."""
        return int(np.floor(self.t_now / self.sample_period))

    def visible_sample_indices(self) -> Tuple[int, int]:
        """First and last sample indices visible in the window (inclusive)."""
        first_k = int(np.ceil(self.t_now / self.sample_period))
        last_k = int(np.floor((self.t_now + self.window) / self.sample_period))
        return first_k, last_k

    def sample_indices_in(self, t0_rel: float, t1_rel: float) -> Tuple[int, int]:
        """First/last sample indices whose time is within ``[now+t0, now+t1]``."""
        first_k = int(np.ceil((self.t_now + t0_rel) / self.sample_period))
        last_k = int(np.floor((self.t_now + t1_rel) / self.sample_period))
        return first_k, last_k

    def sample_index_at(self, t_rel: float) -> int:
        """Index of the most recent sample at or before ``now + t_rel``."""
        return int(np.floor((self.t_now + t_rel) / self.sample_period))

    def sample_value(self, k: int) -> float:
        """Analog value at sample instant ``k`` (absolute time ``k * Ts``)."""
        return float(self._waveform(k * self.sample_period))

    def t_rel_of_index(self, k: int) -> float:
        """Time offset from now of sample index ``k``."""
        return k * self.sample_period - self.t_now

    def curve_points(self, n: int = 240, t0_rel: float = 0.0, t1_rel: float | None = None):
        """Dense samples of the continuous curve over ``[now+t0, now+t1]``.

        Returns a list of ``(t_rel, value)`` where ``t_rel`` is the offset from
        now. Defaults to the full forward window when ``t1_rel`` is ``None``.
        """
        if t1_rel is None:
            t1_rel = self.window
        t_rel = np.linspace(t0_rel, t1_rel, n)
        values = self._waveform(self.t_now + t_rel)
        return list(zip(t_rel.tolist(), values.tolist()))

    def sample_instants(self) -> List[Tuple[float, float]]:
        """Visible ADC sample points as ``(t_rel, value)`` pairs.

        Sample instants are fixed in absolute time (a regular sample clock); as
        time advances they slide down the window and new ones appear at the top.
        Ordered from now (bottom) to future (top).
        """
        first_k = int(np.ceil(self.t_now / self.sample_period))
        last_k = int(np.floor((self.t_now + self.window) / self.sample_period))
        out: List[Tuple[float, float]] = []
        for k in range(first_k, last_k + 1):
            ts = k * self.sample_period
            t_rel = ts - self.t_now
            out.append((t_rel, float(self._waveform(ts))))
        return out
