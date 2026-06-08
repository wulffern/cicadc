"""Renderer-agnostic ADC signal-processing core.

``SignalChain`` owns the *model* side of the demonstration: the analog signal
source, the quantizer, the sigma-delta / leapfrog modulators, the sinc^M
decimation filter, the signal-transfer-function (STF) lock-in, the group-delay
and gain corrections, and the FFT of the digital output. It holds no drawing or
geometry state, so any renderer (the manim desktop scene, the browser canvas,
an iOS view, ...) can drive the *same* authoritative DSP instead of each
re-deriving the filter maths.

All public methods are pure functions of the model state (signal, quantizer,
modulator parameters, ``filter_taps``); time only enters through the signal's
``t_now`` and the absolute sample grid (index ``k`` <-> time ``k * Ts``), so a
given ``k`` always yields the same value across frames.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .quantizer import Quantizer
from .sigma_delta import SigmaDelta1, SigmaDelta2, Leapfrog, _SigmaDeltaBase
from .signal_source import SignalSource


class SignalChain:
    """The ADC model: signal -> modulator -> decimation filter -> digital output."""

    def __init__(
        self,
        signal: Optional[SignalSource] = None,
        quantizer: Optional[Quantizer] = None,
        filter_taps: int = 1,
        adc_mode: str = "nyquist",
        fft_size: int = 1024,
    ) -> None:
        self.signal = signal or SignalSource()
        self.quantizer = quantizer or Quantizer()
        self.filter_taps = filter_taps          # moving-average length (decimator)
        self.adc_mode = adc_mode
        self.sd_dither = 0.0
        self.fft_size = fft_size

        _input_fn = lambda k: self.signal.sample_value(k)  # noqa: E731
        self._modulators: Dict[str, _SigmaDeltaBase] = {
            "sigma_delta": SigmaDelta1(
                input_fn=_input_fn, bits=self.quantizer.bits, vref=self.quantizer.vref
            ),
            "sigma_delta2": SigmaDelta2(
                input_fn=_input_fn, bits=self.quantizer.bits, vref=self.quantizer.vref
            ),
            "leapfrog": Leapfrog(
                input_fn=_input_fn, bits=self.quantizer.bits, vref=self.quantizer.vref
            ),
        }

        # Measured modulator signal-transfer (cached per parameter set), and the
        # composite decimation FIR (cached per (K, M)).
        self._stf: complex | None = None
        self._stf_key: tuple | None = None
        self._fir: np.ndarray | None = None
        self._fir_key: tuple | None = None

    # ------------------------------------------------------------- ADC mode
    def is_sigma_delta(self) -> bool:
        return self.adc_mode in self._modulators

    def modulator(self) -> Optional[_SigmaDeltaBase]:
        """The active modulator, or ``None`` in Nyquist mode."""
        return self._modulators.get(self.adc_mode)

    def set_adc_mode(self, mode: str) -> None:
        self.adc_mode = mode
        self.reset()

    def set_dither(self, amount: float) -> None:
        self.sd_dither = amount
        self.reset()

    def reset(self) -> None:
        """Invalidate every modulator cache (input/quantizer parameters changed)."""
        for sd in self._modulators.values():
            sd.reset()

    def sync(self) -> None:
        """Mirror the live quantizer settings into the active modulator, resetting
        its cache if anything that changes the output sequence has changed."""
        sd = self.modulator()
        if sd is None:
            return
        if sd.bits != self.quantizer.bits or sd.vref != self.quantizer.vref or sd.dither != self.sd_dither:
            sd.bits = self.quantizer.bits
            sd.vref = self.quantizer.vref
            sd.dither = self.sd_dither
            sd.reset()

    def prepare(self, k_lo: int, k_hi: int) -> None:
        """Warm up / cache the active modulator over ``[k_lo, k_hi]``."""
        if self.is_sigma_delta():
            self.sync()
            self.modulator().prepare(k_lo, k_hi)

    # ------------------------------------------------------------- levels
    def raw_level(self, k: int) -> float:
        """Reconstructed (unfiltered) digital level for sample index ``k``.

        For a sigma-delta / leapfrog ADC this is the coarse modulator output;
        otherwise it is the memoryless uniform quantizer level.
        """
        sd = self.modulator()
        if sd is not None:
            return sd.output(k)
        q = self.quantizer
        return q.level_of(q.code_of(self.signal.sample_value(k)))

    def filt_level(self, k: int) -> float:
        """Decimated digital level at sample ``k`` (normalised sinc^M cascade)."""
        K = max(1, int(self.filter_taps))
        if K <= 1:
            return self.raw_level(k)
        h = self.decimation_taps()
        gain = self.filter_gain()
        acc = 0.0
        for j, hj in enumerate(h):
            acc += hj * self.raw_level(k - j)
        return (acc / float(h.sum())) / gain

    def digital_out_sample(self, k: int) -> float:
        """The digital output level at sample ``k`` (filtered when a decimator
        is active, otherwise the raw level)."""
        return self.filt_level(k) if int(self.filter_taps) > 1 else self.raw_level(k)

    def digital_out_at(self, t_rel: float) -> float:
        """Sample-and-held digital output value at relative time ``t_rel``.

        A filtered output is delay-compensated so its held value lines up in
        time with the analog signal it represents.
        """
        if int(self.filter_taps) > 1:
            k = self.signal.sample_index_at(t_rel + self.group_delay())
            return self.filt_level(k)
        return self.raw_level(self.signal.sample_index_at(t_rel))

    # ------------------------------------------------------------- filter
    def filter_order(self) -> int:
        """Order ``M`` of the sinc^M decimation filter: ``modulator order + 1``.

        Nyquist mode uses a plain moving average (sinc^1); a 1st-order modulator
        is matched by sinc^2, a 2nd-order by sinc^3, the leapfrog (order 3) by
        sinc^4 - the textbook "decimator order = modulator order + 1" rule.
        """
        sd = self.modulator()
        return (sd.order + 1) if sd is not None else 1

    def decimation_taps(self) -> np.ndarray:
        """Composite FIR of the sinc^M decimator (M cascaded K-tap boxcars).

        Cached by ``(K, M)``; ``sum(h) == K**M`` so dividing by it is unity at DC.
        """
        K = max(1, int(self.filter_taps))
        M = self.filter_order()
        key = (K, M)
        if self._fir_key != key:
            h = np.ones(1)
            box = np.ones(K)
            for _ in range(M):
                h = np.convolve(h, box)
            self._fir = h
            self._fir_key = key
        return self._fir

    def signal_w(self) -> float:
        """Digital angular frequency (rad/sample) of the input at the sample rate."""
        return 2.0 * np.pi * self.signal.frequency * self.signal.sample_period

    def modulator_stf(self) -> complex:
        """Measured signal transfer ``STF(e^{jw})`` of the active modulator.

        Only the modulator's *noise* transfer is shaped; the *signal* transfer
        has its own in-band gain/phase (unity for Nyquist/1st-order, below 1 and
        lagging for the higher-order loops). Rather than an analytic model, it is
        *measured* by a lock-in of the modulator output against the clean input
        at the signal frequency, and cached (it depends only on the modulator
        parameters and ``f * Ts``, not on time).
        """
        sd = self.modulator()
        sig = self.signal
        w = self.signal_w()
        if sd is None or abs(w) < 1e-6:
            return 1.0 + 0.0j
        key = (
            self.adc_mode, self.quantizer.bits, self.quantizer.vref, self.sd_dither,
            round(sig.frequency, 6), round(sig.sample_period, 6), round(sig.amplitude, 6),
        )
        if self._stf_key == key and self._stf is not None:
            return self._stf

        samp_per_cycle = 1.0 / max(sig.frequency * sig.sample_period, 1e-9)
        n = int(min(max(samp_per_cycle * 24.0, 64.0), 4000.0))
        k0 = sig.sample_index_now()
        self.prepare(k0 - n - 2, k0 + 2)
        ks = np.arange(k0 - n + 1, k0 + 1)
        t = ks * sig.sample_period
        y = np.array([self.raw_level(int(k)) for k in ks], dtype=float)
        ref = sig.amplitude * np.sin(2.0 * np.pi * sig.frequency * t + sig._phase0)
        win = np.hanning(n)
        phasor = np.exp(-1j * 2.0 * np.pi * sig.frequency * t)
        num = np.sum(y * win * phasor)
        den = np.sum(ref * win * phasor)
        h = num / den if abs(den) > 1e-12 else (1.0 + 0.0j)
        self._stf, self._stf_key = h, key
        return h

    def group_delay(self) -> float:
        """Group delay of the reconstruction, in seconds of signal time.

        An ``M``-fold cascade of K-tap boxcars has impulse-response length
        ``M*(K-1)+1`` and a (linear-phase) group delay of half that. With a
        decimator active the modulator's STF phase adds a further (frequency
        dependent) delay ``-arg(STF)/w``.
        """
        K = max(1, int(self.filter_taps))
        M = self.filter_order()
        delay = M * (K - 1) / 2.0 * self.signal.sample_period
        if K > 1:
            w = self.signal_w()
            if abs(w) > 1e-9:
                delay += (-np.angle(self.modulator_stf()) / w) * self.signal.sample_period
        return delay

    def filter_gain(self) -> float:
        """Magnitude of the full signal path at the input frequency.

        Normalising the reconstructed output by this gain makes the in-band
        transfer unity. It is the sinc^M decimator magnitude times the
        modulator's signal-transfer magnitude ``|STF|``.
        """
        K = max(1, int(self.filter_taps))
        if K == 1:
            return 1.0
        M = self.filter_order()
        w = self.signal_w()
        s = np.sin(w / 2.0)
        if abs(s) < 1e-9:
            return 1.0  # near DC, gain is already 1
        mag = (abs(np.sin(K * w / 2.0) / s) / K) ** M
        mag *= abs(self.modulator_stf())
        return float(max(mag, 0.05))  # clamp to avoid blow-up near a filter null

    # ------------------------------------------------------------- readouts
    def noise_full(self) -> float:
        """Value (in FS) at +/- full deflection of the noise strip: one LSB."""
        return max(self.quantizer.step, 1e-6)

    def fft_db(self, k0: int, n: int | None = None) -> np.ndarray:
        """FFT magnitude (dBFS) of the digital output over ``n`` samples ending
        at sample index ``k0``. A full-scale sine peaks at 0 dB.

        With a decimator active the raw levels are filtered with one vectorised
        convolution rather than re-running the FIR per sample.
        """
        N = int(self.fft_size if n is None else n)
        K = max(1, int(self.filter_taps))
        if K > 1:
            h = self.decimation_taps()
            L = len(h)
            gain = self.filter_gain()
            k_start = k0 - N + 1 - (L - 1)
            self.prepare(k_start - 2, k0 + 2)
            raw = np.array([self.raw_level(k) for k in range(k_start, k0 + 1)], dtype=float)
            d = np.convolve(raw, h)[L - 1: L - 1 + N] / (h.sum() * gain)
        else:
            self.prepare(k0 - N + 1 - 2, k0 + 2)
            d = np.array([self.raw_level(k) for k in range(k0 - N + 1, k0 + 1)], dtype=float)

        win = np.hanning(N)
        spec = np.fft.rfft(d * win)
        coherent_gain = win.sum() / 2.0  # so a full-scale sine peaks at 0 dB
        mag = np.abs(spec) / max(coherent_gain, 1e-9)
        return 20.0 * np.log10(mag + 1e-9)
