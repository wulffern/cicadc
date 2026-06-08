"""Offscreen manim renderer for the two-panel ADC visualisation.

Rather than embedding manim's live OpenGL window inside Qt (fragile, especially
on macOS), we drive manim's ``Camera`` directly: every frame we (re)build the
mobjects for the current signal/quantizer state, capture them onto an offscreen
pixel buffer, and hand the resulting RGBA numpy array back to the Qt widget.

The class is named :class:`AdcScene` to match the project plan; it composes a
manim ``Camera`` and manim mobjects rather than subclassing ``Scene`` because
the ``Scene`` machinery is geared towards the CLI render-to-file pipeline.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from manim import (
    Camera,
    Dot,
    ImageMobject,
    Line,
    Rectangle,
    Text,
    Triangle,
    VGroup,
    VMobject,
)

from .quantizer import Quantizer
from .sigma_delta import SigmaDelta1, SigmaDelta2, Leapfrog
from .signal_source import SignalSource

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
CAR_PATH = os.path.join(_ASSETS_DIR, "car.png")

# Retro, 80's arcade-ish palette.
BG = "#0b0f1a"
PANEL_EDGE = "#2b3a63"
GRID = "#1d2742"
CURVE = "#3361e6"   # analog signal (blue, matches the analog car)
QUANT = "#b6e3b6"   # quantizer / modulator output (pale green)
DIGITAL = "#cdd6e6"  # decimated/filtered digital output (white / gray, matches car)
ERROR = "#ff3b3b"   # quantization error
SAMPLE = "#ffd400"  # sample points
CAR = "#3361e6"     # analog "now" marker / car (blue, fallback)
TEXT = "#e6f0ff"
BAR = "#3b82f6"
BAR_NOW = "#39ff14"


class AdcScene:
    def __init__(
        self,
        signal: SignalSource | None = None,
        quantizer: Quantizer | None = None,
        pixel_width: int = 1280,
        pixel_height: int = 1152,
        filter_taps: int = 1,
        adc_mode: str = "nyquist",
    ) -> None:
        self.signal = signal or SignalSource()
        self.quantizer = quantizer or Quantizer()
        self.filter_taps = filter_taps  # moving-average length on the digital output
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height

        # ADC behaviour: "nyquist" (memoryless uniform quantizer) or a
        # sigma-delta modulator ("sigma_delta" = 1st order, "sigma_delta2" = 2nd
        # order) whose coarse bitstream is decimated by the moving-average filter.
        self.adc_mode = adc_mode
        self.sd_dither = 0.0
        _input_fn = lambda k: self.signal.sample_value(k)  # noqa: E731
        self._modulators = {
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

        # Keep ~128 px per frame unit (so 896 px -> 7.0 as before, 1152 -> 9.0):
        # the extra height is reserved for the two analysis strips at the bottom.
        self.frame_height = pixel_height / 128.0
        self.frame_width = self.frame_height * pixel_width / pixel_height

        self.camera = Camera(
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            background_color=BG,
        )
        # The static scenery (panels, grids, axes, labels) is built once and
        # reused every frame so only the moving content is rebuilt; this keeps the
        # per-frame cost down so resolution and frame rate can be raised.
        self._static_mobs: List | None = None
        self._static_key: tuple | None = None

        # The FFT is over a long sample history, so it is recomputed only when a
        # new sample arrives (or a parameter changes) rather than every frame.
        self._fft_mob: VMobject | None = None
        self._fft_key: tuple | None = None

        # Measured modulator signal-transfer (gain/phase), cached per parameter
        # set (it is time-invariant). Used to undo the modulator's in-band
        # gain/phase when reconstructing, so the noise readout shows only noise.
        self._stf: complex | None = None
        self._stf_key: tuple | None = None

        self._text_cache: dict[tuple, Text] = {}
        # Car variants, all keyed to the traces they ride:
        #   "color"  -> blue analog car (the original sprite)
        #   "quant"  -> pale green, the quantizer/modulator (raw) output
        #   "digital"-> white/gray, the decimated/filtered digital output
        # plus translucent "*_t" variants for the left panel's "shadow" markers.
        self._car_arrays = {"color": self._load_car()}
        gray = self._to_gray(self._car_arrays["color"])
        self._car_arrays["quant"] = self._tint(gray, (0.71, 0.89, 0.71))
        self._car_arrays["quant_t"] = self._fade_alpha(self._car_arrays["quant"], 0.4)
        self._car_arrays["digital"] = self._tint(gray, (0.80, 0.84, 0.90))
        self._car_arrays["digital_t"] = self._fade_alpha(self._car_arrays["digital"], 0.45)
        self._car_bases: dict[str, object] = {}
        self._layout()

    @staticmethod
    def _tint(gray: np.ndarray | None, rgb: tuple[float, float, float]) -> np.ndarray | None:
        """Tint a grayscale RGBA array by ``rgb`` (each 0..1), keeping alpha."""
        if gray is None:
            return None
        out = gray.copy()
        lum = gray[..., 0].astype(float)
        for c in range(3):
            out[..., c] = np.clip(lum * rgb[c], 0, 255).astype(np.uint8)
        return out

    @staticmethod
    def _to_gray(arr: np.ndarray | None) -> np.ndarray | None:
        """Desaturate an RGBA array, preserving the alpha channel."""
        if arr is None:
            return None
        out = arr.copy()
        lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
        out[..., 0] = out[..., 1] = out[..., 2] = lum
        return out

    @staticmethod
    def _fade_alpha(arr: np.ndarray | None, factor: float) -> np.ndarray | None:
        """Scale the alpha channel of an RGBA array (for a translucent shadow)."""
        if arr is None:
            return None
        out = arr.copy()
        out[..., 3] = (arr[..., 3].astype(float) * factor).astype(np.uint8)
        return out

    @staticmethod
    def _load_car() -> np.ndarray | None:
        """Load car.png and key out its green chroma background to alpha."""
        if not os.path.exists(CAR_PATH):
            return None
        try:
            from PIL import Image

            img = Image.open(CAR_PATH).convert("RGBA")
            # Crop to the car (drop most of the green border) then downscale, so
            # the camera does not have to resize a 1254x1254 image every frame.
            arr = np.array(img)
            r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
            green = (g > 110) & (g - r > 40) & (g - b > 40)
            ys, xs = np.where(~green)
            if len(xs):
                pad = 8
                x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, arr.shape[1] - 1)
                y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, arr.shape[0] - 1)
                img = img.crop((x0, y0, x1 + 1, y1 + 1))
            img.thumbnail((220, 220), Image.LANCZOS)
            arr = np.array(img)
            r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
            green = (g > 110) & (g - r > 40) & (g - b > 40)
            arr[..., 3] = np.where(green, 0, 255).astype(np.uint8)
            return arr
        except Exception:
            return None

    # ------------------------------------------------------------------ layout
    def _layout(self) -> None:
        half_w = self.frame_width / 2.0
        # Left (analog) panel - now pushed up to leave room for a bottom strip.
        self.lp_x0, self.lp_x1 = -half_w + 0.2, -0.15
        self.lp_y0, self.lp_y1 = -1.8, 4.3
        self.xc_left = (self.lp_x0 + self.lp_x1) / 2.0
        self.aw = (self.lp_x1 - self.lp_x0) / 2.0 - 0.45  # amplitude half-width
        self.yb = -1.2    # bottom of the time axis (the past)
        self.yt = 3.6     # top of the time axis (the future)
        self.y_now = (self.yb + self.yt) / 2.0  # "now" sits in the middle

        # Right (digital) panel - mirrors the left panel's geometry.
        self.rp_x0, self.rp_x1 = 0.15, half_w - 0.2
        self.rp_y0, self.rp_y1 = -1.8, 4.3
        self.xc_right = (self.rp_x0 + self.rp_x1) / 2.0
        self.aw_right = (self.rp_x1 - self.rp_x0) / 2.0 - 0.5

        # Bottom analysis strips (one under each main panel). Time/frequency runs
        # left-to-right with "now" / DC on the appropriate side.
        self.bs_y0, self.bs_y1 = -4.3, -2.15  # shared top/bottom of both strips
        bs_yc = (self.bs_y0 + self.bs_y1) / 2.0

        # Left strip: quantization noise vs time (now on the far right).
        self.ns_x0 = self.lp_x0 + 0.5   # leave room for the +/-FS axis labels
        self.ns_x1 = self.lp_x1 - 0.15
        self.ns_yc = bs_yc - 0.12       # vertical centre (the zero-error line)
        self.ns_hh = 0.62               # half-height of the +/- full-scale band
        self.ns_hist = self.signal.window  # seconds of history shown

        # Right strip: FFT magnitude of the digital output (0 dBFS at the top).
        self.fs_x0 = self.rp_x0 + 0.55  # room for the dB axis labels
        self.fs_x1 = self.rp_x1 - 0.15
        self.fs_yt = self.bs_y1 - 0.55  # top of the dB axis (0 dBFS)
        self.fs_yb = self.bs_y0 + 0.35  # bottom of the dB axis (floor)
        self.fft_floor_db = -80.0
        self.fft_size = 1024  # long window -> fine frequency resolution

    # ------------------------------------------------------------- coord maps
    def _x_amp(self, value: float) -> float:
        return self.xc_left + value * self.aw

    def _x_dig(self, value: float) -> float:
        return self.xc_right + value * self.aw_right

    def _y_time(self, t_rel: float) -> float:
        # "now" (t_rel == 0) is the middle; the window spans +/- window/2 around it.
        half = self.signal.window / 2.0
        return self.y_now + (t_rel / half) * ((self.yt - self.yb) / 2.0)

    def _half_span(self) -> float:
        return self.signal.window / 2.0

    # ----------------------------------------------- bottom-strip coord maps
    def _ns_x(self, t_rel: float) -> float:
        """Map a (negative) time offset to the noise strip's x: now -> right."""
        frac = (t_rel + self.ns_hist) / self.ns_hist  # 0 at oldest, 1 at now
        return self.ns_x0 + frac * (self.ns_x1 - self.ns_x0)

    def _ns_y(self, err: float, full: float) -> float:
        """Map a quantization error (FS) to the noise strip's y, clipped to band."""
        v = max(-1.0, min(1.0, err / full)) if full > 0 else 0.0
        return self.ns_yc + v * self.ns_hh

    def _fft_x(self, frac: float) -> float:
        """Map a 0..1 fraction of the band (DC..Nyquist) to the FFT strip's x."""
        return self.fs_x0 + max(0.0, min(1.0, frac)) * (self.fs_x1 - self.fs_x0)

    def _fft_y(self, db: float) -> float:
        """Map a dBFS value (0 at top, floor at bottom) to the FFT strip's y."""
        frac = (db - self.fft_floor_db) / (0.0 - self.fft_floor_db)
        frac = max(0.0, min(1.0, frac))
        return self.fs_yb + frac * (self.fs_yt - self.fs_yb)

    # ------------------------------------------------- digital output helpers
    def _prepare_sd(self, k_lo: int, k_hi: int) -> None:
        """Warm up / cache the active sigma-delta modulator over ``[k_lo, k_hi]``."""
        if self._is_sigma_delta():
            self._sync_sd()
            self._modulator().prepare(k_lo, k_hi)

    def _digital_out_sample(self, k: int) -> float:
        """The digital output level at sample ``k`` (filtered when a decimator
        is active, otherwise the raw quantizer/modulator level)."""
        return self._filt_level(k) if int(self.filter_taps) > 1 else self._raw_level(k)

    def _digital_out_at(self, t_rel: float) -> float:
        """Sample-and-held digital output value at relative time ``t_rel``.

        Mirrors the staircase drawn on the digital panel: a filtered output is
        delay-compensated so its held value lines up in time with the analog
        signal it represents.
        """
        if int(self.filter_taps) > 1:
            k = self.signal.sample_index_at(t_rel + self._group_delay())
            return self._filt_level(k)
        return self._raw_level(self.signal.sample_index_at(t_rel))

    def _filter_order(self) -> int:
        """Order ``M`` of the sinc^M decimation filter: ``modulator order + 1``.

        Nyquist mode uses a plain moving average (sinc^1); a 1st-order modulator
        is matched by sinc^2 and a 2nd-order modulator by sinc^3 - the textbook
        "decimator order = modulator order + 1" rule.
        """
        sd = self._modulator()
        return (sd.order + 1) if sd is not None else 1

    def _decimation_taps(self) -> np.ndarray:
        """Composite FIR of the sinc^M decimator (M cascaded K-tap boxcars).

        Cached by ``(K, M)``; ``sum(h) == K**M`` so dividing by it is unity at DC.
        """
        K = max(1, int(self.filter_taps))
        M = self._filter_order()
        key = (K, M)
        if getattr(self, "_fir_key", None) != key:
            h = np.ones(1)
            box = np.ones(K)
            for _ in range(M):
                h = np.convolve(h, box)
            self._fir = h
            self._fir_key = key
        return self._fir

    def _signal_w(self) -> float:
        """Digital angular frequency (rad/sample) of the input at the sample rate."""
        return 2.0 * np.pi * self.signal.frequency * self.signal.sample_period

    def _modulator_stf(self) -> complex:
        """Measured signal transfer ``STF(e^{jw})`` of the active modulator.

        The decimation filter recovers the modulator output, but a sigma-delta
        loop does not pass the signal untouched: only its *noise* transfer is
        shaped, while the *signal* transfer has its own in-band gain/phase. For a
        1st-order loop this is unity, but the 2nd-order loop's signal gain dips
        below 1 (and lags) in band; left uncorrected it leaves a residual
        sinusoid in the reconstruction - a slow waveform that, in LSB units,
        looks like it grows with bit depth.

        Rather than an analytic model (the linearised 2nd-order loop is actually
        unstable - it is the quantizer nonlinearity that stabilises it), the
        transfer is *measured* by a lock-in of the modulator output against the
        clean input at the signal frequency. It depends only on the modulator
        parameters and ``f * Ts`` (not on time), so the result is cached.
        """
        sd = self._modulator()
        sig = self.signal
        w = self._signal_w()
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
        self._prepare_sd(k0 - n - 2, k0 + 2)
        ks = np.arange(k0 - n + 1, k0 + 1)
        t = ks * sig.sample_period
        y = np.array([self._raw_level(int(k)) for k in ks], dtype=float)
        ref = sig.amplitude * np.sin(2.0 * np.pi * sig.frequency * t + sig._phase0)
        win = np.hanning(n)
        phasor = np.exp(-1j * 2.0 * np.pi * sig.frequency * t)
        num = np.sum(y * win * phasor)
        den = np.sum(ref * win * phasor)
        h = num / den if abs(den) > 1e-12 else (1.0 + 0.0j)
        self._stf, self._stf_key = h, key
        return h

    def _group_delay(self) -> float:
        """Group delay of the reconstruction, in seconds of signal time.

        An ``M``-fold cascade of K-tap boxcars has impulse-response length
        ``M*(K-1)+1`` and a (linear-phase) group delay of half that. When a
        decimator is active the modulator's STF phase adds a further (frequency
        dependent) delay ``-arg(STF)/w`` so the reconstructed output lines up in
        time with the analog signal it represents.
        """
        K = max(1, int(self.filter_taps))
        M = self._filter_order()
        delay = M * (K - 1) / 2.0 * self.signal.sample_period
        if K > 1:
            w = self._signal_w()
            if abs(w) > 1e-9:
                delay += (-np.angle(self._modulator_stf()) / w) * self.signal.sample_period
        return delay

    def _digital_car_anchor(self):
        """``(t_rel, sample_index)`` for the delay-compensated digital-output car.

        The car trails "now" by the decimator's group delay so the latency is
        visible, but the trail is capped to stay inside the window (long sinc^M
        filters can delay by more than the whole window). It still lands on the
        analog curve because the delay-compensated output is evaluated there.
        """
        delay = self._group_delay()
        d_vis = min(delay, 0.82 * self._half_span())
        k_vis = self.signal.sample_index_at(delay - d_vis)
        return -d_vis, k_vis

    # ------------------------------------------------------------ small helpers
    def _text(self, s: str, font_size: float, color: str = TEXT, rotate: float = 0.0) -> Text:
        # NOTE: rotation must be baked in at creation time. The mobject is cached
        # and reused every frame, so applying .rotate() per frame would accumulate.
        key = (s, round(font_size, 2), color, round(rotate, 4))
        m = self._text_cache.get(key)
        if m is None:
            m = Text(s, font_size=font_size, color=color)
            if rotate:
                m.rotate(rotate)
            self._text_cache[key] = m
        return m

    def _car(self, x: float, y: float, angle: float = 0.0, height: float = 0.7, variant: str = "color"):
        """Return a car marker (image if available, else a triangle), pointing
        "up" (towards the future) by default and rotated by ``angle`` radians.

        ``variant`` is one of "color", "gray" or "shadow". A base mobject is
        built once per variant and copied per frame so the rotation is applied
        fresh (rotating a cached instance would accumulate).
        """
        base = self._car_bases.get(variant)
        if base is None:
            arr = self._car_arrays.get(variant)
            if arr is not None:
                base = ImageMobject(arr)
                base.scale_to_fit_height(height)
            else:
                fallback = {
                    "color": CAR,
                    "quant": QUANT,
                    "quant_t": QUANT,
                    "digital": DIGITAL,
                    "digital_t": DIGITAL,
                }
                base = Triangle(color=fallback[variant], fill_opacity=1.0).scale(0.16).rotate(np.pi)
            self._car_bases[variant] = base
        car = base.copy()
        if angle:
            car.rotate(angle)
        car.move_to([x, y, 0])
        return car

    @staticmethod
    def _polyline(points: List[Tuple[float, float]], color: str, width: float) -> VMobject:
        m = VMobject()
        m.set_points_as_corners([np.array([x, y, 0.0]) for x, y in points])
        m.set_stroke(color=color, width=width)
        return m

    @staticmethod
    def _panel(x0, y0, x1, y1) -> Rectangle:
        rect = Rectangle(width=x1 - x0, height=y1 - y0)
        rect.move_to([(x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.0])
        rect.set_stroke(color=PANEL_EDGE, width=2.0)
        rect.set_fill(opacity=0.0)
        return rect

    # --------------------------------------------------------------- building
    def _static_left(self) -> List:
        """Static scenery for the analog panel (rebuilt only when bits change)."""
        q = self.quantizer
        items: List = [self._panel(self.lp_x0, self.lp_y0, self.lp_x1, self.lp_y1)]

        # Quantization decision thresholds as faint vertical gridlines.
        if q.num_levels <= 64:
            grid = VGroup()
            for k in range(1, q.num_levels):
                xv = -q.vref + k * q.step
                x = self._x_amp(xv)
                grid.add(Line([x, self.yb, 0], [x, self.yt, 0]))
            grid.set_stroke(color=GRID, width=1.0)
            items.append(grid)

        # Amplitude centre axis and the "now" line (now is in the middle).
        items.append(
            Line([self._x_amp(0.0), self.yb, 0], [self._x_amp(0.0), self.yt, 0]).set_stroke(GRID, 1.5)
        )
        items.append(
            Line([self.lp_x0 + 0.2, self.y_now, 0], [self.lp_x1 - 0.2, self.y_now, 0]).set_stroke(TEXT, 2.0)
        )

        title = self._text("ANALOG SIGNAL", 22, CURVE)
        title.move_to([(self.lp_x0 + self.lp_x1) / 2.0, self.lp_y1 - 0.35, 0])
        items.append(title)
        for val, lab in ((-1.0, "-FS"), (0.0, "0"), (1.0, "+FS")):
            t = self._text(lab, 16, TEXT)
            t.move_to([self._x_amp(val), self.yb - 0.3, 0])
            items.append(t)
        now_lbl = self._text("now", 14, TEXT)
        now_lbl.move_to([self.lp_x1 - 0.45, self.y_now + 0.22, 0])
        items.append(now_lbl)
        fut_lbl = self._text("future ->", 14, TEXT, rotate=np.pi / 2)
        fut_lbl.move_to([self.lp_x0 + 0.25, (self.y_now + self.yt) / 2.0, 0])
        items.append(fut_lbl)
        return items

    def _dynamic_left(self) -> List:
        """Moving content for the analog panel (rebuilt every frame)."""
        sig = self.signal
        items: List = []

        # Continuous analog curve across past and future (now in the middle).
        half = self._half_span()
        curve = sig.curve_points(140, -half, half)
        true_pts = [(self._x_amp(v), self._y_time(t)) for t, v in curve]
        items.append(self._polyline(true_pts, CURVE, 4.0))

        # Sample points along the analog curve.
        first_k, last_k = sig.sample_indices_in(-half, half)
        dots = VGroup()
        for k in range(first_k, last_k + 1):
            t_rel = sig.t_rel_of_index(k)
            dots.add(Dot([self._x_amp(sig.sample_value_clean(k)), self._y_time(t_rel), 0], radius=0.045, color=SAMPLE))
        items.append(dots)

        # Cars. The blue car drives along the (blue) analog curve at now. The
        # translucent "shadow" marks the quantized "now" value - pale green (the
        # quantizer/modulator output) when a decimator is active, otherwise
        # white/gray (the digital output). The white/gray digital-output car
        # trails "now" by the decimator's group delay, landing back on the analog
        # curve - so the filter's latency is visible.
        k0 = sig.sample_index_now()
        ang_now = self._tangent_angle(0.0, self.aw)
        raw_variant = "quant_t" if self.filter_taps > 1 else "digital_t"
        items.append(self._car(self._x_amp(self._raw_level(k0)), self.y_now, angle=ang_now, variant=raw_variant))
        if self.filter_taps > 1:  # the digital-output car only when a decimator is active
            t_d, k_vis = self._digital_car_anchor()
            ang_d = self._tangent_angle(t_d, self.aw)
            items.append(self._car(self._x_amp(self._filt_level(k_vis)), self._y_time(t_d), angle=ang_d, variant="digital"))
        items.append(self._car(self._x_amp(sig.clean_value(0.0)), self.y_now, angle=ang_now, variant="color"))
        return items

    def _filter_gain(self) -> float:
        """Magnitude of the full signal path at the input frequency.

        Normalising the reconstructed output by this gain makes the in-band
        transfer unity, so the recovered amplitude matches the analog signal.
        This is the sinc^M decimator magnitude (single-stage moving-average
        magnitude raised to ``M``) times the modulator's signal-transfer
        magnitude ``|STF|`` - the latter being unity for Nyquist/1st-order but
        not for the 2nd-order loop, whose in-band gain would otherwise leak into
        the quantization-noise readout.
        """
        K = max(1, int(self.filter_taps))
        if K == 1:
            return 1.0
        M = self._filter_order()
        w = self._signal_w()
        s = np.sin(w / 2.0)
        if abs(s) < 1e-9:
            return 1.0  # near DC, gain is already 1
        mag = (abs(np.sin(K * w / 2.0) / s) / K) ** M
        mag *= abs(self._modulator_stf())
        return float(max(mag, 0.05))  # clamp to avoid blow-up near a filter null

    # ----------------------------------------------------------------- ADC mode
    def _is_sigma_delta(self) -> bool:
        return self.adc_mode in self._modulators

    def _modulator(self):
        """The active sigma-delta modulator, or ``None`` in Nyquist mode."""
        return self._modulators.get(self.adc_mode)

    def set_adc_mode(self, mode: str) -> None:
        """Switch between the "nyquist", "sigma_delta" and "sigma_delta2" ADCs."""
        self.adc_mode = mode
        self.reset_sd()

    def set_sd_dither(self, amount: float) -> None:
        """Set the sigma-delta dither amplitude (0 disables it)."""
        self.sd_dither = amount
        self.reset_sd()

    def reset_sd(self) -> None:
        """Invalidate every modulator cache (input/quantizer parameters changed)."""
        for sd in self._modulators.values():
            sd.reset()

    def _sync_sd(self) -> None:
        """Mirror the live quantizer settings into the active modulator, resetting
        its cache if anything that changes the output sequence has changed."""
        sd = self._modulator()
        if sd is None:
            return
        if sd.bits != self.quantizer.bits or sd.vref != self.quantizer.vref or sd.dither != self.sd_dither:
            sd.bits = self.quantizer.bits
            sd.vref = self.quantizer.vref
            sd.dither = self.sd_dither
            sd.reset()

    def _raw_level(self, k: int) -> float:
        """Reconstructed (unfiltered) digital level for sample index ``k``.

        For a sigma-delta ADC this is the coarse modulator output; otherwise it
        is the memoryless uniform quantizer level.
        """
        sd = self._modulator()
        if sd is not None:
            return sd.output(k)
        q = self.quantizer
        return q.level_of(q.code_of(self.signal.sample_value(k)))

    def _filt_level(self, k: int) -> float:
        """Decimated digital level at sample ``k`` (normalised sinc^M cascade)."""
        K = max(1, int(self.filter_taps))
        if K <= 1:
            return self._raw_level(k)
        h = self._decimation_taps()
        gain = self._filter_gain()
        acc = 0.0
        for j, hj in enumerate(h):
            acc += hj * self._raw_level(k - j)
        return (acc / float(h.sum())) / gain

    def _tangent_angle(self, t_rel: float, aw: float) -> float:
        """Rotation so a car (nose +y) points along the path tangent at ``t_rel``."""
        dy = (self.yt - self.yb) / self.signal.window
        dx = aw * self.signal.derivative(t_rel)
        return float(np.arctan2(dy, dx) - np.pi / 2.0)

    def _hold_staircase(self, level_fn, x_map, t0_rel: float, t1_rel: float, delay: float = 0.0):
        """Build a sample-and-hold staircase over the time window.

        Each sample's level is plotted at its time minus ``delay`` (used to
        delay-compensate the filtered trace). Returns ``(points, dot_points)``.
        """
        sig = self.signal
        first_k, last_k = sig.sample_indices_in(t0_rel + delay, t1_rel + delay)
        kb = sig.sample_index_at(t0_rel + delay)
        prev = x_map(level_fn(kb))
        pts: List[Tuple[float, float]] = [(prev, self._y_time(t0_rel))]
        dots: List[Tuple[float, float]] = []
        for k in range(first_k, last_k + 1):
            y = self._y_time(sig.t_rel_of_index(k) - delay)
            x_here = x_map(level_fn(k))
            pts.append((prev, y))
            pts.append((x_here, y))
            prev = x_here
            dots.append((x_here, y))
        pts.append((prev, self._y_time(t1_rel)))
        return pts, dots

    def _static_right(self) -> List:
        """Static scenery for the digital panel (rebuilt only when bits change)."""
        q = self.quantizer
        items: List = [self._panel(self.rp_x0, self.rp_y0, self.rp_x1, self.rp_y1)]

        # Quantization levels as faint vertical gridlines (same as the left panel).
        if q.num_levels <= 64:
            grid = VGroup()
            for k in range(1, q.num_levels):
                xv = -q.vref + k * q.step
                x = self._x_dig(xv)
                grid.add(Line([x, self.yb, 0], [x, self.yt, 0]))
            grid.set_stroke(color=GRID, width=1.0)
            items.append(grid)

        items.append(
            Line([self._x_dig(0.0), self.yb, 0], [self._x_dig(0.0), self.yt, 0]).set_stroke(GRID, 1.5)
        )
        items.append(
            Line([self.rp_x0 + 0.2, self.y_now, 0], [self.rp_x1 - 0.2, self.y_now, 0]).set_stroke(TEXT, 2.0)
        )

        title = self._text("DIGITAL SIGNAL", 22, TEXT)
        title.move_to([(self.rp_x0 + self.rp_x1) / 2.0, self.rp_y1 - 0.35, 0])
        items.append(title)
        for val, lab in ((-1.0, "-FS"), (1.0, "+FS")):
            t = self._text(lab, 16, TEXT)
            t.move_to([self._x_dig(val), self.yb - 0.3, 0])
            items.append(t)
        return items

    def _dynamic_right(self) -> List:
        """Right panel moving content: the digital (sample-and-hold) signal vs
        time, mirroring the left panel so the two graphs are directly comparable."""
        q = self.quantizer
        sig = self.signal
        items: List = []

        half = self._half_span()
        delay = self._group_delay()
        K = max(1, int(self.filter_taps))

        # For a stateful sigma-delta ADC, warm up and cache the contiguous range
        # of sample indices this frame will touch (window, the filter's look-back,
        # and the delay-compensated filtered trace) before drawing.
        if self._is_sigma_delta():
            self._sync_sd()
            fir_len = len(self._decimation_taps())  # decimator look-back, in samples
            k_lo = sig.sample_index_at(-half) - fir_len - 1
            k_hi = sig.sample_index_at(half + delay) + 2
            self._modulator().prepare(k_lo, k_hi)

        # Unfiltered (gray) staircase at the true sample times, with sample dots.
        # When the filter is off it is the only output, so draw it more boldly.
        raw_pts, raw_dots = self._hold_staircase(self._raw_level, self._x_dig, -half, half, delay=0.0)
        if K > 1:
            # With a decimator/filter active, the raw trace is the coarse
            # quantizer/modulator output (pale green) and the filtered trace is
            # the digital output (white/gray), delay-compensated to line up in
            # time with the analog signal.
            items.append(self._polyline(raw_pts, QUANT, 2.0))
            filt_pts, _ = self._hold_staircase(self._filt_level, self._x_dig, -half, half, delay=delay)
            items.append(self._polyline(filt_pts, DIGITAL, 3.5))
        else:
            # No filter: the quantizer output is the digital output (white/gray).
            items.append(self._polyline(raw_pts, DIGITAL, 3.5))

        dots = VGroup()
        for x, y in raw_dots:
            dots.add(Dot([x, y, 0], radius=0.045, color=SAMPLE))
        items.append(dots)

        # Subtitle (changes with bits / averaging / ADC mode).
        sd = self._modulator()
        if sd is not None:
            ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(sd.order, f"{sd.order}th")
            label = "leapfrog" if self.adc_mode == "leapfrog" else "\u03a3\u0394"
            sub_text = f"{ordinal}-order {label}  -  {q.bits}-bit"
        else:
            sub_text = f"{q.bits} bits  -  {q.num_levels} levels"
        if K > 1:
            M = self._filter_order()
            sup = {2: "\u00b2", 3: "\u00b3"}.get(M, "")
            prefix = f"sinc{sup} " if M > 1 else ""
            sub_text += f"  -  {prefix}avg {K}"
        sub = self._text(sub_text, 16, TEXT)
        sub.move_to([(self.rp_x0 + self.rp_x1) / 2.0, self.rp_y1 - 0.72, 0])
        items.append(sub)

        # Current readout near the bottom: the digital code (Nyquist) or the
        # coarse modulator output (sigma-delta).
        k0 = sig.sample_index_now()
        if self._is_sigma_delta():
            readout = self._text(f"\u03a3\u0394  {self._raw_level(k0):+.2f}", 26, TEXT)
        else:
            now_code = q.code_of(sig.value_now())
            readout = self._text(f"{q.code_string(now_code)}  ({now_code})", 26, TEXT)
        readout.move_to([(self.rp_x0 + self.rp_x1) / 2.0, self.yb - 0.3, 0])
        items.append(readout)

        # Cars (heading nudged): the quantizer/modulator output at now (pale green
        # when a decimator is active, else white/gray as the digital output); and
        # the white/gray digital-output car, trailing by the group delay.
        def nudge(t_rel):
            return max(-0.45, min(0.45, 0.4 * self._tangent_angle(t_rel, self.aw_right)))

        raw_variant = "quant" if K > 1 else "digital"
        items.append(self._car(self._x_dig(self._raw_level(k0)), self.y_now, angle=nudge(0.0), variant=raw_variant))
        if K > 1:  # the digital-output car only when a decimator is active
            t_d, k_vis = self._digital_car_anchor()
            items.append(self._car(self._x_dig(self._filt_level(k_vis)), self._y_time(t_d), angle=nudge(t_d), variant="digital"))
        return items

    # ------------------------------------------------------- bottom strips
    def _noise_full(self) -> float:
        """Value (in FS) at the top/bottom of the noise strip: one LSB.

        The strip is scaled to the quantizer's LSB (its step size), so the band
        spans +/- 1 LSB and the noise keeps the same relative height whatever the
        bit depth. The axis is therefore labelled directly in LSBs.
        """
        return max(self.quantizer.step, 1e-6)

    def _fft_logfrac(self, norm_freq: float) -> float:
        """Map a normalised frequency ``f / f_s`` to a 0..1 position on the log
        frequency axis (the lowest bin ``1/N`` maps to 0, Nyquist ``0.5`` to 1)."""
        denom = np.log10(self.fft_size / 2.0)
        if denom <= 0:
            return 0.0
        return float(np.log10(max(norm_freq, 1e-12) * self.fft_size) / denom)

    def _static_bottom(self) -> List:
        """Static scenery for the two bottom analysis strips (cached per bits)."""
        items: List = []
        full = self._noise_full()

        # --- Left: quantization-noise-over-time strip ---------------------
        items.append(self._panel(self.lp_x0, self.bs_y0, self.lp_x1, self.bs_y1))
        items.append(
            Line([self.ns_x0, self.ns_yc, 0], [self.ns_x1, self.ns_yc, 0]).set_stroke(GRID, 1.5)
        )
        # "now" edge on the far right.
        items.append(
            Line([self.ns_x1, self.ns_yc - self.ns_hh, 0], [self.ns_x1, self.ns_yc + self.ns_hh, 0]).set_stroke(TEXT, 1.5)
        )
        ntitle = self._text("QUANTIZATION NOISE  (analog - digital)", 15, ERROR)
        ntitle.move_to([(self.lp_x0 + self.lp_x1) / 2.0, self.bs_y1 - 0.26, 0])
        items.append(ntitle)
        lsb_note = self._text(f"1 LSB = {full:.3f} FS", 11, TEXT)
        lsb_note.move_to([(self.ns_x0 + self.ns_x1) / 2.0, self.bs_y0 + 0.18, 0])
        items.append(lsb_note)
        for lab, yy in (("+1 LSB", self.ns_yc + self.ns_hh), ("0", self.ns_yc), ("-1 LSB", self.ns_yc - self.ns_hh)):
            t = self._text(lab, 12, TEXT)
            t.move_to([self.lp_x0 + 0.3, yy, 0])
            items.append(t)
        now_lbl = self._text("now", 12, TEXT)
        now_lbl.move_to([self.ns_x1 - 0.22, self.bs_y0 + 0.18, 0])
        items.append(now_lbl)
        past_lbl = self._text("<- past", 12, TEXT)
        past_lbl.move_to([self.ns_x0 + 0.4, self.bs_y0 + 0.18, 0])
        items.append(past_lbl)

        # --- Right: FFT-of-digital-output strip ---------------------------
        items.append(self._panel(self.rp_x0, self.bs_y0, self.rp_x1, self.bs_y1))
        ftitle = self._text("DIGITAL SPECTRUM", 15, DIGITAL)
        ftitle.move_to([(self.rp_x0 + self.rp_x1) / 2.0, self.bs_y1 - 0.26, 0])
        items.append(ftitle)
        grid = VGroup()
        for db in (0.0, -20.0, -40.0, -60.0, self.fft_floor_db):
            y = self._fft_y(db)
            grid.add(Line([self.fs_x0, y, 0], [self.fs_x1, y, 0]))
        grid.set_stroke(color=GRID, width=1.0)
        items.append(grid)
        for db in (0.0, -40.0, self.fft_floor_db):
            lab = self._text(f"{db:.0f}", 12, TEXT)
            lab.move_to([self.rp_x0 + 0.3, self._fft_y(db), 0])
            items.append(lab)
        db_unit = self._text("dBFS", 12, TEXT)
        db_unit.move_to([self.rp_x0 + 0.3, self.fs_yt + 0.22, 0])
        items.append(db_unit)

        # Logarithmic frequency axis: decade gridlines in f / f_s, from Nyquist
        # (0.5) leftwards down to the lowest resolvable bin (1 / N).
        rmin = 1.0 / self.fft_size
        fgrid = VGroup()
        decades = [0.5]
        r = 0.1
        while r > rmin * 1.0001:
            decades.append(r)
            r /= 10.0
        for rv in decades:
            x = self._fft_x(self._fft_logfrac(rv))
            fgrid.add(Line([x, self.fs_yb, 0], [x, self.fs_yt, 0]))
            lab = self._text(("0.5" if rv == 0.5 else f"{rv:g}"), 11, TEXT)
            lab.move_to([x, self.bs_y0 + 0.18, 0])
            items.append(lab)
        fgrid.set_stroke(color=GRID, width=1.0)
        items.append(fgrid)
        fcap = self._text("f / f_s  (log)", 11, TEXT)
        fcap.move_to([self.fs_x1 - 0.45, self.fs_yt + 0.22, 0])
        items.append(fcap)
        return items

    def _dynamic_bottom(self) -> List:
        """Bottom strips: the (per-frame) noise staircase and the cached FFT."""
        return self._noise_items() + [self._fft_spectrum()]

    def _noise_items(self) -> List:
        """Per-sample (analog - digital output) error, held as a staircase.

        Compared at the sample instants and held between them, like the digital
        panel. A filtered output is delay-compensated so each error lines up in
        time with the analog value it represents. This is cheap (only the visible
        samples) so it is rebuilt every frame to scroll smoothly.
        """
        sig = self.signal
        K = max(1, int(self.filter_taps))
        full = self._noise_full()
        delay = self._group_delay() if K > 1 else 0.0
        hist = self.ns_hist

        first_k, last_k = sig.sample_indices_in(-hist + delay, delay)
        kb = sig.sample_index_at(-hist + delay)
        look = len(self._decimation_taps()) if K > 1 else 0
        self._prepare_sd(kb - look - 2, last_k + 2)

        def err_at(k: int) -> float:
            return sig.clean_value(sig.t_rel_of_index(k) - delay) - self._digital_out_sample(k)

        prev = err_at(kb)
        noise_pts: List[Tuple[float, float]] = [(self._ns_x(-hist), self._ns_y(prev, full))]
        dot_pts: List[Tuple[float, float]] = []
        for k in range(first_k, last_k + 1):
            t = max(-hist, min(0.0, sig.t_rel_of_index(k) - delay))
            x = self._ns_x(t)
            e = err_at(k)
            ye = self._ns_y(e, full)
            noise_pts.append((x, self._ns_y(prev, full)))
            noise_pts.append((x, ye))
            dot_pts.append((x, ye))
            prev = e
        noise_pts.append((self._ns_x(0.0), self._ns_y(prev, full)))
        items: List = [self._polyline(noise_pts, ERROR, 2.5)]
        dots = VGroup()
        for x, y in dot_pts:
            dots.add(Dot([x, y, 0], radius=0.03, color=SAMPLE))
        items.append(dots)
        return items

    def _fft_spectrum(self) -> VMobject:
        """FFT magnitude of the digital output, drawn as a single filled curve.

        The transform runs over ``fft_size`` samples, so it is recomputed only
        when a new sample arrives (or a parameter changes) and otherwise reused;
        the result is a static curve so caching it also keeps per-frame raster
        cost down despite the large bin count.
        """
        sig, q = self.signal, self.quantizer
        N = int(self.fft_size)
        K = max(1, int(self.filter_taps))
        k0 = sig.sample_index_now()
        key = (
            k0, N, K, self.adc_mode, q.bits, q.vref, self.sd_dither,
            round(sig.frequency, 6), round(sig.amplitude, 6),
            round(sig.sample_period, 6), round(sig.noise_amp, 6),
            round(sig._phase0, 6), self.pixel_height,
        )
        if self._fft_key == key and self._fft_mob is not None:
            return self._fft_mob

        # Build the digital-output sequence ending at "now". With a decimator
        # active, filter the raw levels with one vectorised convolution rather
        # than re-running the FIR per sample.
        if K > 1:
            h = self._decimation_taps()
            L = len(h)
            gain = self._filter_gain()
            k_start = k0 - N + 1 - (L - 1)
            self._prepare_sd(k_start - 2, k0 + 2)
            raw = np.array([self._raw_level(k) for k in range(k_start, k0 + 1)], dtype=float)
            d = np.convolve(raw, h)[L - 1: L - 1 + N] / (h.sum() * gain)
        else:
            self._prepare_sd(k0 - N + 1 - 2, k0 + 2)
            d = np.array([self._raw_level(k) for k in range(k0 - N + 1, k0 + 1)], dtype=float)

        win = np.hanning(N)
        spec = np.fft.rfft(d * win)
        coherent_gain = win.sum() / 2.0  # so a full-scale sine peaks at 0 dB
        mag = np.abs(spec) / max(coherent_gain, 1e-9)
        db = 20.0 * np.log10(mag + 1e-9)

        n_bins = mag.shape[0]
        base_y = self._fft_y(self.fft_floor_db)
        # Log frequency axis: bin i maps to f/f_s = i/N; skip DC (i == 0).
        pts: List[Tuple[float, float]] = [(self._fft_x(0.0), base_y)]
        for i in range(1, n_bins):
            x = self._fft_x(self._fft_logfrac(i / float(N)))
            pts.append((x, self._fft_y(float(db[i]))))
        pts.append((self._fft_x(1.0), base_y))
        m = VMobject()
        m.set_points_as_corners([np.array([x, y, 0.0]) for x, y in pts])
        m.set_stroke(color=BAR, width=1.5)
        m.set_fill(color=BAR, opacity=0.35)

        self._fft_mob = m
        self._fft_key = key
        return m

    # ----------------------------------------------------------------- render
    def _static_mobjects(self) -> List:
        """Static scenery mobjects, cached and rebuilt only when they change."""
        key = (self.quantizer.bits, self.quantizer.vref, self.pixel_width, self.pixel_height, self.fft_size)
        if self._static_key != key or self._static_mobs is None:
            self._static_mobs = self._static_left() + self._static_right() + self._static_bottom()
            self._static_key = key
        return self._static_mobs

    def render_frame(self) -> np.ndarray:
        """Render the current state and return an ``(H, W, 4)`` RGBA uint8 array.

        The static scenery (panels, grids, axes, labels) is built once and
        cached; only the moving content is rebuilt each frame. Everything is
        rasterised in a single pass.
        """
        mobjects = (
            self._static_mobjects()
            + self._dynamic_left()
            + self._dynamic_right()
            + self._dynamic_bottom()
        )
        self.camera.reset()
        self.camera.capture_mobjects(mobjects)
        return np.asarray(self.camera.pixel_array, dtype=np.uint8).copy()
