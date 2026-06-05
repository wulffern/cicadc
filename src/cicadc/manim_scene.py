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
from .signal_source import SignalSource

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
CAR_PATH = os.path.join(_ASSETS_DIR, "car.png")

# Retro, 80's arcade-ish palette.
BG = "#0b0f1a"
PANEL_EDGE = "#2b3a63"
GRID = "#1d2742"
CURVE = "#39ff14"   # analog signal (neon green)
STAIR = "#39ff14"   # filtered digital signal (green, matches the green car)
PREFILTER = "#8a93a3"  # unfiltered (raw) digital signal (gray, matches the gray car)
ERROR = "#ff3b3b"   # quantization error
SAMPLE = "#ffd400"  # sample points
CAR = "#ff8c00"     # "now" marker / car
TEXT = "#e6f0ff"
BAR = "#3b82f6"
BAR_NOW = "#39ff14"


class AdcScene:
    def __init__(
        self,
        signal: SignalSource | None = None,
        quantizer: Quantizer | None = None,
        pixel_width: int = 880,
        pixel_height: int = 616,
        filter_taps: int = 1,
    ) -> None:
        self.signal = signal or SignalSource()
        self.quantizer = quantizer or Quantizer()
        self.filter_taps = filter_taps  # moving-average length on the digital output
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height

        self.frame_height = 7.0
        self.frame_width = self.frame_height * pixel_width / pixel_height

        self.camera = Camera(
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            background_color=BG,
        )

        self._text_cache: dict[tuple, Text] = {}
        self._car_arrays = {"color": self._load_car()}
        self._car_arrays["gray"] = self._to_gray(self._car_arrays["color"])
        self._car_arrays["shadow"] = self._fade_alpha(self._car_arrays["gray"], 0.4)
        self._car_arrays["green"] = self._tint(self._car_arrays["gray"], (0.25, 1.0, 0.3))
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
        # Left (analog) panel.
        self.lp_x0, self.lp_x1 = -half_w + 0.2, -0.15
        self.lp_y0, self.lp_y1 = -3.2, 3.2
        self.xc_left = (self.lp_x0 + self.lp_x1) / 2.0
        self.aw = (self.lp_x1 - self.lp_x0) / 2.0 - 0.45  # amplitude half-width
        self.yb = -2.55   # bottom of the time axis (the past)
        self.yt = 2.45    # top of the time axis (the future)
        self.y_now = (self.yb + self.yt) / 2.0  # "now" sits in the middle

        # Right (digital) panel - mirrors the left panel's geometry.
        self.rp_x0, self.rp_x1 = 0.15, half_w - 0.2
        self.rp_y0, self.rp_y1 = -3.2, 3.2
        self.xc_right = (self.rp_x0 + self.rp_x1) / 2.0
        self.aw_right = (self.rp_x1 - self.rp_x0) / 2.0 - 0.5

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

    def _group_delay(self) -> float:
        """Group delay of the K-tap moving average, in seconds of signal time."""
        K = max(1, int(self.filter_taps))
        return (K - 1) / 2.0 * self.signal.sample_period

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
                fallback = {"color": CAR, "gray": "#9aa3b2", "shadow": "#555a66", "green": CURVE}
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
    def _build_left(self) -> List:
        q = self.quantizer
        sig = self.signal
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

        # Continuous analog curve across past and future (now in the middle).
        half = self._half_span()
        curve = sig.curve_points(200, -half, half)
        true_pts = [(self._x_amp(v), self._y_time(t)) for t, v in curve]
        items.append(self._polyline(true_pts, CURVE, 4.0))

        # Sample points along the analog curve.
        first_k, last_k = sig.sample_indices_in(-half, half)
        dots = VGroup()
        for k in range(first_k, last_k + 1):
            t_rel = sig.t_rel_of_index(k)
            dots.add(Dot([self._x_amp(sig.sample_value(k)), self._y_time(t_rel), 0], radius=0.045, color=SAMPLE))
        items.append(dots)

        # Labels.
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

        # Cars. The unfiltered (gray shadow) and true analog (blue) cars sit at
        # now (the middle). The filtered (green) car trails by the filter's group
        # delay, landing back on the analog curve - i.e. delay-compensated.
        k0 = sig.sample_index_now()
        delay = self._group_delay()
        ang_now = self._tangent_angle(0.0, self.aw)
        items.append(self._car(self._x_amp(self._raw_level(k0)), self.y_now, angle=ang_now, variant="shadow"))
        if self.filter_taps > 1:  # the filtered (green) car only when a filter is active
            t_green = sig.t_rel_of_index(k0) - delay
            ang_green = self._tangent_angle(t_green, self.aw)
            items.append(self._car(self._x_amp(self._filt_level(k0)), self._y_time(t_green), angle=ang_green, variant="green"))
        items.append(self._car(self._x_amp(sig.value_continuous(0.0)), self.y_now, angle=ang_now, variant="color"))
        return items

    def _filter_gain(self) -> float:
        """Magnitude response of the K-tap moving average at the signal frequency.

        Normalising the filtered output by this gain makes the filter's transfer
        function unity at the signal frequency, so the filtered amplitude matches
        the analog signal (a plain moving average otherwise attenuates in-band).
        """
        K = max(1, int(self.filter_taps))
        if K == 1:
            return 1.0
        w = 2.0 * np.pi * self.signal.frequency * self.signal.sample_period
        s = np.sin(w / 2.0)
        if abs(s) < 1e-9:
            return 1.0  # near DC, gain is already 1
        mag = abs(np.sin(K * w / 2.0) / s) / K
        return float(max(mag, 0.1))  # clamp to avoid blow-up near a filter null

    def _raw_level(self, k: int) -> float:
        """Reconstructed quantizer level for sample index ``k`` (unfiltered)."""
        q = self.quantizer
        return q.level_of(q.code_of(self.signal.sample_value(k)))

    def _filt_level(self, k: int) -> float:
        """Filtered digital level at sample ``k`` (normalised K-tap moving avg)."""
        K = max(1, int(self.filter_taps))
        gain = self._filter_gain()
        return (sum(self._raw_level(j) for j in range(k - K + 1, k + 1)) / K) / gain

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

    def _build_right(self) -> List:
        """Right panel: the digital (sample-and-hold) signal vs time, mirroring
        the left panel's layout so the two graphs are directly comparable."""
        q = self.quantizer
        sig = self.signal
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

        half = self._half_span()
        delay = self._group_delay()
        K = max(1, int(self.filter_taps))

        # Unfiltered (gray) staircase at the true sample times, with sample dots.
        # When the filter is off it is the only output, so draw it more boldly.
        raw_pts, raw_dots = self._hold_staircase(self._raw_level, self._x_dig, -half, half, delay=0.0)
        if K > 1:
            items.append(self._polyline(raw_pts, PREFILTER, 2.0))
            # Filtered (green) staircase, shifted down by the group delay so it
            # lines up in time with the analog signal (delay-compensated).
            filt_pts, _ = self._hold_staircase(self._filt_level, self._x_dig, -half, half, delay=delay)
            items.append(self._polyline(filt_pts, STAIR, 3.5))
        else:
            items.append(self._polyline(raw_pts, "#aeb6c4", 3.5))

        dots = VGroup()
        for x, y in raw_dots:
            dots.add(Dot([x, y, 0], radius=0.045, color=SAMPLE))
        items.append(dots)

        # Labels.
        title = self._text("DIGITAL SIGNAL", 22, STAIR)
        title.move_to([(self.rp_x0 + self.rp_x1) / 2.0, self.rp_y1 - 0.35, 0])
        items.append(title)
        sub_text = f"{q.bits} bits  -  {q.num_levels} levels"
        if K > 1:
            sub_text += f"  -  avg {K}"
        sub = self._text(sub_text, 16, TEXT)
        sub.move_to([(self.rp_x0 + self.rp_x1) / 2.0, self.rp_y1 - 0.72, 0])
        items.append(sub)
        for val, lab in ((-1.0, "-FS"), (1.0, "+FS")):
            t = self._text(lab, 16, TEXT)
            t.move_to([self._x_dig(val), self.yb - 0.3, 0])
            items.append(t)

        # Current code readout near the bottom.
        k0 = sig.sample_index_now()
        now_code = q.code_of(sig.value_now())
        readout = self._text(f"{q.code_string(now_code)}  ({now_code})", 26, BAR_NOW)
        readout.move_to([(self.rp_x0 + self.rp_x1) / 2.0, self.yb - 0.3, 0])
        items.append(readout)

        # Cars (heading nudged): gray at the unfiltered output at now; green at the
        # filtered output, trailing by the group delay (matches the left panel).
        t_green = sig.t_rel_of_index(k0) - delay

        def nudge(t_rel):
            return max(-0.45, min(0.45, 0.4 * self._tangent_angle(t_rel, self.aw_right)))

        items.append(self._car(self._x_dig(self._raw_level(k0)), self.y_now, angle=nudge(0.0), variant="gray"))
        if K > 1:  # the filtered (green) car only when a filter is active
            items.append(self._car(self._x_dig(self._filt_level(k0)), self._y_time(t_green), angle=nudge(t_green), variant="green"))
        return items

    # ----------------------------------------------------------------- render
    def render_frame(self) -> np.ndarray:
        """Render the current state and return an ``(H, W, 4)`` RGBA uint8 array."""
        mobjects = self._build_left() + self._build_right()
        self.camera.reset()
        self.camera.capture_mobjects(mobjects)
        return np.asarray(self.camera.pixel_array, dtype=np.uint8).copy()
