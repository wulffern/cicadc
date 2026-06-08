"""Main application window: the render view plus a panel of live controls."""

from __future__ import annotations

import os
import time
from typing import Callable, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .manim_scene import AdcScene
from .quantizer import Quantizer
from .render_widget import RenderWidget
from .signal_source import SignalSource

# Chain-bar block colors (mirror the scene palette).
_ANALOG = "#3361e6"   # analog signal (blue, matches the car)
_NOISE = "#ffd400"    # additive noise
_QUANT = "#b6e3b6"    # quantizer / modulator output (pale green)
_DIGITAL = "#cdd6e6"  # decimated/filtered digital output (white / gray)


class ChainBar(QWidget):
    """A horizontal block diagram of the signal chain shown above the panels.

    Analog -> Noise -> ADC -> Filter -> Digital. The Noise and Filter blocks dim
    when they are switched off so the active processing chain is obvious.
    """

    def __init__(self) -> None:
        super().__init__()
        self._blocks: dict[str, QLabel] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)
        layout.addStretch(1)
        order = ["analog", "noise", "adc", "filter", "digital"]
        for i, key in enumerate(order):
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._blocks[key] = lbl
            layout.addWidget(lbl)
            if i < len(order) - 1:
                arrow = QLabel("\u2192")
                arrow.setStyleSheet("color:#55617a; font-size:16px;")
                layout.addWidget(arrow)
        layout.addStretch(1)

    @staticmethod
    def _style(label: QLabel, active: bool, color: str) -> None:
        if active:
            label.setStyleSheet(
                f"QLabel {{ color:{color}; border:1px solid {color}; border-radius:6px;"
                f" padding:4px 10px; font-weight:bold; background:#121a2e; }}"
            )
        else:
            label.setStyleSheet(
                "QLabel { color:#55617a; border:1px solid #2b3a63; border-radius:6px;"
                " padding:4px 10px; background:#0f1626; }"
            )

    def update_state(
        self, noise_on: bool, filter_on: bool, bits: int, taps: int, sd_order: int = 0
    ) -> None:
        is_sd = sd_order > 0
        self._blocks["analog"].setText("ANALOG")
        self._blocks["noise"].setText("+ NOISE" if noise_on else "noise off")
        order_sup = {1: "\u00b9", 2: "\u00b2", 3: "\u00b3"}.get(sd_order, "")
        if sd_order >= 3:  # leapfrog
            adc_text = f"LF{order_sup} \u00b7 {bits} bit"
        elif is_sd:
            adc_text = f"\u03a3\u0394{order_sup} \u00b7 {bits} bit"
        else:
            adc_text = f"ADC \u00b7 {bits} bit"
        self._blocks["adc"].setText(adc_text)
        if is_sd:
            decim_sup = {2: "\u00b2", 3: "\u00b3", 4: "\u2074"}.get(sd_order + 1, "")
            on_text, off_text = f"sinc{decim_sup} \u00b7 {taps}", "decimate off"
        else:
            on_text, off_text = f"FILTER \u00b7 avg {taps}", "filter off"
        self._blocks["filter"].setText(on_text if filter_on else off_text)
        self._blocks["digital"].setText("DIGITAL")

        self._style(self._blocks["analog"], True, _ANALOG)
        self._style(self._blocks["noise"], noise_on, _NOISE)
        self._style(self._blocks["adc"], True, _QUANT)
        self._style(self._blocks["filter"], filter_on, _DIGITAL)
        self._style(self._blocks["digital"], True, _DIGITAL)


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ADC Intro - how an analog-to-digital converter works")

        self.signal = SignalSource(frequency=0.6, amplitude=0.85, speed=1.0, window=4.0, sample_period=0.4)
        self.quantizer = Quantizer(bits=3)
        self.scene = AdcScene(signal=self.signal, quantizer=self.quantizer)
        self.view = RenderWidget(self.scene, fps=30)

        self.chain = ChainBar()
        controls = self._build_controls()
        self._update_chain()

        body = QHBoxLayout()
        body.addWidget(self.view, stretch=1)
        body.addWidget(controls, stretch=0)

        root = QVBoxLayout(self)
        root.addWidget(self.chain, stretch=0)
        root.addLayout(body, stretch=1)

        self._apply_dark_theme()
        self.resize(1280, 800)

    # --------------------------------------------------------------- helpers
    def _slider_row(
        self,
        vmin: float,
        vmax: float,
        step: float,
        value: float,
        suffix: str,
        decimals: int,
        on_change: Callable[[float], None],
    ) -> Tuple[QWidget, QSlider]:
        """Build a float slider (QSlider is integer-only) with a live value label.

        The slider is mapped onto integer ticks of ``step`` and the float value is
        recovered as ``tick * step``; ``on_change`` receives the float value.
        """
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(round(vmin / step)), int(round(vmax / step)))
        slider.setValue(int(round(value / step)))

        label = QLabel(f"{value:.{decimals}f}{suffix}")
        label.setMinimumWidth(58)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        label.setStyleSheet("color:#aab6d0;")

        def handler(tick: int) -> None:
            v = tick * step
            label.setText(f"{v:.{decimals}f}{suffix}")
            on_change(v)

        slider.valueChanged.connect(handler)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, stretch=1)
        h.addWidget(label, stretch=0)
        return row, slider

    # --------------------------------------------------------------- controls
    def _build_controls(self) -> QWidget:
        box = QGroupBox("Controls")
        box.setMaximumWidth(300)
        form = QFormLayout()

        self.input_combo = QComboBox()
        self.input_combo.addItem("Single sinusoid")
        self.input_combo.setEnabled(False)  # only shape available in the MVP
        form.addRow("Input", self.input_combo)

        self.adc_combo = QComboBox()
        self.adc_combo.addItem("Nyquist (uniform)", "nyquist")
        self.adc_combo.addItem("1st-order \u03a3\u0394", "sigma_delta")
        self.adc_combo.addItem("2nd-order \u03a3\u0394", "sigma_delta2")
        self.adc_combo.addItem("3rd-order leapfrog", "leapfrog")
        self.adc_combo.addItem("Leapfrog (control-bounded)", "leapfrog_cb")
        self.adc_combo.currentIndexChanged.connect(self._on_adc_type)
        form.addRow("ADC type", self.adc_combo)

        freq_row, self.freq_slider = self._slider_row(
            0.05, 5.0, 0.05, self.signal.frequency, " Hz", 2, self._on_freq
        )
        form.addRow("Frequency", freq_row)

        amp_row, self.amp_slider = self._slider_row(
            0.0, 1.2, 0.05, self.signal.amplitude, "", 2, self._on_amp
        )
        form.addRow("Amplitude (FS)", amp_row)

        speed_row, self.speed_slider = self._slider_row(
            0.1, 4.0, 0.1, self.signal.speed, " x", 1, self._on_speed
        )
        form.addRow("Scroll speed", speed_row)

        self.bits_slider = QSlider(Qt.Orientation.Horizontal)
        self.bits_slider.setRange(1, 8)
        self.bits_slider.setValue(self.quantizer.bits)
        self.bits_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.bits_slider.valueChanged.connect(self._on_bits)
        self.bits_label = QLabel(self._bits_text())
        form.addRow(self.bits_label, self.bits_slider)

        sample_row, self.sample_slider = self._slider_row(
            0.05, 1.0, 0.05, self.signal.sample_period, " s", 2, self._on_sample
        )
        form.addRow("Sample period", sample_row)

        noise_row, self.noise_slider = self._slider_row(
            0.0, 0.5, 0.02, self.signal.noise_amp, "", 2, self._on_noise
        )
        form.addRow("Noise (FS)", noise_row)

        self.filter_slider = QSlider(Qt.Orientation.Horizontal)
        self.filter_slider.setRange(1, 64)
        self.filter_slider.setValue(self.scene.filter_taps)
        self.filter_slider.setPageStep(4)
        self.filter_slider.setTickInterval(8)
        self.filter_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.filter_slider.valueChanged.connect(self._on_filter)
        self.filter_label = QLabel(self._filter_text())
        form.addRow(self.filter_label, self.filter_slider)

        self.dither_check = QCheckBox("dither (\u03a3\u0394 only)")
        self.dither_check.setChecked(False)
        self.dither_check.setEnabled(False)  # only meaningful in sigma-delta mode
        self.dither_check.toggled.connect(self._on_dither)
        form.addRow("Dither", self.dither_check)

        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._on_play)

        self.record_button = QPushButton("\u25cf Record video")
        self.record_button.setCheckable(True)
        self.record_button.toggled.connect(self._on_record)
        self.record_button.setStyleSheet(
            "QPushButton:checked { background-color:#b3261e; }"
        )

        self.record_label = QLabel("")
        self.record_label.setWordWrap(True)
        self.record_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.record_label.setStyleSheet("color:#ff8a80; font-size:11px;")

        outer = QVBoxLayout(box)
        outer.addLayout(form)
        outer.addWidget(self.play_button)
        outer.addWidget(self.record_button)
        outer.addWidget(self.record_label)

        hint = QLabel(
            "Blue = analog signal (+ car)\n"
            "Pale green = quantizer/modulator out\n"
            "White/gray = digital output (+ car)\n"
            "Yellow dots = samples\n"
            "Bottom left = quantization noise vs time (now at right)\n"
            "Bottom right = FFT of digital output (0 dB = FS)\n"
            "Record video = save the animation to an .mp4"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8aa0c8; font-size:11px;")
        outer.addWidget(hint)
        outer.addStretch(1)

        thanks = QLabel("Big thanks to Domen Visnar for the idea!")
        thanks.setWordWrap(True)
        thanks.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thanks.setStyleSheet("color:#f5c542; font-size:11px; font-style:italic;")
        outer.addWidget(thanks)
        return box

    def _bits_text(self) -> str:
        return f"Bits: {self.quantizer.bits}"

    def _filter_text(self) -> str:
        n = self.scene.filter_taps
        return "Filter: off" if n <= 1 else f"Avg: {n}"

    # ----------------------------------------------------------------- slots
    def _refresh_if_paused(self) -> None:
        if not self.view.is_running():
            self.view.render_once()

    def _update_chain(self) -> None:
        sd_order = {"sigma_delta": 1, "sigma_delta2": 2, "leapfrog": 3,
                    "leapfrog_cb": 3}.get(self.scene.adc_mode, 0)
        self.chain.update_state(
            noise_on=self.signal.noise_amp > 0.0,
            filter_on=self.scene.filter_taps > 1,
            bits=self.quantizer.bits,
            taps=self.scene.filter_taps,
            sd_order=sd_order,
        )

    def _on_freq(self, v: float) -> None:
        self.signal.frequency = v
        self.scene.reset_sd()
        self._refresh_if_paused()

    def _on_amp(self, v: float) -> None:
        self.signal.amplitude = v
        self.scene.reset_sd()
        self._refresh_if_paused()

    def _on_speed(self, v: float) -> None:
        self.signal.speed = v

    def _on_sample(self, v: float) -> None:
        self.signal.sample_period = v
        self.scene.reset_sd()
        self._refresh_if_paused()

    def _on_bits(self, v: int) -> None:
        self.quantizer.bits = v
        self.bits_label.setText(self._bits_text())
        self.scene.reset_sd()
        self._update_chain()
        self._refresh_if_paused()

    def _on_noise(self, v: float) -> None:
        self.signal.noise_amp = v
        self.scene.reset_sd()
        self._update_chain()
        self._refresh_if_paused()

    def _on_filter(self, v: int) -> None:
        self.scene.filter_taps = v
        self.filter_label.setText(self._filter_text())
        self._update_chain()
        self._refresh_if_paused()

    def _on_adc_type(self, index: int) -> None:
        mode = self.adc_combo.itemData(index)
        self.scene.set_adc_mode(mode)
        self.dither_check.setEnabled(mode in ("sigma_delta", "sigma_delta2", "leapfrog"))
        # The control-bounded leapfrog only reconstructs at high oversampling, so
        # drop the signal frequency to put it around OSR ~20 when it is selected.
        if mode == "leapfrog_cb":
            target = 0.5 / (20.0 * self.signal.sample_period)   # OSR ~ 20
            tick = max(self.freq_slider.minimum(),
                       min(self.freq_slider.maximum(), int(round(target / 0.05))))
            self.freq_slider.setValue(tick)   # triggers _on_freq -> resets caches
        self._update_chain()
        self._refresh_if_paused()

    def _on_dither(self, checked: bool) -> None:
        self.scene.set_sd_dither(1.0 if checked else 0.0)
        self._refresh_if_paused()

    def _on_play(self, checked: bool) -> None:
        if checked:
            self.play_button.setText("Pause")
            self.view.start()
        else:
            self.play_button.setText("Play")
            self.view.stop()

    def _on_record(self, checked: bool) -> None:
        if checked:
            default = os.path.join(
                os.path.expanduser("~"), f"cicadc-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
            )
            path, _ = QFileDialog.getSaveFileName(
                self, "Save recording", default, "MP4 video (*.mp4)"
            )
            if not path:
                self.record_button.setChecked(False)  # user cancelled
                return
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            try:
                self.view.start_recording(path)
            except Exception as exc:  # encoder unavailable / bad path
                self.record_button.setChecked(False)
                QMessageBox.critical(
                    self, "Recording failed", f"Could not start recording:\n{exc}"
                )
                return
            self.record_button.setText("\u25a0 Stop recording")
            self.record_label.setText(f"Recording to {os.path.basename(path)}\u2026")
            if not self.view.is_running():  # need motion in the video
                self.play_button.setChecked(True)
        else:
            self.record_button.setText("\u25cf Record video")
            if not self.view.is_recording():
                return
            path, frames = self.view.stop_recording()
            secs = frames / max(1, self.view.fps)
            self.record_label.setText(
                f"Saved {os.path.basename(path)} ({frames} frames, ~{secs:.1f}s)"
            )

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self.view.is_recording():
            self.view.stop_recording()  # flush partial file instead of corrupting it
        super().closeEvent(event)

    # ----------------------------------------------------------------- theme
    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background-color: #0b0f1a; color: #e6f0ff; }
            QGroupBox {
                border: 1px solid #2b3a63; border-radius: 6px; margin-top: 14px;
                font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton {
                background-color: #1a2440; border: 1px solid #2b3a63;
                border-radius: 5px; padding: 8px; font-weight: bold;
            }
            QPushButton:checked { background-color: #1f7a3a; }
            QComboBox {
                background-color: #121a2e; border: 1px solid #2b3a63;
                border-radius: 4px; padding: 3px;
            }
            """
        )
