"""Main application window: the render view plus a panel of live controls."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .manim_scene import AdcScene
from .quantizer import Quantizer
from .render_widget import RenderWidget
from .signal_source import SignalSource


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ADC Intro - how an analog-to-digital converter works")

        self.signal = SignalSource(frequency=0.6, amplitude=0.85, speed=1.0, window=4.0, sample_period=0.4)
        self.quantizer = Quantizer(bits=3)
        self.scene = AdcScene(signal=self.signal, quantizer=self.quantizer)
        self.view = RenderWidget(self.scene, fps=24)

        controls = self._build_controls()

        layout = QHBoxLayout(self)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(controls, stretch=0)

        self._apply_dark_theme()
        self.resize(1280, 760)

    # --------------------------------------------------------------- controls
    def _build_controls(self) -> QWidget:
        box = QGroupBox("Controls")
        box.setMaximumWidth(280)
        form = QFormLayout()

        self.input_combo = QComboBox()
        self.input_combo.addItem("Single sinusoid")
        self.input_combo.setEnabled(False)  # only shape available in the MVP
        form.addRow("Input", self.input_combo)

        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(0.05, 5.0)
        self.freq_spin.setSingleStep(0.05)
        self.freq_spin.setValue(self.signal.frequency)
        self.freq_spin.setSuffix(" Hz")
        self.freq_spin.valueChanged.connect(self._on_freq)
        form.addRow("Frequency", self.freq_spin)

        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0.0, 1.2)
        self.amp_spin.setSingleStep(0.05)
        self.amp_spin.setValue(self.signal.amplitude)
        self.amp_spin.valueChanged.connect(self._on_amp)
        form.addRow("Amplitude (FS)", self.amp_spin)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 4.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(self.signal.speed)
        self.speed_spin.setSuffix(" x")
        self.speed_spin.valueChanged.connect(self._on_speed)
        form.addRow("Scroll speed", self.speed_spin)

        self.bits_slider = QSlider(Qt.Orientation.Horizontal)
        self.bits_slider.setRange(1, 8)
        self.bits_slider.setValue(self.quantizer.bits)
        self.bits_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.bits_slider.valueChanged.connect(self._on_bits)
        self.bits_label = QLabel(self._bits_text())
        form.addRow(self.bits_label, self.bits_slider)

        self.sample_spin = QDoubleSpinBox()
        self.sample_spin.setRange(0.05, 1.0)
        self.sample_spin.setSingleStep(0.05)
        self.sample_spin.setValue(self.signal.sample_period)
        self.sample_spin.setSuffix(" s")
        self.sample_spin.valueChanged.connect(self._on_sample)
        form.addRow("Sample period", self.sample_spin)

        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 0.5)
        self.noise_spin.setSingleStep(0.02)
        self.noise_spin.setValue(self.signal.noise_amp)
        self.noise_spin.valueChanged.connect(self._on_noise)
        form.addRow("Noise (FS)", self.noise_spin)

        self.filter_slider = QSlider(Qt.Orientation.Horizontal)
        self.filter_slider.setRange(1, 16)
        self.filter_slider.setValue(self.scene.filter_taps)
        self.filter_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.filter_slider.valueChanged.connect(self._on_filter)
        self.filter_label = QLabel(self._filter_text())
        form.addRow(self.filter_label, self.filter_slider)

        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._on_play)

        outer = QVBoxLayout(box)
        outer.addLayout(form)
        outer.addWidget(self.play_button)

        hint = QLabel(
            "Green = analog signal\nBlue car = analog now\n"
            "Gray = unfiltered digital\nGreen = filtered digital\n"
            "Yellow dots = samples"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8aa0c8; font-size:11px;")
        outer.addWidget(hint)
        outer.addStretch(1)
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

    def _on_freq(self, v: float) -> None:
        self.signal.frequency = v
        self._refresh_if_paused()

    def _on_amp(self, v: float) -> None:
        self.signal.amplitude = v
        self._refresh_if_paused()

    def _on_speed(self, v: float) -> None:
        self.signal.speed = v

    def _on_sample(self, v: float) -> None:
        self.signal.sample_period = v
        self._refresh_if_paused()

    def _on_bits(self, v: int) -> None:
        self.quantizer.bits = v
        self.bits_label.setText(self._bits_text())
        self._refresh_if_paused()

    def _on_noise(self, v: float) -> None:
        self.signal.noise_amp = v
        self._refresh_if_paused()

    def _on_filter(self, v: int) -> None:
        self.scene.filter_taps = v
        self.filter_label.setText(self._filter_text())
        self._refresh_if_paused()

    def _on_play(self, checked: bool) -> None:
        if checked:
            self.play_button.setText("Pause")
            self.view.start()
        else:
            self.play_button.setText("Play")
            self.view.stop()

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
            QDoubleSpinBox, QSpinBox, QComboBox {
                background-color: #121a2e; border: 1px solid #2b3a63;
                border-radius: 4px; padding: 3px;
            }
            """
        )
