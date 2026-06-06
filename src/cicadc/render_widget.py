"""Qt widget that displays manim-rendered frames and drives the animation loop.

The widget owns a :class:`AdcScene` and a ``QTimer``. On each tick it advances
the signal model by the elapsed wall-clock time and asks the scene to render a
fresh RGBA frame, which it converts to a ``QImage`` and paints (scaled, aspect
preserving) into the widget.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QWidget

from .manim_scene import AdcScene
from .recorder import VideoRecorder


class RenderWidget(QWidget):
    """Displays the manim output and runs the scrolling animation."""

    frameRendered = Signal(int)  # emits the current "now" code after each frame

    def __init__(self, scene: AdcScene, fps: int = 24, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.fps = fps
        self._qimage: QImage | None = None
        self._arr: np.ndarray | None = None
        self._last_t: float | None = None
        self._recorder: VideoRecorder | None = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self.setMinimumSize(720, 504)
        self.render_once()

    # ------------------------------------------------------------- animation
    def is_running(self) -> bool:
        return self._timer.isActive()

    def start(self) -> None:
        if not self._timer.isActive():
            self._last_t = time.perf_counter()
            self._timer.start(int(1000 / self.fps))

    def stop(self) -> None:
        self._timer.stop()
        self._last_t = None

    # ------------------------------------------------------------- recording
    def is_recording(self) -> bool:
        return self._recorder is not None

    def start_recording(self, path: str) -> None:
        """Begin writing each rendered frame to an MP4 at ``path``."""
        self._recorder = VideoRecorder(path, fps=self.fps)

    def stop_recording(self) -> tuple[str, int]:
        """Finalize the recording and return ``(path, frame_count)``."""
        recorder = self._recorder
        self._recorder = None
        if recorder is None:
            return ("", 0)
        recorder.close()
        return (recorder.path, recorder.frame_count)

    def _on_tick(self) -> None:
        now = time.perf_counter()
        dt = 0.0 if self._last_t is None else (now - self._last_t)
        self._last_t = now
        # Clamp dt so a stalled frame does not jump the signal forwards wildly.
        self.scene.signal.advance(min(dt, 0.1))
        self.render_once()

    # ---------------------------------------------------------------- render
    def render_once(self) -> None:
        arr = self.scene.render_frame()
        self._set_array(arr)
        if self._recorder is not None:
            self._recorder.add_frame(self._arr)
        code = self.scene.quantizer.code_of(self.scene.signal.value_now())
        self.frameRendered.emit(int(code))

    def _set_array(self, arr: np.ndarray) -> None:
        self._arr = np.ascontiguousarray(arr)
        h, w, _ = self._arr.shape
        image = QImage(self._arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
        self._qimage = image.copy()  # detach from the numpy buffer
        self.update()

    # ----------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#05070d"))
        if self._qimage is None:
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        target = self.rect()
        scaled = self._qimage.size().scaled(target.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = target.x() + (target.width() - scaled.width()) // 2
        y = target.y() + (target.height() - scaled.height()) // 2
        painter.drawImage(QRect(x, y, scaled.width(), scaled.height()), self._qimage)
