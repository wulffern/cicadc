"""Encode rendered RGBA frames into an H.264 MP4 with PyAV.

PyAV is already a (transitive) dependency and bundles its own ffmpeg
libraries, so screen recording works from the standalone binaries without a
separate ffmpeg install. The recorder is fed the same RGBA arrays the render
widget paints, so the saved video matches the on-screen animation exactly at
the scene's native resolution.
"""

from __future__ import annotations

import numpy as np


class VideoRecorder:
    """Incrementally encode RGBA frames to an MP4 file.

    The output stream dimensions are locked to the first frame, so every frame
    handed to :meth:`add_frame` must share that shape.
    """

    def __init__(self, path: str, fps: int = 30, crf: int = 20) -> None:
        import av  # local import: keep the GUI importable even if av is broken

        self._av = av
        self.path = str(path)
        self.fps = max(1, int(fps))
        self._crf = int(crf)
        self._container = av.open(self.path, mode="w")
        self._stream = None
        self._count = 0

    def _ensure_stream(self, width: int, height: int) -> None:
        if self._stream is not None:
            return
        # H.264 needs even dimensions; round down if a frame is ever odd.
        width -= width % 2
        height -= height % 2
        stream = self._container.add_stream("h264", rate=self.fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(self._crf), "preset": "veryfast"}
        self._stream = stream
        self._width = width
        self._height = height

    def add_frame(self, rgba: np.ndarray) -> None:
        rgb = np.ascontiguousarray(rgba[:, :, :3])
        h, w = rgb.shape[:2]
        self._ensure_stream(w, h)
        rgb = rgb[: self._height, : self._width]
        frame = self._av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self._count += 1

    @property
    def frame_count(self) -> int:
        return self._count

    def close(self) -> None:
        """Flush the encoder and finalize the file."""
        try:
            if self._stream is not None:
                for packet in self._stream.encode():
                    self._container.mux(packet)
        finally:
            self._container.close()
