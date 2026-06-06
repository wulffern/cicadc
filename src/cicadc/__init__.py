"""cicadc - an interactive PySide6 + manim demonstration of how an ADC works.

Two side-by-side panels visualise analog-to-digital conversion: a scrolling
analog signal (with optional noise) on the left and the resulting digital
signal on the right, with an optional moving-average filter, quantization
shadow, and little cars that drive along the signal paths. Two analysis strips
along the bottom show the quantization noise over time (scaled to the LSB) and
the FFT of the digital output (log frequency axis, 0 dBFS reference).
"""

__version__ = "0.3.0"
__author__ = "Carsten Wulff"
__email__ = "carsten@wulff.no"

__all__ = [
    "SignalSource",
    "Quantizer",
]

from .signal_source import SignalSource
from .quantizer import Quantizer
