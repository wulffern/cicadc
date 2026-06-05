"""cicadc - an interactive PySide6 + manim demonstration of how an ADC works.

Two side-by-side panels visualise analog-to-digital conversion: a scrolling
analog signal (with optional noise) on the left and the resulting digital
signal on the right, with an optional moving-average filter, quantization
shadow, and little cars that drive along the signal paths.
"""

__version__ = "0.2.0"
__author__ = "Carsten Wulff"
__email__ = "carsten@wulff.no"

__all__ = [
    "SignalSource",
    "Quantizer",
]

from .signal_source import SignalSource
from .quantizer import Quantizer
