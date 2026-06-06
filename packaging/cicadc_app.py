"""Entry point used when freezing the GUI into a standalone binary.

PyInstaller imports this module and runs :func:`cicadc.app.run`. Kept separate
from ``main.py`` (which manipulates ``sys.path`` for source checkouts) because a
frozen bundle ships ``cicadc`` as a real, importable package.
"""

from __future__ import annotations

import multiprocessing
import sys


def _main() -> int:
    # Some of manim's dependencies (moderngl-window, multiprocessing-based
    # helpers) spawn child processes; this is required for frozen apps on
    # Windows/macOS so the children re-enter cleanly instead of relaunching the
    # whole GUI.
    multiprocessing.freeze_support()
    from cicadc.app import run

    return run()


if __name__ == "__main__":
    sys.exit(_main())
