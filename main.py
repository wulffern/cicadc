#!/usr/bin/env python3
"""Convenience launcher: run the cicadc GUI without installing the package.

Adds ``src`` to the path so ``python main.py`` works from a fresh checkout.
For an installed package, use the ``cicadc`` console script instead.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from cicadc.app import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run())
