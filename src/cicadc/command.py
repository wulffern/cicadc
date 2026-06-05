#!/usr/bin/env python3
######################################################################
##        Copyright (c) 2026 Carsten Wulff Software, Norway
## ###################################################################
##  The MIT License (MIT)
##
##  Permission is hereby granted, free of charge, to any person obtaining a copy
##  of this software and associated documentation files (the "Software"), to deal
##  in the Software without restriction, including without limitation the rights
##  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
##  copies of the Software, and to permit persons to whom the Software is
##  furnished to do so, subject to the following conditions:
##
##  The above copyright notice and this permission notice shall be included in all
##  copies or substantial portions of the Software.
######################################################################
"""Shared helpers (logging setup) for the cicadc CLI."""

from __future__ import annotations

import logging


def setup_logging(color: bool = True, level: int = logging.INFO) -> None:
    """Configure root logging for the application."""
    fmt = "%(levelname)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)
