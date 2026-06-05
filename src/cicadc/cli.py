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
"""cicadc command-line entry point."""

from __future__ import annotations

import logging

import click

from .command import setup_logging


@click.command()
@click.option("--color/--no-color", default=True, help="Enable/disable color output")
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging")
def main(color: bool, debug: bool) -> None:
    """cicadc: an interactive demonstration of how an ADC works.

    Launches a PySide6 + manim window with two side-by-side panels (analog and
    digital), with controls for frequency, amplitude, bit depth, noise and a
    moving-average filter.
    """
    setup_logging(color=color, level=logging.DEBUG if debug else logging.INFO)

    from .app import run

    raise SystemExit(run())


if __name__ == "__main__":
    main()
