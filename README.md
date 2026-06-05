# cicadc

An interactive desktop program that demonstrates how an analog-to-digital
converter (ADC) works. It shows two side-by-side panels rendered with
[manim](https://www.manim.community/) inside a [PySide6](https://doc.qt.io/qtforpython-6/)
window:

- **Left panel (analog)** - the analog signal drawn as a path that scrolls past
  "now" (the middle line), with the future at the top and the past at the
  bottom. Optional noise is added. A blue car drives along the signal and turns
  to follow the path; a translucent gray "shadow" car marks the quantized value.
- **Right panel (digital)** - the digital signal as a sample-and-hold staircase.
  Gray is the unfiltered (raw) quantizer output and green is the optional
  moving-average filtered output (normalised to unity gain at the signal
  frequency). A green car follows the filtered output, trailing by the filter's
  group delay so it lands back on the analog curve (delay-compensated).

## Project layout

```
src/cicadc/          package (src layout, mirrors cicwave)
  cli.py             click console entry point (`cicadc`)
  app.py             QApplication bootstrap
  main_window.py     PySide6 window + controls
  manim_scene.py     offscreen manim renderer
  signal_source.py   analog signal model (sinusoid + noise)
  quantizer.py       N-bit quantizer
  render_widget.py   Qt widget + animation loop
  assets/car.png     car sprite
tests/unittests/     unittest suite
```

## Requirements

- Python 3.9-3.13 (manim does not yet support 3.14). A `python3.12` venv is used
  by the setup steps below.
- A working manim install (Cairo backend). On macOS you may need the system
  libraries `pkg-config cairo pango` via Homebrew if the wheels do not cover your
  platform.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .          # or: pip install -r requirements.txt
```

`make dev-install` runs the editable install for you.

## Run

```bash
cicadc                   # installed console script
# or, from a checkout without installing:
python main.py
```

## Controls

- **Frequency / Amplitude** - the input sinusoid (amplitude is a fraction of full scale).
- **Speed** - how fast the signal scrolls past "now".
- **Bits** - resolution of the quantizer.
- **Sample period** - the ADC sample clock interval.
- **Noise** - additive analog noise amplitude.
- **Avg** - moving-average filter length on the digital output (1 = off).
- **Play / Pause** - start or stop the animation.

## Development

```bash
make test     # run unit tests
make check    # import and print version
make lint     # ruff (if installed)
make build    # build wheel + sdist
```

## Status

Single/− sinusoid input, adjustable bit depth, noise, and a normalised
moving-average filter with delay compensation are implemented. Random /
multi-sinusoid inputs and a configurable bandwidth filter are planned follow-ups.
