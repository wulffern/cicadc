# cicadc

An interactive desktop program that demonstrates how an analog-to-digital
converter (ADC) works. It shows two side-by-side panels rendered with
[manim](https://www.manim.community/) inside a [PySide6](https://doc.qt.io/qtforpython-6/)
window:

- **Left panel (analog)** - the analog signal drawn as a path that scrolls past
  "now" (the middle line), with the future at the top and the past at the
  bottom. A blue car drives along the (clean) analog signal and turns to follow
  the path. A "shadow" car marks the converter output at now - pale green when a
  decimator is active (the coarse quantizer/modulator output) or white/gray
  otherwise (the digital output). Noise is added before the converter, not on
  the displayed analog curve.
- **Right panel (digital)** - the digital signal as a sample-and-hold staircase.
  Pale green is the coarse quantizer/modulator output and white/gray is the
  decimated/filtered digital output (normalised to unity gain at the signal
  frequency). A white/gray car follows the digital output, trailing by the
  filter's group delay so it lands back on the analog curve (delay-compensated).

Two analysis strips run along the bottom of the view:

- **Quantization noise (bottom left)** - the error `analog - digital output`
  sampled at each instant and held as a staircase, with time on the x-axis and
  "now" on the far right. The y-axis is scaled to the quantizer's **LSB**
  (`±1 LSB`), with the LSB-in-FS value annotated.
- **Digital spectrum (bottom right)** - a Hann-windowed FFT of the digital
  output on a **logarithmic frequency** axis (`f / f_s`), with magnitude in
  **dBFS** (0 dB = full scale). The transform uses a long window (1024 points)
  and is recomputed only when a new sample arrives.

The ADC can run as a plain Nyquist-rate uniform quantizer or as a 1st- or
2nd-order **sigma-delta** modulator (with optional dither); a sigma-delta
bitstream is decimated by a cascaded sinc^N (CIC-style) moving-average filter
matched to the modulator order. When reconstructing, both the decimation
filter's gain/group-delay **and** the modulator's measured in-band signal
transfer (gain/phase) are de-embedded, so the quantization-noise strip shows the
true noise rather than a residual signal-transfer error. (The 2nd-order loop's
coefficients are chosen for single-bit stability, so some residual harmonic
distortion of the signal remains visible - a real modulator artifact.)

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

## Prebuilt binaries

Standalone, self-contained bundles (no Python install required) are built for
Linux, Windows and macOS (Apple Silicon) and attached to each
[GitHub Release](https://github.com/wulffern/cicadc/releases). Download the zip
for your platform, unpack it, and run:

- **Linux** - unpack the zip, run `cicadc/cicadc`
- **Windows** - unpack the zip, run `cicadc\cicadc.exe`
- **macOS** - open the `.dmg` and drag `cicadc.app` into Applications.

On macOS the app is **unsigned / not notarized**, so Gatekeeper blocks the first
launch (often with a cascade of "cannot verify developer" prompts for the
bundled libraries). Clear the download quarantine once and it runs normally:

```bash
xattr -dr com.apple.quarantine /Applications/cicadc.app
open /Applications/cicadc.app
```

(Right-click -> Open only whitelists the top-level app, so for this multi-library
bundle the `xattr` one-liner is the reliable fix.)

Intel Macs are not pre-built (GitHub is retiring Intel macOS runners); install
from PyPI/source there with `pip install cicadc`.

The bundles are produced by the `Build binaries` workflow (PyInstaller); you can
also build one locally with `pyinstaller --noconfirm packaging/cicadc.spec`.

## Run

```bash
cicadc                   # installed console script
# or, from a checkout without installing:
python main.py
```

## Controls

- **Frequency / Amplitude** - the input sinusoid (sliders; amplitude is a fraction of full scale).
- **Speed** - how fast the signal scrolls past "now".
- **Bits** - resolution of the quantizer / modulator.
- **Sample period** - the ADC sample clock interval.
- **Noise** - additive noise amplitude, applied before the converter.
- **ADC type** - Nyquist quantizer, or 1st-/2nd-order sigma-delta modulator.
- **Dither** - deterministic dither for the sigma-delta quantizer (ΣΔ modes only).
- **Avg** - decimation/averaging length on the digital output (1 = off); uses a
  sinc^N cascade matched to the ADC type.
- **Play / Pause** - start or stop the animation.

A signal-chain bar above the view highlights the active path
(Analog → Noise → Quantizer/ΣΔ → Filter → Digital).

## Development

```bash
make test     # run unit tests
make check    # import and print version
make lint     # ruff (if installed)
make build    # build wheel + sdist
```

## Acknowledgements

Big thanks to **Domen Visnar** for the idea behind this project!

## Rendering

The manim scene is rasterised offscreen (Cairo) and painted into the Qt widget.
Cost scales with pixel count, so resolution and frame rate trade off; the
default is 1280x1152 at 30 fps (the taller frame makes room for the bottom
analysis strips), with the static scenery and the FFT curve cached so only the
moving content is rebuilt each frame.

## Status

Single sinusoid input, adjustable bit depth, noise, Nyquist and 1st-/2nd-order
sigma-delta ADCs, a sinc^N decimation filter with delay compensation, the
signal-chain bar, and the quantization-noise / FFT analysis strips are
implemented. Random / multi-sinusoid inputs and a configurable bandwidth filter
are planned follow-ups.
