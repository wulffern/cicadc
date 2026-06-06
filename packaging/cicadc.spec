# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the cicadc GUI (one-folder bundle, all platforms).

Run from the repository root:

    pyinstaller --noconfirm packaging/cicadc.spec

The app renders manim frames offscreen (no video export), so ffmpeg and LaTeX
are *not* bundled. manim and its less-common dependencies do not ship
PyInstaller hooks, so we ``collect_all`` them (data files, submodules, dylibs)
and copy the package metadata manim reads at runtime.
"""

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

# Paths in a spec are resolved relative to the spec's directory, so anchor
# everything to the repository root (the spec lives in ``packaging/``).
_SPEC_DIR = SPECPATH  # directory containing this spec file
ROOT = os.path.abspath(os.path.join(_SPEC_DIR, os.pardir))
ENTRY = os.path.join(_SPEC_DIR, "cicadc_app.py")
SRC = os.path.join(ROOT, "src")

# The car sprite is loaded relative to the cicadc package directory.
datas = [(os.path.join(SRC, "cicadc", "assets"), "cicadc/assets")]
binaries = []
hiddenimports = collect_submodules("cicadc") + ["av"]

# manim ecosystem packages that lack built-in PyInstaller hooks. The app uses
# manim's Cairo camera (rendered offscreen), not the OpenGL window backends, so
# moderngl-window / pyglet / Qt-context shims are intentionally left out - they
# only drag in a second Qt binding (PyQt5) and conflict with PySide6.
_COLLECT = [
    "manim",
    "av",  # PyAV: bundles ffmpeg libs used by the video recorder (lazy import)
    "manimpango",
    "skia_pathops",
    "mapbox_earcut",
    "isosurfaces",
    "svgelements",
    "cloup",
    "screeninfo",
    "srt",
    "rich",
    "pygments",
]
for _pkg in _COLLECT:
    try:
        _d, _b, _h = collect_all(_pkg)
        datas += _d
        binaries += _b
        hiddenimports += _h
    except Exception as exc:  # pragma: no cover - best effort on the runner
        print(f"[cicadc.spec] skip collect_all({_pkg}): {exc}")

# Packages whose distribution metadata is consulted at runtime (manim resolves
# its own version and discovers plugins via importlib.metadata).
for _pkg in ["manim", "cloup", "click", "rich", "numpy", "scipy", "Pillow"]:
    try:
        datas += copy_metadata(_pkg)
    except Exception as exc:  # pragma: no cover
        print(f"[cicadc.spec] skip copy_metadata({_pkg}): {exc}")

a = Analysis(
    [ENTRY],
    pathex=[SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Keep a single Qt binding (PySide6); manim's optional OpenGL window
    # backends would otherwise pull in PyQt5 and abort the build.
    excludes=[
        "tkinter", "matplotlib", "pytest", "IPython",
        "PyQt5", "PyQt6", "PySide2",
        "moderngl_window", "pyglet", "glfw",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cicadc",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI app: no terminal window on Windows/macOS
    disable_windowed_traceback=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="cicadc",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="cicadc.app",
        icon=None,
        bundle_identifier="no.wulff.cicadc",
        info_plist={"NSHighResolutionCapable": True},
    )
