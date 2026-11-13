# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Build the warp-metal native runtime dylib (ObjC++ / Metal.framework).

warp-native build style: compile the .mm into a dylib next to this module;
Python side binds via ctypes (see runtime/native.py). Requires the macOS
command-line toolchain (clang with ObjC++ + Metal framework headers).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).parent
DYLIB = MODULE_DIR / "libwarp_metal.dylib"
SOURCE = MODULE_DIR / "warp_metal.mm"


def build(force: bool = False) -> Path:
    if DYLIB.exists() and not force:
        if DYLIB.stat().st_mtime >= SOURCE.stat().st_mtime:
            return DYLIB
    cmd = [
        "clang++",
        "-ObjC++",
        "-arch", "arm64",
        "-dynamiclib",
        "-framework", "Metal",
        "-framework", "Foundation",
        "-fobjc-arc",
        "-O2",
        "-o", str(DYLIB),
        str(SOURCE),
    ]
    print(f"[warp-metal] building native runtime: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return DYLIB


if __name__ == "__main__":
    build(force="--force" in sys.argv)
    print(f"[warp-metal] built {DYLIB}")
