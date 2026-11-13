# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Alias this package as ``warp`` in ``sys.modules`` (KEP-0001 M1.5).

Newton's ~360 ``import warp as wp`` sites stay untouched; the shim is the
single-point switch. Call :func:`install_as_warp` before any ``import warp``
in the process — if real warp-lang is already imported, it fails loud rather
than mixing two warp packages in one interpreter (black-box principle).

Public ``warp.*`` submodules Newton references are mapped onto their
vendored counterparts (``warp.types`` → ``warp_metal.frontend.types``);
``warp._src`` mirrors ``warp_metal.frontend``. Anything outside the mapped
surface raises ImportError — the capability boundary is the same fail-loud
one the package itself enforces.
"""

from __future__ import annotations

import importlib
import sys

_DONE = False


def install_as_warp() -> None:
    """Plant ``sys.modules['warp'] = warp_metal`` plus submodule aliases.

    Idempotent. Refuses to run after real warp-lang has been imported.
    """
    global _DONE
    existing = sys.modules.get("warp")
    if existing is not None:
        import warp_metal

        if existing is warp_metal or getattr(existing, "__name__", "") == "warp_metal":
            return  # already installed
        raise RuntimeError(
            "warp-metal shim: real warp-lang is already imported in this process; "
            "install_as_warp() must run before any 'import warp'"
        )
    if _DONE:
        return

    import warp_metal
    from . import frontend

    # the package itself + every already-loaded submodule
    sys.modules["warp"] = warp_metal
    for name, mod in list(sys.modules.items()):
        if name == "warp_metal" or name.startswith("warp_metal."):
            sys.modules.setdefault("warp" + name[len("warp_metal"):], mod)

    # public submodule names Newton references (real warp re-exports these
    # from warp/_src/*); map onto the vendored modules and pre-import so the
    # sys.modules entries exist even if not loaded yet
    for public, vendored in (
        ("types", "warp_metal.frontend.types"),
        ("codegen", "warp_metal.frontend.codegen"),
        ("builtins", "warp_metal.frontend.builtins"),
        ("context", "warp_metal.frontend.context"),
    ):
        mod = importlib.import_module(vendored)
        sys.modules["warp." + public] = mod
        sys.modules["warp._src." + public] = mod

    # warp._src -> the vendored frontend (Newton does `from warp._src import types`)
    sys.modules["warp._src"] = frontend
    warp_metal._src = frontend  # attribute chain: warp._src.types

    _DONE = True
