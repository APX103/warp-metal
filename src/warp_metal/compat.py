# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""``import warp_metal as wp`` compatibility layer + vendored version sentinel.

Design (KEP-0001):

- This module is the single import surface Newton will use. The ``wp.*`` API
  surface grows incrementally over M0.2–M0.5; nothing may bypass this layer.
- The **version sentinel** is a drift guard for the vendored frontend: the
  frontend ships a manifest with the upstream warp version it was vendored
  from, and this module hard-fails at import time on mismatch. The sentinel
  activates once ``frontend/`` is populated (M0.2); before that it is a no-op
  returning ``None``.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Frozen upstream version the vendored frontend must declare. KEP-0001 pins
# warp-lang 1.19.x (matches the version Newton currently depends on).
EXPECTED_VENDORED_WARP_VERSION = "1.19.0"

# Package-relative path where the vendored frontend writes its version
# manifest (M0.2). Kept here so compat and frontend agree on the contract.
MANIFEST_RELATIVE_PATH = "frontend/_vendored_version.txt"


class VendoredVersionError(RuntimeError):
    """Raised when the vendored frontend is missing or version-mismatched."""


def check_vendored_version() -> str | None:
    """Validate the vendored frontend manifest against the pinned version.

    Returns the vendored version string when a manifest exists and matches,
    ``None`` when the frontend is not yet vendored (M0.1 skeleton state), and
    raises :class:`VendoredVersionError` on any mismatch once a manifest
    exists. M0.2 wires this into an import-time check.
    """
    import importlib.resources

    pkg = importlib.resources.files("warp_metal")
    manifest = pkg.joinpath(MANIFEST_RELATIVE_PATH)
    if not manifest.is_file():
        # Frontend not vendored yet (M0.1 skeleton). Nothing to guard.
        return None
    vendored = manifest.read_text(encoding="utf-8").strip()
    if vendored != EXPECTED_VENDORED_WARP_VERSION:
        raise VendoredVersionError(
            f"vendored frontend version {vendored!r} does not match pinned "
            f"{EXPECTED_VENDORED_WARP_VERSION!r}; re-vendor or update the pin "
            f"(KEP-0001 maintenance line: upstream warp rebase)"
        )
    return vendored
