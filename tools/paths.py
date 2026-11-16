# Machine-local checkout locations for the conformance/bench tooling.
#
# Resolution order for the Newton checkout:
#   1. WARP_METAL_NEWTON_REPO environment variable
#   2. tools/local_paths.py (gitignored — put machine-specific paths there)
#   3. ~/work/newton
#
# Raises RuntimeError with fix-up instructions when none resolves.

import os
from pathlib import Path


def newton_repo() -> Path:
    env = os.environ.get("WARP_METAL_NEWTON_REPO")
    if env:
        return Path(env).expanduser()
    try:
        from local_paths import NEWTON_REPO  # gitignored, machine-specific

        return Path(NEWTON_REPO).expanduser()
    except ImportError:
        pass
    home = Path.home() / "work" / "newton"
    if home.exists():
        return home
    raise RuntimeError(
        "Cannot locate the Newton checkout. Set WARP_METAL_NEWTON_REPO "
        "or create tools/local_paths.py containing: NEWTON_REPO = '/path/to/newton'"
    )
