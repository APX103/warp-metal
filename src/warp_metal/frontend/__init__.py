# Copyright 2026 Jaylin Lee
# Portions Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Vendored Warp compiler frontend.

Placeholder (M0.1 skeleton). M0.2 populates this package by vendoring the
compiler frontend from warp-lang 1.19.0 — ``warp/_src/`` codegen.py,
builtins.py, and the module/AST parts of context.py — with the
nvrtc/ptx/cuda binding paths stripped.

Every vendored file must carry its NVIDIA copyright header (Apache-2.0) and a
prominent notice of local modification, per Apache-2.0 §4(b). The vendored
upstream version is pinned via ``frontend/_vendored_version.txt`` and enforced
by :func:`warp_metal.compat.check_vendored_version`.
"""
