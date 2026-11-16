# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Device registry (KEP-0001 M1.2, bucket D).

Two devices: ``"cpu"`` (host fallback; arrays share unified memory) and
``"metal"`` (Apple Silicon GPU via MTLBuffer). Mirrors the observable
surface of upstream warp's device objects that Newton relies on:
``name``, ``ordinal``, ``is_cpu``, ``is_cuda``, ``is_metal``, ``arch``,
``context``, plus the capture-parity attributes (``stream``,
``is_capturing``, ``captures``, ``is_mempool_enabled``; KEP-0001 M2.5).
"""

from __future__ import annotations


class Stream:
    """Minimal stream token.

    warp-metal eager launches encode into one per-process command pipeline
    and execute at the next host-visible flush (graph replay runs one
    batched command buffer per ``capture_launch``), so a Stream is an
    ordering handle only: it records its device and name,
    carries the device's capture state for parity (:attr:`is_capturing`),
    and is accepted wherever upstream would take one. Each device owns one
    stable implicit stream — ``Device.stream`` and :func:`get_stream` return
    the same object.
    """

    def __init__(self, device=None, name=None):
        if device is None:
            device = "metal"
        if isinstance(device, str):
            from .devices import get_device

            device = get_device(device)
        self.device = device
        self.name = name or "metal_stream"

    @property
    def is_capturing(self) -> bool:
        """A boolean indicating whether a graph capture is currently ongoing
        on this stream (the device's active capture; one implicit stream)."""
        from . import host as _host

        return _host.active_capture(self.device) is not None

    def __repr__(self):
        return f"Stream(name={self.name!r}, device={self.device.name!r})"


class Event:
    """Minimal event handle for host-side timing.

    Upstream events are device queue fences; warp-metal records host
    timestamps, so recording an event is a timestamp and elapsed time is
    the wall-clock difference.
    """

    def __init__(self, device=None, name=None):
        if device is None:
            device = "metal"
        if isinstance(device, str):
            from .devices import get_device

            device = get_device(device)
        self.device = device
        self.name = name
        self._recorded: float | None = None

    def record(self) -> None:
        import time

        self._recorded = time.perf_counter()

    def __repr__(self):
        return f"Event(name={self.name!r}, device={self.device.name!r})"


class _DeviceAllocator:
    """Shared-memory allocator for the vendored frontend array path
    (upstream ``Device.get_allocator`` semantics).

    Every warp-metal allocation lives in shared unified memory
    (MTLBuffer-backed, CPU-accessible pointer), so the upstream
    host/device allocator split collapses to one implementation. The
    vendored ``array._init_new`` allocates through ``allocate(capacity)`` /
    ``deallocate(ptr, capacity)`` raw pointers; this allocator maps those
    pointers to their MTLBuffer handles for release."""

    # ptr -> native MTLBuffer handle (strong refs: the handle must outlive
    # the raw pointer handed out to the vendored array until deallocate)
    _HANDLES: dict[int, object] = {}

    def allocate(self, capacity: int) -> int:
        from .. import native as _mm

        handle = _mm.buffer_create(int(capacity))
        ptr = _mm.buffer_ptr(handle)
        self._HANDLES[ptr] = handle
        return ptr

    def deallocate(self, ptr: int, capacity: int) -> None:
        from .. import native as _mm

        handle = self._HANDLES.pop(int(ptr), None)
        if handle is not None:
            _mm.buffer_release(handle)

    # the runtime Device has no CUDA context; __del__ must call the deleter
    # directly instead of taking the (AttributeError-suppressed) guard path,
    # which would leak every allocation
    deallocate_requires_context_guard = False


class Device:
    """A compute device.

    Mirrors the upstream ``warp.Device`` attributes Newton's device-string
    threading reads: ``name``, ``alias``, ``ordinal``, ``is_cpu``,
    ``is_cuda``, ``is_metal``, ``arch``, ``context``, plus the capture-parity
    surface (``stream``, ``is_capturing``, ``captures``, ``is_mempool_enabled``)
    so Newton's capture-aware guards observe real state instead of relying
    on ``is_cuda`` short-circuits by accident (KEP-0001 M2.5).
    """

    def __init__(self, name: str, ordinal: int = 0, arch: int = 0):
        self.name = name
        self.alias = name  # upstream warp.Device.alias (Newton cache keys read it)
        self.ordinal = ordinal
        self.is_cpu = name == "cpu"
        self.is_cuda = False  # warp-metal strips CUDA entirely (KEP-0001)
        self.is_metal = name == "metal"
        self.arch = arch
        self.context = None
        # One stable implicit stream per device. Upstream ``Device.stream``
        # is a cudaStream handle; warp-metal has one implicit stream per
        # device, so the token is a Stream object whose role is the stable
        # hashable key Newton's ``captures.get(device.stream)`` guards read
        # (documented divergence).
        self._stream = Stream(self, name=f"{name}_default")
        # Upstream Device.is_mempool_enabled is a plain bool Newton reads
        # directly (kamino device info). Mempool allocation is a KEP-0001
        # Non-Goal (MTLBuffer allocations are direct), so it is always False
        # — parity attribute only, so capture/mem guards don't AttributeError.
        self.is_mempool_enabled = False

    @property
    def stream(self) -> Stream:
        """The device's stable implicit stream token.

        Upstream ``Device.stream`` is a cudaStream handle; warp-metal has one
        implicit stream per device, so this is a stable ``Stream`` object —
        the key :attr:`captures` is keyed by and the object
        :func:`get_stream` returns.
        """
        return self._stream

    @property
    def is_capturing(self) -> bool:
        """A boolean indicating whether this device is currently capturing a
        graph.

        Wired to the host runtime's capture state (single source of truth):
        True only while the capture begun on this device is active, like
        upstream's per-device ``captures`` registration.
        """
        from . import host as _host

        return _host.active_capture(self) is not None

    @property
    def captures(self) -> dict:
        """View of the active captures keyed by stream (upstream
        ``Device.captures``).

        ``{self.stream: in-progress Graph}`` while the capture begun on this
        device is recording — the same Graph ``capture_end`` returns — and
        ``{}`` outside capture, so Newton guards observe
        ``captures.get(device.stream) is not None`` exactly as on CUDA.
        Read-only per call (upstream mutates the dict inside warp's own
        capture_begin/capture_end); warp-metal has a single implicit stream,
        so the view never holds more than one entry.
        """
        from . import host as _host

        graph = _host.active_capture(self)
        if graph is None:
            return {}
        return {self._stream: graph}

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        if isinstance(other, Device):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Device(name={self.name!r}, arch={self.arch})"

    def __enter__(self):
        self._prev = get_current_device()
        set_device(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        set_device(self._prev)
        return False

    def get_allocator(self, pinned: bool = False):
        """Host allocator for the vendored frontend array path (upstream
        ``Device.get_allocator``). Pinned allocation is a KEP-0001 Non-Goal
        (the runtime WarpArray rejects it too)."""
        if pinned:
            raise RuntimeError(
                "warp-metal does not support pinned allocation flags (KEP-0001 Non-Goal)"
            )
        return _DeviceAllocator()


def _query_metal_arch() -> int:
    """Best-effort GPU architecture id (0 = unknown)."""
    return 0  # native probe lands with the device registry hardening (M1.4+)


_CPU_DEVICE = Device("cpu", ordinal=0, arch=0)
_METAL_AVAILABLE = None  # lazily probed
_METAL_DEVICE: Device | None = None

_DEVICES: dict[str, Device] = {"cpu": _CPU_DEVICE}

_CURRENT_DEVICE: Device = _CPU_DEVICE


def is_metal_available() -> bool:
    global _METAL_AVAILABLE, _METAL_DEVICE
    if _METAL_AVAILABLE is None:
        try:
            from .. import native as _mm

            _mm.device_name()  # creates the system MTLDevice; raises if unavailable
            _METAL_AVAILABLE = True
            _METAL_DEVICE = Device("metal", ordinal=0, arch=_query_metal_arch())
            _DEVICES["metal"] = _METAL_DEVICE
        except Exception:
            _METAL_AVAILABLE = False
    return _METAL_AVAILABLE


def get_device(ident=None) -> Device:
    """Resolve a device identifier ("cpu"/"metal"/Device/None) to a Device."""
    if ident is None:
        return _CURRENT_DEVICE
    if isinstance(ident, Device):
        return ident
    if isinstance(ident, Stream):
        return ident.device
    if isinstance(ident, str):
        key = ident.lower()
        if key == "cpu":
            return _CPU_DEVICE
        if key in ("metal", "gpu"):
            if not is_metal_available():
                raise RuntimeError(
                    "Metal device is not available on this machine (warp-metal requires Apple Silicon + Metal)"
                )
            return _DEVICES["metal"]
    raise ValueError(f"Unknown device {ident!r} - supported devices are 'cpu' and 'metal'")


def get_current_device() -> Device:
    return _CURRENT_DEVICE


def set_device(ident) -> Device:
    global _CURRENT_DEVICE
    _CURRENT_DEVICE = get_device(ident)
    return _CURRENT_DEVICE


def get_preferred_device() -> Device:
    """Metal when available, else CPU (upstream prefers CUDA when present)."""
    try:
        return get_device("metal")
    except RuntimeError:
        return _CPU_DEVICE


def activate_preferred_device() -> None:
    """Make the preferred GPU current, like upstream ``wp.init``.

    Upstream warp selects ``cuda:0`` as the current device when CUDA is
    present; Newton code (builder finalize, solver internals) launches on
    the current device when no explicit device is passed, so the shim
    activation must mirror that selection. Metal unavailable -> stays CPU.
    """
    global _CURRENT_DEVICE
    try:
        _CURRENT_DEVICE = get_device("metal")
    except RuntimeError:
        pass


def is_cuda_available() -> bool:
    """Always False: warp-metal strips CUDA (KEP-0001 core goal)."""
    return False


def get_cuda_device_count() -> int:
    """Always 0: warp-metal strips CUDA (KEP-0001 core goal).

    Upstream callers (Newton's test runner) only branch on ``> 1``.
    """
    return 0


def get_cuda_devices() -> list[Device]:
    """Always empty: warp-metal strips CUDA (KEP-0001 core goal).

    Upstream helpers iterate this list to drive per-device sweeps and
    mempool cleanup; an empty list degrades both to no-ops.
    """
    return []


def get_devices() -> list[Device]:
    """All registered devices (cpu + metal), like upstream ``wp.get_devices``."""
    return list(_DEVICES.values())


def get_stream(device=None) -> Stream:
    """Return the (single, implicit) stream for a device.

    warp-metal has one implicit stream per device; this returns the same
    object as ``Device.stream`` (upstream likewise returns the device's
    current stream), so the ``captures`` key and ``wp.get_stream`` handles
    stay identical.
    """
    return get_device(device).stream


def set_stream(stream) -> Stream:
    """Accepted for API parity; the single implicit pipeline makes it a no-op."""
    if isinstance(stream, Device) or isinstance(stream, str):
        get_device(stream)  # validate
    return stream


def synchronize_device(device=None) -> None:
    """Wait for outstanding work on a device (upstream ``wp.synchronize_device``).

    Flushes the pending eager command pipeline (commit + watchdog-bounded
    wait); no-op when nothing is pending."""
    get_device(device)
    from . import host as _host

    _host.flush()


def synchronize_stream(stream_or_device=None) -> None:
    synchronize_device(stream_or_device if stream_or_device is not None else None)


def record_event(event=None, external=None) -> Event:
    """Record an event on the current device (host timestamp)."""
    if event is None:
        event = Event(get_current_device())
    event.record()
    return event


def get_event_elapsed_time(start_event: Event, end_event: Event, synchronize=True) -> float:
    """Wall-clock seconds between two recorded events."""
    if start_event._recorded is None or end_event._recorded is None:
        raise RuntimeError("cannot query elapsed time of unrecorded event")
    return max(0.0, end_event._recorded - start_event._recorded)


def is_mempool_enabled(device=None) -> bool:
    """No memory pool: MTLBuffer allocations are direct."""
    return False


def get_mempool_used_mem_current(device=None) -> int:
    return 0


def get_mempool_used_mem_high(device=None) -> int:
    return 0


def can_access_device(a, b) -> bool:
    """Unified memory: every device can access every other."""
    return True


class ScopedDevice:
    """Context manager that temporarily sets the current device."""

    def __init__(self, device):
        self.device = get_device(device) if device is not None else None
        self._prev = None

    def __enter__(self):
        self._prev = _CURRENT_DEVICE
        if self.device is not None:
            set_device(self.device)
        return self.device

    def __exit__(self, exc_type, exc, tb):
        set_device(self._prev)
        return False


class ScopedStream:
    """Context manager that temporarily sets the current stream (no-op)."""

    def __init__(self, stream):
        self.stream = stream
        self._prev = None

    def __enter__(self):
        self._prev = self.stream
        return self.stream

    def __exit__(self, exc_type, exc, tb):
        return False
