// Copyright 2026 Jaylin Lee
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at http://www.apache.org/licenses/LICENSE-2.0.
//
// warp-metal native runtime prototype (KEP-0001 M0.4).
// Original work. C API for ctypes binding, warp-native style (no pybind):
//   - MTLBuffer allocation with storageModeShared (unified memory, CPU-side
//     numpy views are plain pointers into the buffer)
//   - runtime MSL -> MTLLibrary compilation, PSO cache is Python-side
//   - eager dispatch (M2.6): ONE process-lifetime device + ONE command
//     queue; launches ENCODE into a persistent command buffer WITHOUT
//     waiting and execute at the next explicit flush (wmm_flush: commit +
//     watchdog-bounded wait; host.py calls it at every host-visible
//     read/sync boundary). Zero/fill and array-to-array copies CAN also
//     encode as COMPUTE sections on the SAME compute encoder
//     (wmm_compute_fill / wmm_compute_copy, M2.6 round 2b — the upstream
//     CUDA semantics: device-side memsets/memcpys on the stream, ordered
//     with the launches around them), so no host write is needed and
//     pending work stays unflushed across them — OPT-IN only since round
//     2c (WARP_METAL_EAGER_GPU_FILLS=1): the eager default is the
//     conveyor-safe round-1 ordered host write (host.py flushes, then
//     writes shared storage). Round-2b: the earlier blit
//     encodings (wmm_blit_fill / wmm_blit_copy) regressed the Newton
//     conveyor ~35x, while every compute-only eager command buffer
//     (round-1 launches, round-2b compute fills) measures healthy — but the
//     compute-fill eager route itself is unproven at Newton scale, so it is
//     OPT-IN (WARP_METAL_EAGER_GPU_FILLS=1) since round 2c and the blit
//     entries remain as eager-triage/measure paths, off every default route.
//     WARP_METAL_SYNC_EAGER=1 reverts eager launches to the pre-M2.6
//     per-dispatch commit+wait for triage (Python keeps zero_/fill_ on the
//     ordered host write there), and wmm_dispatch_sync /
//     wmm_dispatch_tiled_sync expose that old behavior as explicit entry
//     points (the WARP_METAL_BATCH=0 replay path).
//   - capture replay (M2.5) batches consecutive records into one command
//     buffer via wmm_dispatch_batch — kernels as compute sections,
//     eligible array copies / zero-fills as blit sections on the same CB

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static char _last_error[1024] = {0};

static void _set_error(NSString *msg) {
    strncpy(_last_error, msg.UTF8String ?: "unknown error", sizeof(_last_error) - 1);
}

const char *wmm_last_error(void) {
    return _last_error;
}

// One MTLDevice for the whole process (M2.6): MTLCreateSystemDefaultDevice
// returns the same underlying device but a fresh retain on every call, and
// the per-dispatch creation leaked one per launch. Lazily created, reused.
static id<MTLDevice> _device = nil;

static id<MTLDevice> _shared_device(void) {
    if (_device == nil) {
        _device = MTLCreateSystemDefaultDevice();
        if (!_device) {
            _set_error(@"no Metal device");
        }
    }
    return _device;
}

const char *wmm_device_name(void) {
    id<MTLDevice> dev = _shared_device();
    if (!dev) {
        return NULL;
    }
    return dev.name.UTF8String;  // device outlives process; name is static
}

// ---- persistent eager pipeline (M2.6) ----

// One command queue for the process; the uncommitted command buffer + its
// open compute encoder accumulate eager dispatches until the next flush.
// _pending_done carries the completion signal (handler attached before
// commit, per Metal requirement). _pending_kept pins every MTLBuffer the
// uncommitted work references (one explicit +1 each, raw handles boxed in
// NSValue) and _pending_dropped marks those Python already released while
// pending: flush teardown removes exactly those from the residency set
// (removing a buffer the GPU may still dereference would silently zero its
// reads on macOS 15+) and consumes their ownership, and drops only the pin
// for the rest. This is what lets host.py free its params/argbuf/arena
// temporaries and let arrays GC while a command buffer is still
// uncommitted — the buffers outlive it.
static id<MTLCommandQueue> _eager_queue = nil;
static id<MTLCommandBuffer> _pending_cb = nil;  // nil = nothing pending
// At most ONE encoder is open at a time (Metal requires sequential encoders
// on one command buffer); _pending_enc holds the compute one, _pending_blit
// the blit one (zero/fill/copy sections). Record order = encoder creation
// order = execution order, exactly like the batched replay path.
static id<MTLComputeCommandEncoder> _pending_enc = nil;
static id<MTLBlitCommandEncoder> _pending_blit = nil;
static dispatch_semaphore_t _pending_done = nil;
static uint32_t _pending_count = 0;
static NSMutableArray *_pending_kept = nil;    // NSValue-boxed void* handles
static NSMutableSet *_pending_dropped = nil;   // of those, Python-released
// Watchdog-abandoned flushes keep their buffer pins here forever (leak by
// design — the runaway command buffer still references them; freeing under
// it would be a use-after-free, and a flush timeout already means the GPU
// likely faulted and the process is headed down).
static NSMutableArray *_orphaned_buffers = nil;

// WARP_METAL_SYNC_EAGER=1: eager dispatches revert to the pre-M2.6
// per-dispatch commit+wait (triage escape hatch; "0" or unset = deferred).
static bool _sync_eager(void) {
    const char *s = getenv("WARP_METAL_SYNC_EAGER");
    return s && *s && strcmp(s, "0") != 0;
}

// ---- buffers (storageModeShared: one allocation, CPU+GPU same memory) ----

// Metal residency for address-referenced buffers: a buffer that never appears
// in an encoder's binding list but is dereferenced through a raw device
// address (the host-built mesh table today; BVH nodes from M2.1) is owed no
// residency by the driver, and reads through a non-resident address silently
// return zeros on Apple Silicon. Keep every allocation in one residency set
// and mark it used on every command buffer ([cb useResidencySet:], macOS
// 15+; older systems have no residency API — their shared-buffer contract
// predates driver paging, so address indirection is safe there).
static id<MTLResidencySet> _residency = nil;

static void _residency_ensure(id<MTLDevice> dev) {
    if (_residency != nil) return;
    if (![dev respondsToSelector:@selector(newResidencySetWithDescriptor:error:)]) return;
    MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
    desc.label = @"warp-metal allocations";
    NSError *err = nil;
    id<MTLResidencySet> rs = [dev newResidencySetWithDescriptor:desc error:&err];
    if (!rs) {
        _set_error([NSString stringWithFormat:@"residency set creation failed: %@",
                    err.localizedDescription ?: @"unknown error"]);
        return;
    }
    _residency = rs;
}

static void _residency_add(id<MTLDevice> dev, void *alloc_handle) {
    _residency_ensure(dev);
    if (_residency == nil) return;
    [_residency addAllocation:(__bridge id<MTLAllocation>)alloc_handle];
    [_residency commit];
}

static void _residency_remove(void *alloc_handle) {
    if (_residency == nil) return;
    [_residency removeAllocation:(__bridge id<MTLAllocation>)alloc_handle];
    [_residency commit];
}

// mark the set used on a command buffer (no-op pre-macOS 15)
static void _residency_use(id<MTLCommandBuffer> cb) {
    if (_residency != nil && [cb respondsToSelector:@selector(useResidencySet:)]) {
        [cb useResidencySet:_residency];
    }
}

void *wmm_buffer_create(uint64_t bytes) {
    id<MTLDevice> dev = _shared_device();
    if (!dev) {
        return NULL;
    }
    if (bytes == 0) bytes = 16;  // MTLBuffer requires nonzero length
    id<MTLBuffer> buf = [dev newBufferWithLength:bytes
                                         options:MTLResourceStorageModeShared];
    if (!buf) {
        _set_error(@"buffer allocation failed");
        return NULL;
    }
    _residency_add(dev, (__bridge void *)buf);
    return (__bridge_retained void *)buf;
}

void *wmm_buffer_ptr(void *handle) {
    if (!handle) return NULL;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)handle;
    return buf.contents;
}

uint64_t wmm_buffer_gpu_address(void *handle) {
    if (!handle) return 0;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)handle;
    return buf.gpuAddress;
}

uint64_t wmm_buffer_length(void *handle) {
    if (!handle) return 0;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)handle;
    return buf.length;
}

void wmm_buffer_release(void *handle) {
    if (!handle) return;
    if (_pending_kept != nil) {
        NSValue *boxed = [NSValue valueWithPointer:handle];
        if ([_pending_kept containsObject:boxed]) {
            // The uncommitted eager command buffer still references this buffer
            // (host.py releases its params/argbuf/arena temporaries right after
            // every dispatch, and arrays can GC mid-frame). Defer the residency
            // removal and the release to flush teardown — freeing or
            // un-residing now would pull storage the GPU has not executed yet.
            [_pending_dropped addObject:boxed];
            return;
        }
    }
    _residency_remove(handle);
    id<MTLBuffer> owned = (__bridge_transfer id<MTLBuffer>)handle;
    (void)owned;  // release
}

// ---- library compilation ----

void *wmm_library_create(const char *msl_source) {
    id<MTLDevice> dev = _shared_device();
    if (!dev) {
        return NULL;
    }
    NSString *src = [NSString stringWithUTF8String:msl_source];
    // KEP-0001 baseline: macOS 26+ / Metal 4 (fallback Metal 3). Pin the
    // language version so 64-bit atomics (atomic_ulong, Metal 3.1+) resolve
    // instead of silently missing overloads under the default (nil) version.
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    if (@available(macOS 26.0, *)) {
        opts.languageVersion = MTLLanguageVersion4_0;
    } else {
        opts.languageVersion = MTLLanguageVersion3_2;
    }
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        _set_error(err.localizedDescription ?: @"library compile failed");
        return NULL;
    }
    return (__bridge_retained void *)lib;
}

void wmm_library_release(void *handle) {
    if (!handle) return;
    id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)handle;
    (void)lib;
}

void *wmm_function_get(void *lib_handle, const char *name) {
    if (!lib_handle) return NULL;
    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)lib_handle;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) {
        _set_error([NSString stringWithFormat:@"function '%s' not found in library", name]);
        return NULL;
    }
    return (__bridge_retained void *)fn;
}

void wmm_function_release(void *handle) {
    if (!handle) return;
    id<MTLFunction> fn = (__bridge_transfer id<MTLFunction>)handle;
    (void)fn;
}

void *wmm_pso_create(void *fn_handle) {
    id<MTLDevice> dev = _shared_device();
    id<MTLFunction> fn = (__bridge id<MTLFunction>)fn_handle;
    if (!dev || !fn) {
        _set_error(@"pso create: missing device or function");
        return NULL;
    }
    NSError *err = nil;
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn
                                                                        error:&err];
    if (!pso) {
        _set_error(err.localizedDescription ?: @"pso create failed");
        return NULL;
    }
    return (__bridge_retained void *)pso;
}

void wmm_pso_release(void *handle) {
    if (!handle) return;
    id<MTLComputePipelineState> pso = (__bridge_transfer id<MTLComputePipelineState>)handle;
    (void)pso;
}

uint64_t wmm_pso_max_threads_per_group(void *pso_handle) {
    if (!pso_handle) return 0;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pso_handle;
    return pso.maxTotalThreadsPerThreadgroup;
}

// ---- dispatch: deferred eager pipeline + explicit sync entries ----

// Watchdog window for a completion wait, in seconds, read at every flush so
// tests can tune it via the environment. Default 120 s per PENDING dispatch —
// a flush of n pending launches gets 120*n so each launch keeps its
// unbatched watchdog budget. WARP_METAL_DISPATCH_TIMEOUT_S overrides
// absolutely (0 disables the watchdog and restores the unbounded
// waitUntilCompleted). Since M2.6 the window bounds a whole FLUSH (the
// commit + completion wait of every pending eager dispatch), not a single
// dispatch. Without this bound a faulted dispatch (out-of-bounds device
// access etc.) hangs the host forever inside the wait — a fast, named
// failure is far more actionable.
static double _dispatch_timeout_seconds(uint32_t n_records) {
    const char *s = getenv("WARP_METAL_DISPATCH_TIMEOUT_S");
    if (s && *s) {
        char *end = NULL;
        double v = strtod(s, &end);
        if (end != s && v >= 0.0) return v;
    }
    return 120.0 * n_records;
}

// Encode one dispatch section: pipeline state, buffer bindings, threadgroup
// sizing, grid. Shared by every dispatch entry point (the deferred eager
// pipeline and the batched replay path loop this per record onto one
// encoder instead of duplicating the binding code). Does NOT end the
// encoding — the caller owns the encoder lifecycle and ends it once, after
// its last section. tg_x == 0 → threadgroup size auto (PSO max, clamped to
// the grid); tg_x > 0 → exact threadgroup size (tiled launches and
// plain-launch block_dim scheduling hints), validated against the PSO
// limit by the caller.
static void _encode_section(id<MTLComputeCommandEncoder> enc,
                            id<MTLComputePipelineState> pso,
                            void **buffer_handles, uint32_t n_buffers,
                            uint32_t threads_x, uint32_t tg_x) {
    [enc setComputePipelineState:pso];

    for (uint32_t i = 0; i < n_buffers; i++) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer_handles[i];
        [enc setBuffer:buf offset:0 atIndex:i];
    }

    NSUInteger tg = tg_x;
    if (tg == 0) {
        // 1-D dispatch; threadgroup size clamped to PSO limit and grid
        tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg > threads_x) tg = threads_x;
    }
    MTLSize grid = MTLSizeMake(threads_x, 1, 1);
    MTLSize tgs = MTLSizeMake(tg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
}

// threadgroup validation shared by the sync and deferred entries (the
// error text is byte-identical across both so triage output does not
// change shape between modes)
static int _check_threadgroup(id<MTLComputePipelineState> pso, uint32_t tg_x,
                              bool require_tg, const char *api) {
    if (tg_x > pso.maxTotalThreadsPerThreadgroup || (require_tg && tg_x == 0)) {
        _set_error([NSString stringWithFormat:
                    @"%s: block_dim %u exceeds the PSO threadgroup limit %lu%s",
                    api, tg_x, (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                    require_tg ? "" : " (threadgroup sizes above the limit are invalid)"]);
        return -3;
    }
    return 0;
}

// Open a pending-pipeline cycle: one command buffer + a fresh completion
// semaphore (fresh per cycle — a stale signal from an abandoned CB must
// never satisfy a later flush's wait). Encoders are opened on demand by
// _pending_to_compute / _pending_to_blit, so a cycle can hold any mix of
// compute and blit sections in record order.
static int _defer_begin(id<MTLDevice> dev) {
    if (_eager_queue == nil) {
        _eager_queue = [dev newCommandQueue];
        if (!_eager_queue) {
            _set_error(@"eager pipeline: command queue creation failed");
            return -1;
        }
    }
    _pending_cb = [_eager_queue commandBuffer];
    if (!_pending_cb) {
        _set_error(@"eager pipeline: command buffer creation failed");
        return -1;
    }
    // fresh semaphore per cycle (a stale signal from an abandoned command
    // buffer must never satisfy a later flush's wait); the completion block
    // captures it by value, so it outlives the flush teardown on the
    // watchdog-abandon path too
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    _pending_done = done;
    [_pending_cb addCompletedHandler:^(id<MTLCommandBuffer> _completed) {
        dispatch_semaphore_signal(done);
    }];
    _residency_use(_pending_cb);
    _pending_enc = nil;
    _pending_blit = nil;
    _pending_kept = [NSMutableArray array];
    _pending_dropped = [NSMutableSet set];
    _pending_count = 0;
    return 0;
}

// Ensure the open encoder is (or becomes) the compute one, ending an open
// blit encoder first (sequential-encoder rule; ended sections still run in
// creation order).
static int _pending_to_compute(void) {
    if (_pending_enc == nil) {
        if (_pending_blit != nil) {
            [_pending_blit endEncoding];
            _pending_blit = nil;
        }
        _pending_enc = [_pending_cb computeCommandEncoder];
        if (!_pending_enc) {
            _set_error(@"eager pipeline: compute encoder creation failed");
            return -1;
        }
    }
    return 0;
}

// Ensure the open encoder is (or becomes) the blit one, ending an open
// compute encoder first.
static int _pending_to_blit(void) {
    if (_pending_blit == nil) {
        if (_pending_enc != nil) {
            [_pending_enc endEncoding];
            _pending_enc = nil;
        }
        _pending_blit = [_pending_cb blitCommandEncoder];
        if (!_pending_blit) {
            _set_error(@"eager pipeline: blit encoder creation failed");
            return -1;
        }
    }
    return 0;
}

// Flush the pending eager pipeline: end the open encoder (compute or blit),
// commit the command buffer, and wait bounded by the watchdog (window = the
// whole flush window over the pending record count — launches and blits
// alike). No-op when nothing is pending. Failures name how many records
// were pending (batch attribution — per-record attribution inside one
// command buffer is impossible, the SYNC_EAGER escape hatch replaces it).
static int _flush_internal(void) {
    if (_pending_cb == nil) return 0;
    id<MTLCommandBuffer> cb = _pending_cb;
    id<MTLComputeCommandEncoder> enc = _pending_enc;
    id<MTLBlitCommandEncoder> blit = _pending_blit;
    dispatch_semaphore_t done = _pending_done;
    uint32_t n = _pending_count;
    NSMutableArray *kept = _pending_kept;
    NSMutableSet *dropped = _pending_dropped;

    if (enc) [enc endEncoding];
    if (blit) [blit endEncoding];
    double timeout = _dispatch_timeout_seconds(n);
    [cb commit];

    int rc = 0;
    if (timeout > 0.0) {
        long waited = dispatch_semaphore_wait(
            done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout * NSEC_PER_SEC)));
        if (waited != 0) {
            // Deadline hit while the flush is still in flight — it keeps
            // running in the background, but the host fails loud instead of
            // hanging. Buffer pins move to the orphan list (leaked, not
            // freed under the runaway command buffer).
            _set_error([NSString stringWithFormat:
                        @"flush: %u pending record(s) (launches + fills/copies, indices 0..%u) did "
                        "not complete within %.1f s (watchdog; WARP_METAL_DISPATCH_TIMEOUT_S "
                        "bounds a whole flush window — set it to tune, 0 disables). Command-"
                        "buffer status %ld, no error reported yet — the GPU likely faulted "
                        "(out-of-bounds device access) or a kernel never terminates. The "
                        "in-flight command buffer keeps running; its buffer pins are "
                        "retained to avoid use-after-free.",
                        n, n - 1, timeout, (long)cb.status]);
            rc = -4;
        }
    } else {
        // watchdog disabled: the documented unbounded waitUntilCompleted
        // (without a wait, the status check below would race the in-flight
        // command buffer and report "command buffer failed" for healthy work)
        [cb waitUntilCompleted];
    }
    if (rc == 0 && (cb.status != MTLCommandBufferStatusCompleted || cb.error)) {
        _set_error([NSString stringWithFormat:
                    @"flush: batch of %u pending record(s) (launches + fills/copies, indices "
                    "0..%u) failed: %@ (command-buffer status %ld; per-record attribution "
                    "inside one command buffer is impossible — set WARP_METAL_SYNC_EAGER=1 "
                    "to run eager dispatches one-by-one and isolate the faulting launch)",
                    n, n - 1,
                    cb.error.localizedDescription ?: @"no error reported", (long)cb.status]);
        rc = -2;
    }

    // teardown the cycle
    if (rc == -4) {
        if (_orphaned_buffers == nil) _orphaned_buffers = [NSMutableArray array];
        [_orphaned_buffers addObjectsFromArray:kept];  // transfers the pins
    } else {
        for (NSValue *boxed in kept) {
            void *h = boxed.pointerValue;
            // A handle can appear in kept MORE THAN ONCE (bound by several
            // sections of this cycle: e.g. a same-buffer blit copy pins it as
            // both src and dst, an in-place launch binds it as input and
            // output). Python's deferred ownership drop is consumed EXACTLY
            // ONCE — membership in dropped, not per kept entry — otherwise
            // the extra CFRelease underflows the refcount and traps in
            // CFRelease at flush teardown.
            if ([dropped containsObject:boxed]) {
                // Python released it while pending: residency removal was
                // deferred with the release — do both now that the GPU is
                // done (residency removal first, then both ownership drops)
                _residency_remove(h);
                CFRelease(h);  // consume Python's deferred ownership
                [dropped removeObject:boxed];
            }
            CFRelease(h);  // drop one pipeline pin
        }
    }
    _pending_cb = nil;
    _pending_enc = nil;
    _pending_blit = nil;
    _pending_done = nil;
    _pending_kept = nil;
    _pending_dropped = nil;
    _pending_count = 0;
    return rc;
}

int wmm_flush(void) {
    return _flush_internal();
}

// Encode one eager dispatch into the persistent command buffer WITHOUT
// committing (M2.6 eager path): consecutive launches share one command
// queue, one command buffer and one compute encoder (Metal runs a command
// buffer's encoders sequentially, so record order is execution order,
// exactly like the batched replay path). Execution happens at the next
// flush. Every bound buffer is pinned (+1) in the pending state so a
// Python-side release cannot free storage the uncommitted work references.
static int _dispatch_defer(void *pso_handle, void **buffer_handles, uint32_t n_buffers,
                           uint32_t threads_x, uint32_t tg_x, const char *api, bool require_tg) {
    id<MTLDevice> dev = _shared_device();
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pso_handle;
    if (!dev || !pso) {
        _set_error([NSString stringWithFormat:@"%s: missing device or pso", api]);
        return -1;
    }
    int vrc = _check_threadgroup(pso, tg_x, require_tg, api);
    if (vrc != 0) return vrc;
    if (_pending_cb == nil && _defer_begin(dev) != 0) return -1;
    if (_pending_to_compute() != 0) return -1;
    _encode_section(_pending_enc, pso, buffer_handles, n_buffers, threads_x, tg_x);
    for (uint32_t i = 0; i < n_buffers; i++) {
        void *h = buffer_handles[i];
        // +1 for the pending pipeline (released at flush teardown, or leaked
        // with the orphan list on a watchdog-abandoned flush)
        CFRetain(h);
        [_pending_kept addObject:[NSValue valueWithPointer:h]];
    }
    _pending_count++;
    return 0;
}

// ---- eager blit sections (M2.6 round 2): GPU-encoded zero/fill + array ----
// ---- copies. TRIAGE/MEASURE ONLY since round 2b: the knob-gated eager ----
// ---- path (WARP_METAL_EAGER_GPU_FILLS=1) encodes fills/copies as ----
// ---- compute sections (see below), and the blit entries remain so the ----
// ---- microbench can keep measuring the blit mechanism against it       ----
// ---- (tests/test_deferred_micro.py configs B/B2/E)                     ----

// Encode one eager fillBuffer section into the pending command buffer
// WITHOUT committing (the deferred counterpart of the replay path's fill
// records). Upstream zero_/fill_ are device memsets on the stream — the
// blit section sits between the launches around it in record order, so a
// later launch observes the fill without any host flush. The Python side
// has already validated the value's 8-bit pattern (fillBuffer replicates
// one byte); a zero-size fill encodes nothing. OFF THE EAGER DEFAULT
// PATH since round 2b (compute_fill — knob-gated — replaces it — see
// wmm_compute_fill); kept as an eager-triage/measure entry point.
int wmm_blit_fill(void *dst_handle, uint64_t offset_bytes, uint64_t nbytes, uint8_t value) {
    if (nbytes == 0) return 0;  // zero-size fill: nothing to encode
    id<MTLDevice> dev = _shared_device();
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dst_handle;
    if (!dev || !dst) {
        _set_error(@"blit_fill: missing device or buffer");
        return -1;
    }
    if (offset_bytes > dst.length || nbytes > dst.length - offset_bytes) {
        _set_error([NSString stringWithFormat:
                    @"blit_fill: range (%llu bytes at offset %llu) exceeds the buffer "
                    "length (%llu)",
                    (unsigned long long)nbytes, (unsigned long long)offset_bytes,
                    (unsigned long long)dst.length]);
        return -3;
    }
    if (_pending_cb == nil && _defer_begin(dev) != 0) return -1;
    if (_pending_to_blit() != 0) return -1;
    [_pending_blit fillBuffer:dst
                        range:NSMakeRange((NSUInteger)offset_bytes, (NSUInteger)nbytes)
                        value:value];
    // +1 for the pending pipeline (same pin discipline as compute dispatches)
    CFRetain(dst_handle);
    [_pending_kept addObject:[NSValue valueWithPointer:dst_handle]];
    _pending_count++;
    return 0;
}

// Encode one eager copyFromBuffer section into the pending command buffer
// WITHOUT committing (the deferred counterpart of the replay path's copy
// records): upstream wp.copy array→array is a stream memcpy. Byte
// offsets/sizes, eligibility (equal dtype/element size, same device, no
// same-buffer overlap) is the Python caller's — re-checked here fail-loud
// so an eligibility bug cannot corrupt via undefined blit aliasing.
// OFF THE EAGER DEFAULT PATH since round 2b (compute_copy — knob-gated —
// replaces it — see wmm_compute_copy); kept as an eager-triage/measure
// entry point.
int wmm_blit_copy(void *src_handle, void *dst_handle, uint64_t src_off_bytes,
                  uint64_t dst_off_bytes, uint64_t nbytes) {
    if (nbytes == 0) return 0;  // zero-size copy: nothing to encode
    id<MTLDevice> dev = _shared_device();
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)src_handle;
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dst_handle;
    if (!dev || !src || !dst) {
        _set_error(@"blit_copy: missing device or buffer");
        return -1;
    }
    if (dst_off_bytes > dst.length || nbytes > dst.length - dst_off_bytes) {
        _set_error([NSString stringWithFormat:
                    @"blit_copy: destination range (%llu bytes at offset %llu) exceeds the "
                    "buffer length (%llu)",
                    (unsigned long long)nbytes, (unsigned long long)dst_off_bytes,
                    (unsigned long long)dst.length]);
        return -3;
    }
    if (src_off_bytes > src.length || nbytes > src.length - src_off_bytes) {
        _set_error([NSString stringWithFormat:
                    @"blit_copy: source range (%llu bytes at offset %llu) exceeds the "
                    "buffer length (%llu)",
                    (unsigned long long)nbytes, (unsigned long long)src_off_bytes,
                    (unsigned long long)src.length]);
        return -3;
    }
    if (src == dst && src_off_bytes < dst_off_bytes + nbytes &&
        dst_off_bytes < src_off_bytes + nbytes) {
        // overlapping ranges in one buffer have undefined blit semantics —
        // the Python caller segments these to the host copy; reaching here
        // means that check was bypassed
        _set_error([NSString stringWithFormat:
                    @"blit_copy: overlapping src/dst ranges (%llu..%llu vs %llu..%llu) in "
                    "one buffer — must run on the host path",
                    (unsigned long long)src_off_bytes,
                    (unsigned long long)(src_off_bytes + nbytes),
                    (unsigned long long)dst_off_bytes,
                    (unsigned long long)(dst_off_bytes + nbytes)]);
        return -3;
    }
    if (_pending_cb == nil && _defer_begin(dev) != 0) return -1;
    if (_pending_to_blit() != 0) return -1;
    [_pending_blit copyFromBuffer:src
                     sourceOffset:(NSUInteger)src_off_bytes
                         toBuffer:dst
                destinationOffset:(NSUInteger)dst_off_bytes
                             size:(NSUInteger)nbytes];
    CFRetain(src_handle);
    [_pending_kept addObject:[NSValue valueWithPointer:src_handle]];
    CFRetain(dst_handle);
    [_pending_kept addObject:[NSValue valueWithPointer:dst_handle]];
    _pending_count++;
    return 0;
}

// ---- eager compute-section fills/copies (M2.6 round 2b): the knob-gated ----
// ---- eager zero_/fill_/wp.copy GPU encoding (round-2c default: host write) ----
//
// The eager pipeline stays COMPUTE-ONLY: a fill/copy is a tiny compute
// section on the SAME compute encoder as the launches around it — no
// encoder transition, no blit/DMA engine involvement, same in-stream
// ordering as the blit encoding. Evidence (round-2b): the Newton conveyor
// regressed ~35x (1287.7 vs 36.9 ms/frame) exactly when round 2 moved its
// eager fills/copies into blit sections, while the round-2b microbench
// (tests/test_deferred_micro.py) shows the isolated eager pipeline is
// healthy in every configuration — interleaved blit fills cost ~15 us per
// compute<->blit encoder transition at micro scale, and every
// compute-only eager command buffer (round-1 launches, round-2b compute
// fills) measures fast end to end. Whatever amplifies the conveyor's
// per-substep command buffer under mixed encoders was never reproduced at
// micro scale, so since round 2c this compute-only route is OPT-IN
// (WARP_METAL_EAGER_GPU_FILLS=1) and the eager DEFAULT is the round-1
// ordered host write. The graph-replay batch path keeps its blit
// sections (measured fine in shapes/pendulum) — untouched.
//
// Kernels: byte-granularity with host-selected aligned fast paths (4-byte
// fill / 16-byte copy per thread) carried in a per-op setBytes args block
// — no per-op buffer allocation, the bytes ride inside the command
// stream. MSL builtin scalar/vector names only, no fp64. PSOs compile
// lazily once per process (first fill/copy pays one library compile).

static const char *const _wmm_fill_msl =
    "struct wmm_fill_args_t {"
    "  unsigned int offset;"  // byte offset into dst
    "  unsigned int count;"   // thread work-item count (units depend on mode)
    "  unsigned int mode;"    // 0 = one byte per thread, 1 = one uint per thread
    "  unsigned int value;"   // mode 0: replicated byte; mode 1: byte x4 word
    "};"
    "kernel void wmm_fill(device unsigned char *dst [[buffer(0)]],"
    "                     constant wmm_fill_args_t &args [[buffer(1)]],"
    "                     unsigned int gid [[thread_position_in_grid]]) {"
    "  if (gid >= args.count) { return; }"
    "  if (args.mode == 1u) {"
    "    device unsigned int *d = (device unsigned int *)(dst + args.offset);"
    "    d[gid] = args.value;"
    "    return;"
    "  }"
    "  dst[args.offset + gid] = (unsigned char)args.value;"
    "}";

static const char *const _wmm_copy_msl =
    "struct wmm_copy_args_t {"
    "  unsigned int src_off;"  // byte offset into src
    "  unsigned int dst_off;"  // byte offset into dst
    "  unsigned int count;"    // thread work-item count (units depend on mode)
    "  unsigned int mode;"     // 0 = byte, 1 = one uint (4B), 2 = four uints (16B) per thread
    "};"
    "kernel void wmm_copy(device unsigned char *src [[buffer(0)]],"
    "                     device unsigned char *dst [[buffer(1)]],"
    "                     constant wmm_copy_args_t &args [[buffer(2)]],"
    "                     unsigned int gid [[thread_position_in_grid]]) {"
    "  if (gid >= args.count) { return; }"
    "  if (args.mode == 2u) {"
    "    device unsigned int *s = (device unsigned int *)(src + args.src_off);"
    "    device unsigned int *d = (device unsigned int *)(dst + args.dst_off);"
    "    d[gid * 4u + 0u] = s[gid * 4u + 0u];"
    "    d[gid * 4u + 1u] = s[gid * 4u + 1u];"
    "    d[gid * 4u + 2u] = s[gid * 4u + 2u];"
    "    d[gid * 4u + 3u] = s[gid * 4u + 3u];"
    "    return;"
    "  }"
    "  if (args.mode == 1u) {"
    "    device unsigned int *s = (device unsigned int *)(src + args.src_off);"
    "    device unsigned int *d = (device unsigned int *)(dst + args.dst_off);"
    "    d[gid] = s[gid];"
    "    return;"
    "  }"
    "  dst[args.dst_off + gid] = src[args.src_off + gid];"
    "}";

static id<MTLLibrary> _fill_copy_lib = nil;
static id<MTLFunction> _fill_fn = nil;
static id<MTLFunction> _copy_fn = nil;
static id<MTLComputePipelineState> _fill_pso = nil;
static id<MTLComputePipelineState> _copy_pso = nil;

// Compile one MSL source with the same pinned language version as
// wmm_library_create (Metal 4 on macOS 26+, else Metal 3.2 — 64-bit atomics
// resolve; irrelevant for these kernels but kept uniform). Returns nil with
// _last_error set on failure.
static id<MTLLibrary> _compile_source(id<MTLDevice> dev, NSString *src) {
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    if (@available(macOS 26.0, *)) {
        opts.languageVersion = MTLLanguageVersion4_0;
    } else {
        opts.languageVersion = MTLLanguageVersion3_2;
    }
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        _set_error(err.localizedDescription ?: @"library compile failed");
        return nil;
    }
    return lib;
}

static int _fill_copy_pso_ensure(id<MTLDevice> dev) {
    if (_fill_pso != nil && _copy_pso != nil) return 0;
    if (_fill_copy_lib == nil) {
        _fill_copy_lib = _compile_source(
            dev, [NSString stringWithFormat:@"%s\n%s", _wmm_fill_msl, _wmm_copy_msl]);
        if (!_fill_copy_lib) return -1;
    }
    if (_fill_fn == nil) {
        _fill_fn = [_fill_copy_lib newFunctionWithName:@"wmm_fill"];
        if (!_fill_fn) {
            _set_error(@"compute_fill: kernel 'wmm_fill' not found in library");
            return -1;
        }
    }
    if (_copy_fn == nil) {
        _copy_fn = [_fill_copy_lib newFunctionWithName:@"wmm_copy"];
        if (!_copy_fn) {
            _set_error(@"compute_copy: kernel 'wmm_copy' not found in library");
            return -1;
        }
    }
    if (_fill_pso == nil) {
        NSError *err = nil;
        _fill_pso = [dev newComputePipelineStateWithFunction:_fill_fn error:&err];
        if (!_fill_pso) {
            _set_error(err.localizedDescription ?: @"compute_fill: pso create failed");
            return -1;
        }
    }
    if (_copy_pso == nil) {
        NSError *err = nil;
        _copy_pso = [dev newComputePipelineStateWithFunction:_copy_fn error:&err];
        if (!_copy_pso) {
            _set_error(err.localizedDescription ?: @"compute_copy: pso create failed");
            return -1;
        }
    }
    return 0;
}

// Encode one eager fill as a COMPUTE section into the pending command
// buffer WITHOUT committing (the eager zero_/fill_ GPU encoding, round 2b,
// opt-in via WARP_METAL_EAGER_GPU_FILLS=1 since round 2c): same stream
// semantics as the blit fill it replaces — the
// section orders with the launches around it, no host write, no flush —
// but it rides the SAME compute encoder as those launches, so the eager
// command buffer never mixes encoders. One thread per replicated byte
// (per 4-byte word on the aligned fast path); the value rides the command
// stream via setBytes. DEFERRED MODE ONLY: the Python caller keeps
// SYNC_EAGER on the ordered host write.
int wmm_compute_fill(void *dst_handle, uint64_t offset_bytes, uint64_t nbytes, uint8_t value) {
    if (nbytes == 0) return 0;  // zero-size fill: nothing to encode
    id<MTLDevice> dev = _shared_device();
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dst_handle;
    if (!dev || !dst) {
        _set_error(@"compute_fill: missing device or buffer");
        return -1;
    }
    if (offset_bytes > dst.length || nbytes > dst.length - offset_bytes) {
        _set_error([NSString stringWithFormat:
                    @"compute_fill: range (%llu bytes at offset %llu) exceeds the buffer "
                    "length (%llu)",
                    (unsigned long long)nbytes, (unsigned long long)offset_bytes,
                    (unsigned long long)dst.length]);
        return -3;
    }
    if (_fill_copy_pso_ensure(dev) != 0) return -1;
    if (_pending_cb == nil && _defer_begin(dev) != 0) return -1;
    if (_pending_to_compute() != 0) return -1;

    // aligned 4-byte fast path; byte path otherwise (element strides and
    // struct sizes keep Newton's fills 4-aligned in practice)
    unsigned int mode = 0;
    uint64_t count = nbytes;
    unsigned int value32 = (unsigned int)value;
    if ((offset_bytes & 3u) == 0 && (nbytes & 3u) == 0) {
        mode = 1;
        count = nbytes / 4;
        value32 = 0x01010101u * (unsigned int)value;
    }
    if (count > 0xFFFFFFFFull) {
        _set_error([NSString stringWithFormat:
                    @"compute_fill: %llu-byte range exceeds one dispatch (%llu work "
                    "items) — split it",
                    (unsigned long long)nbytes, (unsigned long long)count]);
        return -3;
    }
    struct {
        unsigned int offset;
        unsigned int count;
        unsigned int mode;
        unsigned int value;
    } args = {(unsigned int)offset_bytes, (unsigned int)count, mode, value32};

    [_pending_enc setComputePipelineState:_fill_pso];
    [_pending_enc setBuffer:dst offset:0 atIndex:0];
    [_pending_enc setBytes:&args length:sizeof(args) atIndex:1];
    NSUInteger tg = _fill_pso.maxTotalThreadsPerThreadgroup;
    if (tg > count) tg = (NSUInteger)count;
    [_pending_enc dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    // +1 for the pending pipeline (same pin discipline as compute dispatches)
    CFRetain(dst_handle);
    [_pending_kept addObject:[NSValue valueWithPointer:dst_handle]];
    _pending_count++;
    return 0;
}

// Encode one eager array→array copy as a COMPUTE section into the pending
// command buffer WITHOUT committing (the eager wp.copy/clone GPU encoding,
// round 2b, opt-in via WARP_METAL_EAGER_GPU_FILLS=1 since round 2c): same
// stream semantics and eligibility matrix as the
// blit copy it replaces (equal dtype/element size, same device, no
// same-buffer overlap — checked by the Python caller, re-checked here
// fail-loud). One thread per copied byte (per 16-byte uchar16 on the
// aligned fast path). DEFERRED MODE ONLY (see wmm_compute_fill).
int wmm_compute_copy(void *src_handle, void *dst_handle, uint64_t src_off_bytes,
                     uint64_t dst_off_bytes, uint64_t nbytes) {
    if (nbytes == 0) return 0;  // zero-size copy: nothing to encode
    id<MTLDevice> dev = _shared_device();
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)src_handle;
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dst_handle;
    if (!dev || !src || !dst) {
        _set_error(@"compute_copy: missing device or buffer");
        return -1;
    }
    if (dst_off_bytes > dst.length || nbytes > dst.length - dst_off_bytes) {
        _set_error([NSString stringWithFormat:
                    @"compute_copy: destination range (%llu bytes at offset %llu) exceeds "
                    "the buffer length (%llu)",
                    (unsigned long long)nbytes, (unsigned long long)dst_off_bytes,
                    (unsigned long long)dst.length]);
        return -3;
    }
    if (src_off_bytes > src.length || nbytes > src.length - src_off_bytes) {
        _set_error([NSString stringWithFormat:
                    @"compute_copy: source range (%llu bytes at offset %llu) exceeds the "
                    "buffer length (%llu)",
                    (unsigned long long)nbytes, (unsigned long long)src_off_bytes,
                    (unsigned long long)src.length]);
        return -3;
    }
    if (src == dst && src_off_bytes < dst_off_bytes + nbytes &&
        dst_off_bytes < src_off_bytes + nbytes) {
        // overlapping ranges in one buffer are undefined for a byte-thread
        // copy too (read/write ordering per byte is arbitrary) — the Python
        // caller segments these to the host copy; reaching here means that
        // check was bypassed
        _set_error([NSString stringWithFormat:
                    @"compute_copy: overlapping src/dst ranges (%llu..%llu vs %llu..%llu) "
                    "in one buffer — must run on the host path",
                    (unsigned long long)src_off_bytes,
                    (unsigned long long)(src_off_bytes + nbytes),
                    (unsigned long long)dst_off_bytes,
                    (unsigned long long)(dst_off_bytes + nbytes)]);
        return -3;
    }
    if (_fill_copy_pso_ensure(dev) != 0) return -1;
    if (_pending_cb == nil && _defer_begin(dev) != 0) return -1;
    if (_pending_to_compute() != 0) return -1;

    // aligned 16-byte fast path (4 uint words per thread); aligned 4-byte
    // path; byte path otherwise
    unsigned int mode = 0;
    uint64_t count = nbytes;
    if ((src_off_bytes & 15u) == 0 && (dst_off_bytes & 15u) == 0 && (nbytes & 15u) == 0) {
        mode = 2;
        count = nbytes / 16;
    } else if ((src_off_bytes & 3u) == 0 && (dst_off_bytes & 3u) == 0 && (nbytes & 3u) == 0) {
        mode = 1;
        count = nbytes / 4;
    }
    if (count > 0xFFFFFFFFull) {
        _set_error([NSString stringWithFormat:
                    @"compute_copy: %llu-byte range exceeds one dispatch (%llu work "
                    "items) — split it",
                    (unsigned long long)nbytes, (unsigned long long)count]);
        return -3;
    }
    struct {
        unsigned int src_off;
        unsigned int dst_off;
        unsigned int count;
        unsigned int mode;
    } args = {(unsigned int)src_off_bytes, (unsigned int)dst_off_bytes,
              (unsigned int)count, mode};

    [_pending_enc setComputePipelineState:_copy_pso];
    [_pending_enc setBuffer:src offset:0 atIndex:0];
    [_pending_enc setBuffer:dst offset:0 atIndex:1];
    [_pending_enc setBytes:&args length:sizeof(args) atIndex:2];
    NSUInteger tg = _copy_pso.maxTotalThreadsPerThreadgroup;
    if (tg > count) tg = (NSUInteger)count;
    [_pending_enc dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    CFRetain(src_handle);
    [_pending_kept addObject:[NSValue valueWithPointer:src_handle]];
    CFRetain(dst_handle);
    [_pending_kept addObject:[NSValue valueWithPointer:dst_handle]];
    _pending_count++;
    return 0;
}

// Encode, commit, and wait bounded by the watchdog. The pre-M2.6 single
// dispatch behavior, kept as the explicit wmm_dispatch_sync /
// wmm_dispatch_tiled_sync entry points (WARP_METAL_BATCH=0 replay triage)
// and as WARP_METAL_SYNC_EAGER=1's eager mode. Pending eager work is
// flushed first so ordering never inverts across the two modes.
// require_tg selects the tiled entry point's strict validation (0 is an
// error there).
static int _dispatch_sync(void *pso_handle, void **buffer_handles, uint32_t n_buffers,
                          uint32_t threads_x, uint32_t tg_x, const char *api, bool require_tg) {
    (void)_flush_internal();  // pending deferred work commits + completes first

    id<MTLDevice> dev = _shared_device();
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pso_handle;
    if (!dev || !pso) {
        _set_error([NSString stringWithFormat:@"%s: missing device or pso", api]);
        return -1;
    }
    int vrc = _check_threadgroup(pso, tg_x, require_tg, api);
    if (vrc != 0) return vrc;

    id<MTLCommandQueue> queue = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [queue commandBuffer];

    // Completion handler must be attached BEFORE commit; it gives a
    // deadline-bounded wait without polling latency (waitUntilCompleted
    // has no timeout form).
    double timeout = _dispatch_timeout_seconds(1);
    dispatch_semaphore_t done = nil;
    if (timeout > 0.0) {
        done = dispatch_semaphore_create(0);
        [cb addCompletedHandler:^(id<MTLCommandBuffer> _completed) {
            dispatch_semaphore_signal(done);
        }];
    }

    _residency_use(cb);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    _encode_section(enc, pso, buffer_handles, n_buffers, threads_x, tg_x);
    [enc endEncoding];

    [cb commit];
    if (done != nil) {
        long waited = dispatch_semaphore_wait(
            done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout * NSEC_PER_SEC)));
        if (waited != 0) {
            // Deadline hit while the command buffer is still in flight — it
            // keeps running in the background (later dispatches use a fresh
            // command queue), but the host fails loud instead of hanging.
            _set_error([NSString stringWithFormat:
                        @"%s: dispatch did not complete within %.1f s "
                        "(watchdog; set WARP_METAL_DISPATCH_TIMEOUT_S to tune, 0 disables). "
                        "Command-buffer status %ld, no error reported yet — the GPU likely "
                        "faulted (out-of-bounds device access) or the kernel never terminates.",
                        api, timeout, (long)cb.status]);
            return -4;
        }
    } else {
        // watchdog disabled: the documented unbounded waitUntilCompleted
        // (without a wait, the status check below would race the in-flight
        // command buffer and report "command buffer failed" for healthy work)
        [cb waitUntilCompleted];
    }
    if (cb.status != MTLCommandBufferStatusCompleted || cb.error) {
        _set_error(cb.error.localizedDescription ?: @"command buffer failed");
        return -2;
    }
    return 0;
}

// eager launch (default): encode into the pending command buffer; SYNC_EAGER
// reverts to the per-dispatch commit+wait
int wmm_dispatch(void *pso_handle, void **buffer_handles, uint32_t n_buffers,
                 uint32_t threads_x, uint32_t tg_x) {
    if (_sync_eager()) {
        return _dispatch_sync(pso_handle, buffer_handles, n_buffers, threads_x, tg_x,
                              "dispatch", false);
    }
    return _dispatch_defer(pso_handle, buffer_handles, n_buffers, threads_x, tg_x,
                           "dispatch", false);
}

// tiled eager launch (wp.launch_tiled): the caller fixes the threadgroup
// size so the kernel's block/lane tid mapping and threadgroup reductions
// are exact; deferred like wmm_dispatch
int wmm_dispatch_tiled(void *pso_handle, void **buffer_handles, uint32_t n_buffers,
                       uint32_t threads_x, uint32_t tg_x) {
    if (_sync_eager()) {
        return _dispatch_sync(pso_handle, buffer_handles, n_buffers, threads_x, tg_x,
                              "dispatch_tiled", true);
    }
    return _dispatch_defer(pso_handle, buffer_handles, n_buffers, threads_x, tg_x,
                           "dispatch_tiled", true);
}

// explicit synchronous single dispatch (pre-M2.6 behavior; triage entry)
int wmm_dispatch_sync(void *pso_handle, void **buffer_handles, uint32_t n_buffers,
                      uint32_t threads_x, uint32_t tg_x) {
    return _dispatch_sync(pso_handle, buffer_handles, n_buffers, threads_x, tg_x,
                          "dispatch_sync", false);
}

int wmm_dispatch_tiled_sync(void *pso_handle, void **buffer_handles, uint32_t n_buffers,
                            uint32_t threads_x, uint32_t tg_x) {
    return _dispatch_sync(pso_handle, buffer_handles, n_buffers, threads_x, tg_x,
                          "dispatch_tiled_sync", true);
}

// One batch record (M2.5): a compute dispatch section or a blit op (buffer
// copy / zero-fill), executed in array order on one command buffer. Compute
// records carry the pipeline + buffer bindings (+ thread sizing); blit
// records carry raw buffer handles with BYTE offsets/sizes — the Python
// side converts element offsets to bytes (offset_elems included) and has
// already validated blit eligibility (same device, byte-identical element
// layout, no overlapping src/dst ranges), segmenting everything else back
// to the host replay path. The stride is queryable
// (wmm_batch_record_stride) so the ctypes mirror can fail loud on any ABI
// drift instead of misreading payload fields.
#define WMM_KIND_COMPUTE 0
#define WMM_KIND_COPY 1
#define WMM_KIND_FILL 2

typedef struct {
    void *pso;           // compute: pipeline state
    void **buffers;      // compute: flattened per-record buffer handles
    void *src;           // blit copy: source buffer
    void *dst;           // blit copy/fill: destination buffer
    uint64_t src_off;    // blit copy: source byte offset
    uint64_t dst_off;    // blit copy/fill: destination byte offset
    uint64_t nbytes;     // blit: byte count (0 = encode nothing)
    uint32_t n_buffers;  // compute: handle count
    uint32_t threads_x;  // compute: grid threads
    uint32_t tg_x;       // compute: threadgroup size (0 = auto)
    uint8_t require_tg;  // compute: exact threadgroup size required (tiled)
    uint8_t kind;        // WMM_KIND_*
    uint8_t value;       // blit fill: 8-bit pattern (eligible fills: 0)
} wmm_batch_record;

uint64_t wmm_batch_record_stride(void) {
    return sizeof(wmm_batch_record);
}

// Batched dispatch (M2.5 capture replay): one command queue and one command
// buffer for a whole run of recorded ops — kernel launches as compute
// sections, eligible array copies and zero-fills as blit sections
// (copyFromBuffer / fillBuffer on MTLBlitCommandEncoder). Records execute in
// array order; Metal runs the encoders of a single command buffer strictly
// sequentially, so record i observes record i-1's writes exactly like the
// per-dispatch path, and at most one encoder is open at any time (compute
// sections merge into one compute encoder, blits into one blit encoder;
// switching kinds ends the open encoder before the new one starts).
// Compute record sizes/threadgroup limits and blit handle/bounds/overlap
// are pre-flighted before the first encode, so those failures still name
// their record index. Once committed, per-record attribution inside one
// command buffer is impossible — failures report the batch size and record
// range and point at record-by-record replay (WARP_METAL_BATCH=0 on the
// Python side) for isolation. One watchdog wait bounds the whole batch;
// after it returns, shared-storage host reads are exact.
int wmm_dispatch_batch(const wmm_batch_record *records, uint32_t n_records) {
    if (n_records == 0) return 0;
    // the batch CB uses its own queue, so ordering against the pending eager
    // pipeline is not guaranteed cross-queue — drain it first (no-op when idle)
    (void)_flush_internal();
    id<MTLDevice> dev = _shared_device();
    if (!dev) {
        _set_error(@"dispatch_batch: no Metal device");
        return -1;
    }

    // pre-flight every record: a bad record fails here, attributed to its
    // index, before anything encodes or executes
    for (uint32_t r = 0; r < n_records; r++) {
        const wmm_batch_record *rec = &records[r];
        if (rec->kind == WMM_KIND_COMPUTE) {
            id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)rec->pso;
            if (!pso) {
                _set_error([NSString stringWithFormat:
                            @"dispatch_batch: record %u/%u: missing pso", r, n_records]);
                return -1;
            }
            if (rec->tg_x > pso.maxTotalThreadsPerThreadgroup ||
                (rec->require_tg && rec->tg_x == 0)) {
                _set_error([NSString stringWithFormat:
                            @"dispatch_batch: record %u/%u: block_dim %u exceeds the PSO "
                            @"threadgroup limit %lu%s",
                            r, n_records, rec->tg_x,
                            (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                            rec->require_tg ? "" : " (threadgroup sizes above the limit are invalid)"]);
                return -3;
            }
        } else if (rec->kind == WMM_KIND_COPY || rec->kind == WMM_KIND_FILL) {
            id<MTLBuffer> src = (__bridge id<MTLBuffer>)rec->src;
            id<MTLBuffer> dst = (__bridge id<MTLBuffer>)rec->dst;
            if (!dst || (rec->kind == WMM_KIND_COPY && !src)) {
                _set_error([NSString stringWithFormat:
                            @"dispatch_batch: record %u/%u: missing blit buffer", r, n_records]);
                return -1;
            }
            if (rec->dst_off + rec->nbytes > dst.length ||
                (rec->kind == WMM_KIND_COPY && rec->src_off + rec->nbytes > src.length)) {
                const BOOL src_over = rec->kind == WMM_KIND_COPY &&
                                      rec->src_off + rec->nbytes > src.length;
                _set_error([NSString stringWithFormat:
                            @"dispatch_batch: record %u/%u: blit %s range (%llu bytes at "
                            @"offset %llu) exceeds the %s buffer length (%llu)",
                            r, n_records, src_over ? "source" : "destination",
                            (unsigned long long)rec->nbytes,
                            (unsigned long long)(src_over ? rec->src_off : rec->dst_off),
                            src_over ? "source" : "destination",
                            (unsigned long long)(src_over ? src.length : dst.length)]);
                return -3;
            }
            if (rec->kind == WMM_KIND_COPY && src == dst &&
                rec->src_off < rec->dst_off + rec->nbytes &&
                rec->dst_off < rec->src_off + rec->nbytes) {
                // overlapping ranges in one buffer have undefined blit
                // semantics — the Python replay segments these to the host
                // copy; reaching here means that check was bypassed
                _set_error([NSString stringWithFormat:
                            @"dispatch_batch: record %u/%u: overlapping src/dst ranges "
                            "(%llu..%llu vs %llu..%llu) in one buffer — must replay on the "
                            "host path",
                            r, n_records, (unsigned long long)rec->src_off,
                            (unsigned long long)(rec->src_off + rec->nbytes),
                            (unsigned long long)rec->dst_off,
                            (unsigned long long)(rec->dst_off + rec->nbytes)]);
                return -3;
            }
        } else {
            _set_error([NSString stringWithFormat:
                        @"dispatch_batch: record %u/%u: unknown kind %u", r, n_records, rec->kind]);
            return -1;
        }
    }

    id<MTLCommandQueue> queue = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [queue commandBuffer];

    // Completion handler must be attached BEFORE commit; one completion
    // handler + watchdog for the whole batch (the default window scales with
    // the record count — see _dispatch_timeout_seconds).
    double timeout = _dispatch_timeout_seconds(n_records);
    dispatch_semaphore_t done = nil;
    if (timeout > 0.0) {
        done = dispatch_semaphore_create(0);
        [cb addCompletedHandler:^(id<MTLCommandBuffer> _completed) {
            dispatch_semaphore_signal(done);
        }];
    }

    _residency_use(cb);
    // Exactly one encoder open at a time (Metal requires sequential encoders
    // on one command buffer). Consecutive same-kind records share one
    // encoder; a kind switch ends the open one before the next starts.
    id<MTLComputeCommandEncoder> comp = nil;
    id<MTLBlitCommandEncoder> blit = nil;
    int open = 0;  // 0 none, 1 compute, 2 blit
    for (uint32_t r = 0; r < n_records; r++) {
        const wmm_batch_record *rec = &records[r];
        if (rec->kind == WMM_KIND_COMPUTE) {
            if (open != 1) {
                if (open == 2) [blit endEncoding];
                comp = [cb computeCommandEncoder];
                open = 1;
            }
            id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)rec->pso;
            _encode_section(comp, pso, rec->buffers, rec->n_buffers,
                            rec->threads_x, rec->tg_x);
        } else if (rec->nbytes > 0) {  // zero-size blit: encode nothing
            if (open != 2) {
                if (open == 1) [comp endEncoding];
                blit = [cb blitCommandEncoder];
                open = 2;
            }
            if (rec->kind == WMM_KIND_COPY) {
                [blit copyFromBuffer:(__bridge id<MTLBuffer>)rec->src
                       sourceOffset:(NSUInteger)rec->src_off
                           toBuffer:(__bridge id<MTLBuffer>)rec->dst
                  destinationOffset:(NSUInteger)rec->dst_off
                               size:(NSUInteger)rec->nbytes];
            } else {
                [blit fillBuffer:(__bridge id<MTLBuffer>)rec->dst
                           range:NSMakeRange((NSUInteger)rec->dst_off, (NSUInteger)rec->nbytes)
                           value:rec->value];
            }
        }
    }
    if (open == 1) [comp endEncoding];
    else if (open == 2) [blit endEncoding];

    [cb commit];
    if (done != nil) {
        long waited = dispatch_semaphore_wait(
            done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout * NSEC_PER_SEC)));
        if (waited != 0) {
            // Deadline hit while the batch is still in flight — it keeps
            // running in the background (later dispatches use a fresh
            // command queue), but the host fails loud instead of hanging.
            _set_error([NSString stringWithFormat:
                        @"dispatch_batch: %u record(s) (indices 0..%u) did not complete "
                        "within %.1f s (watchdog scales with the batch; set "
                        "WARP_METAL_DISPATCH_TIMEOUT_S to pin it, 0 disables). Command-buffer "
                        "status %ld, no error reported yet — the GPU likely faulted "
                        "(out-of-bounds device access) or a kernel never terminates. "
                        "Per-record attribution inside one command buffer is impossible; "
                        "replay with WARP_METAL_BATCH=0 to isolate the faulting record.",
                        n_records, n_records - 1, timeout, (long)cb.status]);
            return -4;
        }
    } else {
        // watchdog disabled: the documented unbounded waitUntilCompleted
        // (without a wait, the status check below would race the in-flight
        // command buffer and report "command buffer failed" for healthy work)
        [cb waitUntilCompleted];
    }
    if (cb.status != MTLCommandBufferStatusCompleted || cb.error) {
        _set_error([NSString stringWithFormat:
                    @"dispatch_batch: batch of %u record(s) (indices 0..%u) failed: %@ "
                    "(command-buffer status %ld; per-record attribution inside one command "
                    "buffer is impossible — replay with WARP_METAL_BATCH=0 to isolate the "
                    "faulting record)",
                    n_records, n_records - 1,
                    cb.error.localizedDescription ?: @"no error reported", (long)cb.status]);
        return -2;
    }
    return 0;
}

void wmm_synchronize(void) {
    // M2.6: flush the pending eager pipeline (commit + watchdog-bounded wait).
    // Void ABI — failures are recorded in wmm_last_error; the Python side
    // raises through wmm_flush (native.flush) instead.
    (void)_flush_internal();
}

#ifdef __cplusplus
}  // extern "C"
#endif
