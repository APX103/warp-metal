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
//   - single synchronous compute dispatch (batched command buffers land in
//     M2.5; every dispatch here commits and waits)

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

const char *wmm_device_name(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        _set_error(@"no Metal device");
        return NULL;
    }
    return dev.name.UTF8String;  // device outlives process; name is static
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
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        _set_error(@"no Metal device");
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
    _residency_remove(handle);
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)handle;
    (void)buf;  // release
}

// ---- library compilation ----

void *wmm_library_create(const char *msl_source) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        _set_error(@"no Metal device");
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
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
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

// ---- dispatch (synchronous single kernel; batched commit is M2.5) ----

// Watchdog window for the completion wait, in seconds, read on EVERY
// dispatch so tests can tune it via the environment. Default 120 s;
// WARP_METAL_DISPATCH_TIMEOUT_S overrides (0 disables the watchdog and
// restores the unbounded waitUntilCompleted). Without this bound a faulted
// dispatch (out-of-bounds device access etc.) hangs the host forever inside
// waitUntilCompleted — a fast, named failure is far more actionable.
static double _dispatch_timeout_seconds(void) {
    const char *s = getenv("WARP_METAL_DISPATCH_TIMEOUT_S");
    if (s && *s) {
        char *end = NULL;
        double v = strtod(s, &end);
        if (end != s && v >= 0.0) return v;
    }
    return 120.0;
}

// Encode, commit, and wait bounded by the watchdog. Shared by every dispatch
// entry point. tg_x == 0 → threadgroup size auto (PSO max, clamped to the
// grid); tg_x > 0 → exact threadgroup size (tiled launches and plain-launch
// block_dim scheduling hints), validated against the PSO limit. require_tg
// selects the tiled entry point's strict validation (0 is an error there).
static int _dispatch_sync(void *pso_handle, void **buffer_handles, uint32_t n_buffers,
                          uint32_t threads_x, uint32_t tg_x, const char *api, bool require_tg) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pso_handle;
    if (!dev || !pso) {
        _set_error([NSString stringWithFormat:@"%s: missing device or pso", api]);
        return -1;
    }
    if (tg_x > pso.maxTotalThreadsPerThreadgroup || (require_tg && tg_x == 0)) {
        _set_error([NSString stringWithFormat:
                    @"%s: block_dim %u exceeds the PSO threadgroup limit %lu%s",
                    api, tg_x, (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                    require_tg ? "" : " (threadgroup sizes above the limit are invalid)"]);
        return -3;
    }

    id<MTLCommandQueue> queue = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [queue commandBuffer];

    // Completion handler must be attached BEFORE commit; it gives a
    // deadline-bounded wait without polling latency (waitUntilCompleted
    // has no timeout form).
    double timeout = _dispatch_timeout_seconds();
    dispatch_semaphore_t done = nil;
    if (timeout > 0.0) {
        done = dispatch_semaphore_create(0);
        [cb addCompletedHandler:^(id<MTLCommandBuffer> _completed) {
            dispatch_semaphore_signal(done);
        }];
    }

    _residency_use(cb);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
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
    }
    if (cb.status != MTLCommandBufferStatusCompleted || cb.error) {
        _set_error(cb.error.localizedDescription ?: @"command buffer failed");
        return -2;
    }
    return 0;
}

int wmm_dispatch(void *pso_handle, void **buffer_handles, uint32_t n_buffers,
                 uint32_t threads_x, uint32_t tg_x) {
    return _dispatch_sync(pso_handle, buffer_handles, n_buffers, threads_x, tg_x,
                          "dispatch", false);
}

// tiled launch (wp.launch_tiled): the caller fixes the threadgroup size so
// the kernel's block/lane tid mapping and threadgroup reductions are exact
int wmm_dispatch_tiled(void *pso_handle, void **buffer_handles, uint32_t n_buffers,
                       uint32_t threads_x, uint32_t tg_x) {
    return _dispatch_sync(pso_handle, buffer_handles, n_buffers, threads_x, tg_x,
                          "dispatch_tiled", true);
}

void wmm_synchronize(void) {
    // every dispatch above already waits; kept as the future batch-commit sync
    // point (M2.5)
}

#ifdef __cplusplus
}  // extern "C"
#endif
