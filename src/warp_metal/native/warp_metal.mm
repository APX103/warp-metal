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

#include <stdint.h>
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

int wmm_dispatch(void *pso_handle, void **buffer_handles, uint32_t n_buffers,
                 uint32_t threads_x) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pso_handle;
    if (!dev || !pso) {
        _set_error(@"dispatch: missing device or pso");
        return -1;
    }

    id<MTLCommandQueue> queue = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];

    for (uint32_t i = 0; i < n_buffers; i++) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer_handles[i];
        [enc setBuffer:buf offset:0 atIndex:i];
    }

    // spike kernels are 1D; threadgroup size clamped to PSO limit
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if (tg > threads_x) tg = threads_x;
    MTLSize grid = MTLSizeMake(threads_x, 1, 1);
    MTLSize tgs = MTLSizeMake(tg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
    [enc endEncoding];

    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted || cb.error) {
        _set_error(cb.error.localizedDescription ?: @"command buffer failed");
        return -2;
    }
    return 0;
}

void wmm_synchronize(void) {
    // every dispatch above already waits; kept as the future batch-commit sync
    // point (M2.5)
}

#ifdef __cplusplus
}  // extern "C"
#endif
