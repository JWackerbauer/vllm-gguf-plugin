// Minimal stand-in for llama.cpp's ggml/src/ggml-cuda/common.cuh, providing
// only what the vendored vecdotq.cuh needs. Definitions are copied from
// llama.cpp; see README.md for the source commit.
#pragma once

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define GGML_CUDA_CC_DP4A 610
#define FP8_AVAILABLE

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    return __dp4a(a, b, c);
#else
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t x) {
#if CUDART_VERSION >= 12080
    const nv_bfloat16 e = __nv_cvt_e8m0_to_bf16raw(x);
    return (float) e;
#else
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
#endif
}

static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32(uint8_t x) {
    const uint32_t bits = x * (x != 0x7F && x != 0xFF); // Convert NaN to 0.0f to match CPU implementation.
    const __nv_fp8_e4m3 xf = *reinterpret_cast<const __nv_fp8_e4m3 *>(&bits);
    return static_cast<float>(xf) / 2;
}
