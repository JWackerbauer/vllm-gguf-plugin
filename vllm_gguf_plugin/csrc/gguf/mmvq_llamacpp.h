#pragma once

#include <cuda_runtime.h>

#define MMVQ_LLAMACPP_MAX_BATCH 8

enum {
  MMVQ_LLAMACPP_DST_F32 = 0,
  MMVQ_LLAMACPP_DST_F16 = 1,
  MMVQ_LLAMACPP_DST_BF16 = 2,
};

// dst[j, row] = dot(W[row, :], y[j, :]) for j < ncols_dst, row < nrows_x.
// vy_q8_1 holds ncols_dst activation vectors quantized to block_q8_1, each
// stride_col_y blocks apart. Returns false for unsupported types.
bool mmvq_llamacpp(int type, const void* vx, const void* vy_q8_1, void* dst,
                   int dst_dtype, int ncols_x, int nrows_x, int ncols_dst,
                   int stride_col_y, cudaStream_t stream);
