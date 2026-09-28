// Quantized matrix x vector product (MMVQ) using llama.cpp's current CUDA
// kernels. Adapted from llama.cpp ggml/src/ggml-cuda/mmvq.cu (see
// llamacpp/README.md for the commit): the dense, non-fused path only, with the
// NVIDIA "generic" launch parameters (Ampere and newer).
//
// Compared with mmvq.cuh (llama.cpp b2899) this spreads each row over several
// warps, computes up to 8 input vectors per block against a single pass over
// the weights, and uses the newer vec_dot implementations for the i-quants.

// torch's extension build disables implicit half/bfloat16 conversions, which
// llama.cpp's kernels rely on. This translation unit does not use torch types.
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_CONVERSIONS__

#include <cuda_runtime.h>

#include "llamacpp/common.cuh"
#include "llamacpp/vecdotq.cuh"

#include "mmvq_llamacpp.h"

namespace {

constexpr int kWarpSize = 32;

typedef float (*vec_dot_q_cuda_t)(const void* __restrict__ vbq,
                                  const block_q8_1* __restrict__ bq8_1,
                                  const int& kbx, const int& iqs);

// GGML type ids (enum ggml_type) as used by the plugin.
enum : int {
  T_Q4_0 = 2,
  T_Q4_1 = 3,
  T_Q5_0 = 6,
  T_Q5_1 = 7,
  T_Q8_0 = 8,
  T_Q2_K = 10,
  T_Q3_K = 11,
  T_Q4_K = 12,
  T_Q5_K = 13,
  T_Q6_K = 14,
  T_IQ2_XXS = 16,
  T_IQ2_XS = 17,
  T_IQ3_XXS = 18,
  T_IQ1_S = 19,
  T_IQ4_NL = 20,
  T_IQ3_S = 21,
  T_IQ2_S = 22,
  T_IQ4_XS = 23,
  T_IQ1_M = 29,
};

template <int type>
struct mmvq_type;
#define MMVQ_TYPE(T, QK, QI, VDR, FN)               \
  template <>                                       \
  struct mmvq_type<T> {                             \
    static constexpr int qk = QK;                   \
    static constexpr int qi = QI;                   \
    static constexpr int vdr = VDR;                 \
    static constexpr vec_dot_q_cuda_t vec_dot = FN; \
  };
MMVQ_TYPE(T_Q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1)
MMVQ_TYPE(T_Q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1)
MMVQ_TYPE(T_Q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1)
MMVQ_TYPE(T_Q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1)
MMVQ_TYPE(T_Q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1)
MMVQ_TYPE(T_Q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
MMVQ_TYPE(T_Q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
MMVQ_TYPE(T_Q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1)
MMVQ_TYPE(T_Q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
MMVQ_TYPE(T_Q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)
MMVQ_TYPE(T_IQ2_XXS, QK_K, QI2_XXS, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1)
MMVQ_TYPE(T_IQ2_XS, QK_K, QI2_XS, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1)
MMVQ_TYPE(T_IQ2_S, QK_K, QI2_S, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1)
MMVQ_TYPE(T_IQ3_XXS, QK_K, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1)
MMVQ_TYPE(T_IQ3_S, QK_K, QI3_S, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1)
MMVQ_TYPE(T_IQ1_S, QK_K, QI1_S, 1, vec_dot_iq1_s_q8_1)
MMVQ_TYPE(T_IQ1_M, QK_K, QI1_S, 1, vec_dot_iq1_m_q8_1)
MMVQ_TYPE(T_IQ4_NL, QK4_NL, QI4_NL, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1)
MMVQ_TYPE(T_IQ4_XS, QK_K, QI4_XS, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1)
#undef MMVQ_TYPE

// llama.cpp MMVQ_PARAMETERS_GENERIC
constexpr int calc_nwarps(int ncols_dst) { return ncols_dst <= 4 ? 4 : 2; }
constexpr int calc_rows_per_block(int ncols_dst) {
  return ncols_dst == 1 ? 1 : 2;
}

template <typename T>
__device__ __forceinline__ T from_float(float x);
template <>
__device__ __forceinline__ float from_float<float>(float x) {
  return x;
}
template <>
__device__ __forceinline__ half from_float<half>(float x) {
  return __float2half(x);
}
template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

template <int type, int ncols_dst, typename dst_t>
__launch_bounds__(calc_nwarps(ncols_dst) * kWarpSize, 1) __global__
    void mul_mat_vec_q(const void* __restrict__ vx,
                       const block_q8_1* __restrict__ vy,
                       dst_t* __restrict__ dst, const int ncols_x,
                       const int nrows_x, const int stride_col_y,
                       const int stride_col_dst) {
  constexpr int qk = mmvq_type<type>::qk;
  constexpr int qi = mmvq_type<type>::qi;
  constexpr int vdr = mmvq_type<type>::vdr;
  constexpr vec_dot_q_cuda_t vec_dot_q_cuda = mmvq_type<type>::vec_dot;
  constexpr int nwarps = calc_nwarps(ncols_dst);
  constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst);

  const int tid = kWarpSize * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * nwarps * kWarpSize / qi;

  // partial sum for each thread
  float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

  const int kbx_offset = row0 * blocks_per_row_x;

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);  // y block index that aligns with kbx

    // x block quant index when casting the quants to int
    const int kqs = vdr * (tid % (qi / vdr));

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        // The last block may overhang nrows_x; clamp the row so loads stay in
        // bounds.
        const int row_off = row0 + i < nrows_x ? i : 0;
        tmp[j][i] +=
            vec_dot_q_cuda(vx, &vy[j * stride_col_y + kby],
                           kbx_offset + row_off * blocks_per_row_x + kbx, kqs);
      }
    }
  }

  __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst]
                             [rows_per_cuda_block][kWarpSize];
  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
      for (int l = 0; l < nwarps - 1; ++l) {
        tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
      }
#pragma unroll
      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        tmp[j][i] += __shfl_xor_sync(0xffffffff, tmp[j][i], offset, kWarpSize);
      }
      if (threadIdx.x == i && row0 + i < nrows_x) {
        dst[j * stride_col_dst + row0 + i] = from_float<dst_t>(tmp[j][i]);
      }
    }
  }
}

template <int type, int ncols_dst, typename dst_t>
void launch(const void* vx, const block_q8_1* vy, dst_t* dst, const int ncols_x,
            const int nrows_x, const int stride_col_y, const int stride_col_dst,
            cudaStream_t stream) {
  constexpr int rpb = calc_rows_per_block(ncols_dst);
  const dim3 block_nums((nrows_x + rpb - 1) / rpb, 1, 1);
  const dim3 block_dims(kWarpSize, calc_nwarps(ncols_dst), 1);
  mul_mat_vec_q<type, ncols_dst, dst_t><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst);
}

template <int type, typename dst_t>
void switch_ncols(const void* vx, const block_q8_1* vy, dst_t* dst,
                  const int ncols_x, const int nrows_x, const int ncols_dst,
                  const int stride_col_y, cudaStream_t stream) {
  // dst is [ncols_dst, nrows_x] row-major, so consecutive vectors are nrows_x
  // apart.
  for (int j0 = 0; j0 < ncols_dst; j0 += MMVQ_LLAMACPP_MAX_BATCH) {
    const int n = ncols_dst - j0 < MMVQ_LLAMACPP_MAX_BATCH
                      ? ncols_dst - j0
                      : MMVQ_LLAMACPP_MAX_BATCH;
    const block_q8_1* y = vy + j0 * stride_col_y;
    dst_t* d = dst + (size_t)j0 * nrows_x;
    switch (n) {
      case 1:
        launch<type, 1>(vx, y, d, ncols_x, nrows_x, stride_col_y, nrows_x,
                        stream);
        break;
      case 2:
        launch<type, 2>(vx, y, d, ncols_x, nrows_x, stride_col_y, nrows_x,
                        stream);
        break;
      case 3:
        launch<type, 3>(vx, y, d, ncols_x, nrows_x, stride_col_y, nrows_x,
                        stream);
        break;
      case 4:
        launch<type, 4>(vx, y, d, ncols_x, nrows_x, stride_col_y, nrows_x,
                        stream);
        break;
      case 5:
        launch<type, 5>(vx, y, d, ncols_x, nrows_x, stride_col_y, nrows_x,
                        stream);
        break;
      case 6:
        launch<type, 6>(vx, y, d, ncols_x, nrows_x, stride_col_y, nrows_x,
                        stream);
        break;
      case 7:
        launch<type, 7>(vx, y, d, ncols_x, nrows_x, stride_col_y, nrows_x,
                        stream);
        break;
      default:
        launch<type, 8>(vx, y, d, ncols_x, nrows_x, stride_col_y, nrows_x,
                        stream);
        break;
    }
  }
}

template <typename dst_t>
bool switch_type(const int type, const void* vx, const void* vy, dst_t* dst,
                 const int ncols_x, const int nrows_x, const int ncols_dst,
                 const int stride_col_y, cudaStream_t stream) {
  const block_q8_1* y = (const block_q8_1*)vy;
#define MMVQ_CASE(T)                                                       \
  case T:                                                                  \
    switch_ncols<T>(vx, y, dst, ncols_x, nrows_x, ncols_dst, stride_col_y, \
                    stream);                                               \
    return true
  switch (type) {
    MMVQ_CASE(T_Q4_0);
    MMVQ_CASE(T_Q4_1);
    MMVQ_CASE(T_Q5_0);
    MMVQ_CASE(T_Q5_1);
    MMVQ_CASE(T_Q8_0);
    MMVQ_CASE(T_Q2_K);
    MMVQ_CASE(T_Q3_K);
    MMVQ_CASE(T_Q4_K);
    MMVQ_CASE(T_Q5_K);
    MMVQ_CASE(T_Q6_K);
    MMVQ_CASE(T_IQ2_XXS);
    MMVQ_CASE(T_IQ2_XS);
    MMVQ_CASE(T_IQ2_S);
    MMVQ_CASE(T_IQ3_XXS);
    MMVQ_CASE(T_IQ3_S);
    MMVQ_CASE(T_IQ1_S);
    MMVQ_CASE(T_IQ1_M);
    MMVQ_CASE(T_IQ4_NL);
    MMVQ_CASE(T_IQ4_XS);
    default:
      return false;
  }
#undef MMVQ_CASE
}

}  // namespace

bool mmvq_llamacpp(const int type, const void* vx, const void* vy_q8_1,
                   void* dst, const int dst_dtype, const int ncols_x,
                   const int nrows_x, const int ncols_dst,
                   const int stride_col_y, cudaStream_t stream) {
  switch (dst_dtype) {
    case MMVQ_LLAMACPP_DST_F32:
      return switch_type(type, vx, vy_q8_1, (float*)dst, ncols_x, nrows_x,
                         ncols_dst, stride_col_y, stream);
    case MMVQ_LLAMACPP_DST_F16:
      return switch_type(type, vx, vy_q8_1, (half*)dst, ncols_x, nrows_x,
                         ncols_dst, stride_col_y, stream);
    case MMVQ_LLAMACPP_DST_BF16:
      return switch_type(type, vx, vy_q8_1, (nv_bfloat16*)dst, ncols_x, nrows_x,
                         ncols_dst, stride_col_y, stream);
    default:
      return false;
  }
}
