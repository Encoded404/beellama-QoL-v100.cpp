#include <algorithm>
#include <cstdint>
#include <cfloat>

#include "mars_stats.cuh"
#include "common.cuh"

// Per-row reduction over the logits: dst[0] = largest value, dst[1] = k-th
// largest (k clamped to [1, 8] at op creation). Deterministic: each thread
// builds a local top-k array, the warp leaders merge them in shared memory,
// no atomics.
static __global__ void mars_stats_f32(
        const float * __restrict__ x,
        float * __restrict__ dst,
        const int64_t ncols,
        const int32_t k) {
    const int64_t row = blockIdx.x;
    const float * rowx = x + row * ncols;

    float topk[8];
    for (int32_t j = 0; j < k; ++j) {
        topk[j] = -FLT_MAX;
    }

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > topk[k - 1]) {
            int32_t j = k - 1;
            while (j > 0 && val > topk[j - 1]) {
                topk[j] = topk[j - 1];
                --j;
            }
            topk[j] = val;
        }
    }

    // merge the per-thread arrays: leader threads of each warp write their
    // top-k into shared memory, then the first warp merges them
    constexpr int max_warps = 1024 / 32;
    __shared__ float shared_topk[max_warps * 8];

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    const int n_warps = (blockDim.x + 31) / 32;
    for (int32_t j = 0; j < k; ++j) {
        shared_topk[warp_id * 8 + j] = topk[j];
    }
    __syncthreads();

    if (warp_id == 0) {
        float merged[8];
        for (int32_t j = 0; j < k; ++j) {
            merged[j] = -FLT_MAX;
        }
        for (int32_t w = 0; w < n_warps; ++w) {
            for (int32_t j = 0; j < k; ++j) {
                const float val = shared_topk[w * 8 + j];
                if (val > merged[k - 1]) {
                    int32_t t = k - 1;
                    while (t > 0 && val > merged[t - 1]) {
                        merged[t] = merged[t - 1];
                        --t;
                    }
                    merged[t] = val;
                }
            }
        }
        if (lane_id == 0) {
            dst[row * 2 + 0] = merged[0];
            dst[row * 2 + 1] = merged[k - 1];
        }
    }
}

void ggml_cuda_mars_stats(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t k = std::max(1, std::min((int32_t) dst->op_params[0], 8));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t n_threads = std::min<int64_t>(256, std::max<int64_t>(32, ncols));
    mars_stats_f32<<<nrows, n_threads, 0, stream>>>(src0_d, dst_d, ncols, k);
}