#include "common.cuh"
#include "convert.cuh"
#include "fattn.cuh"

#include <cstring>

#if defined(GGML_CUDA_SM70_D256)

#include "fattn-sm70-d256.cuh"

#include <cstdio>
#include <cstdlib>

using namespace ggml_cuda_fattn_sm70_d256;

// SM70 (Volta) D256 split-D prefill FlashAttention; see fattn-sm70-d256.cuh.

static bool sm70_d256_env_enabled() {
    // Opt-in for now: this kernel is compile-verified but has not been validated on a
    // real V100, and the failure mode is silently wrong prefill output rather than a
    // crash. Set LLAMA_SM70_D256=1 to enable it. Once a V100 passes the correctness
    // gates this inverts to opt-out, matching the other Volta tuning knobs.
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_SM70_D256");
        return e != nullptr && e[0] != '0';
    }();
    return enabled;
}

// Route diagnostics. The first decision is always printed; LLAMA_SM70_D256_DEBUG=1
// prints every decision.
static void sm70_d256_probe(const char * reason, const int cc, const ggml_tensor * dst) {
    static const bool verbose = getenv("LLAMA_SM70_D256_DEBUG") != nullptr;
    static int printed = 0;
    if (!verbose && printed > 0) {
        return;
    }
    printed++;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    fprintf(stderr,
            "[sm70-d256] %s | cc=%d Q=(%lld,%lld,%lld,%lld) K=(%lld,%lld,%lld,%lld) Ktype=%d "
            "Vtype=%d mask=%p Mkv=%lld q_len=%lld\n",
            reason, cc,
            (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
            (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3],
            (int) K->type, (int) V->type, (const void *) mask,
            mask ? (long long) mask->ne[0] : -1LL, (long long) Q->ne[1]);
}

// Can this K/V tensor reach the kernel, either directly (F16) or through the same
// to_fp16 / to_fp16_nc conversion the stock mma_f16 route uses?
static bool sm70_d256_kv_source_ok(const ggml_tensor * t) {
    if (!ggml_cuda_fattn_kv_type_supported(t->type)) {
        return false;
    }
    if (t->type == GGML_TYPE_F16) {
        // Read directly by the kernel, which indexes in half2 units.
        return t->nb[0] == sizeof(half) &&
               t->nb[1] % sizeof(half2) == 0 &&
               t->nb[2] % sizeof(half2) == 0 &&
               t->nb[3] % sizeof(half2) == 0;
    }
    if (ggml_is_contiguously_allocated(t)) {
        return ggml_get_to_fp16_cuda(t->type) != nullptr;
    }
    // to_fp16_nc walks element strides, so rows must be element-contiguous and every
    // stride an exact element multiple.
    const size_t ts = ggml_type_size(t->type);
    return t->nb[0] == ts && ggml_get_to_fp16_nc_cuda(t->type) != nullptr &&
           t->nb[1] % ts == 0 && t->nb[2] % ts == 0 && t->nb[3] % ts == 0;
}

// Shared by the kernel selector, the alloc-size query and supported_op. Must stay
// identical between those three or the scratch carve would be inconsistent.
bool ggml_cuda_sm70_d256_supported(const int cc, const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    // volta_mma_available() requires the highest compiled arch to be exactly sm_70, so
    // the route disappears on builds that do not compile sm_70 (for example CUDA >= 13).
    if (!volta_mma_available(cc)) {
        return false;
    }
    if (!sm70_d256_env_enabled()) {
        sm70_d256_probe("REJECT: not enabled (set LLAMA_SM70_D256=1)", cc, dst);
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (Q == nullptr || K == nullptr || V == nullptr) {
        return false;
    }
    // Q is read and dst is written as float2, so both need element-contiguous rows and
    // float2-multiple edge strides. The stock kernel asserts the same for Q.
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            Q->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            Q->nb[1] % sizeof(float2) != 0 || Q->nb[2] % sizeof(float2) != 0 || Q->nb[3] % sizeof(float2) != 0 ||
            dst->nb[1] % sizeof(float2) != 0 || dst->nb[2] % sizeof(float2) != 0 || dst->nb[3] % sizeof(float2) != 0) {
        sm70_d256_probe("REJECT: Q/dst layout", cc, dst);
        return false;
    }
    if (Q->ne[0] != kHeadDim || K->ne[0] != kHeadDim || V->ne[0] != kHeadDim) {
        sm70_d256_probe("REJECT: head_dim != 256", cc, dst);
        return false;
    }
    // Prefill only: decode, MTP and small batches stay on the stock path. The kernel
    // handles arbitrary additive masks, not just causal ones.
    if (mask == nullptr || sinks != nullptr || Q->ne[1] < kMinQLen) {
        sm70_d256_probe("REJECT: no mask / sinks / small batch", cc, dst);
        return false;
    }
    // The exact-tail path (dst->src[5..11]) runs its packed body through this op with the
    // per-(row, head) max/sum meta attached as src[8] and folds the body result into the
    // tail pass. The stock kernels publish that meta; this kernel computes the same row
    // max/sum but does not write it, so the tail pass would read uninitialized weights.
    // Those body passes keep the stock route (the outer tail op is dispatched before this
    // route is consulted at all), which costs the tail path the split-D kernel but keeps
    // it correct.
    if (dst->src[8] != nullptr) {
        sm70_d256_probe("REJECT: exact-tail body meta (src[8])", cc, dst);
        return false;
    }
    // Same mask contract the exact-tail body pass uses. The mask must be F16 (the stock
    // kernel asserts the same) and its batch axis must broadcast or match exactly.
    if (mask->type != GGML_TYPE_F16 ||
            mask->ne[0] != K->ne[1] || mask->ne[1] < Q->ne[1] || mask->ne[2] != 1 ||
            (mask->ne[3] != 1 && mask->ne[3] != Q->ne[3])) {
        sm70_d256_probe("REJECT: mask shape / type", cc, dst);
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0) {
        sm70_d256_probe("REJECT: gqa ratio", cc, dst);
        return false;
    }
    // The kernel indexes K, V and the mask with the Q batch index.
    if (Q->ne[3] != K->ne[3] || K->ne[3] != V->ne[3]) {
        sm70_d256_probe("REJECT: batch mismatch", cc, dst);
        return false;
    }

    // The kernel implements neither ALiBi nor a logit softcap.
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        sm70_d256_probe("REJECT: max_bias / softcap", cc, dst);
        return false;
    }

    // The full FA-supported cache range: F16 is read directly, every quantized type
    // (Q8_0 down to Q2_0S/Q2_1, plus BF16, IQ4_NL and F32) is materialized into the
    // f16 mirror by the launcher.
    if (!sm70_d256_kv_source_ok(K) || !sm70_d256_kv_source_ok(V)) {
        sm70_d256_probe("REJECT: kv type / layout", cc, dst);
        return false;
    }

    sm70_d256_probe("ACCEPT: sm70 d256 split-d prefill", cc, dst);
    return true;
}

namespace {

struct sm70_d256_kv_view {
    const half * data;
    int64_t row_stride;   // half2 units
    int64_t head_stride;
    int64_t batch_stride;
};

// Materialize one K/V tensor the way launch_fattn does for the stock mma_f16 route:
// a whole-tensor convert when the allocation is contiguous, otherwise the strided
// to_fp16_nc walk. Both branches emit [batch][head][kv][D], so the resulting strides
// are derived from the tensor shape rather than from whichever branch ran.
static sm70_d256_kv_view sm70_d256_kv_f16(
        const ggml_tensor * t, const uintptr_t mirror, const cudaStream_t stream) {
    sm70_d256_kv_view view;

    if (t->type == GGML_TYPE_F16) {
        view.data         = (const half *) t->data;
        view.row_stride   = t->nb[1] / (int64_t) sizeof(half2);
        view.head_stride  = t->nb[2] / (int64_t) sizeof(half2);
        view.batch_stride = t->nb[3] / (int64_t) sizeof(half2);
        return view;
    }

    GGML_ASSERT(mirror != 0);
    half * const dst = (half *) mirror;
    if (ggml_is_contiguously_allocated(t)) {
        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(t->type);
        GGML_ASSERT(to_fp16 != nullptr);
        to_fp16((const char *) t->data, dst, ggml_nelements(t), stream);
    } else {
        const size_t ts = ggml_type_size(t->type);
        GGML_ASSERT(t->nb[0] == ts);
        to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(t->type);
        GGML_ASSERT(to_fp16 != nullptr);
        to_fp16((const char *) t->data, dst, t->ne[0], t->ne[1], t->ne[2], t->ne[3],
                t->nb[1] / ts, t->nb[2] / ts, t->nb[3] / ts, stream);
    }

    view.data         = dst;
    view.row_stride   = t->ne[0] / 2;
    view.head_stride  = t->ne[1] * t->ne[0] / 2;
    view.batch_stride = t->ne[2] * t->ne[1] * t->ne[0] / 2;
    return view;
}

} // namespace

void ggml_cuda_flash_attn_ext_sm70_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    GGML_ASSERT(volta_mma_available(cc));
    GGML_ASSERT(ggml_cuda_sm70_d256_supported(cc, dst));

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int q_len  = (int) Q->ne[1];
    const int kv_len = (int) K->ne[1];
    const int gqa    = (int) (Q->ne[2] / K->ne[2]);
    const int batch  = (int) Q->ne[3];

    const cudaStream_t stream = ctx.stream();

    // Matches the needs_f16 flag set for BEST_FATTN_KERNEL_SM70_D256.
    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, true, true);

    const bool V_is_K_view = V->view_src != nullptr &&
        (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const sm70_d256_kv_view kv = sm70_d256_kv_f16(K, f16_extra.K, stream);
    const sm70_d256_kv_view vv = V_is_K_view ? kv : sm70_d256_kv_f16(V, f16_extra.V, stream);

    // kSmemBytes exceeds the 48 KiB default dynamic limit on Volta.
    static bool smem_raised = false;
    if (!smem_raised) {
        CUDA_CHECK(cudaFuncSetAttribute(sm70_d256_splitd_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kSmemBytes));
        smem_raised = true;
    }

    const int  ntiles_q = (q_len + kBlockM - 1) / kBlockM;
    const dim3 grid((unsigned) ntiles_q, (unsigned) batch, (unsigned) Q->ne[2]);
    // (warp_size, nwarps): the ggml_cuda_mma tile helpers use threadIdx.x as the lane.
    const dim3 block(32, kThreads / 32);

    // Optional KV tile bound: the same on-GPU mask scan launch_fattn runs for the
    // tile/vec routes (fattn-common.cuh, flash_attn_mask_to_KV_max). Every KV block at
    // or above the result is masked out for all rows of its Q tile, so the kernel can
    // stop there instead of walking the whole cache. The scan starts at the newest KV
    // tile and walks down to the first one holding a visible column, so it reads only
    // the all-masked tiles it then lets the kernel skip. A mask layout or length the
    // scan cannot handle leaves the bound null and the kernel scans every KV block,
    // which is always correct.
    const int mask_n_batch = (int) mask->ne[3];

    ggml_cuda_pool_alloc<int> kv_max_alloc(ctx.pool());
    const bool use_kv_max =
        K->ne[1] % FATTN_KQ_STRIDE == 0 &&                  // 256-column scan tiles
        mask->nb[0] == sizeof(half) &&                      // columns contiguous in half
        mask->nb[1] % (int64_t) sizeof(half2) == 0 &&       // rows half2-aligned
        mask->nb[3] % (int64_t) sizeof(half2) == 0 &&       // mask batches half2-aligned
        (int64_t) ntiles_q * kBlockM <= mask->ne[1];        // scanned rows inside the mask
    if (use_kv_max) {
        kv_max_alloc.alloc((size_t) ntiles_q * mask_n_batch);
        const dim3 blocks_num_KV_max((unsigned) ntiles_q, (unsigned) mask_n_batch, 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);
        ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<kBlockM>,
            ggml_cuda_kernel_launch_params(blocks_num_KV_max, block_dim_KV_max, 0, stream),
            (const half2 *) mask->data, kv_max_alloc.ptr, (int) (K->ne[1] / FATTN_KQ_STRIDE),
            mask->nb[1] / (int64_t) sizeof(half2), mask->nb[3] / (int64_t) sizeof(half2));
        CUDA_CHECK(cudaGetLastError());
    }

    sm70_d256_splitd_kernel<<<grid, block, kSmemBytes, stream>>>(
            (const float *) Q->data, kv.data, vv.data,
            (const half *) mask->data, (float *) dst->data,
            Q->nb[1] / (int64_t) sizeof(float2), Q->nb[2] / (int64_t) sizeof(float2), Q->nb[3] / (int64_t) sizeof(float2),
            kv.row_stride, kv.head_stride, kv.batch_stride,
            vv.row_stride, vv.head_stride, vv.batch_stride,
            mask->nb[1] / (int64_t) sizeof(half),
            mask->nb[3] / (int64_t) sizeof(half), mask_n_batch,
            use_kv_max ? kv_max_alloc.ptr : nullptr,
            // dst is not laid out like Q: ggml_flash_attn_ext returns permute(0, 2, 1, 3), so the
            // result is [D][heads_q][q_len][batch] while Q is [D][q_len][heads_q][batch]. The row
            // stride therefore lives in nb[2] and the head stride in nb[1] - the opposite of Q.
            // They only coincide when heads_q == 1, so mixing them up is invisible on single-head
            // shapes and scrambles the output on every real model.
            dst->nb[2] / (int64_t) sizeof(float2), dst->nb[1] / (int64_t) sizeof(float2), dst->nb[3] / (int64_t) sizeof(float2),
            q_len, kv_len, gqa, scale);
    CUDA_CHECK(cudaGetLastError());
}

#else // defined(GGML_CUDA_SM70_D256)

// The route is always reachable from the kernel selector, so the translation unit is
// always compiled and stubs out when the feature is not built in. Same pattern as
// fattn-kvarn-dispatch.cu.

bool ggml_cuda_sm70_d256_supported(const int cc, const ggml_tensor * dst) {
    GGML_UNUSED(cc);
    GGML_UNUSED(dst);
    return false;
}

void ggml_cuda_flash_attn_ext_sm70_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("SM70 D256 split-D prefill was not compiled in (-DGGML_CUDA_SM70_D256=ON)");
}

#endif // defined(GGML_CUDA_SM70_D256)
