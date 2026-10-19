#pragma once

#include "fattn-common.cuh"
#include "fattn-tail.cuh"

// In-place current segment.
//
// The KV cache holds exactly the sliding window (the "body"), while the ubatch
// in flight is attended straight from the graph tensors in src[10]/src[11].
// Nothing is packed and no run descriptor is involved: the two sources are each
// executed by the ordinary FlashAttention kernels and the normalized partials
// are merged with the same FP32 rescaling the exact-tail route uses.
//
// This is the CUDA half of LLAMA_SWA_INPLACE_CURRENT; the CPU reference lives in
// ggml-cpu/ops.cpp.

static bool ggml_cuda_flash_attn_ext_current_pass_sources(ggml_tensor & pass, const ggml_tensor * dst) {
    // Clear every source the FlashAttention kernels do not use, so the nested
    // dispatch cannot re-enter this route or the exact-tail route.
    for (int i = 5; i < GGML_MAX_SRC; ++i) {
        pass.src[i] = nullptr;
    }
    pass.src[0] = dst->src[0];
    pass.view_src = nullptr;
    pass.view_offs = 0;
    return pass.src[0] != nullptr;
}

bool ggml_cuda_flash_attn_ext_current_supported(int device, const ggml_tensor * dst) {
    ggml_tensor * kc = dst->src[10];
    ggml_tensor * vc = dst->src[11];

    if (!kc || !vc || dst->src[5] != nullptr) {
        return false;
    }
    if (dst->src[3] == nullptr || dst->src[7] == nullptr) {
        return false;
    }
    if (dst->src[0]->ne[3] != 1) {
        // The merge below is the identity query order, i.e. one stream.
        return false;
    }

    // Both passes must be executable by the ordinary kernels on their own.
    ggml_tensor body = *dst;
    ggml_cuda_flash_attn_ext_current_pass_sources(body, dst);
    body.src[1] = dst->src[1];
    body.src[2] = dst->src[2];
    body.src[3] = dst->src[3];
    if (!ggml_cuda_flash_attn_ext_supported(device, &body)) {
        return false;
    }

    ggml_tensor current = body;
    current.src[1] = kc;
    current.src[2] = vc;
    current.src[3] = dst->src[7];
    return ggml_cuda_flash_attn_ext_supported(device, &current);
}

static void ggml_cuda_flash_attn_ext_current(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // Tensor sources are non-const handles even when only read here.
    ggml_tensor * q  = dst->src[0];
    ggml_tensor * kb = dst->src[1];
    ggml_tensor * vb = dst->src[2];
    ggml_tensor * mb = dst->src[3];
    ggml_tensor * mc = dst->src[7];
    ggml_tensor * kc = dst->src[10];
    ggml_tensor * vc = dst->src[11];

    GGML_ASSERT(q && kb && vb && mb && mc && kc && vc);
    GGML_ASSERT(q->ne[3] == 1 && "the in-place current segment is single-stream");

    const int d_v     = int(dst->ne[0]);
    const int n_query = int(q->ne[1]);
    const int n_head  = int(q->ne[2]);

    ggml_cuda_pool & pool = ctx.pool();

    const auto run_pass = [&](ggml_tensor & pass, ggml_tensor & meta, ggml_tensor * k,
                              ggml_tensor * v, ggml_tensor * mask, bool with_sinks,
                              ggml_cuda_pool_alloc<float2> & meta_alloc,
                              ggml_cuda_pool_alloc<uint8_t> & out_alloc) {
        ggml_cuda_flash_attn_ext_current_pass_sources(pass, dst);
        pass.src[1] = k;
        pass.src[2] = v;
        pass.src[3] = mask;
        // Sinks belong to exactly one partial; the merge rescales the body's
        // normalized result, so only the body may carry them.
        pass.src[4] = with_sinks ? dst->src[4] : nullptr;

        meta_alloc.alloc(size_t(n_query)*n_head);
        meta = *dst;
        meta.data = meta_alloc.get();
        ggml_cuda_tail_make_contiguous(meta, 2, n_head, n_query, 1, sizeof(float));

        pass.src[8] = &meta;

        const size_t alloc_size = ggml_cuda_tail_pass_alloc_size(ctx, pass);
        out_alloc.alloc(alloc_size);
        pass.data = out_alloc.get();

        ggml_cuda_flash_attn_ext_dispatch(ctx, &pass);
    };

    // ggml_cuda_pool_leg requires deallocation in reverse order of allocation,
    // and the destructors run in reverse declaration order. Declare them in the
    // order run_pass below allocates them, not one type at a time.
    ggml_cuda_pool_alloc<float2> body_meta_alloc(pool);
    ggml_cuda_pool_alloc<uint8_t> body_alloc(pool);
    ggml_cuda_pool_alloc<float2> cur_meta_alloc(pool);
    ggml_cuda_pool_alloc<uint8_t> cur_alloc(pool);

    ggml_tensor body_meta = {};
    ggml_tensor cur_meta  = {};

    // LLAMA_SWA_INPLACE_DEBUG=1 synchronizes after each launch so a fault is
    // attributed to the pass that caused it instead of the next CUDA API call.
    // It breaks CUDA graph capture, so it is strictly a debugging aid.
    const char * env_debug = getenv("LLAMA_SWA_INPLACE_DEBUG");
    const bool debug_sync = env_debug && env_debug[0] != '\0' && strcmp(env_debug, "0") != 0;
    const auto check_launch = [&](const char * what) {
        if (!debug_sync) {
            return;
        }
        const cudaError_t err = cudaStreamSynchronize(ctx.stream());
        if (err != cudaSuccess) {
            GGML_ABORT("%s: CUDA error after the %s pass: %s\n",
                    __func__, what, cudaGetErrorString(err));
        }
    };

    ggml_tensor body_pass = *dst;
    run_pass(body_pass, body_meta, kb, vb, mb, /*with_sinks =*/ true, body_meta_alloc, body_alloc);
    check_launch("body");

    ggml_tensor cur_pass = *dst;
    run_pass(cur_pass, cur_meta, kc, vc, mc, /*with_sinks =*/ false, cur_meta_alloc, cur_alloc);
    check_launch("current");

    // One logical softmax: both partials are normalized and joined in FP32.
    const dim3 grid(n_query, n_head, 1);
    k_flash_attn_ext_tail_partials_merge<<<grid, 256, 0, ctx.stream()>>>(
        (const float *) body_pass.data, (const float *) cur_pass.data, (float *) dst->data,
        body_meta_alloc.get(), cur_meta_alloc.get(), /*query_order =*/ nullptr,
        d_v, n_query, n_head, /*n_stream =*/ 1, /*q_max =*/ n_query, /*n_active =*/ 1,
        /*body_packed =*/ false,
        body_pass.nb[1], body_pass.nb[2], body_pass.nb[3],
        cur_pass.nb[1], cur_pass.nb[2], cur_pass.nb[3],
        dst->nb[1], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
    check_launch("merge");
}
