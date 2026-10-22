#pragma once

// Mask-origin contract for GGML_OP_FLASH_ATTN_EXT.
//
// When the graph knows that the leading `origin` K/V columns of the batch all
// carry one uniform mask value, it omits them from the mask tensor: src[3]
// column j then holds the mask for K/V column origin + j. The uniform value is
// selected by GGML_FLASH_ATTN_EXT_OP_PARAM_KQ_MASK_PREFIX_VISIBLE
// (1 => 0.0f, i.e. keep; 0 => -inf, i.e. drop).
//
// A kernel must therefore read every mask entry through this helper, and must
// never read src[3] for a column below origin. origin == 0 is the ordinary
// "mask spans the whole of src[1]" contract and must stay on the existing code
// path.
//
// note: the KV_max scanners are separate; see fattn-common.cuh. They must clamp
// their result to at least `origin` when the prefix is visible, otherwise the
// visible prefix columns are dropped from the attention window.

#include "common.cuh"

#include <cmath>

struct fattn_mask_origin {
    int  origin;
    half prefix;
};

static inline fattn_mask_origin fattn_mask_origin_get(const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    fattn_mask_origin res;
    res.origin = ggml_get_op_params_i32(dst, GGML_FLASH_ATTN_EXT_OP_PARAM_KQ_MASK_ORIGIN);
    GGML_ASSERT(res.origin >= 0);

    const bool prefix_visible =
            ggml_get_op_params_i32(dst, GGML_FLASH_ATTN_EXT_OP_PARAM_KQ_MASK_PREFIX_VISIBLE) != 0;
    res.prefix = prefix_visible ? __float2half(0.0f) : __float2half(-INFINITY);

    if (res.origin > 0) {
        const ggml_tensor * k    = dst->src[1];
        const ggml_tensor * mask = dst->src[3];
        GGML_ASSERT(k != nullptr && mask != nullptr);
        GGML_ASSERT(mask->type == GGML_TYPE_F16);
        GGML_ASSERT(mask->ne[0] + res.origin == k->ne[1]);
    }

    return res;
}

// Mask value for the absolute K/V column `ik`. `row` points at column 0 of the
// mask row, i.e. at K/V column `origin`.
__device__ __forceinline__ half fattn_mask_at(
        const char * row, int64_t nb0, int64_t ik, const fattn_mask_origin m) {
    return ik < m.origin ? m.prefix : *reinterpret_cast<const half *>(row + (ik - m.origin)*nb0);
}

// Same, for a row already expressed as a half pointer with an element stride.
__device__ __forceinline__ half fattn_mask_at_h(
        const half * row, int64_t nb0_h, int64_t ik, const fattn_mask_origin m) {
    return ik < m.origin ? m.prefix : row[(ik - m.origin)*nb0_h];
}
