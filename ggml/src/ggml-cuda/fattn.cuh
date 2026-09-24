#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_tail_supported(
        ggml_type body_k, ggml_type body_v, ggml_type tail_k, ggml_type tail_v, int64_t d_k, int64_t d_v);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

const char * ggml_cuda_fa_build_policy();

bool ggml_cuda_fa_pair_compiled(ggml_type type_K, ggml_type type_V);

// The K/V cache types the CUDA FlashAttention path accepts (F32 is canonicalized to
// F16). Shared with the SM70 D256 route so both use one type contract.
bool ggml_cuda_fattn_kv_type_supported(ggml_type type);

// Volta (SM70) D256 split-D prefill route; see fattn-sm70-d256.cuh.
bool ggml_cuda_sm70_d256_supported(int cc, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_sm70_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
