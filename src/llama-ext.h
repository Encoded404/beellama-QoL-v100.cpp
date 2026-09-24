#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"
#include "llama-kv-memory-stats.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);
LLAMA_API llama_kv_memory_stats llama_get_kv_memory_stats(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which hidden state llama_get_embeddings_nextn() returns. false (the
// default) is the post-output-norm hidden, i.e. the LM-head input, which is what
// the reference drafter consumes. true is the raw last-layer hidden before that
// norm, for drafters trained against the pre-norm form.
LLAMA_API void llama_set_embeddings_nextn_raw(struct llama_context * ctx, bool value);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// per-layer KV dumping
//
// When enabled, each decode copies the rows that the selected layers hand to
// their KV cache into host buffers: one row per token in the batch, holding the
// post-rope K and the post-norm V in the cache's own layout (but before any
// quantization implied by the cache type). Those rows are the attention inputs a
// shared-KV draft head consumes (for example the Gemma 4 MTP assistant), so a
// frozen target can be recorded once and replayed offline by a trainer.
//
// layers is a list of layer indices; they are validated against the model and
// de-duplicated, and the selection is kept in ascending order. An empty list
// disables dumping.
LLAMA_API void llama_set_kv_dump_layers(struct llama_context * ctx, const int32_t * layers, size_t n_layers);

// Select which basis the K/V dump records. false (the default) records the rows as
// they are stored, i.e. after any cache-domain transform. A quantized cache type
// rotates K/V into a basis that is cheaper to quantize; that rotation cancels out
// at attention time because the query is rotated with it, so the stored rows are
// correct for inspecting a cache, but they are not the rows a trainer or a draft
// head consumes. true records the model basis, i.e. the same rows from before the
// rotation. Layers whose attention route does not transform K/V are unaffected
// either way. Also settable with LLAMA_DUMP_KV_PRE_ROTATION.
LLAMA_API void llama_set_kv_dump_pre_rotation(struct llama_context * ctx, bool value);

// Which basis the K/V dump currently records: false for the rows as the cache
// stores them, true for the model basis. Set by llama_set_kv_dump_pre_rotation()
// or LLAMA_DUMP_KV_PRE_ROTATION. A consumer that labels a dump should read this
// instead of assuming which switch was used.
LLAMA_API bool llama_get_kv_dump_pre_rotation(struct llama_context * ctx);

// Number of layers currently selected for KV dumping.
LLAMA_API size_t llama_get_kv_dump_n_layers(struct llama_context * ctx);

// Layer index of the i-th selected layer, in ascending order.
LLAMA_API int32_t llama_get_kv_dump_layer(struct llama_context * ctx, size_t i);

// Per-token row width of the dumped K and V for the i-th selected layer.
LLAMA_API uint32_t llama_get_kv_dump_n_embd_k(struct llama_context * ctx, size_t i);
LLAMA_API uint32_t llama_get_kv_dump_n_embd_v(struct llama_context * ctx, size_t i);

// Host buffers holding n_tokens rows of n_embd_k/v floats for the i-th selected
// layer, in batch token order. Valid until the next decode.
// Returns nullptr for a layer that does not emit the requested tensor - V is
// absent for layers that cache K only and derive V from it.
LLAMA_API float * llama_get_kv_dump_k(struct llama_context * ctx, size_t i);
LLAMA_API float * llama_get_kv_dump_v(struct llama_context * ctx, size_t i);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
