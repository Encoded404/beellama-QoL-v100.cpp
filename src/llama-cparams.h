#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_outputs_max_per_seq;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    // expose the raw last-layer hidden instead of the post-norm state. gemma4.cpp
    // reads this on every nextn build, so it has to be deterministic, and every
    // draft head and recorded corpus so far has been fed the raw form: true keeps
    // that, while the post-norm LM-head input (what the reference drafter consumes)
    // is the alternative and would need the corpora re-dumped.
    bool embeddings_nextn_raw = true;
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    // [n_layer()] extract the K/V rows that each selected layer writes to its KV
    // cache. used to record a frozen target model's attention inputs so a draft
    // head can be trained against them offline.
    std::vector<bool> kv_dump_layers;

    // also record the Q each selected layer computes and the attention output it
    // produces, so the recorded K/V can be checked against the model's own
    // attention. only reachable from LLAMA_DUMP_ATTN_IO / LLAMA_DUMP_KV_LAYERS,
    // which record all three together
    bool attn_io_dump = false;

    // record the rows above before the cache-domain transform instead of after it.
    // a quantized cache type rotates Q, K and V into a basis that is cheaper to
    // quantize; the rotation cancels out at attention time because the query is
    // rotated with it, so the stored rows are the right thing to record for
    // inspecting a cache but are not the rows a trainer or a draft head consumes.
    // false (the default) records the stored rows; true records the model basis -
    // Q and K/V from before the rotation and the attention output from after the
    // un-rotation - so a dump never mixes bases. has no effect on layers whose
    // attention route does not transform K/V.
    bool kv_dump_pre_rotation = false;

    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;

    // Structured KVarN cache settings.  Kept in the internal context params so
    // memory creation does not need to depend on the public params object.
    llama_kvarn_params kvarn;

    uint32_t  kv_tail_tokens = 0;
    uint32_t  kv_tail_tokens_swa = 0;
    uint32_t  kv_tail_tokens_requested = 0;
    uint32_t  kv_tail_tokens_swa_requested = 0;
    bool      kv_tail_native_exact = false;
    bool      kv_tail_native_exact_swa = false;
    uint32_t  kv_tail_rollback_tokens = 0;
    ggml_type kv_tail_type   = GGML_TYPE_COUNT;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
