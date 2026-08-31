#pragma once

#include "llama-memory.h"

#include <cstdint>
#include <map>
#include <memory>

struct llama_hparams;
struct llama_model;
class llama_kv_cache_kvarn;
struct llama_kvarn_borrow_config;

// Probe every structured cache reachable from mem for a KVarN store that an
// auxiliary (draft/MTP) context can borrow. Returns nullptr when no
// shareable KVarN store is present:
//   - direct llama_kv_cache_kvarn
//   - llama_kv_cache_iswa whose base (or SWA group) is a llama_kv_cache_kvarn
//   - anything else (hybrid, DSA, DSV4, plain) -> nullptr
const llama_kv_cache_kvarn * llama_memory_probe_kvarn(const llama_memory_i * mem);

// True when the auxiliary context can borrow the target context's KVarN
// store instead of allocating its own cache. Requires:
//   - a probeable KVarN store on the target side,
//   - a unified target cache (single shared stream),
//   - a lossless F16 staging region large enough for the speculative window,
//   - n_seq_max within the global sequence cap (the aux writes all stay
//     inside the frontier window).
bool llama_kvarn_aux_borrow_supported(
        const llama_memory_i * mem_other,
        uint32_t n_seq_max);

// v1 constants for the borrow state descriptor.
#define LLAMA_KVARN_BORROW_MAGIC 0x4B56424F // "KVBO"
#define LLAMA_KVARN_BORROW_VERSION 2 // v2: descriptor carries n_seq_max + max_draft_tokens

// Environment kill-switch: GGML_KVARN_BORROW=0 falls back to the classic
// private/plain-cache draft behavior.
bool llama_kvarn_borrow_disabled_by_env();

// Borrowing auxiliary KV memory.
//
// Semantics
// ---------
// The auxiliary context (draft/MTP/assistant) shares ONE position-addressed
// KVarN store with the target context:
//
//   * reads:   every attention op of the auxiliary graph resolves record,
//              staging and tail spans from the target store using the
//              auxiliary's own logical cursor. Record-domain reads are
//              clamped to the target's last fully VERIFIED record group
//              (seal_clamp_groups), so speculative rows still living in the
//              F16 staging region are never read through unsealed groups.
//
//   * writes:  the auxiliary ubatch rows are stored with eager_records =
//              false and advance_fill = false: stage rows are written at
//              their positions, but no record group is sealed and the
//              stream's committed fill only moves through the target's own
//              eager stores at verification.
//
//   * rollback: rejected draft rows are discarded with seq_rm over the
//              draft range. No physical cleanup is required for correctness:
//              reads are bounded by the cursor and every position the
//              auxiliary writes is re-written by the target at verification.
//
//   * state:   the auxiliary owns no tensors. state_write() emits a
//              descriptor that must match the live target store; a restore
//              mismatch throws so the caller falls back to re-prefilling
//              the auxiliary context (the established draft-restore path).
//
// Invariants (see also llama-kv-cache-kvarn.h, tail_groups comment)
// -----------
//   I1. The auxiliary writes exclusively at the frontier [t, t+k) with
//       k <= max_draft_tokens; the lossless F16 staging region always
//       covers [t, t+k) so no speculative row is ever sealed by the
//       auxiliary.
//   I2. Sealing is exclusively driven by the target's eager stores; the
//       auxiliary stores never move the committed fill (advance_fill=false)
//       and never complete records (eager_records=false).
//   I3. Reads are bounded by the auxiliary cursor and additionally by
//       seal_clamp_groups on the record domain, so stale rows after
//       rejections are unreachable.
//   I4. The scheduler thread serializes all cache access; each
//       llama_kv_cache_kvarn_context instance carries its own
//       llama_kvarn_borrow_config (target contexts never see borrow
//       configs).
//   I5. scope: unified single-stream target, non-SWA or SWA KVarN stores,
//       any auxiliary sequence count up to LLAMA_MAX_SEQ, no DSA/DSV4/
//       hybrid targets, no
//       context shifting (KVarN caches already refuse shifts).
class llama_kv_cache_kvarn_borrow : public llama_memory_i {
public:
    llama_kv_cache_kvarn_borrow(
            const llama_model & model,
            const llama_hparams & hparams,
            llama_kv_cache_kvarn * target,
            uint32_t n_ubatch,
            uint32_t n_seq_max,
            uint32_t max_draft_tokens);
    llama_kv_cache_kvarn_borrow(const llama_kv_cache_kvarn_borrow &) = delete;
    llama_kv_cache_kvarn_borrow & operator=(const llama_kv_cache_kvarn_borrow &) = delete;

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override { return false; }

    void clear(bool data) override;
    bool can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const override;
    seq_rm_capability get_seq_rm_capability() const override;
    bool seq_rm_plan(
            llama_seq_id seq_id, llama_pos p0, llama_pos p1,
            llama_pos & planned_p0, llama_pos & planned_p1) const override;
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    bool seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) override;
    int cells_at_pos(llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) override;
    void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    GGML_NORETURN void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    GGML_NORETURN void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;
    llama_kv_memory_stats kv_memory_stats() const override { return {}; }
    ggml_type get_kv_tail_type() const override;

    bool requires_state_for_partial_restore() const override;
    bool state_seq_can_save(llama_seq_id seq_id) const override;
    bool state_seq_can_restore(llama_seq_id seq_id) const override;
    bool state_seq_can_save(llama_seq_id seq_id, llama_state_seq_flags flags) const override;
    bool state_seq_can_restore(llama_seq_id seq_id, llama_state_seq_flags flags) const override;
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read(llama_io_read_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    // Auxiliary logical cursor in the shared position space.
    uint32_t get_used_pos() const { return used_pos; }

private:
    llama_kvarn_borrow_config make_config() const;

    llama_kv_cache_kvarn * target;
    const uint32_t n_seq_max;
    const uint32_t max_draft_tokens;
    uint32_t used_pos = 0; // aux logical cursor (positions 0..used_pos authored)
};

// No-op context for init_update(): borrows never carry pending updates.
class llama_kv_cache_kvarn_borrow_noop_context : public llama_memory_context_i {
public:
    llama_kv_cache_kvarn_borrow_noop_context();
    bool next() override { return false; }
    bool apply() override { return true; }
    const llama_ubatch & get_ubatch() const override;
    llama_memory_status get_status() const override { return LLAMA_MEMORY_STATUS_NO_UPDATE; }
};