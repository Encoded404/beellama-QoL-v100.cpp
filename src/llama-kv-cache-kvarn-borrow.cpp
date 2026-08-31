#include "llama-kv-cache-kvarn-borrow.h"

#include "llama-kv-cache-iswa.h"
#include "llama-io.h"
#include "llama-kv-cache-kvarn.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <cstdlib>
#include <stdexcept>

// ---------------------------------------------------------------------------
// KVarN store probing
// ---------------------------------------------------------------------------

const llama_kv_cache_kvarn * llama_memory_probe_kvarn(const llama_memory_i * mem) {
    if (mem == nullptr) {
        return nullptr;
    }
    if (const auto * kvarn = dynamic_cast<const llama_kv_cache_kvarn *>(mem)) {
        return kvarn;
    }
    if (const auto * iswa = dynamic_cast<const llama_kv_cache_iswa *>(mem)) {
        if (const auto * base = dynamic_cast<const llama_kv_cache_kvarn *>(iswa->get_base())) {
            return base;
        }
        if (const auto * swa = dynamic_cast<const llama_kv_cache_kvarn *>(iswa->get_swa())) {
            return swa;
        }
    }
    return nullptr;
}

bool llama_kvarn_aux_borrow_supported(
        const llama_memory_i * mem_other,
        uint32_t n_seq_max) {
    if (llama_kvarn_borrow_disabled_by_env()) {
        return false;
    }
    // The store tracks per-sequence membership like the plain caches, so
    // any number of auxiliary sequences up to the global cap is supported
    // (multi-slot servers and block-style drafters). The aux's writes all
    // stay inside the frontier window (invariant I1).
    if (n_seq_max == 0 || n_seq_max > LLAMA_MAX_SEQ) {
        return false;
    }
    const llama_kv_cache_kvarn * kvarn = llama_memory_probe_kvarn(mem_other);
    if (kvarn == nullptr) {
        return false;
    }
    // v1 borrows the shared (unified) stream only.
    if (kvarn->get_kv_n_stream() != 1) {
        return false;
    }
    // The speculative rows must stay inside the lossless F16 staging region
    // until the target re-verifies them. stage_groups >= 2 is the minimum
    // the rollback-safe staging layout guarantees.
    if (kvarn->get_stage_groups() < 2) {
        return false;
    }
    return true;
}

bool llama_kvarn_borrow_disabled_by_env() {
    const char * env = std::getenv("GGML_KVARN_BORROW");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

// ---------------------------------------------------------------------------
// Borrowing auxiliary KV memory
// ---------------------------------------------------------------------------
//
// Semantics
// ---------
// The auxiliary context (draft/MTP/assistant) shares ONE position-addressed
// KVarN store with the target context:
//
//   * reads:   every attention op of the auxiliary graph resolves record,
//              staging and tail spans from the target store using the
//              auxiliary's own logical cursor (target verified frontier +
//              draft rows). Record-domain reads are clamped to the target's
//              last fully verified record group (seal_clamp_groups) so
//              speculative rows that still live in the F16 staging region
//              are never read through unsealed record groups.
//
//   * writes:  the auxiliary ubatch rows are stored with eager_records =
//              false and advance_fill = false: stage (and tail) rows are
//              written at their positions, but no record group is sealed
//              and the stream's committed fill only moves through the
//              target's own eager stores at verification.
//
//   * rollback: rejected draft rows are discarded with seq_rm over the
//              draft range. No physical cleanup is required for correctness:
//              reads are bounded by the cursor, and every position the
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

llama_kv_cache_kvarn_borrow::llama_kv_cache_kvarn_borrow(
        const llama_model & /*model*/,
        const llama_hparams & /*hparams*/,
        llama_kv_cache_kvarn * target,
        uint32_t /*n_ubatch*/,
        uint32_t n_seq_max,
        uint32_t max_draft_tokens) :
    target(target),
    n_seq_max(n_seq_max),
    max_draft_tokens(max_draft_tokens) {
    GGML_ASSERT(target != nullptr);
    GGML_ASSERT(n_seq_max >= 1 && n_seq_max <= LLAMA_MAX_SEQ &&
            "KVarN borrow supports at least one auxiliary sequence up to the global cap");
    // I1: the draft window must fit inside the lossless F16 staging region.
    // Non-SWA staging depth is stage_groups*128 tokens; SWA keeps the fixed
    // local tail. Requiring max_draft_tokens <= stage_groups*KVAR_N_GROUP
    // keeps every speculative row in F16 even for the weakest layout.
    GGML_ASSERT(max_draft_tokens <= target->get_stage_groups() * KVAR_N_GROUP &&
            "KVarN borrow window exceeds the lossless F16 staging region");
}

llama_kvarn_borrow_config llama_kv_cache_kvarn_borrow::make_config() const {
    llama_kvarn_borrow_config cfg;
    // Seal clamp: the last fully VERIFIED record group = target frontier /
    // 128. The target's own writes define the frontier (eager stores), and
    // the auxiliary never advances it.
    const llama_pos verified = target->seq_pos_max(/* seq_id */ 0);
    if (verified > 0) {
        cfg.seal_clamp_groups = uint32_t((verified + 1) / int64_t(KVAR_N_GROUP));
    }
    cfg.eager_records = false;
    cfg.advance_fill  = false;
    return cfg;
}

llama_memory_context_ptr llama_kv_cache_kvarn_borrow::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    // The shared store is driven through the target's own batch machinery;
    // the aux ubatch positions [t, t+k) map onto the shared cells exactly
    // like the plain ctx_other sharing path does.
    auto ctx = target->init_batch(balloc, n_ubatch, embd_all);
    if (ctx != nullptr) {
        if (auto * kctx = dynamic_cast<llama_kv_cache_kvarn_context *>(ctx.get())) {
            const llama_kvarn_borrow_config cfg = make_config();
            kctx->set_borrow(cfg);
            // The cursor is the last verified target frontier at batch start;
            // the ubatch rows [frontier, frontier+n) are added by apply().
            used_pos = uint32_t(std::max<llama_pos>(0, target->seq_pos_max(0)));
        }
    }
    return ctx;
}

llama_memory_context_ptr llama_kv_cache_kvarn_borrow::init_full() {
    // The auxiliary owns no storage, so the worst-case compute buffer plan
    // comes from the shared store with the same borrow clamp.
    auto ctx = target->init_full();
    if (ctx != nullptr) {
        if (auto * kctx = dynamic_cast<llama_kv_cache_kvarn_context *>(ctx.get())) {
            kctx->set_borrow(make_config());
        }
    }
    return ctx;
}

llama_memory_context_ptr llama_kv_cache_kvarn_borrow::init_update(
        llama_context * /*lctx*/, bool /*optimize*/) {
    // Borrows never carry pending updates; the target context owns all
    // stream copies and shifts (KVARN refuses shifts anyway).
    return std::make_unique<llama_kv_cache_kvarn_borrow_noop_context>();
}



// No-op context for init_update(): borrows never carry pending updates.
llama_kv_cache_kvarn_borrow_noop_context::llama_kv_cache_kvarn_borrow_noop_context() = default;

const llama_ubatch & llama_kv_cache_kvarn_borrow_noop_context::get_ubatch() const {
    static const llama_ubatch empty{};
    return empty;
}

void llama_kv_cache_kvarn_borrow::clear(bool /*data*/) {
    // The auxiliary owns no storage: clearing drops its cursor only. Shared
    // rows are authoritative target content and must survive; stale draft
    // rows are unreachable (I3) and are re-written by the target.
    used_pos = 0;
}

bool llama_kv_cache_kvarn_borrow::can_seq_rm(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    return target->can_seq_rm(seq_id, p0, p1);
}

llama_memory_i::seq_rm_capability llama_kv_cache_kvarn_borrow::get_seq_rm_capability() const {
    // Suffix rollbacks (draft rejections) are the primary op; the target
    // store supports arbitrary ranges.
    return target->get_seq_rm_capability();
}

bool llama_kv_cache_kvarn_borrow::seq_rm_plan(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1,
        llama_pos & planned_p0, llama_pos & planned_p1) const {
    return target->seq_rm_plan(seq_id, p0, p1, planned_p0, planned_p1);
}

bool llama_kv_cache_kvarn_borrow::seq_rm(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool ok = target->seq_rm(seq_id, p0, p1);
    if (ok && seq_id == 0) {
        // Rollback: the shared cells/rows in [p0, p1) are released by the
        // store; the auxiliary cursor must not extend past the surviving
        // prefix. Positions >= p0 will be re-authored by the target's
        // verification writes.
        used_pos = std::min(used_pos, uint32_t(std::max<llama_pos>(p0, 0)));
    }
    return ok;
}

bool llama_kv_cache_kvarn_borrow::seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) {
    // Cell-level removal on the shared store; the cursor clamp is handled by
    // the positional seq_rm path used by draft rejection flows.
    return target->seq_rm_cell(seq_id, cell_idx);
}

int llama_kv_cache_kvarn_borrow::cells_at_pos(
        llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) {
    return target->cells_at_pos(seq_id, pos, cell_indices, n_max);
}

void llama_kv_cache_kvarn_borrow::seq_cp(
        llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    target->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_kvarn_borrow::seq_keep(llama_seq_id seq_id) {
    target->seq_keep(seq_id);
}

GGML_NORETURN void llama_kv_cache_kvarn_borrow::seq_add(
        llama_seq_id, llama_pos, llama_pos, llama_pos) {
    GGML_ABORT("KVarN caches do not support context shifting");
}

GGML_NORETURN void llama_kv_cache_kvarn_borrow::seq_div(
        llama_seq_id, llama_pos, llama_pos, int) {
    GGML_ABORT("KVarN caches do not support context shifting");
}

llama_pos llama_kv_cache_kvarn_borrow::seq_pos_min(llama_seq_id seq_id) const {
    return target->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_kvarn_borrow::seq_pos_max(llama_seq_id seq_id) const {
    return target->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_kvarn_borrow::memory_breakdown() const {
    // The auxiliary owns no tensors; memory is accounted on the target.
    return {};
}

ggml_type llama_kv_cache_kvarn_borrow::get_kv_tail_type() const {
    return target->get_kv_tail_type();
}

bool llama_kv_cache_kvarn_borrow::requires_state_for_partial_restore() const {
    return target->requires_state_for_partial_restore();
}

bool llama_kv_cache_kvarn_borrow::state_seq_can_save(llama_seq_id seq_id) const {
    // The auxiliary owns no per-sequence state beyond the shared cursor; the
    // descriptor is bound to the target store, so any sequence may be
    // checkpointed (multi-slot servers checkpoint each slot's aux state).
    return seq_id >= -1;
}

bool llama_kv_cache_kvarn_borrow::state_seq_can_restore(llama_seq_id seq_id) const {
    return seq_id >= -1;
}

bool llama_kv_cache_kvarn_borrow::state_seq_can_save(llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);
    return state_seq_can_save(seq_id);
}

bool llama_kv_cache_kvarn_borrow::state_seq_can_restore(llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);
    return state_seq_can_restore(seq_id);
}

void llama_kv_cache_kvarn_borrow::state_write(
        llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);
    // The auxiliary owns no physical state, so the descriptor only pins the
    // borrow contract to the target store. Restore runs after the target
    // context restored its own state (server ordering), then validates.
    const uint32_t magic = LLAMA_KVARN_BORROW_MAGIC;
    const uint32_t version = LLAMA_KVARN_BORROW_VERSION;
    const uint32_t key_bits = target->get_key_bits();
    const uint32_t value_bits = target->get_value_bits();
    const uint32_t stage_groups = target->get_stage_groups();
    const uint32_t tail_groups = target->get_tail_groups();
    const uint32_t groups_per_stream = target->get_record_groups_per_stream();
    const uint32_t exact_tail_tokens = target->get_exact_tail_tokens();
    const uint32_t kv_size = target->get_kv_size();
    const uint32_t n_stream = target->get_kv_n_stream();
    const llama_pos verified = seq_id < 0 ? -1 : target->seq_pos_max(seq_id);

    io.write(&magic, sizeof(magic));
    io.write(&version, sizeof(version));
    io.write(&key_bits, sizeof(key_bits));
    io.write(&value_bits, sizeof(value_bits));
    io.write(&stage_groups, sizeof(stage_groups));
    io.write(&tail_groups, sizeof(tail_groups));
    io.write(&groups_per_stream, sizeof(groups_per_stream));
    io.write(&exact_tail_tokens, sizeof(exact_tail_tokens));
    io.write(&kv_size, sizeof(kv_size));
    io.write(&n_stream, sizeof(n_stream));
    io.write(&n_seq_max, sizeof(n_seq_max));
    io.write(&max_draft_tokens, sizeof(max_draft_tokens));
    io.write(&verified, sizeof(verified));
    io.write(&used_pos, sizeof(used_pos));
}

void llama_kv_cache_kvarn_borrow::state_read(
        llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t key_bits = 0;
    uint32_t value_bits = 0;
    uint32_t stage_groups = 0;
    uint32_t tail_groups = 0;
    uint32_t groups_per_stream = 0;
    uint32_t exact_tail_tokens = 0;
    uint32_t kv_size = 0;
    uint32_t n_stream = 0;
    llama_pos verified = -1;
    uint32_t saved_used_pos = 0;
    uint32_t saved_n_seq_max = 0;
    uint32_t saved_max_draft_tokens = 0;

    io.read(&magic, sizeof(magic));
    if (magic != LLAMA_KVARN_BORROW_MAGIC) {
        throw std::runtime_error("KVarN borrow state has an invalid magic");
    }
    io.read(&version, sizeof(version));
    if (version != LLAMA_KVARN_BORROW_VERSION) {
        throw std::runtime_error("KVarN borrow state has an unsupported version");
    }
    io.read(&key_bits, sizeof(key_bits));
    io.read(&value_bits, sizeof(value_bits));
    io.read(&stage_groups, sizeof(stage_groups));
    io.read(&tail_groups, sizeof(tail_groups));
    io.read(&groups_per_stream, sizeof(groups_per_stream));
    io.read(&exact_tail_tokens, sizeof(exact_tail_tokens));
    io.read(&kv_size, sizeof(kv_size));
    io.read(&n_stream, sizeof(n_stream));
    io.read(&saved_n_seq_max, sizeof(saved_n_seq_max));
    io.read(&saved_max_draft_tokens, sizeof(saved_max_draft_tokens));
    io.read(&verified, sizeof(verified));
    io.read(&saved_used_pos, sizeof(saved_used_pos));

    // Validate the descriptor against the live (already restored) target
    // store. Any mismatch means the borrow contract no longer holds; the
    // caller falls back to re-prefilling the auxiliary context.
    if (key_bits != target->get_key_bits() ||
            value_bits != target->get_value_bits() ||
            stage_groups != target->get_stage_groups() ||
            tail_groups != target->get_tail_groups() ||
            groups_per_stream != target->get_record_groups_per_stream() ||
            exact_tail_tokens != target->get_exact_tail_tokens() ||
            kv_size != target->get_kv_size() ||
            n_stream != target->get_kv_n_stream() ||
            saved_n_seq_max != n_seq_max ||
            saved_max_draft_tokens != max_draft_tokens) {
        throw std::runtime_error("KVarN borrow state does not match the target cache");
    }
    if (verified >= 0 && seq_id >= 0 && target->seq_pos_max(seq_id) != verified) {
        throw std::runtime_error("KVarN borrow verified frontier does not match the target cache");
    }
    used_pos = saved_used_pos;
}