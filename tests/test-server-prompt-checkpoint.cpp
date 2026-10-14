#include "../tools/server/server-task.h"

#undef NDEBUG
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <list>
#include <string>
#include <vector>

static constexpr size_t KIB = 1024;

static void speculative_rollback_checkpoint_boundary() {
    constexpr uint32_t reserve = 8;

    assert(!server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_NO, reserve, reserve + 1));
    assert(!server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_PART, reserve, reserve + 1));
    assert( server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_FULL, reserve, 1));
    assert(!server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_RS, reserve, 1));
    assert(!server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_RS, reserve, reserve));
    assert( server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_RS, reserve, reserve + 1));
    assert( server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_RS, 0, 1));

    assert(!server_prompt_reuse_requires_checkpoint_search(false, 0, 528));
    assert( server_prompt_reuse_requires_checkpoint_search(false, 512, 512));
    assert( server_prompt_reuse_requires_checkpoint_search(true, 0, 528));

    assert(server_prompt_reuse_alignment(0) == 1);
    assert(server_prompt_reuse_alignment(128) == 128);
    assert(server_prompt_reuse_alignment(192) == 192);

    assert(server_prompt_checkpoint_boundary(795, 132,   1) == 663);
    assert(server_prompt_checkpoint_boundary(795, 132, 128) == 640);
    assert(server_prompt_checkpoint_boundary(795,   4, 128) == 768);
    assert(server_prompt_checkpoint_boundary(3,     4, 128) == 0);

    assert(server_prompt_checkpoint_interval(131072, 32,    0, 128) == 4096);
    assert(server_prompt_checkpoint_interval(131072, 32, 8192, 128) == 8192);
    assert(server_prompt_checkpoint_interval(120000, 32,    0, 128) == 3840);
    assert(server_prompt_checkpoint_interval(0,      32, 8192, 128) == 0);
    assert(server_prompt_checkpoint_interval(131072,  0, 8192, 128) == 0);

    assert(!server_draft_context_owns_state(false, false));
    assert( server_draft_context_owns_state(true,  false));
    assert(!server_draft_context_owns_state(true,  true));
}

static server_prompt make_prompt(const llama_tokens & tokens) {
    server_prompt prompt;
    prompt.tokens = server_tokens(tokens, false);
    return prompt;
}

static server_prompt make_prompt_with_checkpoints(std::initializer_list<int64_t> n_tokens) {
    server_prompt prompt = make_prompt({1, 2, 3});
    for (const int64_t n : n_tokens) {
        prompt.checkpoints.emplace_back().n_tokens = n;
    }
    return prompt;
}

static std::list<common_prompt_checkpoint> make_checkpoints(std::initializer_list<int64_t> n_tokens) {
    std::list<common_prompt_checkpoint> checkpoints;
    for (const int64_t n : n_tokens) {
        checkpoints.emplace_back().n_tokens = n;
    }
    return checkpoints;
}

static std::vector<int64_t> checkpoint_positions(const std::list<common_prompt_checkpoint> & checkpoints) {
    std::vector<int64_t> result;
    for (const auto & checkpoint : checkpoints) {
        result.push_back(checkpoint.n_tokens);
    }
    return result;
}

static void checkpoint_max_step_cadence_is_off_by_default() {
    assert(!server_prompt_checkpoint_max_step_due(9000, 100, 0));
    assert(!server_prompt_checkpoint_max_step_due(9000, 100, -1));

    assert( server_prompt_checkpoint_max_step_due(2048, 0, 2048));
    assert(!server_prompt_checkpoint_max_step_due(2047, 0, 2048));
    assert( server_prompt_checkpoint_max_step_due(4096, 2048, 2048));
    assert(!server_prompt_checkpoint_max_step_due(3000, 2048, 2048));
}

static void checkpoint_max_step_cadence_seeds_the_first_checkpoint() {
    const std::list<common_prompt_checkpoint> empty;

    // an empty list anchors at the start of the prompt so the cadence can seed the
    // first checkpoint; the first batch still has nothing to capture
    assert(server_prompt_checkpoint_max_step_anchor(empty) == 0);
    assert(!server_prompt_checkpoint_max_step_due(
            0, server_prompt_checkpoint_max_step_anchor(empty), 4096));
    assert(server_prompt_checkpoint_max_step_due(
            4096, server_prompt_checkpoint_max_step_anchor(empty), 4096));

    // once a checkpoint exists the cadence measures from the newest one
    const auto checkpoints = make_checkpoints({2000});
    assert(server_prompt_checkpoint_max_step_anchor(checkpoints) == 2000);
    assert(!server_prompt_checkpoint_max_step_due(
            4000, server_prompt_checkpoint_max_step_anchor(checkpoints), 4096));
    assert(server_prompt_checkpoint_max_step_due(
            6096, server_prompt_checkpoint_max_step_anchor(checkpoints), 4096));

    // the cadence stays off when disabled, even with an empty list
    assert(!server_prompt_checkpoint_max_step_due(
            99999, server_prompt_checkpoint_max_step_anchor(empty), 0));
}

static void checkpoint_reuse_selection_prefers_latest_valid_boundary() {
    const auto prompt = make_prompt_with_checkpoints({4096, 12000, 18000});

    const auto pick = [&](int64_t n_tokens_lcp, int64_t n_tokens_new, int32_t alignment) {
        const auto it = prompt.find_reusable_checkpoint(n_tokens_lcp, n_tokens_new, alignment);
        return it == prompt.checkpoints.end() ? int64_t(-1) : it->n_tokens;
    };

    assert(pick(15000, 21000, 1) == 12000);
    assert(pick(20000, 21000, 1) == 18000);

    // a checkpoint past the lexical common prefix can never be restored
    assert(pick(4000, 21000, 1) == -1);

    // an identical prompt still has to evaluate one token for logits
    assert(pick(20000, 20000, 1) == 18000);

    // a zero-length checkpoint is not a usable restore point
    const auto zero = make_prompt_with_checkpoints({0, 100});
    const auto zero_it = zero.find_reusable_checkpoint(200, 300, 1);
    assert(zero_it != zero.checkpoints.end());
    assert(zero_it->n_tokens == 100);

    const auto empty = make_prompt_with_checkpoints({});
    assert(empty.find_reusable_checkpoint(200, 300, 1) == empty.checkpoints.end());
}

static void checkpoint_reuse_selection_respects_descriptor_alignment() {
    const auto prompt = make_prompt_with_checkpoints({4096, 9000});

    // dense caches may resume anywhere
    const auto any = prompt.find_reusable_checkpoint(10000, 20000, 1);
    assert(any != prompt.checkpoints.end());
    assert(any->n_tokens == 9000);

    // KVarN can only resume on a complete descriptor group, so the 9000 token
    // checkpoint is skipped in favour of the aligned 4096 one
    const auto aligned = prompt.find_reusable_checkpoint(10000, 20000, 128);
    assert(aligned != prompt.checkpoints.end());
    assert(aligned->n_tokens == 4096);

    // nothing at all when only the unaligned boundary would qualify
    const auto unaligned_only = make_prompt_with_checkpoints({9000});
    assert(unaligned_only.find_reusable_checkpoint(10000, 20000, 128) ==
            unaligned_only.checkpoints.end());
}

static void checkpoint_eviction_preserves_prompt_anchors() {
    {
        auto checkpoints = make_checkpoints({100, 300, 600, 900});
        const auto evict = server_prompt_evict_checkpoint(checkpoints);

        // the densest interior region is thinned, not the oldest anchor
        assert(evict->n_tokens == 300);

        checkpoints.erase(evict);
        assert(checkpoint_positions(checkpoints) == std::vector<int64_t>({100, 600, 900}));
    }

    {
        // a cluster near the start must not cost us the newest anchor
        auto checkpoints = make_checkpoints({6400, 6500, 6600, 12000});
        const auto evict = server_prompt_evict_checkpoint(checkpoints);

        assert(evict->n_tokens == 6500);

        checkpoints.erase(evict);
        assert(checkpoint_positions(checkpoints) == std::vector<int64_t>({6400, 6600, 12000}));
    }

    {
        // a cap of two has no interior to thin, so the oldest entry goes
        auto checkpoints = make_checkpoints({100, 900});
        assert(server_prompt_evict_checkpoint(checkpoints)->n_tokens == 100);
    }

    {
        auto checkpoints = make_checkpoints({100});
        assert(server_prompt_evict_checkpoint(checkpoints)->n_tokens == 100);
    }
}

static common_memory_seq_rm_result test_seq_rm_suffix(
        llama_seq_id seq_id,
        llama_pos requested_p0,
        const server_tokens & prompt_tokens,
        const common_memory_seq_rm_io & io,
        llama_pos & planned_p0) {
    const auto normalize_p0 = [&](llama_pos value) {
        return value > 0 ? prompt_tokens.pos_next(prompt_tokens.size_up_to_pos(value)) : value;
    };
    return common_memory_seq_rm_suffix(seq_id, requested_p0, io, normalize_p0, planned_p0);
}

static void prompt_cache_load_target_success_draft_failure_is_atomic() {
    server_prompt_cache cache(1, 0);
    server_prompt_cache_state saved;
    saved.prompt = make_prompt({1, 2, 3});
    saved.data.main.resize(16);
    saved.data.drft.resize(8);
    cache.states.push_back(std::move(saved));

    server_prompt current = make_prompt({9});
    current.checkpoints.emplace_back().data_tgt.resize(4);
    server_tokens requested(llama_tokens {1, 2, 4}, false);

    bool restored_main = false;
    bool restored_draft = false;
    bool cleared_main = false;
    bool cleared_draft = false;
    int main_state = 90;
    int draft_state = 80;
    server_prompt_cache_state_io io {
        /*.has_draft =*/ true,
        /*.has_speculative =*/ false,
        /*.restore_transaction =*/ [&](const uint8_t * main, size_t main_size,
                                       const uint8_t * drft, size_t drft_size,
                                       const uint8_t *, size_t) {
            server_prompt_restore_transaction_io tx {
                /*.restore_target =*/ true,
                /*.restore_draft =*/ true,
                /*.restore_speculative =*/ false,
                /*.prepare =*/ [&](server_prompt_state_kind kind, server_prompt_state_view state) {
                    if (kind == SERVER_PROMPT_STATE_MAIN) {
                        return state.data == main && state.size == main_size;
                    }
                    return state.data != drft || state.size != drft_size;
                },
                /*.commit =*/ [&](server_prompt_state_kind kind) {
                    if (kind == SERVER_PROMPT_STATE_MAIN) {
                        restored_main = true;
                        main_state = int(main_size);
                    } else {
                        restored_draft = true;
                        draft_state = int(drft_size);
                    }
                },
            };
            return server_prompt_restore_transaction(
                    { main, main_size }, { drft, drft_size }, {}, tx);
        },
    };

    assert(!cache.load(current, requested, 0, 1, io));
    assert(!restored_main && !restored_draft);
    assert(!cleared_main && !cleared_draft);
    assert(main_state == 90 && draft_state == 80);
    assert(cache.states.size() == 1);
    assert(current.tokens.size() == 1);
    assert(current.tokens[0] == 9);
    assert(current.checkpoints.size() == 1);
}

static void restore_transaction_validation_failures_are_atomic() {
    const server_prompt_state_view states[] = {
        { reinterpret_cast<const uint8_t *>("target"), 6 },
        { reinterpret_cast<const uint8_t *>("draft"), 5 },
        { reinterpret_cast<const uint8_t *>("spec"), 4 },
    };
    const server_prompt_state_kind kinds[] = {
        SERVER_PROMPT_STATE_MAIN,
        SERVER_PROMPT_STATE_DRAFT,
        SERVER_PROMPT_STATE_SPECULATIVE,
    };

    for (const auto failed_kind : kinds) {
        int prepared = 0;
        int committed = 0;
        server_prompt_restore_transaction_io io {
            /*.restore_target =*/ true,
            /*.restore_draft =*/ true,
            /*.restore_speculative =*/ true,
            /*.prepare =*/ [&](server_prompt_state_kind kind, server_prompt_state_view) {
                ++prepared;
                return kind != failed_kind;
            },
            /*.commit =*/ [&](server_prompt_state_kind) { ++committed; },
        };
        assert(!server_prompt_restore_transaction(states[0], states[1], states[2], io));
        assert(prepared >= 1 && prepared <= 3);
        assert(committed == 0);
    }
}

static void restore_transaction_validation_failure_identifies_prepare_leg() {
    const server_prompt_state_view states[] = {
        { reinterpret_cast<const uint8_t *>("target"), 6 },
        { reinterpret_cast<const uint8_t *>("draft"), 5 },
        { reinterpret_cast<const uint8_t *>("spec"), 4 },
    };
    const server_prompt_state_kind kinds[] = {
        SERVER_PROMPT_STATE_MAIN,
        SERVER_PROMPT_STATE_DRAFT,
        SERVER_PROMPT_STATE_SPECULATIVE,
    };

    for (const auto failed_kind : kinds) {
        int committed = 0;
        server_prompt_restore_transaction_io io {
            /*.restore_target =*/ true,
            /*.restore_draft =*/ true,
            /*.restore_speculative =*/ true,
            /*.prepare =*/ [&](server_prompt_state_kind kind, server_prompt_state_view) {
                return kind != failed_kind;
            },
            /*.commit =*/ [&](server_prompt_state_kind) { ++committed; },
        };
        const auto result = server_prompt_restore_transaction_diagnostic(
                states[0], states[1], states[2], io);
        assert(!result.success);
        assert(result.component == failed_kind);
        assert(result.reason == SERVER_PROMPT_RESTORE_PREPARE_REJECTED);
        assert(committed == 0);
    }
}

static void prompt_cache_ranks_safe_restorable_prefix_before_lexical_lcp() {
    server_prompt_cache cache(1, 0);
    server_prompt_cache_state saved;
    saved.prompt = make_prompt({1, 2});
    saved.data.main = {0x2a};
    cache.states.push_back(std::move(saved));

    // The live slot has the larger lexical prefix, but no durable checkpoint
    // at the divergent boundary.  The self-contained RAM prompt is fully
    // restorable and must therefore win despite its shorter lexical LCP.
    server_prompt current = make_prompt({1, 2, 3, 4});
    server_tokens requested(llama_tokens {1, 2, 3, 9}, false);
    bool restored = false;
    server_prompt_cache_state_io io {
        /*.has_draft =*/ false,
        /*.has_speculative =*/ false,
        /*.restore_transaction =*/ [&](const uint8_t * main, size_t main_size,
                                       const uint8_t *, size_t,
                                       const uint8_t *, size_t) {
            restored = main_size == 1 && main[0] == 0x2a;
            return restored;
        },
    };

    assert(cache.load(current, requested, 0, 128, io));
    assert(restored);
    assert(current.tokens.size() == 2);
}

static void prompt_cache_does_not_restore_just_admitted_live_state() {
    server_prompt_cache cache(1, 0);
    server_prompt current = make_prompt({1, 2, 3, 4});
    server_prompt_data data;
    data.main = {0x2a};
    const auto * admitted = cache.insert(current, std::move(data));
    assert(admitted != nullptr);

    bool restored = false;
    server_prompt_cache_state_io io {
        /*.has_draft =*/ false,
        /*.has_speculative =*/ false,
        /*.restore_transaction =*/ [&](const uint8_t *, size_t,
                                       const uint8_t *, size_t,
                                       const uint8_t *, size_t) {
            restored = true;
            return true;
        },
    };
    server_tokens requested(llama_tokens {1, 2, 9}, false);
    assert(cache.load(current, requested, 0, 1, io, admitted));
    assert(!restored);
    assert(cache.restore_attempts == 0);
    assert(current.tokens.size() == 4);
}

static void checkpoint_failed_target_save_cannot_reuse_stale_bytes() {
    common_prompt_checkpoint checkpoint;
    checkpoint.data_tgt.resize(32, 0x5a);

    checkpoint.update_tgt(nullptr, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    assert(checkpoint.data_tgt.empty());
    assert(checkpoint.empty());
}

static void speculative_draft_rollback_uses_draft_axis_and_recovers() {
    assert(server_speculative_draft_rollback_p0(4096, 63) == 4096);
    assert(server_speculative_draft_rollback_p0(0, 63) == 64);

    std::vector<std::pair<llama_pos, llama_pos>> removals;
    server_speculative_draft_rollback_io exact_io {
        /*.plan =*/ {},
        /*.remove =*/ [&](llama_pos p0, llama_pos p1) {
            removals.emplace_back(p0, p1);
            return true;
        },
    };
    llama_pos applied_p0 = -1;
    assert(server_speculative_draft_rollback(4096, exact_io, applied_p0) ==
            SERVER_SPECULATIVE_DRAFT_ROLLBACK_EXACT);
    const std::vector<std::pair<llama_pos, llama_pos>> exact_expected = {{4096, -1}};
    assert(applied_p0 == 4096 && removals == exact_expected);

    removals.clear();
    server_speculative_draft_rollback_io widened_io {
        /*.plan =*/ [](llama_pos p0, llama_pos p1, llama_pos & planned_p0, llama_pos & planned_p1) {
            assert(p0 == 4096 && p1 == -1);
            planned_p0 = 3968;
            planned_p1 = -1;
            return true;
        },
        /*.remove =*/ [&](llama_pos p0, llama_pos p1) {
            removals.emplace_back(p0, p1);
            return p0 == 3968 && p1 == -1;
        },
    };
    assert(server_speculative_draft_rollback(4096, widened_io, applied_p0) ==
            SERVER_SPECULATIVE_DRAFT_ROLLBACK_WIDENED);
    assert(applied_p0 == 3968);
    const std::vector<std::pair<llama_pos, llama_pos>> widened_expected = {{4096, -1}, {3968, -1}};
    assert(removals == widened_expected);

    removals.clear();
    server_speculative_draft_rollback_io clear_io {
        /*.plan =*/ [](llama_pos, llama_pos, llama_pos & planned_p0, llama_pos & planned_p1) {
            planned_p0 = -1;
            planned_p1 = -1;
            return true;
        },
        /*.remove =*/ [&](llama_pos p0, llama_pos p1) {
            removals.emplace_back(p0, p1);
            return p0 == -1 && p1 == -1;
        },
    };
    assert(server_speculative_draft_rollback(4096, clear_io, applied_p0) ==
            SERVER_SPECULATIVE_DRAFT_ROLLBACK_CLEARED);
    assert(applied_p0 == -1);
    const std::vector<std::pair<llama_pos, llama_pos>> clear_expected = {{4096, -1}, {-1, -1}};
    assert(removals == clear_expected);

    server_speculative_draft_rollback_io failed_io {
        /*.plan =*/ [](llama_pos, llama_pos, llama_pos & planned_p0, llama_pos & planned_p1) {
            planned_p0 = 3968;
            planned_p1 = -1;
            return true;
        },
        /*.remove =*/ [](llama_pos, llama_pos) { return false; },
    };
    assert(server_speculative_draft_rollback(4096, failed_io, applied_p0) ==
            SERVER_SPECULATIVE_DRAFT_ROLLBACK_FAILED);
}

static void server_unsupported_removal_falls_back_to_full_reprocess() {
    server_tokens prompt_tokens(llama_tokens(5626, 1), false);
    int partial_removals = 0;
    int full_clears = 0;
    common_memory_seq_rm_io io {
        /*.has_draft =*/ true,
        /*.plan =*/ [](common_memory_context_kind kind, llama_seq_id, llama_pos, llama_pos,
                       llama_pos & planned_p0, llama_pos & planned_p1) {
            if (kind == COMMON_MEMORY_CONTEXT_DRAFT) {
                return false;
            }
            planned_p0 = 5504;
            planned_p1 = -1;
            return true;
        },
        /*.can_remove =*/ [](common_memory_context_kind, llama_seq_id, llama_pos, llama_pos) { return true; },
        /*.remove =*/ [&](common_memory_context_kind, llama_seq_id, llama_pos p0, llama_pos p1) {
            if (p0 == -1 && p1 == -1) {
                ++full_clears;
            } else {
                ++partial_removals;
            }
            return true;
        },
    };
    llama_pos planned_p0 = -1;
    const auto result = test_seq_rm_suffix(0, 5626, prompt_tokens, io, planned_p0);
    assert(result == COMMON_MEMORY_SEQ_RM_FULL_REPROCESS);
    assert(planned_p0 == 0);
    assert(partial_removals == 0);
    assert(full_clears == 2);
}

static void server_post_preflight_mutation_failure_clears_both_contexts() {
    server_tokens prompt_tokens(llama_tokens(5626, 1), false);
    bool main_partial = false;
    bool draft_partial = false;
    bool main_cleared = false;
    bool draft_cleared = false;
    common_memory_seq_rm_io io {
        /*.has_draft =*/ true,
        /*.plan =*/ [](common_memory_context_kind, llama_seq_id, llama_pos p0, llama_pos p1,
                       llama_pos & planned_p0, llama_pos & planned_p1) {
            planned_p0 = p0;
            planned_p1 = p1;
            return true;
        },
        /*.can_remove =*/ [](common_memory_context_kind, llama_seq_id, llama_pos, llama_pos) { return true; },
        /*.remove =*/ [&](common_memory_context_kind kind, llama_seq_id, llama_pos p0, llama_pos p1) {
            if (p0 == -1 && p1 == -1) {
                (kind == COMMON_MEMORY_CONTEXT_TARGET ? main_cleared : draft_cleared) = true;
                return true;
            }
            if (kind == COMMON_MEMORY_CONTEXT_TARGET) {
                main_partial = true;
                return true;
            }
            draft_partial = true;
            return false;
        },
    };
    llama_pos planned_p0 = -1;
    const auto result = test_seq_rm_suffix(0, 5626, prompt_tokens, io, planned_p0);
    assert(result == COMMON_MEMORY_SEQ_RM_MUTATION_FAILED);
    assert(main_partial && draft_partial);
    assert(main_cleared && draft_cleared);
}

static void server_planned_removal_preserves_atomic_media_chunks() {
    mtmd::input_chunks chunks(mtmd_test_create_input_chunks());
    server_tokens prompt_tokens(chunks, true);

    const llama_pos requested_p0 = prompt_tokens.pos_next();
    const llama_pos inside_media = 6;
    const llama_pos media_end = prompt_tokens.pos_next(prompt_tokens.size_up_to_pos(inside_media));
    assert(media_end > inside_media);

    llama_pos removed_p0 = -1;
    common_memory_seq_rm_io io {
        /*.has_draft =*/ false,
        /*.plan =*/ [&](common_memory_context_kind, llama_seq_id, llama_pos, llama_pos,
                       llama_pos & planned_p0, llama_pos & planned_p1) {
            planned_p0 = inside_media;
            planned_p1 = -1;
            return true;
        },
        /*.can_remove =*/ [](common_memory_context_kind, llama_seq_id, llama_pos, llama_pos) { return true; },
        /*.remove =*/ [&](common_memory_context_kind, llama_seq_id, llama_pos p0, llama_pos) {
            removed_p0 = p0;
            return true;
        },
    };

    llama_pos planned_p0 = -1;
    const auto result = test_seq_rm_suffix(0, requested_p0, prompt_tokens, io, planned_p0);
    assert(result == COMMON_MEMORY_SEQ_RM_APPLIED);
    assert(planned_p0 == media_end);
    assert(removed_p0 == media_end);
}

static void cache_phase_timings_are_public() {
    server_slot_stats stats;
    stats.cache_slot_ms = 12.5;
    stats.cache_ram_save_ms = 3.0;
    stats.cache_ram_load_ms = 7.0;
    stats.cache_ram_restore_prepare_ms = 5.0;
    stats.cache_ram_restore_commit_ms = 2.0;
    stats.cache_ram_update_ms = 0.5;
    stats.cache_checkpoint_restore_ms = 2.0;
    stats.cache_checkpoint_prepare_ms = 1.5;
    stats.cache_checkpoint_commit_ms = 0.5;

    const auto value = stats.to_json();
    assert(value.at("cache_slot_ms") == 12.5);
    assert(value.at("cache_ram_save_ms") == 3.0);
    assert(value.at("cache_ram_load_ms") == 7.0);
    assert(value.at("cache_ram_restore_prepare_ms") == 5.0);
    assert(value.at("cache_ram_restore_commit_ms") == 2.0);
    assert(value.at("cache_ram_update_ms") == 0.5);
    assert(value.at("cache_checkpoint_restore_ms") == 2.0);
    assert(value.at("cache_checkpoint_prepare_ms") == 1.5);
    assert(value.at("cache_checkpoint_commit_ms") == 0.5);
}

static void prompt_cache_snapshot_restore_evict_stress() {
    server_prompt_cache cache(0, 0);

    int logical_state = 0;
    server_prompt_cache_state_io io {
        /*.has_draft =*/ false,
        /*.has_speculative =*/ false,
        /*.restore_transaction =*/ [&](const uint8_t * main, size_t main_size,
                                       const uint8_t *, size_t,
                                       const uint8_t *, size_t) {
            assert(main_size > 0);
            logical_state = main[0];
            return true;
        },
    };

    for (int i = 0; i < 10000; ++i) {
        server_prompt source = make_prompt({i + 1, i + 20001});
        auto & checkpoint = source.checkpoints.emplace_back();
        checkpoint.n_tokens = 2;
        checkpoint.data_tgt.resize(4*KIB);

        auto * admitted = cache.alloc(source, 32*KIB, 0);
        assert(admitted != nullptr);
        admitted->data.main.front() = uint8_t(i);

        server_prompt destination;
        server_tokens requested(llama_tokens {i + 1, i + 20001, i + 40001}, false);
        assert(cache.load(destination, requested, 0, 1, io));
        assert(destination.tokens.size() == source.tokens.size());
        assert(logical_state == uint8_t(i));
        assert(cache.states.size() == 1);
        assert(cache.erase(admitted));
        assert(cache.states.empty());
        assert(cache.accounted_size() == 0);
    }

    assert(cache.admission_successes == 10000);
    assert(cache.restore_successes == 10000);
    assert(cache.admission_failures == 0);
    assert(cache.restore_failures == 0);
}

//
// automatic disk swap tier
//

static std::string disk_test_dir() {
    static int counter = 0;

    const char * env = getenv("SERVER_PROMPT_DISK_TEST_DIR");
    const std::filesystem::path base = env != nullptr
            ? std::filesystem::u8path(env)
            : std::filesystem::temp_directory_path();

    const std::filesystem::path dir = base / ("beellama-prompt-disk-test-" + std::to_string(counter++));

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    assert(!ec);

    return dir.u8string() + "/";
}

static void disk_test_remove(const std::string & dir) {
    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::u8path(dir), ec);
}

// Insert two entries large enough to exceed a 1 MiB RAM budget, which demotes the
// oldest one to disk. Returns the demoted file path.
static std::string disk_test_make_demoted(server_prompt_cache & cache) {
    assert(cache.disk.enabled());

    server_prompt_data data;
    data.main = std::vector<uint8_t>(700*KIB, 0x11);
    data.drft = std::vector<uint8_t>(64, 0x22);
    for (size_t i = 0; i < 64; ++i) {
        data.main[i] = uint8_t(i);
    }
    assert(cache.insert(make_prompt({1, 2, 3, 4}), std::move(data)) != nullptr);
    assert(cache.disk.bytes_on_disk() == 0);

    server_prompt_data data2;
    data2.main = std::vector<uint8_t>(700*KIB, 0x33);
    assert(cache.insert(make_prompt({5, 6, 7, 8}), std::move(data2)) != nullptr);

    // spills are published before their bytes land, so drain to observe files
    cache.disk_flush();

    assert(cache.states.size() == 2);
    assert(cache.states.front().disk.on_disk());
    assert(cache.disk.entries_on_disk() == 1);

    const std::string file = cache.states.front().disk.path;
    assert(!file.empty());
    assert(std::filesystem::exists(std::filesystem::u8path(file)));
    return file;
}

static void disk_spill_restore_round_trip() {
    const std::string dir = disk_test_dir();

    const std::vector<uint8_t> main_bytes = []{
        std::vector<uint8_t> v(700*KIB, 0x11);
        for (size_t i = 0; i < 64; ++i) {
            v[i] = uint8_t(i);
        }
        return v;
    }();
    const std::vector<uint8_t> drft_bytes(64, 0x22);

    {
        server_prompt_disk::config cfg;
        cfg.path      = dir;
        cfg.limit_mib = 64;

        server_prompt_cache cache(1, 0, cfg, 0xabcdef01u);

        server_prompt_data data;
        data.main = main_bytes;
        data.drft = drft_bytes;
        assert(cache.insert(make_prompt({1, 2, 3, 4}), std::move(data)) != nullptr);

        server_prompt_data data2;
        data2.main = std::vector<uint8_t>(700*KIB, 0x33);
        assert(cache.insert(make_prompt({5, 6, 7, 8}), std::move(data2)) != nullptr);

        // the RAM payload is handed to the disk tier at submission time
        assert(cache.states.size() == 2);
        const auto & demoted = cache.states.front();
        assert(demoted.disk.on_disk());
        assert(demoted.data.main.empty());
        assert(demoted.data.drft.empty());
        assert(cache.disk.bytes_on_disk() > 0);

        cache.disk_flush();
        assert(cache.disk.entries_on_disk() == 1);
        assert(cache.disk.spill_successes.load() == 1);
        assert(cache.disk.spill_failures.load() == 0);

        bool called = false;
        server_prompt_cache_state_io io {
            /*.has_draft =*/ true,
            /*.has_speculative =*/ false,
            /*.restore_transaction =*/ [&](const uint8_t * main, size_t main_size,
                                           const uint8_t * drft, size_t drft_size,
                                           const uint8_t *, size_t) {
                called = true;
                return main_size == main_bytes.size() &&
                        memcmp(main, main_bytes.data(), main_size) == 0 &&
                        drft_size == drft_bytes.size() &&
                        memcmp(drft, drft_bytes.data(), drft_size) == 0;
            },
        };

        server_prompt current = make_prompt({9});
        const server_tokens requested(llama_tokens {1, 2, 3, 4, 5, 6}, false);

        assert(cache.load(current, requested, 0, 1, io));
        assert(called);
        assert(current.n_tokens() == 4);
        assert(cache.restore_disk_successes.load() == 1);
        assert(cache.restore_successes == 1);
        assert(cache.restore_failures == 0);

        // the disk entry stays restorable and is not consumed by a read
        assert(cache.states.front().disk.on_disk());
    }

    disk_test_remove(dir);
}

static void disk_identity_mismatch_is_rejected() {
    const std::string dir = disk_test_dir();

    {
        server_prompt_disk::config cfg;
        cfg.path      = dir;
        cfg.limit_mib = 64;

        server_prompt_cache cache(1, 0, cfg, 0xabcdef01u);
        const std::string file = disk_test_make_demoted(cache);

        {
            std::fstream fs(std::filesystem::u8path(file), std::ios::in | std::ios::out | std::ios::binary);
            assert(fs.is_open());
            const uint64_t other = 0x0123456789abcdefull;
            fs.seekp(8);
            fs.write((const char *) &other, sizeof(other));
            fs.close();
        }

        bool called = false;
        server_prompt_cache_state_io io {
            /*.has_draft =*/ true,
            /*.has_speculative =*/ false,
            /*.restore_transaction =*/ [&](const uint8_t *, size_t, const uint8_t *, size_t, const uint8_t *, size_t) {
                called = true;
                return true;
            },
        };

        server_prompt current = make_prompt({9});
        const server_tokens requested(llama_tokens {1, 2, 3, 4, 5, 6}, false);

        assert(!cache.load(current, requested, 0, 1, io));
        assert(!called);
        // only the unusable disk entry is dropped; the RAM entry stays
        assert(cache.states.size() == 1);
        assert(!cache.states.front().disk.on_disk());
        assert(!std::filesystem::exists(std::filesystem::u8path(file)));
        assert(cache.disk.bytes_on_disk() == 0);
        assert(cache.disk.entries_on_disk() == 0);
        assert(cache.disk.restore_failures.load() >= 1);
    }

    disk_test_remove(dir);
}

enum class disk_corruption {
    magic,
    version,
    identity,
    offsets,
    payload_byte,
    truncate,
};

static void disk_corruption_is_rejected(disk_corruption kind) {
    const std::string dir = disk_test_dir();

    {
        server_prompt_disk::config cfg;
        cfg.path      = dir;
        cfg.limit_mib = 64;

        server_prompt_cache cache(1, 0, cfg, 0xabcdef01u);
        const std::string file = disk_test_make_demoted(cache);

        {
            std::fstream fs(std::filesystem::u8path(file), std::ios::in | std::ios::out | std::ios::binary);
            assert(fs.is_open());

            const auto put64 = [&](std::streamoff off, uint64_t v) {
                fs.seekp(off);
                fs.write((const char *) &v, sizeof(v));
            };
            const auto put32 = [&](std::streamoff off, uint32_t v) {
                fs.seekp(off);
                fs.write((const char *) &v, sizeof(v));
            };

            switch (kind) {
                case disk_corruption::magic:        put32(0, 0xdeadbeef);  break;
                case disk_corruption::version:      put32(4, 99);          break;
                case disk_corruption::identity:     put64(8, 0xfeedface);  break;
                case disk_corruption::offsets:      put64(32, 4096);       break;
                case disk_corruption::payload_byte: put32(128 + 3, 0xffffffff); break;
                case disk_corruption::truncate:     break;
            }
            fs.close();
        }

        if (kind == disk_corruption::truncate) {
            const auto size = std::filesystem::file_size(std::filesystem::u8path(file));
            assert(size > 1);
            std::filesystem::resize_file(std::filesystem::u8path(file), size - 1);
        }

        bool called = false;
        server_prompt_cache_state_io io {
            /*.has_draft =*/ true,
            /*.has_speculative =*/ false,
            /*.restore_transaction =*/ [&](const uint8_t *, size_t, const uint8_t *, size_t, const uint8_t *, size_t) {
                called = true;
                return true;
            },
        };

        server_prompt current = make_prompt({9});
        const server_tokens requested(llama_tokens {1, 2, 3, 4, 5, 6}, false);

        assert(!cache.load(current, requested, 0, 1, io));
        assert(!called);
        assert(cache.states.size() == 1);
        assert(!cache.states.front().disk.on_disk());
        assert(!std::filesystem::exists(std::filesystem::u8path(file)));
        assert(cache.disk.bytes_on_disk() == 0);
    }

    disk_test_remove(dir);
}

static void disk_truncated_and_corrupt_files_are_rejected() {
    for (const auto kind : {
            disk_corruption::magic,
            disk_corruption::version,
            disk_corruption::identity,
            disk_corruption::offsets,
            disk_corruption::payload_byte,
            disk_corruption::truncate }) {
        disk_corruption_is_rejected(kind);
    }
}

static void disk_budget_eviction() {
    const std::string dir = disk_test_dir();

    {
        server_prompt_disk::config cfg;
        cfg.path      = dir;
        cfg.limit_mib = 1;

        server_prompt_cache cache(1, 0, cfg, 0xabcdef01u);

        for (llama_token i = 0; i < 3; ++i) {
            server_prompt_data data;
            data.main = std::vector<uint8_t>(600*KIB, uint8_t(i));
            assert(cache.insert(make_prompt({i, i + 100, i + 200, i + 300}), std::move(data)) != nullptr);
        }

        cache.disk_flush();
        assert(cache.disk.bytes_on_disk() <= cache.disk.limit_bytes());
        assert(cache.accounted_size() <= cache.limit_size);
        assert(cache.disk.entries_on_disk() == 1);

        size_t n_disk = 0;
        for (const auto & state : cache.states) {
            n_disk += state.disk.on_disk() ? 1 : 0;
        }
        assert(n_disk == cache.disk.entries_on_disk());
    }

    disk_test_remove(dir);
}

static void disk_checkpoint_appendix_round_trip() {
    const std::string dir = disk_test_dir();

    {
        server_prompt_disk::config cfg;
        cfg.path      = dir;
        cfg.limit_mib = 64;

        server_prompt_cache cache(1, 0, cfg, 0xabcdef01u);

        server_prompt prompt = make_prompt({1, 2, 3, 4});

        // one durable checkpoint with a payload on every axis
        auto & ckpt = prompt.checkpoints.emplace_back();
        ckpt.n_tokens = 2;
        ckpt.pos_min  = 0;
        ckpt.pos_max  = 1;
        ckpt.data_tgt.resize(256);
        ckpt.data_dft.resize(128);
        ckpt.data_spec.resize(64);
        for (size_t i = 0; i < ckpt.data_tgt.size(); ++i) {
            ckpt.data_tgt.data()[i] = uint8_t(i);
        }

        server_prompt_data data;
        data.main = std::vector<uint8_t>(700*KIB, 0x11);
        data.drft = std::vector<uint8_t>(64, 0x22);
        assert(cache.insert(prompt, std::move(data)) != nullptr);

        server_prompt_data data2;
        data2.main = std::vector<uint8_t>(700*KIB, 0x33);
        assert(cache.insert(make_prompt({5, 6, 7, 8}), std::move(data2)) != nullptr);

        const auto & demoted = cache.states.front();
        assert(demoted.disk.on_disk());
        assert(demoted.disk.size_ckpt > 0);
        // checkpoint payloads are released together with the main payload
        assert(demoted.prompt.checkpoints.size() == 1);
        assert(demoted.prompt.checkpoints.front().empty());

        bool called = false;
        server_prompt_cache_state_io io {
            /*.has_draft =*/ true,
            /*.has_speculative =*/ false,
            /*.restore_transaction =*/ [&](const uint8_t *, size_t, const uint8_t *, size_t, const uint8_t *, size_t) {
                called = true;
                return true;
            },
        };

        server_prompt current = make_prompt({9});
        const server_tokens requested(llama_tokens {1, 2, 3, 4, 5, 6}, false);

        assert(cache.load(current, requested, 0, 1, io));
        assert(called);

        assert(current.checkpoints.size() == 1);
        const auto & restored = current.checkpoints.front();
        assert(restored.n_tokens == 2);
        assert(restored.pos_min == 0);
        assert(restored.pos_max == 1);
        assert(restored.data_tgt.size() == 256);
        assert(restored.data_dft.size() == 128);
        assert(restored.data_spec.size() == 64);
        assert(restored.data_tgt.data()[7] == 7);
    }

    disk_test_remove(dir);
}

static void disk_erase_and_destructor_clean_the_directory() {
    const std::string dir = disk_test_dir();

    std::string file;
    {
        server_prompt_disk::config cfg;
        cfg.path      = dir;
        cfg.limit_mib = 64;

        server_prompt_cache cache(1, 0, cfg, 0xabcdef01u);
        file = disk_test_make_demoted(cache);

        assert(cache.erase(&cache.states.front()));
        assert(cache.states.size() == 1);
        assert(!std::filesystem::exists(std::filesystem::u8path(file)));
        assert(cache.disk.bytes_on_disk() == 0);
    }

    // nothing left behind: the store wipes its own files on destruction
    for (const auto & entry : std::filesystem::directory_iterator(std::filesystem::u8path(dir))) {
        const std::string name = entry.path().filename().u8string();
        assert(!(name.size() > 4 && name.compare(0, 4, "pcd-") == 0));
    }

    disk_test_remove(dir);
}

static void disk_disabled_keeps_eviction_behaviour() {
    const std::string dir = disk_test_dir();

    {
        server_prompt_cache cache(1, 0);
        assert(!cache.disk.enabled());

        server_prompt_data data;
        data.main = std::vector<uint8_t>(700*KIB, 0x11);
        assert(cache.insert(make_prompt({1, 2, 3, 4}), std::move(data)) != nullptr);

        server_prompt_data data2;
        data2.main = std::vector<uint8_t>(700*KIB, 0x33);
        assert(cache.insert(make_prompt({5, 6, 7, 8}), std::move(data2)) != nullptr);

        // with the tier disabled the oldest entry is dropped, not demoted
        assert(cache.states.size() == 1);
        assert(cache.accounted_size() <= cache.limit_size);
        assert(cache.disk.bytes_on_disk() == 0);
    }

    disk_test_remove(dir);
}

static void disk_mapping_modes_agree() {
    const std::string dir = disk_test_dir();

    std::vector<uint8_t> main_bytes(4096);
    for (size_t i = 0; i < main_bytes.size(); ++i) {
        main_bytes[i] = uint8_t(i*7 + 1);
    }
    const std::vector<uint8_t> drft_bytes = { 1, 2, 3, 4 };
    const std::vector<uint8_t> spec_bytes = { 9, 8, 7 };

    std::list<common_prompt_checkpoint> checkpoints;
    {
        auto & ckpt = checkpoints.emplace_back();
        ckpt.n_tokens = 2;
        ckpt.pos_min  = 0;
        ckpt.pos_max  = 1;
        ckpt.data_tgt.resize(32);
        ckpt.data_dft.resize(16);
    }

    for (const bool use_mmap : { false, true }) {
        server_prompt_disk::config cfg;
        cfg.path      = dir;
        cfg.limit_mib = 64;
        cfg.use_mmap  = use_mmap;

        server_prompt_disk::store store(cfg, 0x1111222233334444ull);

        server_prompt_disk::entry entry;
        assert(store.spill(entry,
                    main_bytes.data(), main_bytes.size(),
                    drft_bytes.data(), drft_bytes.size(),
                    spec_bytes.data(), spec_bytes.size(),
                    checkpoints, 4, 0xabc));
        assert(entry.on_disk());

        auto m = store.open(entry);
        assert(m != nullptr);

        // the requested mode is used whenever the platform supports mapping
        assert(m->mapped() == (use_mmap && server_prompt_disk::store::mapping_supported()));

        assert(m->main_size() == main_bytes.size());
        assert(memcmp(m->main(), main_bytes.data(), main_bytes.size()) == 0);
        assert(m->drft_size() == drft_bytes.size());
        assert(memcmp(m->drft(), drft_bytes.data(), drft_bytes.size()) == 0);
        assert(m->spec_size() == spec_bytes.size());
        assert(memcmp(m->spec(), spec_bytes.data(), spec_bytes.size()) == 0);

        std::list<common_prompt_checkpoint> parsed;
        size_t n_consumed = 0;
        assert(server_prompt_disk::read_checkpoints(m->ckpt(), m->ckpt_size(), parsed, SIZE_MAX, &n_consumed));
        assert(n_consumed == m->ckpt_size());
        assert(parsed.size() == 1);
        assert(parsed.front().n_tokens == 2);
        assert(parsed.front().data_tgt.size() == 32);
        assert(parsed.front().data_dft.size() == 16);

        // releasing the mapping must not invalidate the entry
        m.reset();
        assert(entry.on_disk());
        assert(store.bytes_on_disk() > 0);

        // a read that cannot be served must fail closed and invalidate the entry
        std::filesystem::remove(std::filesystem::u8path(entry.path));
        assert(store.open(entry) == nullptr);
        assert(!entry.on_disk());
        assert(store.bytes_on_disk() == 0);
    }

    disk_test_remove(dir);
}

static void disk_async_lifecycle() {
    // erasing a published entry whose write is still in flight must not leak a file
    {
        const std::string dir = disk_test_dir();

        server_prompt_disk::config cfg;
        cfg.path         = dir;
        cfg.limit_mib    = 64;
        cfg.async_writes = true;

        server_prompt_disk::store store(cfg, 0xabcdef01u);

        server_prompt_disk::entry entry;
        std::vector<uint8_t> main(256*KIB, 0x77);
        assert(store.spill(entry, std::move(main), {}, {}, {}, 4, 0x1));

        // the entry is published immediately and the writer owns the bytes
        assert(entry.on_disk());
        assert(store.entries_on_disk() == 1);

        store.erase(entry);
        assert(!entry.on_disk());

        // whatever the writer had done, draining must leave nothing behind
        store.flush();

        size_t n_files = 0;
        for (const auto & de : std::filesystem::directory_iterator(std::filesystem::u8path(dir))) {
            const std::string name = de.path().filename().u8string();
            if (name.size() > 4 && name.compare(0, 4, "pcd-") == 0) {
                ++n_files;
            }
        }
        assert(n_files == 0);
        assert(store.bytes_on_disk() == 0);
        assert(store.entries_on_disk() == 0);

        disk_test_remove(dir);
    }

    // reading a published entry before its bytes land must drain, not fail
    {
        const std::string dir = disk_test_dir();

        server_prompt_disk::config cfg;
        cfg.path         = dir;
        cfg.limit_mib    = 64;
        cfg.async_writes = true;

        server_prompt_disk::store store(cfg, 0xabcdef01u);

        std::vector<uint8_t> main(512*KIB, 0x11);
        for (size_t i = 0; i < 64; ++i) {
            main[i] = uint8_t(i);
        }

        server_prompt_disk::entry entry;
        assert(store.spill(entry, std::move(main), {}, {}, {}, 4, 0x2));

        auto m = store.open(entry);
        assert(m != nullptr);
        assert(m->main_size() == 512*KIB);
        assert(m->main()[7] == 7);
        m.reset();

        store.flush();
        assert(store.spill_successes.load() == 1);
        assert(store.spill_failures.load() == 0);

        disk_test_remove(dir);
    }
}

static void disk_read_checkpoints_structure() {
    // an absent appendix is a valid empty appendix
    {
        std::list<common_prompt_checkpoint> out;
        size_t n_consumed = 123;
        assert(server_prompt_disk::read_checkpoints(nullptr, 0, out, SIZE_MAX, &n_consumed));
        assert(n_consumed == 0);
        assert(out.empty());
    }

    std::vector<uint8_t> appendix;
    {
        std::list<common_prompt_checkpoint> checkpoints;
        auto & ckpt = checkpoints.emplace_back();
        ckpt.n_tokens = 3;
        ckpt.pos_min  = 1;
        ckpt.pos_max  = 2;
        ckpt.data_tgt.resize(8, 0x5a);
        server_prompt_disk::write_checkpoints(appendix, checkpoints);
    }
    assert(!appendix.empty());

    // truncation is rejected without touching the caller's list
    for (const size_t keep : { appendix.size() - 1, size_t(4), size_t(1) }) {
        std::list<common_prompt_checkpoint> out;
        out.emplace_back().n_tokens = 99;
        assert(!server_prompt_disk::read_checkpoints(appendix.data(), keep, out, SIZE_MAX, nullptr));
        assert(out.size() == 1 && out.front().n_tokens == 99);
    }

    {
        std::vector<uint8_t> bad = appendix;
        bad[0] ^= 0xff;
        std::list<common_prompt_checkpoint> out;
        assert(!server_prompt_disk::read_checkpoints(bad.data(), bad.size(), out, SIZE_MAX, nullptr));
    }
    {
        std::vector<uint8_t> bad = appendix;
        bad[4] = 99;
        std::list<common_prompt_checkpoint> out;
        assert(!server_prompt_disk::read_checkpoints(bad.data(), bad.size(), out, SIZE_MAX, nullptr));
    }

    // a valid appendix round-trips
    {
        std::list<common_prompt_checkpoint> parsed;
        size_t n_consumed = 0;
        assert(server_prompt_disk::read_checkpoints(appendix.data(), appendix.size(), parsed, SIZE_MAX, &n_consumed));
        assert(n_consumed == appendix.size());
        assert(parsed.size() == 1);
        assert(parsed.front().n_tokens == 3);
        assert(parsed.front().pos_min == 1);
        assert(parsed.front().pos_max == 2);
        assert(parsed.front().data_tgt.size() == 8);
        assert(parsed.front().data_tgt.data()[0] == 0x5a);

        // the limit truncates the oldest entries
        std::list<common_prompt_checkpoint> limited;
        assert(server_prompt_disk::read_checkpoints(appendix.data(), appendix.size(), limited, 0, nullptr));
        assert(limited.empty());
    }

    // semantic validation against the prompt
    {
        server_prompt prompt = make_prompt({1, 2, 3, 4});
        std::list<common_prompt_checkpoint> checkpoints;
        auto & ckpt = checkpoints.emplace_back();
        ckpt.n_tokens = 2;
        ckpt.pos_min  = 0;
        ckpt.pos_max  = 1;
        ckpt.data_tgt.resize(8);

        assert(server_prompt_disk::checkpoints_match_prompt(
                checkpoints, 1, prompt.n_tokens(), prompt.tokens.pos_next()));
        assert(!server_prompt_disk::checkpoints_match_prompt(
                checkpoints, 2, prompt.n_tokens(), prompt.tokens.pos_next()));

        checkpoints.front().pos_max = prompt.tokens.pos_next();
        assert(!server_prompt_disk::checkpoints_match_prompt(
                checkpoints, 1, prompt.n_tokens(), prompt.tokens.pos_next()));

        checkpoints.front().pos_max = 1;
        checkpoints.front().data_tgt.clear();
        assert(!server_prompt_disk::checkpoints_match_prompt(
                checkpoints, 1, prompt.n_tokens(), prompt.tokens.pos_next()));
    }
}

int main() {
    prompt_cache_ranks_safe_restorable_prefix_before_lexical_lcp();
    prompt_cache_load_target_success_draft_failure_is_atomic();
    prompt_cache_does_not_restore_just_admitted_live_state();
    restore_transaction_validation_failures_are_atomic();
    restore_transaction_validation_failure_identifies_prepare_leg();
    speculative_rollback_checkpoint_boundary();
    checkpoint_max_step_cadence_is_off_by_default();
    checkpoint_max_step_cadence_seeds_the_first_checkpoint();
    checkpoint_reuse_selection_prefers_latest_valid_boundary();
    checkpoint_reuse_selection_respects_descriptor_alignment();
    checkpoint_eviction_preserves_prompt_anchors();
    checkpoint_failed_target_save_cannot_reuse_stale_bytes();
    speculative_draft_rollback_uses_draft_axis_and_recovers();
    server_unsupported_removal_falls_back_to_full_reprocess();
    server_post_preflight_mutation_failure_clears_both_contexts();
    server_planned_removal_preserves_atomic_media_chunks();
    cache_phase_timings_are_public();
    prompt_cache_snapshot_restore_evict_stress();
    disk_spill_restore_round_trip();
    disk_identity_mismatch_is_rejected();
    disk_truncated_and_corrupt_files_are_rejected();
    disk_budget_eviction();
    disk_checkpoint_appendix_round_trip();
    disk_erase_and_destructor_clean_the_directory();
    disk_disabled_keeps_eviction_behaviour();
    disk_mapping_modes_agree();
    disk_async_lifecycle();
    disk_read_checkpoints_structure();
    {
        common_prompt_checkpoint ckpt;
        ckpt.n_tokens = 3;
        ckpt.pos_min = 1;
        ckpt.pos_max = 2;
        ckpt.data_tgt.resize(128);
        ckpt.data_dft.resize(64);
        ckpt.data_spec.resize(32);
        assert(ckpt.size() == 224);

        ckpt.clear();
        assert(ckpt.n_tokens == 0);
        assert(ckpt.pos_min == 0);
        assert(ckpt.pos_max == 0);
        assert(ckpt.empty());
        assert(ckpt.size() == 0);
    }

    {
        server_prompt prompt = make_prompt({1, 2, 3});
        auto & ckpt = prompt.checkpoints.emplace_back();
        ckpt.n_tokens = 3;
        ckpt.data_tgt.resize(16);
        ckpt.data_dft.resize(8);

        const server_prompt clone = prompt.clone();
        assert(clone.n_tokens() == 3);
        assert(clone.checkpoints.size() == 1);
        assert(clone.checkpoints.front().data_tgt.storage_id() ==
                prompt.checkpoints.front().data_tgt.storage_id());

        const void * shared_storage = clone.checkpoints.front().data_tgt.storage_id();
        prompt.checkpoints.front().data_tgt.resize(24);
        assert(prompt.checkpoints.front().data_tgt.storage_id() != shared_storage);
        assert(clone.checkpoints.front().data_tgt.size() == 16);

        server_prompt_cache_state state {
            /*.prompt =*/ std::move(prompt),
            /*.data =*/ {
                /*.main =*/ std::vector<uint8_t>(64),
                /*.drft =*/ std::vector<uint8_t>(32),
                /*.spec =*/ { },
            },
            /*.disk =*/ { },
        };
        assert(state.accounted_size() == 128);
    }

    {
        server_prompt_cache cache(1, 0);
        server_prompt_cache_state existing;
        existing.prompt = make_prompt({1, 2});
        existing.data.main.resize(700*KIB);
        cache.states.push_back(std::move(existing));

        server_prompt current = make_prompt({3, 4});
        auto * saved = cache.alloc(current, 600*KIB, 0);

        assert(saved != nullptr);
        assert(cache.accounted_size() <= cache.limit_size);
        assert(cache.states.size() == 1);
        assert(cache.states.back().data.main.size() == 600*KIB);
    }

    {
        server_prompt_cache cache(1, 0);
        server_prompt current = make_prompt({3, 4});
        auto & ckpt = current.checkpoints.emplace_back();
        ckpt.n_tokens = 4;
        ckpt.data_tgt.resize(200*KIB);

        auto * saved = cache.alloc(current, 800*KIB, 0);
        assert(saved != nullptr);
        assert(saved->prompt.checkpoints.size() == 1);
        assert(saved->accounted_size() == 1000*KIB);
        assert(cache.accounted_size() == saved->accounted_size());
    }

    {
        server_prompt_cache cache(1, 0);
        server_prompt_cache_state existing;
        existing.prompt = make_prompt({1, 2});
        existing.data.main.resize(100*KIB);
        cache.states.push_back(std::move(existing));

        server_prompt current = make_prompt({3, 4});
        auto & ckpt = current.checkpoints.emplace_back();
        ckpt.n_tokens = 4;
        ckpt.data_tgt.resize(200*KIB);

        assert(cache.alloc(current, 900*KIB, 0) == nullptr);
        assert(cache.states.size() == 1);
        assert(cache.accounted_size() == 100*KIB);
    }

    {
        server_prompt_cache cache(1, 0);
        server_prompt current = make_prompt({3, 4});
        assert(cache.alloc(current, 100*KIB, 0) != nullptr);
        assert(cache.alloc(current, 100*KIB, 0) == nullptr);
        assert(cache.states.size() == 1);
    }

    {
        server_prompt source = make_prompt({7, 8, 9});
        source.checkpoints.emplace_back().data_tgt.resize(200*KIB);
        server_prompt_cache cache(0, 0);
        server_prompt_cache_state first;
        first.prompt = source.clone();
        first.data.main.resize(10*KIB);
        server_prompt_cache_state second;
        second.prompt = source.clone();
        second.data.main.resize(20*KIB);
        cache.states.push_back(std::move(first));
        cache.states.push_back(std::move(second));
        assert(cache.accounted_size() == 230*KIB);
    }

    return 0;
}
