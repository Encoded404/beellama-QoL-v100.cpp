# BeeLlama v0.4.4 argument reference

This page covers Bee-owned arguments and the upstream arguments whose behavior
BeeLlama extends. Run `llama-server --help` or `llama-cli --help` for the full
upstream surface. See [BeeLlama features](beellama-features.md) for use cases,
limits, and measurement guidance.

## KVarN cache types and SWA overrides

KVarN values are `kvarn2`, `kvarn3`, `kvarn4`, `kvarn5`, `kvarn6`, and
`kvarn8`. K and V may use different bit widths.

CUDA, ROCm/HIP, Vulkan, and CPU consume compressed KVarN records directly in
native FlashAttention paths. Vulkan requires shader Int64 and
buffer-device-address support for its direct route. An explicitly supported
materialization fallback retains compressed persistent storage when a native
route is unavailable. Pre-Turing NVIDIA GPUs use CUDA's portable rotated-domain
body-plus-tail route and require a CUDA 12.4 build or release package. CUDA
13.3 packages target Turing and newer architectures.

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `-ctk TYPE`, `--cache-type-k TYPE` | `LLAMA_ARG_CACHE_TYPE_K` | `f16` | Selects the target K cache. Bee adds the six KVarN values and standard `q6_0`, `q6_1`, `q3_0`, `q3_1`, `q2_0`, and `q2_1`. If only K or V is KVarN, the other side is promoted to the same KVarN width with a warning. |
| `-ctv TYPE`, `--cache-type-v TYPE` | `LLAMA_ARG_CACHE_TYPE_V` | `f16` | Selects the target V cache with the same values and one-sided promotion rule as `--cache-type-k`. |
| `--cache-type-k-swa TYPE` | `LLAMA_ARG_CACHE_TYPE_K_SWA` | Same as `--cache-type-k` | Overrides KVarN K precision for SWA layers. Accepts only the six `kvarnN` values, requires target KVarN, and must be paired with the V override. |
| `--cache-type-v-swa TYPE` | `LLAMA_ARG_CACHE_TYPE_V_SWA` | Same as `--cache-type-v` | Overrides KVarN V precision for SWA layers. Accepts only the six `kvarnN` values, requires target KVarN, and must be paired with the K override. |

## KV cache precision tail for quantized caches

The KV cache precision tail (KVCPT) makes the newest attention-visible entries exact in F16 or BF16 for
standard quantized and KVarN target caches. A partial tail keeps the complete
quantized body and adds a compact exact-history ring. The active ubatch remains
a separate graph-local exact source. A full-window SWA request uses a compact
native-exact ring and omits the unread compressed SWA body. Draft and auxiliary
contexts remain on standard cache types and do not inherit the target tail.

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `--kv-tail-tokens SPEC` | `LLAMA_ARG_KV_TAIL_TOKENS` | `0` | For standard caches, `0` keeps the ordinary cache path. For KVarN, omitted or `0` retains the intrinsic 128-token exact suffix. A number applies to every canonical group; KVarN rounds positive values upward to complete 128-token groups. `N0,N1` follows canonical group order, while `full=N,swa=N` accepts unique role aliases or structural IDs such as `full@l0`. Invalid, duplicate, incomplete, ambiguous, or wrong-length specifications fail context creation. `auto` requests 1024 exact tokens per applicable target-cache group, capped by that group's effective context or attention window. |
| `--kv-tail-type TYPE` | `LLAMA_ARG_KV_TAIL_TYPE` | `bf16` for standard caches; `f16` for KVarN | Selects `f16`, `bf16`, or `q8_0` storage for the compact history and compact-native SWA of a standard cache, or `f16`/`bf16` for a KVarN cache. An explicit value overrides the cache-family default in either direction. The `q8_0` tail keeps the recent-token rows quantized at half the memory of `f16`/`bf16`; it is available for standard caches only and has no automatic fallback, so an unsupported route fails closed rather than changing representation. Requesting `q8_0` while a KVarN cache is active is rejected during context creation. Other types are rejected. |

An omitted tail type remains automatic until context placement. If the standard
BF16 default lacks a complete Metal or SYCL route but F16 is complete, automatic
selection warns and resolves once to F16. Explicit `--kv-tail-type bf16` fails
instead of changing the requested representation. An explicit `--kv-tail-type
q8_0` is never downgraded: it either finds a complete route (CUDA and CPU, via
the quantized FlashAttention vector pair and the row-convertible CPU reference)
or fails context creation so the requested representation is not silently
replaced. Because a KVarN tail lives in the rotated WHT domain, `q8_0` is
rejected outright for KVarN caches (including the model-bound
`--kv-tail-request` path); only F16/BF16 tails are valid there.

Explicit values are capped by the group's effective attention window and context
capacity. KVarN values are also rounded upward to 128-token groups. Startup logs
show raw, requested, effective, and window lengths, the structural group ID,
participating layers, selected compact-overlay or compact-native-exact
representation, actual body and exact types, logical history rows, rollback
rows, graph-local body execution rows, owner backend, current-segment
presence, transient estimate, and memory increments. Native routes are checked
again against the final constructed operation. A mismatch fails context/graph
construction instead of allowing the scheduler to move that layer silently.

The CLI is parsed once into an immutable, model-independent request. Fit probes
and the final context bind that same request to the model's canonical cache
groups, so `auto`, positional, named, and KVarN-minimum policies cannot diverge
between estimation and allocation. Public callers that need the same behavior
can create a request with `llama_kv_tail_request_init`, keep it alive through
context creation, assign the borrowed pointer to
`llama_context_params::kv_tail_request`, and then free it with
`llama_kv_tail_request_free`.

Let `N` be the resolved exact length, `U` the physical ubatch limit, `R` the
advertised suffix-rollback horizon, and `S` the number of exact streams. Compact
persistent exact capacity is `(N + R) * S` rows and is independent of `U`.
`U` sizes only graph inputs and reusable transient workspace. Backend buffer
alignment may round bytes but does not add logical rows. Exact history remains
per logical sequence even with `--kv-unified`. Positive tails on K-only MLA or
DSA attention are rejected during context creation.

`R` comes from the context's rollback requirement. Contexts that otherwise
request no rollback retain one row for the common one-token capability probe;
it never defaults to `U`. The memory capability API reports this bound, and a
larger speculative removal must use the checkpoint/reprocess path before cache
metadata is mutated.

Partial exact overlays are compatible with `--split-mode layer` and
`--split-mode tensor`. Layer mode keeps each shadow with its ordinary K/V body.
Tensor mode shards standard body/shadow rows, KVarN records and staging, and
exact history at complete KV-head boundaries through the model's meta split
descriptor. Invalid or unsupported component splits fail during cache
construction rather than after graph execution starts.

KVarN's physical staging depth is independent of this logical policy. Increasing
`-ub` may increase transient work but never increases persistent exact coverage.
Completed 128-token records are committed eagerly for partial tails, while the
canonical exact history stores only `N + R` rows. A fully covered SWA group uses
`--kv-tail-type` for its `W + R` compact-native ring and allocates no SWA KVarN
records or stage; non-SWA and partially covered groups remain KVarN.

Partial SWA tails retain the upstream-aligned compressed `W + U` body because
older visible rows still use it. Full-window compact-native SWA has no body and
stores exactly `W + R` persistent rows. In both cases current K/V is consumed
directly by the same attention softmax before an explicitly ordered history
commit.

`llama-bench --kv-memory` reports cache-owned bytes directly. The
`kv_k_payload_bytes`, `kv_v_payload_bytes`, `kv_exact_history_bytes`,
`kv_rollback_reserve_bytes`, `kv_staging_bytes`, `kv_padding_bytes`, and
`kv_resident_bytes` fields describe persistent ownership.
`kv_transient_bytes` is the observed reusable CUDA-pool high water and
`kv_peak_bytes` is resident plus transient. The per-route layer counters show
native bodyless, native mixed, planned device-fallback, and CPU layers. These
fields are more precise than deriving cache memory from whole-process VRAM;
the separate CUDA/WDDM fields remain useful for reconciliation and spill
detection.

Overlay state uses a framed standard-memory section. Exact restore requires the
same structural group, resolved length, representation, and exact type. Native
exact state is carried by the ordinary body and does not serialize a duplicate
shadow. The extended full and sequence state APIs accept
`LLAMA_STATE_SEQ_FLAGS_BODY_ONLY` to deliberately omit overlay shadows; loading
that state into a tail-enabled context is valid, but the coverage API reports
`LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE` until new writes refill the recent
window. Server metrics expose requested/exact token totals, complete/partial/no
coverage group counts, and degraded-sequence counts.

KVarN state version 15 stores sequence-selective logical compressed record
groups, compact exact payloads, selected stage rows, and destination remapping
independently of ubatch workspace, so state may move between `ub=128` and
`ub=512`. Live checkpoint state may retain sealed history already owned by the
context; `LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED` exports every payload needed
after the source is removed. Compatible version 12 and 13 state remains
readable; version 11 is rejected rather than reinterpreting its old physical
workspace layout. Tail length, type, preset, rollback horizon, representation,
and structural-group mismatches fail closed.

Sequence state writes precision-tail manifest version 5, including exact source
cell/generation identities, exact-tail local slots, insertion order, the
per-sequence write cursor, the compact representation, and rollback horizon,
and supports host or on-device tensor transfer. Version 4 remains readable;
manifest version 2 remains readable for non-compact layouts, while version 1
restores conservative degraded provenance. Immediate body
membership and position changes after sequence copy are preserved; pending
exact rows materialize as one batch when state data is requested.

Restore publishes no tensor or metadata changes until the complete state frame
has validated. A truncated, corrupt, mismatched, or failed backend transfer is
cancelled. Deferred precision-tail copy failures propagate through immediate state
save and subsequent decode instead of being reported as successful.

Prompt-cache message boundaries do not reset the suffix. Standard and KVarN
state is sequence-selective in unified and non-unified layouts. The planner
records lexical, restorable, and committed token counts and restores target,
draft, and speculative state as one prepared transaction. Live slots and RAM
entries are compared by safe restorable tokens before existing tie-breaks.
Standard and recurrent caches keep upstream prompt batching; KVarN alone adds
its descriptor-boundary eligibility rule. RAM entries use self-contained
immutable state, are repeatably restorable, and clear an idle unified slot only
after successful admission. Tail length affects quality, memory, and transfer
cost, not logical prompt-cache eligibility.

KVarN durable prompt checkpoints remain on complete 128-token descriptor
boundaries for the current G128 presets. The transient live exact frontier may still service the
existing bounded speculative micro-rollback contract; it does not turn sealed
history into arbitrary-position records. Standard KV has no KVarN group
constraint and may restore any validated logical position. For standard caches
with a precision tail, a hot partial checkpoint references the still-live body
and transfers its logical manifest and exact overlay. The manifest remains
proportional to the logical prefix, so this is not a native sealed-block arena
or a strictly frontier-only operation. Copy-on-write sharing applies to
serialized checkpoint byte buffers, not native KV blocks. Self-contained RAM
state continues to own and transfer the body because it must survive source
removal.

Completion timing JSON includes `cache_lcp_n`, `cache_planned_n`,
`cache_reprocessed_n`, `cache_source`, and `cache_reason`. Prometheus exports
`prompt_cache_admission_*_total`, `prompt_cache_restore_*_total`,
`prompt_cache_accounted_bytes`, `n_busy_slots_per_decode`, and the
`kv_tail_*` coverage/degradation gauges. A nonzero restore-failure or degraded
tail metric is actionable rather than silently counted as a hit. Accounted
bytes are serialized payload accounting, not exact process-resident memory.

## Worst-case workspace and output reservation

The number of sequences (`-np`, `--parallel`) controls KV slot capacity and RAM
prompt-cache slots. `--max-concurrent-streams` reclaims the `-np`-scaled
reservation for deployments that only ever run fewer concurrent streams than
slots. It caps the worst-case number of simultaneously-active streams, and from
that derives everything that scales with it:

- **Decode workspace**: the graph activations / QKV buffers of the
  token-generation graph scale with the number of active streams times the
  per-sequence speculative expansion (`n_seq_active * (1 + n_draft)` tokens).
  With `--kv-unified` the KV body is a single shared stream, so raising `-np`
  no longer grows the KV body; it still grows this decode workspace because
  each active slot emits one stream per decode step.
- **Output buffers**: the host logits/sampling buffers and the graph's output
  rows are derived as `n_seq_active * n_outputs_max_per_seq`, floored at one
  output per sequence (`n_seq_max`), a hard structural minimum that
  `output_reserve()` requires.

The cap applies to the target and any draft/MTP context, and at runtime the
server never batches more active streams than it into a single decode. A value
of `0` keeps the auto `-np` behavior. Reducing it below `-np` is only safe when
you never actually batch more concurrent streams than the cap allows, otherwise
a transient over-cap batch must re-reserve or fails closed.

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `-mcs N`, `--max-concurrent-streams N` | `LLAMA_ARG_MAX_CONCURRENT_STREAMS` | `0` (= `-np`) | Caps the worst-case number of simultaneously-active streams used to reserve the decode workspace (graph activations / QKV buffers) and the host logits/sampling output buffers. Must be `<= -np`; at runtime the server never batches more active streams than this into a single decode. |

## DFlash and adaptive draft depth

The first five rows are upstream speculative controls with Bee-specific DFlash
behavior. The `--spec-dm-*` rows are Bee server additions.

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `--spec-type draft-dflash` | `LLAMA_ARG_SPEC_TYPE` | `none` | Enables upstream DFlash. |
| `--spec-draft-model FNAME`, `-md FNAME` | `LLAMA_ARG_SPEC_DRAFT_MODEL` | Unused | Loads an upstream-format `dflash` draft GGUF. |
| `--spec-draft-n-max N` | `LLAMA_ARG_SPEC_DRAFT_N_MAX` | Upstream: `3`; omitted DFlash: `dflash.block_size - 1` | Sets the maximum draft depth. An explicit CLI or env value always wins; upstream clamps values above the drafter's trained limit. A block-16 drafter therefore defaults to 15 only when this setting is omitted. |
| `--spec-draft-n-min N` | `LLAMA_ARG_SPEC_DRAFT_N_MIN` | `0` | Sets the minimum number of draft tokens used by upstream speculation. |
| `--spec-draft-p-min P`, `--draft-p-min P` | `LLAMA_ARG_SPEC_DRAFT_P_MIN` | `0.0` | Stops an individual greedy draft when its probability falls below `P`; this is independent of the profit controller. |
| `--spec-dm-controller MODE` | `LLAMA_ARG_SPEC_DM_CONTROLLER` | `profit` | For DFlash1, `profit` adapts depth from measured cycle profit and `off` keeps the resolved or explicit maximum static. DFlash2 always uses its fixed trained block limit and selector confidence; other speculative modes are unchanged. |
| `--spec-dm-profit-min F` | `LLAMA_ARG_SPEC_DM_PROFIT_MIN` | `0.05` | Sets the minimum margin over the no-spec baseline before clearing disable dwell. Range: `0.0` to `0.50`. |
| `--spec-dm-profit-raise-margin F` | `LLAMA_ARG_SPEC_DM_PROFIT_RAISE_MARGIN` | `0.05` | Sets the relative profit margin required to raise draft depth. Range: `0.0` to `1.0`. |
| `--spec-dm-profit-lower-margin F` | `LLAMA_ARG_SPEC_DM_PROFIT_LOWER_MARGIN` | `0.05` | Sets the relative profit margin required to lower draft depth. Range: `0.0` to `1.0`. |
| `--spec-dm-profit-ewma-alpha F` | `LLAMA_ARG_SPEC_DM_PROFIT_EWMA_ALPHA` | `0.15` | Sets the EWMA weight for profit statistics. Range: `0.01` to `1.0`. |
| `--spec-dm-profit-min-samples N` | `LLAMA_ARG_SPEC_DM_PROFIT_MIN_SAMPLES` | `3` | Sets the samples required before a depth's profit statistics are ready. Range: `1` to `64`. |
| `--spec-dm-profit-warmup N` | `LLAMA_ARG_SPEC_DM_PROFIT_WARMUP` | `0` | Sets measured samples for each initial positive-depth probe. `0` uses `--spec-dm-profit-min-samples`; range: `0` to `64`. |
| `--spec-dm-profit-baseline-interval N` | `LLAMA_ARG_SPEC_DM_PROFIT_BASELINE_INTERVAL` | `1024` | Sets active controller cycles between no-spec baseline probes. `0` disables periodic probes; range: `0` to `4096`. |

## MARS relaxed verification

MARS (margin-aware relaxed) verification is a lossy speculative acceptance
rule: when the draft token is not the target's sampled pick, it is accepted
anyway if it ranks within `--spec-verify-mars-topk` of the raw target logits
and its softmax-probability ratio vs the raw top-1 is at least
`--spec-verify-mars-theta`. Because the accepted token deviates from the
target's own distribution, use it when higher draft acceptance is worth a
measurable output-distribution change. It is temperature-invariant and works
with every draft type.

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `--spec-verify-mars`, `--no-spec-verify-mars` | `LLAMA_ARG_SPEC_VERIFY_MARS` | Disabled | Enables MARS relaxed verification. With target backend sampling (`-bs`) the per-row top-k stats are computed on the device by a `GGML_OP_MARS_STATS` backend sampler; otherwise the raw logits are scanned on the host. |
| `--spec-verify-mars-theta P` | `LLAMA_ARG_SPEC_VERIFY_MARS_THETA` | `0.9` | Accepts a draft token ranked within `--spec-verify-mars-topk` when `exp(z_draft - z_top1) >= theta` (temperature-invariant). Range: positive. |
| `--spec-verify-mars-topk N` | `LLAMA_ARG_SPEC_VERIFY_MARS_TOPK` | `2` | How far down the raw target logit ranking a draft token may sit to be eligible for relaxed acceptance. Ranks are internally capped at 8. Range: positive. |

## Reasoning loop guard

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `--reasoning-loop-guard MODE` | `LLAMA_ARG_REASONING_LOOP_GUARD` | `force-close` | `off` disables checks, `force-close` asks the reasoning sampler to end hidden reasoning, and `stop` ends generation when a loop triggers. |
| `--reasoning-loop-min-tokens N` | `LLAMA_ARG_REASONING_LOOP_MIN_TOKENS` | `512` | Delays hidden-reasoning checks until `N` reasoning tokens have been seen. Must be non-negative and at least the minimum coverage. |
| `--reasoning-loop-window N` | `LLAMA_ARG_REASONING_LOOP_WINDOW` | `1024` | Sets the token-tail window inspected for repetition. Must be positive and at least the minimum coverage. |
| `--reasoning-loop-max-period N` | `LLAMA_ARG_REASONING_LOOP_MAX_PERIOD` | `128` | Sets the longest periodic loop checked. Must be positive and no more than one third of the window. |
| `--reasoning-loop-min-coverage N` | `LLAMA_ARG_REASONING_LOOP_MIN_COVERAGE` | `256` | Sets the repeated-token coverage required to trigger. Must be positive. |
| `--reasoning-loop-check-interval N` | `LLAMA_ARG_REASONING_LOOP_CHECK_INTERVAL` | `64` | Runs a check after each `N` accepted reasoning tokens. Must be positive. |
| `--reasoning-loop-interventions N` | `LLAMA_ARG_REASONING_LOOP_INTERVENTIONS` | `2` | Sets the maximum successful force-close interventions before a later trigger stops generation. Must be non-negative. |

## Realtime reasoning control

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| Chat request JSON `"reasoning_control": true` | — | `false` | Arms a live `/v1/chat/completions` request for external reasoning control. The chat template must expose a reasoning end sequence. |
| `POST /v1/chat/completions/control` with `{"id":"chatcmpl-...","action":"reasoning_end"}` | — | Disabled per request | Forces the armed completion's reasoning sampler toward its final-answer phase. Unknown or completed ids return a non-success result; `reasoning_end` is the only accepted action. |

## Presets

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `--models-preset PATH` | `LLAMA_ARG_MODELS_PRESET` | Disabled | Loads an INI file containing model presets for router-server mode. Command-line values override values loaded from a preset. |
| Preset key `load-on-startup` | Preset-only | False when absent | A truthy value autoloads that model when router mode starts; the number of startup models may not exceed `--models-max`. |
| Preset key `stop-timeout` | Preset-only | `10` seconds | Force-kills a child model process after this many seconds of graceful shutdown. Invalid values fall back to 10. |

`GET /models` lists model identity, status, source, aliases, tags, and
capabilities. Matching upstream, each entry's `status` exposes the child argv
(`status.args`) and, for preset-backed models, the resolved INI preset
(`status.preset`) with sensitive options stripped; these may contain local
paths for custom-path preset models. It ignores former reload query parameters.
Refresh model sources
with `POST /models/reload`; when `--api-key` is configured this mutation
requires the same `Authorization: Bearer ...` or `X-Api-Key` authentication as
other non-public routes. `--hf-token` is a sensitive option: router children
receive it through `HF_TOKEN`, never through argv or serialized presets.

See [INI presets](preset.md) for syntax, inheritance, remote presets, and a Bee
configuration example.

## KLD measurement

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `--save-all-logits FNAME`, `--kl-divergence-base FNAME` | — | Unused | Without `--kl-divergence`, writes the base run's compressed log probabilities to `FNAME`. |
| `--kl-divergence` | — | Off | Compares the current run with the file supplied by `--kl-divergence-base` and returns a nonzero exit code on read or evaluation failure. |

Use the same corpus, context, logical batch, and physical ubatch for both KLD legs.

## CUDA FlashAttention build policy

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `-DGGML_CUDA_FA_ALL_QUANTS=ON` | — | Off | Expands the CUDA vector matrix from 50 to all 169 standard cache pairs and, when `GGML_CUDA_KVARN=ON`, KVarN fast-decode instances from 15 balanced pairs to all 36 ordered bit pairs. Valid KVarN pairs outside the fast matrix use descriptor-native MMA. |
| `-DGGML_CUDA_FA_NO_BF16=ON` | — | Off | Skips every BF16 CUDA vector pair (BF16 hardware requires sm_80+): 49 pairs instead of 50 by default, 144 instead of 169 with `GGML_CUDA_FA_ALL_QUANTS=ON`. BF16 K/V then reports as uncompiled and KV tails automatically downgrade to F16. |
| `-DGGML_CUDA_KVARN=ON/OFF` | — | On | Compiles or omits the shared CUDA/HIP KVarN kernels and CUDA native-attention template instances. When enabled, `GGML_CUDA_FA_ALL_QUANTS` selects 15 default or all 36 CUDA fast-decode pairs. CUDA devices without the specialized Turing MMA contract use the portable direct-record route when their warp, thread-block, shared-memory, head-dimension, and tail-type capabilities pass. |

Release packages are built with CUDA 12.4 and 13.3. CUDA 12.4 can emit the
Maxwell, Pascal, and Volta PTX targets used by the portable KVarN route; CUDA
13.3 covers Turing and newer architectures. The release workflow no longer has
an exhaustive per-architecture CUDA compile gate. For a local or CI build,
select the intended target explicitly with `CMAKE_CUDA_ARCHITECTURES` when the
build host cannot detect it. Pre-Turing support remains runtime-unqualified
until matching real devices pass the KVarN parity, memory, and model-smoke
tests.

## Volta (sm_70) CUDA decode tuning

On Volta, GQA decode keeps more HBM2 load streams in flight with two optional
behaviors, both off by default where the default is performance-neutral:

- `GGML_CUDA_FATTN_VEC_GQA_HEADS` — `1` (auto, default) lets the FlashAttention
  vector kernel process two Q heads that share one K/V pair per block, halving
  K reads and reusing every dequantized V row. The auto mode enables this only
  for contexts above 128K tokens, where the K/V re-read amortization outweighs
  the two-column block overhead; `0` forces it off and any value `>= 2` forces
  it on.
- `GGML_CUDA_FATTN_VEC_NTHREADS` — `128` or `256`. The default is 256 (8 warps)
  so the streaming quantized K/V loads have enough warps in flight on the
  4-scheduler SMs; `128` restores the legacy 4-warp behavior. D=512 f16/bf16 V
  instances keep 4 warps regardless to fit the shared staging buffer.
- `GGML_CUDA_FATTN_VEC_PREFETCH` — `1` (default) prefetches the next iteration's
  K/V rows into L2 from inside the vector kernel (Volta has no `cp.async`);
  `0` disables the `prefetch.global.L2` hints.
- `GGML_CUDA_MMVQ_GENERIC` — `1` reverts Volta's `mul_mat_vec_q` decode to the
  generic (Pascal-style) parameter table. The default Volta table uses 8-warp
  blocks via the `halve_iters` mechanism for the trivial vec-dot types
  (`q4_0`/`q4_1`/`q5_0`/`q5_1`/`q6_0`/`q6_1`/`q8_0`/`iq4_nl`) so more decode
  streams stay in flight; complex K/IQ vec-dots stay at 4 warps to limit
  register pressure.

## KVarN runtime env

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| — | `GGML_KVARN_BORROW` | `1` (on) | `0` disables borrowing the target's KVarN store for a Gemma-4 assistant (draft/MTP) auxiliary context; the auxiliary then uses a private cache. |

## Migration from earlier versions

| Earlier spelling or surface | v0.4.0 behavior | Replacement |
|---|---|---|
| Target cache `turbo2`, `turbo3`, `turbo4`, or `_tcq` variants | Warns and redirects by width to `kvarn2`, `kvarn3`, or `kvarn4`. | Use the `kvarnN` name directly. |
| Draft cache `turbo2`, `turbo3`, `turbo4`, or `_tcq` variants | Warns and redirects by width to `q2_0`, `q3_0`, or `q4_0`. | Use the standard q-cache name directly. |
| TurboQuant/TCQ GGUF cache formats and TQ3/TQ4 weight formats | Unsupported; legacy TQ file-type ids fail with a re-quantization error. | Re-quantize from source into a retained format. |
| `--spec-type dflash` | Rejected as an unknown speculative type. | `--spec-type draft-dflash` |
| `copyspec`, `suffix`, or `recycle` speculative types | Rejected with a migration error. | Use `draft-dflash` or an upstream n-gram mode. |
| `--draft`, `--draft-n`, `--draft-max` | Rejected as removed. | `--spec-draft-n-max` or `--spec-ngram-mod-n-max` |
| `--draft-min`, `--draft-n-min` | Rejected as removed. | `--spec-draft-n-min` or `--spec-ngram-mod-n-min` |
| `--spec-dflash-default`, `--dflash-max-slots`, `--tree-budget`, `--draft-topk`, `--draft-model`, `--spec-replace`, `--spec-draft-replace` | Removed with the fork DFlash verifier and tree paths. | Use upstream `--spec-*` controls where an equivalent exists. |
| `--spec-dflash-cross-ctx`, `--spec-branch-budget`, `--spec-draft-temp`, `GGML_DFLASH_*` | Removed with the fork ring, capture, and verifier implementation. | No direct replacement. |
| `GGML_CUDA_FA_HALF_QUANTS` | Removed. | Use the default matrix or `GGML_CUDA_FA_ALL_QUANTS=ON`. |
| `GGML_CUDA_KVARN_FA`, `GGML_CUDA_KVARN_FAST_DECODE_ALL_PAIRS` | Removed. | Use the default-on `GGML_CUDA_KVARN`; `GGML_CUDA_FA_ALL_QUANTS` selects 15 or 36 fast-decode pairs. |
