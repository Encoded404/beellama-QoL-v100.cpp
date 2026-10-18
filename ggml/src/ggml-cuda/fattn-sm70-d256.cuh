#pragma once

#include "common.cuh"
#include "fattn-common.cuh" // FATTN_KQ_MAX_OFFSET, SOFTMAX_FTZ_THRESHOLD
#include "mma.cuh" // mma tiles and shared-memory ldmatrix helpers (upstream folded the smem-swizzle header into mma.cuh)

// Volta (SM70) D256 split-D prefill FlashAttention.
//
// The stock mma_f16 prefill path for D256 on Volta (case <256,256,32,2>) keeps the
// complete DV=256 PV accumulator in every warp, which drives the kernel into a
// register spill. This family pairs two warps on the same Q rows and splits the
// work so each warp owns only half of the output dimension:
//
//   * KQ: the two warps of a pair split the KV columns (16 each).
//   * PV: the two warps of a pair split the 256 output dimensions (128 each) and
//         read the same pair-shared P tile from shared memory.
//
// Neither half duplicates work: 2 x (32 rows x 16 cols) = 32 x 32 for KQ and
// 2 x (32 rows x 128 dims) = 32 x 256 for PV.
//
// A Volta m8n8k4 tile<32, 8, float> is one warp covering 32 Q rows x 8 KV columns,
// so 32 rows per warp is the granularity the KQ side must keep; the split therefore
// lives on the KV-column and output-dimension axes. Both mma atoms this needs already
// exist for Volta in mma.cuh (the .row.col f32 QK atom and the .row.row f16 PV atom),
// so no third-party headers and no mma.cuh changes are required.
//
// Background: ggml-org/llama.cpp#28037 documents that Volta's gap to Turing/Ampere is
// architectural (m8n8k4, no ldmatrix, no cp.async) and that config tuning does not
// close it. The split-D design is the remaining lever (the 1Cat-vLLM FLASH_ATTN_V100
// route and the sm70-attn llama.cpp port land the same idea through a CUTLASS-based
// dataflow, and both record large prefill gains on V100).

namespace ggml_cuda_fattn_sm70_d256 {

static constexpr int kHeadDim        = 256;
static constexpr int kBlockM         = 128; // Q rows per CTA (4 pairs x 32 rows)
static constexpr int kBlockN         = 32;  // KV rows per softmax round
static constexpr int kThreads        = 256; // 8 warps = 4 pairs
static constexpr int kWarpsPerGroup  = 2;
static constexpr int kGroups         = kThreads / (kWarpsPerGroup * 32); // 4
static constexpr int kGroupRows      = kBlockM / kGroups;                // 32
static constexpr int kQkColsPerWarp  = kBlockN / kWarpsPerGroup;         // 16
static constexpr int kOwnedDims      = kHeadDim / kWarpsPerGroup;        // 128

// Prefill only: below this Q length the stock path is preferred.
static constexpr int kMinQLen        = 256;

// ---------------------------------------------------------------------------
// Device side
//
// The tile aliases and constants are arch-independent and stay visible to the
// host pass; only the kernel body is gated on VOLTA_MMA_AVAILABLE, which is
// defined during the sm_70 device pass alone. Other arch passes compile the body
// to NO_DEVICE_CODE, exactly like the stock flash_attn_ext_f16 kernels.
// ---------------------------------------------------------------------------
using namespace ggml_cuda_mma;

// m8n8k4 atoms. A QK mma covers 32 Q rows x 8 KV columns with an 8-deep K axis;
// a PV mma covers 32 rows x 8 output dims with an 8-deep KV axis. Both atoms
// already exist for Volta in mma.cuh, so nothing there needs to change.
using QkA = tile<32, 4, half2, DATA_LAYOUT_I_MAJOR>;           // Q operand
using QkB = tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;  // K operand
using QkC = tile<32, 8, float, DATA_LAYOUT_I_MAJOR>;           // scores
using PvA = tile<32, 4, half2, DATA_LAYOUT_I_MAJOR>;           // P operand
using PvB = tile< 8, 4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED>;  // V operand
using PvC = tile<32, 4, half2, DATA_LAYOUT_I_MAJOR>;           // output accumulator

static constexpr int kKvPerMma    = 8;                     // KV columns per QK mma
static constexpr int kDimsPerMma  = 8;                     // K-axis depth per mma
static constexpr int kQkGroups    = kQkColsPerWarp / kKvPerMma; // 2
static constexpr int kDimSteps    = kHeadDim / kDimsPerMma;     // 32
static constexpr int kPvKSteps    = kBlockN / kKvPerMma;        // 4
static constexpr int kPvDimTiles  = kOwnedDims / kDimsPerMma;   // 16

static constexpr int kStrideK_h2  = kHeadDim / 2; // half2 per KV row in smem

// Shared memory:
//   smem_K    2 x half  [kBlockN][kHeadDim]      double buffered
//   smem_V    2 x half  [kBlockN][kHeadDim]      double buffered
//   smem_mask half    [kBlockM][kBlockN]        single buffer, prefetched
//   smem_S    float [kBlockM][kSmemSPitch]      scores, shared by the pair
//
// smem_S uses a padded row pitch. With the natural kBlockN pitch every softmax
// read (thread lane reads row `lane`, iterating columns) addresses differ by
// exactly 32 banks across the warp, so all 32 lanes of a warp hit the same bank
// on every load - a full 32-way conflict three times per KV tile (max scan, sum,
// P build). The +1 float pitch spreads those reads over distinct banks. The
// same trick must NOT be escalated into a swizzle that repacks the mma
// accumulators (1Cat-vLLM measured that combination regressing ~30%).
static constexpr int kSmemSPitch        = kBlockN + 1;
static constexpr int kSmemK_Halfs       = kBlockN * kHeadDim;
static constexpr int kSmemV_Halfs       = kBlockN * kHeadDim;
static constexpr int kSmemMask_Halfs    = kBlockM * kBlockN;
static constexpr int kSmemS_Floats      = kBlockM * kSmemSPitch;

// 2x K + 2x V double buffers, one mask buffer, one padded score tile.
// Volta allows up to 96 KiB of dynamic shared memory per block; the kernel
// stays in that budget (occupancy is register-bound at one CTA per SM anyway).
static constexpr size_t kSmemBytes =
    (size_t) (2 * kSmemK_Halfs + 2 * kSmemV_Halfs + kSmemMask_Halfs) * sizeof(half) +
    (size_t) kSmemS_Floats * sizeof(float);
static_assert(kSmemBytes <= 96 * 1024, "sm70_d256 smem exceeds the Volta 96 KiB per-block limit");

// GQA head mapping: gridDim.z enumerates Q heads, the KV head is derived.

// Fill one K or V tile (kBlockN rows x kHeadDim) cooperatively and coalesced
// into half2-addressable shared memory. `base` pre-resolves the (KV head, batch)
// offset of the source tensor and `row_stride` is its row stride in half2 units.
static __device__ __forceinline__ void sm70_d256_fill_kv_tile(
        half2 * __restrict__ smem_dst, const half2 * __restrict__ base,
        const int64_t row_stride, const int kv0, const int kv_len, const int tid) {
    constexpr int kHalf2PerRow = kHeadDim / 2;
#pragma unroll 4
    for (int i = tid; i < kBlockN * kHalf2PerRow; i += kThreads) {
        const int kv_row = i / kHalf2PerRow;
        const int d2     = i % kHalf2PerRow;
        const int kv     = kv0 + kv_row;
        half2 v = make_half2(0.0f, 0.0f);
        if (kv < kv_len) {
            v = base[(int64_t) kv * row_stride + d2];
        }
        smem_dst[i] = v;
    }
}

// Fill one additive-mask tile (kBlockM rows x kBlockN cols) cooperatively.
// Masked and beyond-kv_len/beyond-q_len positions become -inf so the softmax
// flushes them.
static __device__ __forceinline__ void sm70_d256_fill_mask_tile(
        half * __restrict__ smem_mask, const half * __restrict__ mask,
        const int64_t mask_stride,        // strides in half units
        const int64_t mask_batch_stride,
        const int     mask_n_batch,       // mask batch count; broadcasts when 1
        const int batch, const int q_tile0, const int q_len,
        const int kv0, const int kv_len, const int tid) {
#pragma unroll 4
    for (int i = tid; i < kSmemMask_Halfs; i += kThreads) {
        const int r  = i / kBlockN;
        const int c  = i % kBlockN;
        const int kv = kv0 + c;
        const int qr = q_tile0 + r;
        half m = __float2half_rn(-INFINITY);
        if (kv < kv_len && qr < q_len) {
            // The mask carries its own batch axis, which broadcasts when it is 1;
            // the stock kernel offsets the same way (mask + nb33*(sequence % ne33)).
            const int mb = batch % mask_n_batch;
            m = mask[(int64_t) mb * mask_batch_stride + (int64_t) qr * mask_stride + kv];
        }
        smem_mask[i] = m;
    }
}

// Volta flash attention, D256, split-D prefill.
//
// Warp layout: warp = pair * 2 + n_warp. A pair owns kGroupRows consecutive Q rows
// and its two warps share those rows:
//   * KQ: warp n_warp computes KV columns [n_warp*kQkColsPerWarp, +kQkColsPerWarp).
//   * the f32 scores go through the pair-shared smem_S tile, so both warps see the
//     complete score row and derive the softmax independently (no cross-warp
//     reduction is needed at all).
//   * PV: warp n_warp owns output dims [n_warp*kOwnedDims, +kOwnedDims).
// Each thread ends up holding exactly one output row for the PV accumulator, which
// makes the online-softmax rescale thread-local.
//
// K/V are read with explicit strides so both the native F16 cache and the
// dequantized f16 mirror share one code path.
//
// Scheduling: K and V are double buffered and the next tile's K/V (and the next
// mask tile, single buffered) are prefetched while tile n computes, so the
// global-memory fill of tile n+1 overlaps the KQ, softmax and PV phases of tile
// n instead of serializing into it. Volta has no cp.async, so the fill is plain
// gmem -> registers -> smem; the smem stores only need the next round's KQ (and
// the fold) to be separated from them by one barrier, which the two per-round
// barriers below provide:
//
//   KQ(K[buf], mask) -> __syncthreads() [A] -> prefetch(K/V/mask of n+1) ->
//   softmax(smem_S)  -> __syncthreads() [B] -> PV(V[buf])
//
// [A] publishes the scores and retires every read of the current mask tile (the
// fold already consumed it), so the single mask buffer can be refilled; [B]
// retires all smem_S reads (the next KQ overwrites them) and completes the
// prefetch stores, which the next KQ (K[(n+1)&1]) and fold (mask) then read.
// The PV phase reads V[buf] while the prefetch writes V[buf^1], so no barrier
// after PV is needed.
__global__ __launch_bounds__(kThreads, 1)
void sm70_d256_splitd_kernel(
        const float * __restrict__ Q,      // [D][q_len][heads_q][batch] f32
        const half  * __restrict__ K,      // [D][kv_len][heads_kv][batch] f16
        const half  * __restrict__ V,      // [D][kv_len][heads_kv][batch] f16
        const half  * __restrict__ mask,   // [kv_len][q_len] f16, additive
        float       * __restrict__ dst,    // [D][heads_q][q_len][batch] f32 (note: not the Q layout)
        const int64_t q_row_stride,        // strides in float2 units
        const int64_t q_head_stride,
        const int64_t q_batch_stride,
        const int64_t k_row_stride,        // strides in half2 units
        const int64_t k_head_stride,
        const int64_t k_batch_stride,
        const int64_t v_row_stride,
        const int64_t v_head_stride,
        const int64_t v_batch_stride,
        const int64_t mask_stride,         // stride between Q rows, in half units
        const int64_t mask_batch_stride,   // stride between mask batches, in half units
        const int     mask_n_batch,        // mask batch count; the mask broadcasts when 1
        const int   * __restrict__ kv_max, // [mask batch][Q tile] exclusive KV supremum, or nullptr
        float2      * __restrict__ dst_meta, // [2][heads_q][q_len][batch] final (row_max, row_sum), or nullptr
        const int64_t dst_row_stride,      // strides in float2 units; dst is head-inner, so the
        const int64_t dst_head_stride,     // caller passes nb[2] and nb[1] here, not nb[1] and nb[2]
        const int64_t dst_batch_stride,
        const int   q_len,
        const int   kv_len,
        const int   gqa,
        const float scale) {
#if defined(VOLTA_MMA_AVAILABLE)
    // The ggml_cuda_mma tile helpers read threadIdx.x as the LANE index (the stock
    // kernels launch a (warp_size, nwarps) block for exactly this reason), so this
    // kernel keeps that 2-D shape and derives a flat thread id for the cooperative
    // shared-memory loads.
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = lane + warp * 32;

    const int pair   = warp / kWarpsPerGroup;
    const int n_warp = warp % kWarpsPerGroup;

    const int pair_row0 = pair * kGroupRows;      // first Q row of this pair
    const int q_tile0   = blockIdx.x * kBlockM;   // first Q row of this CTA
    const int head_q    = blockIdx.z;
    const int head_kv   = head_q / gqa;
    const int batch     = blockIdx.y;

    extern __shared__ __align__(16) char smem_raw[];
    half  * const smem_K0   = (half  *) smem_raw;
    half  * const smem_K1   = smem_K0 + kSmemK_Halfs;
    half  * const smem_V0   = smem_K1 + kSmemK_Halfs;
    half  * const smem_V1   = smem_V0 + kSmemV_Halfs;
    half  * const smem_mask = smem_V1 + kSmemV_Halfs;
    float * const smem_S    = (float *) (smem_mask + kSmemMask_Halfs);

    // ---------------------------------------------------------------- Q -> regs
    // Thread `lane` holds Q row (pair_row0 + lane) for the whole head dimension,
    // scaled into f16 exactly like the stock kernel (scale applied in half2).
    const half2 scale_h2 = make_half2(scale, scale);
    QkA Q_reg[kDimSteps];
    {
        const int row = q_tile0 + pair_row0 + lane;
        const bool row_ok = row < q_len;
        const float2 * const q_base = (const float2 *) Q
            + (row_ok ? (int64_t) row * q_row_stride : 0)
            + (int64_t) head_q * q_head_stride
            + (int64_t) batch  * q_batch_stride;
#pragma unroll
        for (int k = 0; k < kDimSteps; ++k) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const float2 v = row_ok ? q_base[4*k + l] : make_float2(0.0f, 0.0f);
                Q_reg[k].x[l] = scale_h2 * make_half2(v.x, v.y);
            }
        }
    }

    // ------------------------------------------------------------ KV tile bounds
    // The FA op carries no causality guarantee: the additive mask is authoritative
    // (test-backend-ops and custom callers use arbitrary masks). The launcher derives a
    // per-Q-tile KV supremum from the mask with the same on-GPU scan launch_fattn runs
    // for the tile/vec routes (flash_attn_mask_to_KV_max): every KV block at or above the
    // result is masked out for all rows of this tile, so stopping there cannot change the
    // output. Without a supremum (mask layout the scan cannot handle) every KV block is
    // processed and the mask zeroes the excluded columns.
    int kv_eff = kv_len;
    if (kv_max != nullptr) {
        // Same broadcast rule the mask tile below uses.
        const int mb     = batch % mask_n_batch;
        const int kv_sup = kv_max[mb * gridDim.x + blockIdx.x];
        kv_eff = kv_sup < kv_len ? (kv_sup > 0 ? kv_sup : 0) : kv_len;
    }
    const int n_block_max = (kv_eff + kBlockN - 1) / kBlockN;

    // PV accumulator: thread `lane` owns output row (pair_row0 + lane).
    PvC VKQ_C[kPvDimTiles];
#pragma unroll
    for (int d = 0; d < kPvDimTiles; ++d) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            VKQ_C[d].x[l] = make_half2(0.0f, 0.0f);
        }
    }

    // Per-thread softmax state for this thread's output row.
    // Initialized to a finite value rather than -INFINITY so that a KV tile whose
    // mask row is fully -inf (an entirely masked segment) contributes
    // expf(-inf - row_max) = 0 without ever materializing -FLT_MAX - -FLT_MAX;
    // the expf result itself is already 0 here, but the finite init keeps any
    // -inf - -inf = NaN hazard out of the online-softmax chain (the SplitKV3
    // implementations in the Volta ports hit exactly that edge case).
    float row_max = -FLT_MAX / 2.0f;
    float row_sum = 0.0f;

    const int row_global = q_tile0 + pair_row0 + lane;
    const bool row_ok    = row_global < q_len;

    // Pre-resolved (KV head, batch) bases for the K and V fills.
    const half2 * const k_base = (const half2 *) K
        + (int64_t) head_kv * k_head_stride + (int64_t) batch * k_batch_stride;
    const half2 * const v_base = (const half2 *) V
        + (int64_t) head_kv * v_head_stride + (int64_t) batch * v_batch_stride;

    // ---------------------------------------------------------------- prologue
    if (n_block_max > 0) {
        sm70_d256_fill_kv_tile((half2 *) smem_K0, k_base, k_row_stride, 0, kv_len, tid);
        sm70_d256_fill_kv_tile((half2 *) smem_V0, v_base, v_row_stride, 0, kv_len, tid);
        sm70_d256_fill_mask_tile(smem_mask, mask, mask_stride, mask_batch_stride, mask_n_batch,
            batch, q_tile0, q_len, 0, kv_len, tid);
    }
    __syncthreads();

    for (int n_block = 0; n_block < n_block_max; ++n_block) {
        const int buf      = n_block & 1;
        const half * cur_K = buf ? smem_K1 : smem_K0;
        const half * cur_V = buf ? smem_V1 : smem_V0;
        const int kv0 = n_block * kBlockN;

        // ------------------------------------------------------------------ KQ
        // This warp covers KV columns [n_warp*kQkColsPerWarp, +kQkColsPerWarp).
        QkC KQ_C[kQkGroups];
#pragma unroll
        for (int g = 0; g < kQkGroups; ++g) {
#pragma unroll
            for (int l = 0; l < QkC::ne; ++l) {
                KQ_C[g].x[l] = 0.0f;
            }
        }
#pragma unroll
        for (int k = 0; k < kDimSteps; ++k) {
#pragma unroll
            for (int g = 0; g < kQkGroups; ++g) {
                QkB K_A;
                ggml_cuda_mma::load_ldmatrix(
                        K_A, (const half2 *) cur_K + (n_warp * kQkColsPerWarp + g * kKvPerMma) * kStrideK_h2 + 4 * k,
                        kStrideK_h2);
                mma(KQ_C[g], Q_reg[k], K_A);
            }
        }

        // Scores -> pair-shared smem tile, with the additive mask folded in.
        // Note the padded smem_S row pitch: see kSmemSPitch.
#pragma unroll
        for (int g = 0; g < kQkGroups; ++g) {
#pragma unroll
            for (int l = 0; l < QkC::ne; ++l) {
                const int r   = pair_row0 + QkC::get_i(l);
                const int col = n_warp * kQkColsPerWarp + g * kKvPerMma + QkC::get_j(l);
                smem_S[r * kSmemSPitch + col] =
                    KQ_C[g].x[l] + __half2float(smem_mask[r * kBlockN + col]);
            }
        }
        __syncthreads(); // [A] scores visible; every read of the current mask tile is done

        // ---------------------------------------------------------------- prefetch
        // Fill the other K/V buffer and the single mask buffer with tile n+1 while
        // tile n computes below (see the scheduling comment on the kernel).
        if (n_block + 1 < n_block_max) {
            const int kv0_next = (n_block + 1) * kBlockN;
            half * const next_K = buf ? smem_K0 : smem_K1;
            half * const next_V = buf ? smem_V0 : smem_V1;
            sm70_d256_fill_kv_tile((half2 *) next_K, k_base, k_row_stride, kv0_next, kv_len, tid);
            sm70_d256_fill_kv_tile((half2 *) next_V, v_base, v_row_stride, kv0_next, kv_len, tid);
            sm70_d256_fill_mask_tile(smem_mask, mask, mask_stride, mask_batch_stride, mask_n_batch,
                batch, q_tile0, q_len, kv0_next, kv_len, tid);
        }

        // -------------------------------------------------------------- softmax
        // Thread `lane` owns output row (pair_row0 + lane) and sees all kBlockN
        // score columns, so the row reduction is entirely thread-local.
        PvA P_reg[kPvKSteps];
        if (row_ok) {
            const float * const srow = smem_S + (pair_row0 + lane) * kSmemSPitch;
            float m = -FLT_MAX / 2.0f;
#pragma unroll
            for (int c = 0; c < kBlockN; ++c) {
                m = fmaxf(m, srow[c]);
            }
            const float row_max_new = fmaxf(row_max, m + FATTN_KQ_MAX_OFFSET);
            float scale_prev = expf(row_max - row_max_new);
            // Flush tiny rescale factors to zero to avoid NaNs, as the stock kernel does.
            *((uint32_t *) &scale_prev) *= (row_max - row_max_new) >= SOFTMAX_FTZ_THRESHOLD;
            row_max = row_max_new;

            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < kBlockN; ++c) {
                const float p = expf(srow[c] - row_max_new);
                sum += p;
            }
            row_sum = scale_prev * row_sum + sum;

            // Rescale the previous PV accumulation (thread-local, one row per thread).
            const half2 scale_prev_h2 = make_half2(scale_prev, scale_prev);
#pragma unroll
            for (int d = 0; d < kPvDimTiles; ++d) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    VKQ_C[d].x[l] = VKQ_C[d].x[l] * scale_prev_h2;
                }
            }

            // P operand: 32 rows x 8 KV per mma, thread `lane` holds its own row.
#pragma unroll
            for (int k = 0; k < kPvKSteps; ++k) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const float p0 = expf(srow[k*kKvPerMma + 2*l + 0] - row_max_new);
                    const float p1 = expf(srow[k*kKvPerMma + 2*l + 1] - row_max_new);
                    P_reg[k].x[l] = make_half2(p0, p1);
                }
            }
        } else {
#pragma unroll
            for (int k = 0; k < kPvKSteps; ++k) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    P_reg[k].x[l] = make_half2(0.0f, 0.0f);
                }
            }
        }
        // [B] all threads saw the complete smem_S tile (the next KQ overwrites it) and
        // the prefetch stores of tile n+1 are complete; no barrier after PV is needed
        // because PV reads V[buf] while the next prefetch writes V[buf^1].
        __syncthreads();

        // ------------------------------------------------------------------ PV
        // Output dims [n_warp*kOwnedDims, +kOwnedDims), all kBlockN KV rows.
#pragma unroll
        for (int k = 0; k < kPvKSteps; ++k) {
#pragma unroll
            for (int d = 0; d < kPvDimTiles; ++d) {
                PvB V_B;
                ggml_cuda_mma::load_ldmatrix(
                        V_B, (const half2 *) cur_V + (k * kKvPerMma) * kStrideK_h2 + (n_warp * (kOwnedDims / 2) + 4 * d),
                        kStrideK_h2);
                mma(VKQ_C[d], P_reg[k], V_B);
            }
        }
    }

    // --------------------------------------------------------------- write back
    if (!row_ok) {
        return;
    }
    // Normalize in f32; the stock route divides in f32 in flash_attn_combine_results,
    // so doing it here in half2 would be a needless precision loss.
    const float inv_sum = 1.0f / row_sum;
    // Exact-tail meta: publish the final (max, sum) for this (row, head) in the same
    // layout the stock MMA_F16 kernels use:
    //   dst_meta[(sequence*ne01 + row)*ne02 + head] = float2(row_max, row_sum)
    // where ne02 is the Q-head count and ne03 (the batch axis here) multiplies
    // ne01*ne02. The tail merge consumes it exactly like stock-published meta.
    if (dst_meta != nullptr) {
        dst_meta[(int64_t) batch * q_len * gridDim.z + (int64_t) row_global * gridDim.z + head_q] =
            make_float2(row_max, row_sum);
    }
    float2 * const drow = (float2 *) dst
        + (int64_t) row_global * dst_row_stride
        + (int64_t) head_q     * dst_head_stride
        + (int64_t) batch      * dst_batch_stride;
#pragma unroll
    for (int d = 0; d < kPvDimTiles; ++d) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float2 v = __half22float2(VKQ_C[d].x[l]);
            drow[n_warp * (kOwnedDims / 2) + 4 * d + l] =
                make_float2(v.x * inv_sum, v.y * inv_sum);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, dst,
        q_row_stride, q_head_stride, q_batch_stride,
        k_row_stride, k_head_stride, k_batch_stride,
        v_row_stride, v_head_stride, v_batch_stride,
        mask_stride, mask_batch_stride, mask_n_batch, kv_max, dst_meta,
        dst_row_stride, dst_head_stride, dst_batch_stride,
        q_len, kv_len, gqa, scale);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

} // namespace ggml_cuda_fattn_sm70_d256