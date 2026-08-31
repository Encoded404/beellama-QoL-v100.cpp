#include "common.cuh"
#include "fattn-common.cuh"

#ifndef GGML_USE_HIP
// Volta (sm_70) default: prefetch the next iteration's K/V rows into L2 from
// inside the VEC kernel. GGML_CUDA_FATTN_VEC_PREFETCH=0 disables (per device).
__device__ int g_ggml_cuda_fattn_vec_prefetch_l2 = 1;

static void ggml_cuda_fattn_vec_init_env_host(const int device) {
    static int initialized = 0;
    if (initialized) {
        return;
    }
    initialized = 1;

    const char * env = getenv("GGML_CUDA_FATTN_VEC_PREFETCH");
    int prefetch = 1;
    if (env != nullptr && env[0] != '\0') {
        prefetch = atoi(env);
    }

    const int n_devices = ggml_cuda_info().device_count;
    for (int d = 0; d < n_devices; ++d) {
        ggml_cuda_set_device(d);
        cudaMemcpyToSymbol(g_ggml_cuda_fattn_vec_prefetch_l2, &prefetch, sizeof(prefetch));
        CUDA_CHECK(cudaGetLastError());
    }
    ggml_cuda_set_device(device);
}

// Bytes per element for the VEC K/V loads; used only to size L2 prefetches.
template <ggml_type type>
static constexpr __device__ int ggml_cuda_fattn_vec_elem_bytes() {
    if constexpr (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        return 2;
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        return (int) sizeof(block_q8_0) / (int) QK8_0;
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        return (int) sizeof(block_q4_0) / (int) QK4_0;
    } else if constexpr (type == GGML_TYPE_Q4_1) {
        return (int) sizeof(block_q4_1) / (int) QK4_1;
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        return (int) sizeof(block_q5_0) / (int) QK5_0;
    } else if constexpr (type == GGML_TYPE_Q5_1) {
        return (int) sizeof(block_q5_1) / (int) QK5_1;
    } else if constexpr (type == GGML_TYPE_Q6_0) {
        return (int) sizeof(block_q6_0) / (int) QK6_0;
    } else if constexpr (type == GGML_TYPE_Q6_1) {
        return (int) sizeof(block_q6_1) / (int) QK6_1;
    } else if constexpr (type == GGML_TYPE_Q3_0) {
        return (int) sizeof(block_q3_0) / (int) QK3_0;
    } else if constexpr (type == GGML_TYPE_Q3_1) {
        return (int) sizeof(block_q3_1) / (int) QK3_1;
    } else if constexpr (type == GGML_TYPE_Q2_0S) {
        return (int) sizeof(block_q2_0s) / (int) QK2_0S;
    } else if constexpr (type == GGML_TYPE_Q2_1) {
        return (int) sizeof(block_q2_1) / (int) QK2_1;
    } else {
        return 2;
    }
}
#endif // GGML_USE_HIP

static int ggml_cuda_fattn_vec_get_nthreads_host(const int cc, const int D) {
    // Volta default: 256 threads (8 warps) so the streaming quantized K/V
    // loads have enough warps in flight to hide DRAM latency on the 4-scheduler
    // SMs. GGML_CUDA_FATTN_VEC_NTHREADS=128|256 overrides (clamped to the
    // compiled __launch_bounds__ maximum for the architecture; 128 restores the
    // legacy behavior on Volta).
    static const int env_nthreads = []() {
        const char * env = getenv("GGML_CUDA_FATTN_VEC_NTHREADS");
        return (env != nullptr && env[0] != '\0') ? atoi(env) : 0;
    }();

    // D=512 with fp16/bf16 V would push the shared staging buffer past the
    // 48 KB static limit at 8 warps (ne_combine = nwarps*V_cols_per_iter*D
    // = 8*4*512*4 B = 64 KB), so those instances keep 4 warps.
    const int nthreads_max = (cc == GGML_CUDA_CC_VOLTA && D <= 256) ? 256 : 128;
    if (env_nthreads >= 128 && env_nthreads <= nthreads_max && env_nthreads % 128 == 0) {
        return env_nthreads;
    }
    return nthreads_max;
}

template <int D>
static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
#ifndef GGML_USE_HIP
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
    // Must match the host side: D=512 f16/bf16 V instances keep 4 warps so the
    // shared staging buffer fits the 48 KB static limit.
    if constexpr (D <= 256) {
        return 256;
    }
#endif
#endif // GGML_USE_HIP
    return 128;
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device<D>(), 1)
static __global__ void flash_attn_ext_vec(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        float2     * dst_final_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#ifdef FLASH_ATTN_AVAILABLE
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;
    float2     * GGML_CUDA_RESTRICT dst_final_meta = dst_final_meta_ptr;

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256 || D == 512)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, dst_final_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

#ifdef GGML_USE_HIP
#ifdef RDNA
    constexpr int nthreads_KQ_q = 2;
#else
    constexpr int nthreads_KQ_q = 4;
#endif // RDNA
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#else
    constexpr int nthreads_KQ_q = (D/4 < 32 ? D/4 : 32);
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#endif // GGML_USE_HIP

    constexpr int nthreads    = ggml_cuda_fattn_vec_get_nthreads_device<D>();
    constexpr int nthreads_KQ = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16) ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr int nthreads_V  = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 128 / cpy_nb : nthreads_V_q;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 2*cpy_ne : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16 && type_K != GGML_TYPE_BF16;
#ifdef V_DOT2_F32_F16_AVAILABLE
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, half,  V_rows_per_thread>();
#else
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();
#endif // V_DOT2_F32_F16_AVAILABLE

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    // GQA-shared decode (Volta): the launch compresses the grid's z-dimension
    // by ncols2=2 so one block covers two Q heads sharing a single K/V pair,
    // halving the K reads and reusing every dequantized V row across both
    // columns. Token-mode launches always enumerate every Q head
    // (gridDim.z == ne02*ne03), so the mode is detected structurally.
    const bool gqa_shared = gridDim.z != ne02*ne03;
    const int z_per_seq = gqa_shared ? gridDim.z / ne03 : ne02;
    const int sequence = blockIdx.z / z_per_seq;
    const int head = gqa_shared ? (blockIdx.z - sequence*z_per_seq)*ncols : blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    // Column addressing: token mode strides columns over Q->ne[1] (tokens);
    // GQA-shared mode strides over Q->ne[2] (heads sharing the same K/V pair).
    const int64_t q_col_stride    = gqa_shared ? nb02 : nb01;
    const int     col0            = gqa_shared ? head : ic0;
    const int     col_max         = gqa_shared ? int(ne02) : int(ne01.z);
    const int64_t mask_col_stride = gqa_shared ? 0 : ne11;

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    // nthreads is the compile-time maximum for the architecture (it sizes the
    // shared staging buffer and __launch_bounds__); the launch may use fewer
    // warps (env-tunable), so the active counts are derived from blockDim.
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int nwarps_active    = blockDim.y;
    const int nthreads_active  = WARP_SIZE*nwarps_active;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#else
    float2           VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#endif // V_DOT2_F32_F16_AVAILABLE

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.
#else
    __align__(16) float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // V_DOT2_F32_F16_AVAILABLE
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2  Q_ds[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];

    ggml_cuda_pdl_sync();
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps_active) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps_active > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 1 && col0 + j >= col_max) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + j*q_col_stride);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*q_col_stride);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;

                __align__(16) float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (ncols == 1 || col0 + j < col_max) {
                    ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*q_col_stride);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                if (ncols == 1 || col0 + j < col_max) {
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x *= scale;
                Q_reg[j][k].y *= scale;
            }
        }
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*nthreads_active * nb11;
    V     += blockIdx.y*nthreads_active * nb21;
    maskh += blockIdx.y*nthreads_active;
    for (int k_VKQ_0 = blockIdx.y*nthreads_active; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads_active,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads_active*nb11, V += gridDim.y*nthreads_active*nb21, maskh += gridDim.y*nthreads_active) {

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (mask && (ncols == 1 || col0 + j < col_max)) {
                    sum += slope*__half2float(maskh[j*mask_col_stride + i_KQ]);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif // GGML_USE_HIP

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

#ifdef V_DOT2_F32_F16_AVAILABLE
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                if constexpr (type_V == GGML_TYPE_BF16) {
                    float2 tmp_f[V_rows_per_thread/2];
                    dequantize_V(V + k*nb21, tmp_f,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                        tmp[i_VKQ_1] = __float22half2_rn(tmp_f[i_VKQ_1]);
                    }
                } else {
                    dequantize_V(V + k*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                    }
                }
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }

#ifndef GGML_USE_HIP
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
        // Volta has no cp.async; prefetch the next iteration's K/V rows into
        // L2 while the current KQ/VKQ work is still in flight. The offsets
        // mirror this block's load pattern (best-effort; sm_70 supports
        // prefetch.global.L2). GGML_CUDA_FATTN_VEC_PREFETCH=0 disables.
        if (g_ggml_cuda_fattn_vec_prefetch_l2 != 0) {
            const char * K_next = K + gridDim.y*nthreads_active*nb11;
            const char * V_next = V + gridDim.y*nthreads_active*nb21;
            constexpr int k_elem_bytes = ggml_cuda_fattn_vec_elem_bytes<type_K>();
            constexpr int v_elem_bytes = ggml_cuda_fattn_vec_elem_bytes<type_V>();
#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
                const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;
                asm volatile("prefetch.global.L2 [%0];" :: "l"(K_next + (int64_t) i_KQ*k_elem_bytes));
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                const int64_t i_VKQ = 2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread;
                asm volatile("prefetch.global.L2 [%0];" :: "l"(V_next + i_VKQ*v_elem_bytes));
            }
        }
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
#endif // GGML_USE_HIP
    }

    if (sinks && blockIdx.y == 0) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps_active) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps_active > ncols && j >= ncols) {
                break;
            }

            const float sink = ((const float *) sinks)[head + (gqa_shared ? j : 0)];

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && col0 + j_VKQ >= col_max) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

#ifdef V_DOT2_F32_F16_AVAILABLE
        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // V_DOT2_F32_F16_AVAILABLE

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads_active <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads_active) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps_active; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                const int j_token = gqa_shared ? 0    : (ic0 + j_VKQ);
                const int j_head  = gqa_shared ? head + j_VKQ : head;
                dst[(((sequence*int(ne01.z) + j_token)*ne02 + j_head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    float2 * const meta_out = gridDim.y == 1 ? dst_final_meta : dst_meta;
    if (meta_out != nullptr && tid < ncols && (ncols == 1 || col0 + tid < col_max)) {
        const int j_token = gqa_shared ? 0    : (ic0 + tid);
        const int j_head  = gqa_shared ? head + tid : head;
        meta_out[((sequence*int(ne01.z) + j_token)*ne02 + j_head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, dst_final_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool gqa_shared) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

#ifndef GGML_USE_HIP
    ggml_cuda_fattn_vec_init_env_host(ggml_cuda_get_device());
#endif // GGML_USE_HIP

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc, D);
    const int nwarps   = nthreads / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    if (gqa_shared) {
        // One block, two GQA-shared Q heads: ncols1=1 (single token), ncols2=2
        // compresses the grid's z-dimension, which the kernel detects.
        launch_fattn<D, 1, 2>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
    } else {
        launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
    }
}

// GQA-shared VEC decode (Volta only): process two Q heads sharing one K/V pair
// per block, halving the K reads and reusing every dequantized V row across
// both columns.
// GGML_CUDA_FATTN_VEC_GQA_HEADS:
//   0    -> force off
//   2+   -> force on
//   unset or 1 -> auto: on only for contexts above 128K, where the K/V re-read
//   amortization outweighs the two-column block overhead (measured
//   neutral-to-loss at <=32K on V100).
static bool ggml_cuda_fattn_vec_gqa_shared(const int cc, const ggml_tensor * dst) {
    if (cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    if (Q->ne[1] != 1) {
        return false;
    }
    const int gqa_ratio = Q->ne[2] / K->ne[2];
    if (gqa_ratio < 2 || gqa_ratio % 2 != 0) {
        return false;
    }
    static const int env_gqa_heads = []() {
        const char * env = getenv("GGML_CUDA_FATTN_VEC_GQA_HEADS");
        return (env != nullptr && env[0] != '\0') ? atoi(env) : 1;
    }();
    if (env_gqa_heads == 0) {
        return false;
    }
    if (env_gqa_heads >= 2) {
        return true;
    }

    constexpr int64_t gqa_min_context = 128*1024; // 128K tokens
    if (K->ne[1] >= gqa_min_context) {
        return true;
    }

    static const bool warned = [](const int64_t context) {
        GGML_LOG_WARN("ggml-cuda: GGML_CUDA_FATTN_VEC_GQA_HEADS: GQA-shared VEC decode disabled for contexts <= 128K "
                      "(current context: %lld tokens). Set GGML_CUDA_FATTN_VEC_GQA_HEADS=2 to force enable, "
                      "or =0 to force disable.\n", (long long) context);
        return true;
    }(K->ne[1]);
    GGML_UNUSED(warned);

    return false;
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    const bool gqa_shared = ggml_cuda_fattn_vec_gqa_shared(ggml_cuda_info().devices[ggml_cuda_get_device()].cc, dst);

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (gqa_shared) {
            if (logit_softcap == 0.0f) {
                constexpr bool use_logit_softcap = false;
                ggml_cuda_flash_attn_ext_vec_case_impl<D, 2, type_K, type_V, use_logit_softcap>(ctx, dst, true);
            } else {
                constexpr bool use_logit_softcap = true;
                ggml_cuda_flash_attn_ext_vec_case_impl<D, 2, type_K, type_V, use_logit_softcap>(ctx, dst, true);
            }
        } else {
            if (logit_softcap == 0.0f) {
                constexpr bool use_logit_softcap = false;
                ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst, false);
            } else {
                constexpr bool use_logit_softcap = true;
                ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst, false);
            }
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst, false);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst, false);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q6_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q6_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q3_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q3_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q2_0S); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q2_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q6_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q6_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q3_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q3_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q2_0S)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q2_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q6_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q6_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q3_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q3_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q2_0S)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q2_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q6_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q6_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q3_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q3_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q2_0S)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q2_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q6_0)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q6_1)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q3_0)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q3_1)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q2_0S)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q2_1)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_BF16)
