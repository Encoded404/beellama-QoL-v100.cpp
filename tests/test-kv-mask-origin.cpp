#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

// Verifies the KQ mask-origin contract end to end on the CPU backend: a mask
// that omits its leading `origin` columns must produce the same attention as
// the equivalent full-width mask, both when those columns are visible (0.0f)
// and when they are dropped (-inf).

static constexpr int D     = 64;
static constexpr int N_Q   = 24;
static constexpr int N_KV  = 1024;
static constexpr int N_HEAD = 4;

[[noreturn]] static void fail(const char * msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static void fill_q(std::vector<float> & q) {
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = 0.25f * sinf(0.017f * float(i)) + 0.1f * cosf(0.011f * float(i));
    }
}

static void fill_kv(std::vector<ggml_fp16_t> & kv) {
    for (size_t i = 0; i < kv.size(); ++i) {
        const float v = 0.3f * sinf(0.013f * float(i + 1)) + 0.2f * cosf(0.007f * float(i));
        kv[i] = ggml_fp32_to_fp16(v);
    }
}

// mask value for K/V column `ikv` and query `iq`; the mask is causal with a few
// arbitrary drop columns, and the leading `origin` columns are uniform
static float mask_value(int ikv, int iq, int origin, bool prefix_visible) {
    if (ikv < origin) {
        return prefix_visible ? 0.0f : -INFINITY;
    }
    if ((ikv % 7) == 3 && iq % 3 == 1) {
        return -INFINITY;
    }
    return ikv <= iq * (N_KV / N_Q) ? 0.0f : -INFINITY;
}

// mask tensor data; ne0 (the K/V column) is the contiguous axis, so the element
// for (query, K/V column) lives at q*rows + column. `col_base` is the K/V column
// corresponding to element 0: 0 for the full mask, `origin` for the compact one.
// Both forms describe the same attention, including the uniform prefix.
static std::vector<ggml_fp16_t> make_mask(int rows, int col_base, int origin, bool prefix_visible) {
    std::vector<ggml_fp16_t> m(size_t(rows)*N_Q);
    for (int iq = 0; iq < N_Q; ++iq) {
        for (int c = 0; c < rows; ++c) {
            m[size_t(iq)*rows + c] =
                    ggml_fp32_to_fp16(mask_value(col_base + c, iq, origin, prefix_visible));
        }
    }
    return m;
}

static std::vector<float> run(bool compact, int origin, bool prefix_visible) {
    const int mask_rows = compact ? N_KV - origin : N_KV;

    ggml_init_params params = {
        /* .mem_size   = */ 64u*1024u*1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fail("failed to initialize ggml context");
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, N_Q, N_HEAD, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, N_KV, N_HEAD, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, N_KV, N_HEAD, 1);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, mask_rows, N_Q);
    ggml_set_input(q);
    ggml_set_input(k);
    ggml_set_input(v);
    ggml_set_input(m);

    std::vector<float> qd(size_t(D)*N_Q*N_HEAD);
    fill_q(qd);
    memcpy(q->data, qd.data(), qd.size()*sizeof(float));

    std::vector<ggml_fp16_t> kv(size_t(D)*N_KV*N_HEAD);
    fill_kv(kv);
    memcpy(k->data, kv.data(), kv.size()*sizeof(ggml_fp16_t));
    memcpy(v->data, kv.data(), kv.size()*sizeof(ggml_fp16_t));

    // the compact form stores only K/V columns [origin, N_KV); the full form
    // stores every column, and the two must describe the same attention
    const auto md = make_mask(mask_rows, compact ? origin : 0, origin, prefix_visible);
    memcpy(m->data, md.data(), md.size()*sizeof(ggml_fp16_t));

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf(float(D)), 0.0f, 0.0f);
    if (compact) {
        ggml_flash_attn_ext_set_kq_mask_origin(out, origin, prefix_visible);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        fail("failed to initialize CPU backend");
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fail("graph compute failed");
    }

    std::vector<float> res(size_t(ggml_nelements(out)));
    memcpy(res.data(), out->data, res.size()*sizeof(float));

    ggml_backend_free(backend);
    ggml_free(ctx);
    return res;
}

static void check_case(int origin, bool prefix_visible) {
    const auto ref = run(false, origin, prefix_visible);
    const auto cmp = run(true,  origin, prefix_visible);

    if (ref.size() != cmp.size()) {
        fail("output size mismatch");
    }

    float max_abs = 0.0f;
    float max_ref = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(cmp[i])) {
            fprintf(stderr, "mask origin produced a non-finite output (origin=%d visible=%d)\n", origin, prefix_visible);
            exit(1);
        }
        max_abs = std::max(max_abs, std::fabs(ref[i] - cmp[i]));
        max_ref = std::max(max_ref, std::fabs(ref[i]));
    }

    const float tol = 2e-3f * std::max(1.0f, max_ref);
    if (max_abs > tol) {
        fprintf(stderr, "mask origin mismatch (origin=%d visible=%d): max_abs=%g tol=%g max_ref=%g\n",
                origin, prefix_visible, double(max_abs), double(tol), double(max_ref));
        for (size_t i = 0; i < std::min<size_t>(ref.size(), 4); ++i) {
            fprintf(stderr, "  [%zu] ref=%.6f cmp=%.6f\n", i, double(ref[i]), double(cmp[i]));
        }
        exit(1);
    }

    printf("  origin=%-5d prefix=%-8s max_abs_diff=%g\n",
            origin, prefix_visible ? "visible" : "dropped", double(max_abs));
}

int main() {
    ggml_backend_load_all();

    for (const int origin : { 0, 256, 512, N_KV - 256 }) {
        if (origin == 0) {
            // an origin of 0 must be a no-op: the compact and full forms coincide
            check_case(0, true);
            continue;
        }
        check_case(origin, true);
        // the dropped prefix exercises the -inf side of the contract
        check_case(origin, false);
    }

    printf("KV mask origin: OK\n");
    return 0;
}
