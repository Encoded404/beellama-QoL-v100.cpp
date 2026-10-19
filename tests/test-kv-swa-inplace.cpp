// Verifies the in-place current segment ("body + current" FlashAttention) that
// backs the position-mapped SWA ring.
//
// The body is the persistent sliding window; the current rows are the ubatch in
// flight and are read straight from the graph, with no packed arena. The
// backends must produce the same normalized result as one softmax over the
// union of the visible body and current rows.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

[[noreturn]] static void fail(const char * message) {
    std::fprintf(stderr, "test-kv-swa-inplace: %s\n", message);
    std::exit(1);
}

static constexpr int D       = 8;
static constexpr int N_HEAD  = 2;
static constexpr int N_BODY  = 3;
static constexpr int N_CUR   = 5;
static constexpr int N_QUERY = N_CUR;

static float value_at(int seed, int index) {
    // deterministic, modest range
    const uint32_t h = uint32_t(seed*2654435761u) ^ uint32_t(index*40503u);
    return float(int(h % 2001) - 1000)/500.0f;
}

// Reference: one softmax over [body | current] restricted to the visible rows.
static void reference(
        const std::vector<float> & q,
        const std::vector<float> & kb, const std::vector<float> & vb,
        const std::vector<float> & kc, const std::vector<float> & vc,
        const std::vector<ggml_fp16_t> & mb, const std::vector<ggml_fp16_t> & mc,
        float scale, std::vector<float> & out) {
    out.assign(size_t(D)*N_QUERY*N_HEAD, 0.0f);

    for (int ih = 0; ih < N_HEAD; ++ih) {
        for (int iq = 0; iq < N_QUERY; ++iq) {
            std::vector<double> scores;
            std::vector<const float *> rows_v;

            const auto push = [&](const float * kbase, const float * vbase, int row, int n_rows, float mask) {
                if (!std::isfinite(mask)) {
                    return;
                }
                double dot = 0.0;
                for (int d = 0; d < D; ++d) {
                    dot += double(q[size_t(d) + size_t(D)*(iq + N_QUERY*ih)]) *
                           double(kbase[size_t(d) + size_t(D)*(row + n_rows*ih)]);
                }
                scores.push_back(dot*scale + double(mask));
                rows_v.push_back(vbase + size_t(D)*(row + n_rows*ih));
            };

            for (int r = 0; r < N_BODY; ++r) {
                push(kb.data(), vb.data(), r, N_BODY, ggml_fp16_to_fp32(mb[size_t(r) + size_t(N_BODY)*iq]));
            }
            for (int r = 0; r < N_CUR; ++r) {
                push(kc.data(), vc.data(), r, N_CUR, ggml_fp16_to_fp32(mc[size_t(r) + size_t(N_CUR)*iq]));
            }

            if (scores.empty()) {
                continue;
            }

            double m = -INFINITY;
            for (double s : scores) {
                m = std::max(m, s);
            }
            double sum = 0.0;
            std::vector<double> w(scores.size());
            for (size_t i = 0; i < scores.size(); ++i) {
                w[i] = std::exp(scores[i] - m);
                sum += w[i];
            }
            for (int d = 0; d < D; ++d) {
                double acc = 0.0;
                for (size_t i = 0; i < scores.size(); ++i) {
                    acc += w[i]*double(rows_v[i][d]);
                }
                // ggml_flash_attn_ext returns [D, n_head, n_query, n_batch]
                out[size_t(d) + size_t(D)*(ih + N_HEAD*iq)] = float(acc/sum);
            }
        }
    }
}

int main() {
    // Body rows are visible to late queries only (sliding window shape), and
    // the current block is causal, so the union reproduces a real prefill.
    std::vector<ggml_fp16_t> mb(size_t(N_BODY)*N_QUERY);
    std::vector<ggml_fp16_t> mc(size_t(N_CUR)*N_QUERY);
    for (int iq = 0; iq < N_QUERY; ++iq) {
        for (int r = 0; r < N_BODY; ++r) {
            const bool visible = r >= N_BODY - iq; // a growing suffix of history
            mb[size_t(r) + size_t(N_BODY)*iq] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
        }
        for (int r = 0; r < N_CUR; ++r) {
            const bool visible = r <= iq;
            mc[size_t(r) + size_t(N_CUR)*iq] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
        }
    }

    std::vector<float> q(size_t(D)*N_QUERY*N_HEAD);
    std::vector<float> kb(size_t(D)*N_BODY*N_HEAD);
    std::vector<float> vb(kb.size());
    std::vector<float> kc(size_t(D)*N_CUR*N_HEAD);
    std::vector<float> vc(kc.size());

    for (size_t i = 0; i < q.size();  ++i) { q[i]  = value_at(1, int(i)); }
    for (size_t i = 0; i < kb.size(); ++i) { kb[i] = value_at(2, int(i)); }
    for (size_t i = 0; i < vb.size(); ++i) { vb[i] = value_at(3, int(i)); }
    for (size_t i = 0; i < kc.size(); ++i) { kc[i] = value_at(4, int(i)); }
    for (size_t i = 0; i < vc.size(); ++i) { vc[i] = value_at(5, int(i)); }

    const float scale = 1.0f/std::sqrt(float(D));

    std::vector<float> expected;
    reference(q, kb, vb, kc, vc, mb, mc, scale, expected);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        fail("failed to init CPU backend");
    }

    ggml_init_params params = { /*.mem_size =*/ 16*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * t_q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, N_QUERY, N_HEAD, 1);
    ggml_tensor * t_kb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, N_BODY, N_HEAD, 1);
    ggml_tensor * t_vb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, N_BODY, N_HEAD, 1);
    ggml_tensor * t_kc = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, N_CUR, N_HEAD, 1);
    ggml_tensor * t_vc = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, N_CUR, N_HEAD, 1);
    ggml_tensor * t_mb = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, N_BODY, N_QUERY, 1, 1);
    ggml_tensor * t_mc = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, N_CUR,  N_QUERY, 1, 1);

    // the body is exactly the window; the ubatch lives only in the graph
    ggml_tensor * body_k = t_kb;
    ggml_tensor * body_v = t_vb;

    std::memcpy(t_q->data,  q.data(),  q.size()*sizeof(float));
    std::memcpy(t_kc->data, kc.data(), kc.size()*sizeof(float));
    std::memcpy(t_vc->data, vc.data(), vc.size()*sizeof(float));
    std::memcpy(t_mb->data, mb.data(), mb.size()*sizeof(ggml_fp16_t));
    std::memcpy(t_mc->data, mc.data(), mc.size()*sizeof(ggml_fp16_t));
    std::memcpy(t_kb->data, kb.data(), kb.size()*sizeof(float));
    std::memcpy(t_vb->data, vb.data(), vb.size()*sizeof(float));

    ggml_tensor * attn = ggml_flash_attn_ext(ctx, t_q, body_k, body_v, t_mb, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_n_kv_max(attn, N_BODY);
    ggml_flash_attn_ext_set_kv_current(attn, t_kc, t_vc, t_mc);
    ggml_set_name(attn, "attn_current");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, attn);

    // host tensors from a plain context: the CPU backend reads tensor->data
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fail("graph compute failed");
    }

    std::vector<float> actual(size_t(D)*N_QUERY*N_HEAD);
    std::memcpy(actual.data(), attn->data, actual.size()*sizeof(float));

    double worst = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double d = std::fabs(actual[i] - expected[i]);
        if (d > worst) { worst = d; worst_i = i; }
    }

    std::printf("test-kv-swa-inplace: max |diff| = %.3e\n", worst);

    ggml_free(ctx);
    ggml_backend_free(backend);

    if (!(worst < 1e-4)) {
        std::fprintf(stderr, "test-kv-swa-inplace: body+current attention mismatch\n");
        return 1;
    }

    std::printf("test-kv-swa-inplace: OK\n");
    return 0;
}
