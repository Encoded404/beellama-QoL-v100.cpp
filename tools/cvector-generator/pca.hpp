#include "common.h"
#include "llama.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#define DEBUG_POS 5

static void print_debug_tensor(struct ggml_tensor * t, bool with_data = true) {
    printf("%s: %s (%s): [%d, %d]\n", __func__, t->name, ggml_type_name(t->type), (int) t->ne[0], (int) t->ne[1]);
    if (!with_data) return;
    printf("%s: %s[0] = [", __func__, t->name);
    for (size_t i = 0; i <= DEBUG_POS; i++) {
        printf(" %f,", ggml_get_f32_nd(t, i, 0, 0, 0));
    }
    printf(" ... ]\n");
}

namespace PCA {

// input params for PCA computations
struct pca_params {
    int n_threads = 1;
    int n_batch = 20;     // only used to derive the null run's iteration budget; the batched-graph
                          // path that used it for the iteration count was removed
    int n_iterations = 1000;
    float tolerance = 1e-7;

    // seed for the power-iteration start vector. fixed so that runs are bit-reproducible
    unsigned int seed = 12345u;

    // for debugging
    int i_layer = 0;
    int n_layers = 0;
    bool verbose = true;
};

struct pca_model {
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx; // context holding the device tensors

    // tensor shapes
    int64_t n_samples = 0;
    int64_t n_embd    = 0;

    // tensors on target device
    struct ggml_tensor * dev_input;
    struct ggml_tensor * dev_square;
    struct ggml_tensor * dev_eigenvector;

    pca_model(struct ggml_tensor * t_input) {
#ifdef GGML_USE_CUDA
        // printed once rather than once per layer (and again per null permutation)
        static bool logged_backend = false;
        if (!logged_backend) {
            fprintf(stderr, "%s: using CUDA backend\n", __func__);
            logged_backend = true;
        }
        backend = ggml_backend_cuda_init(0); // init device 0
        if (!backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
#endif

// TODO: enable Metal support when support for GGML_OP_SQRT is added
// #ifdef GGML_USE_METAL
//         fprintf(stderr, "%s: using Metal backend\n", __func__);
//         backend = ggml_backend_metal_init();
//         if (!backend) {
//             fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
//         }
// #endif

        // if there aren't GPU Backends fallback to CPU backend
        if (!backend) {
            backend = ggml_backend_cpu_init();
        }

        struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * 3,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);

        n_samples = t_input->ne[0];
        n_embd    = t_input->ne[1];

        dev_input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_samples, n_embd);
        dev_square      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,    n_embd);
        dev_eigenvector = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        ggml_set_name(dev_input,       "dev_input");
        ggml_set_name(dev_square,      "dev_square");
        ggml_set_name(dev_eigenvector, "dev_eigenvector");
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(dev_input, t_input->data, 0, ggml_nbytes(t_input));
    }

    ~pca_model() {
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
    }
};

// scratch graph context for a single matvec, reused across iterations so that neither the
// metadata context nor the allocator is rebuilt on every step
struct matvec_graph {
    std::vector<uint8_t> buf;
    struct ggml_context * ctx    = nullptr;
    struct ggml_cgraph  * gf     = nullptr;
    ggml_gallocr_t        allocr = nullptr;
    struct ggml_tensor  * out    = nullptr;
    pca_model           * model  = nullptr;

    matvec_graph(pca_model & m) : model(&m) {
        buf.resize(ggml_tensor_overhead()*64 + ggml_graph_overhead());
        struct ggml_init_params p = {
            /*.mem_size   =*/ buf.size(),
            /*.mem_buffer =*/ buf.data(),
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(p);
        gf  = ggml_new_graph(ctx);
        out = ggml_mul_mat(ctx, m.dev_square, m.dev_eigenvector);
        ggml_set_name(out, "matvec_out");
        ggml_build_forward_expand(gf, out);
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        ggml_gallocr_alloc_graph(allocr, gf);
    }

    ~matvec_graph() {
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
    }

    // y = S * v, on device -> host
    std::vector<float> run(const struct pca_params & params, const std::vector<float> & v) {
        ggml_backend_tensor_set(model->dev_eigenvector, v.data(), 0, v.size() * sizeof(float));
        if (ggml_backend_is_cpu(model->backend)) {
            ggml_backend_cpu_set_n_threads(model->backend, params.n_threads);
        }
        GGML_ASSERT(ggml_backend_graph_compute(model->backend, gf) == GGML_STATUS_SUCCESS);
        std::vector<float> y(model->n_embd, 0.0f);
        ggml_backend_tensor_get(out, y.data(), 0, y.size() * sizeof(float));
        return y;
    }
};

// compute S = X X^T once and keep it in model.dev_square for the power iterations below
static void compute_square(
        const struct pca_params & params,
        pca_model & model) {
    static size_t buf_size = ggml_tensor_overhead()*64 + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params p = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(p);
    struct ggml_cgraph  * gf   = ggml_new_graph(ctx0);

    struct ggml_tensor * square = ggml_mul_mat(ctx0, model.dev_input, model.dev_input);
    ggml_set_name(square, "tmp_square");
    ggml_build_forward_expand(gf, square);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, params.n_threads);
    }

    GGML_ASSERT(ggml_backend_graph_compute(model.backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_copy(square, model.dev_square);

    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
}

// dev_square -= lambda * v v^T, so the next power iteration finds the next component
static void deflate(
        const struct pca_params & params,
        pca_model & model,
        const std::vector<float> & vec,
        float lambda) {
    static size_t buf_size = ggml_tensor_overhead()*64 + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params p = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(p);
    struct ggml_cgraph  * gf   = ggml_new_graph(ctx0);

    struct ggml_tensor * t_vec = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, model.n_embd);
    ggml_set_name(t_vec, "defl_vec");

    struct ggml_tensor * outer  = ggml_out_prod(ctx0, t_vec, t_vec); // [n_embd, n_embd]
    struct ggml_tensor * scaled = ggml_scale(ctx0, outer, lambda);
    struct ggml_tensor * result = ggml_sub(ctx0, model.dev_square, scaled);
    ggml_set_name(result, "defl_result");
    ggml_build_forward_expand(gf, result);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, params.n_threads);
    }

    ggml_backend_tensor_set(t_vec, vec.data(), 0, ggml_nbytes(t_vec));
    GGML_ASSERT(ggml_backend_graph_compute(model.backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_copy(result, model.dev_square);

    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
}

// power iteration for the dominant eigenvector of the current dev_square.
// returns the unit eigenvector on the host and, via lambda_out, the exact Rayleigh quotient v^T S v.
// resid_out (optional) receives the final ||v_next - v||, so callers can tell whether the iteration
// actually converged instead of silently trusting an unconverged vector.
//
// Only the matvec runs on the device. Normalisation, the convergence check, and the eigenvalue all
// happen on the host.
//
// The previous implementation chained every iteration inside one graph and read the final iterate
// back off the device afterwards. That is unsound in ggml: liveness is decided by the allocator
// from use counts, not by graph membership, so a tensor that is not marked with ggml_set_output()
// is considered dead once its last consumer has run and a LATER tensor is assigned the same
// address (ggml-alloc.c: the early return is gated on GGML_TENSOR_FLAG_OUTPUT, and
// ggml_build_forward_expand does not set that flag). Measured directly: intermediate iterates read
// back with norm 0.9965 instead of 1.0 and their first element equal to the following distance
// node; on a near-degenerate spectrum the returned vector collapsed to ~0 with a Rayleigh quotient
// below trace/rank, which is impossible for a real eigenvector.
static std::vector<float> power_iteration(
        const struct pca_params & params,
        pca_model & model,
        float * lambda_out,
        float * resid_out = nullptr) {
    // deterministic zero-mean start vector.
    //
    // the distribution matters: a uniform-positive vector has poor overlap with the top
    // eigenvectors of residual-stream difference matrices (which are high-frequency), so the
    // iteration would start far from the answer. a time-based seed would also make runs
    // non-reproducible.
    std::vector<float> v(model.n_embd, 0.0f);
    {
        std::default_random_engine generator(params.seed);
        std::normal_distribution<float> distribution(0.0f, 1.0f);
        double sum_sqr = 0.0;
        for (int64_t i = 0; i < model.n_embd; ++i) {
            const float f = distribution(generator);
            v[i] = f;
            sum_sqr += (double) f * (double) f;
        }
        const double nrm = std::sqrt(sum_sqr);
        if (nrm > 0.0) {
            for (int64_t i = 0; i < model.n_embd; ++i) {
                v[i] = (float) ((double) v[i] / nrm);
            }
        }
    }

    matvec_graph g(model);

    const int n_iter = params.n_iterations > 0 ? params.n_iterations : 1;
    double final_delta = 0.0;

    for (int iter = 0; iter < n_iter; ++iter) {
        const std::vector<float> y = g.run(params, v);

        double sum_sqr = 0.0;
        for (int64_t i = 0; i < model.n_embd; ++i) {
            sum_sqr += (double) y[i] * (double) y[i];
        }
        const double nrm = std::sqrt(sum_sqr);
        if (!(nrm > 0.0)) {
            break; // S v == 0, nothing left to extract
        }

        double delta = 0.0;
        for (int64_t i = 0; i < model.n_embd; ++i) {
            const float next = (float) ((double) y[i] / nrm);
            const double d = (double) next - (double) v[i];
            delta += d * d;
            v[i] = next;
        }

        const double delta_norm = std::sqrt(delta);
        final_delta = delta_norm;
        if (params.verbose && (iter % 100 == 0)) {
            printf("%s: layer %d/%d, iteration %d/%d (delta = %.3g)\n",
                   __func__, params.i_layer + 1, params.n_layers, iter + 1, n_iter, delta_norm);
        }
        if (delta_norm < params.tolerance) {
            break;
        }
    }

    if (resid_out != nullptr) {
        *resid_out = (float) final_delta;
    }

    if (lambda_out != NULL) {
        // exact Rayleigh quotient of the returned unit vector: lambda = v . (S v)
        const std::vector<float> sv = g.run(params, v);
        double lambda = 0.0;
        for (int64_t i = 0; i < model.n_embd; ++i) {
            lambda += (double) v[i] * (double) sv[i];
        }
        *lambda_out = (float) lambda;
    }

    return v;
}

// orient a direction so that it points towards the "positive" side of the data.
// an eigenvector is only defined up to sign, so without this the emitted vector is a coin flip.
// reference: mean_diff = mean(pos) - mean(neg); we want dot(v, mean_diff) > 0
static bool orient_direction(std::vector<float> & v, const std::vector<float> & mean_diff) {
    GGML_ASSERT(v.size() == mean_diff.size());
    double dot = 0.0;
    double norm_md = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        dot     += (double) v[i] * (double) mean_diff[i];
        norm_md += (double) mean_diff[i] * (double) mean_diff[i];
    }
    // if the mean shift is essentially zero the sign is not recoverable - leave the vector as-is
    if (norm_md <= 1e-12) {
        return false;
    }
    if (dot < 0.0) {
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = -v[i];
        }
    }
    return true;
}

// extract the top n_components directions of a single layer.
//
// input       : shape [n_samples, n_embd], host-side F32 data (the scatter matrix is built from it)
// outputs     : n_components vectors of length n_embd, filled with the L2-normalized directions.
//               NOTE: the sign of a principal component is arbitrary; the caller is responsible
//               for orienting it (see orient_direction() above)
// eigenvalues : filled with the corresponding eigenvalue per component
// collect_eigenvalues_only : if true, outputs is ignored and only the spectrum is computed
// residuals : optional; receives the final ||v_next - v|| per component, so the caller can tell
//             whether the requested iteration budget was actually enough
static void run_layer(
        const struct pca_params & params,
        struct ggml_tensor * input,
        int n_components,
        bool collect_eigenvalues_only,
        std::vector<std::vector<float>> & outputs,
        std::vector<float> & eigenvalues,
        std::vector<float> * residuals = nullptr) {
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    GGML_ASSERT(n_components > 0);

    pca_model model(input);
    compute_square(params, model);

    outputs.clear();
    eigenvalues.clear();
    outputs.resize(n_components);

    // at most rank(S) <= min(n_samples, n_embd) components exist
    const int64_t max_comp = std::min<int64_t>(model.n_samples, model.n_embd);
    if (n_components > max_comp) {
        fprintf(stderr, "%s: warning: requested %d components but the data has rank at most %d; clamping\n",
                __func__, n_components, (int) max_comp);
        outputs.resize(max_comp);
    }

    if (residuals != nullptr) {
        residuals->clear();
    }

    for (size_t k = 0; k < outputs.size(); ++k) {
        float lambda = 0.0f;
        float resid  = 0.0f;
        std::vector<float> v = power_iteration(params, model, &lambda, &resid);

        eigenvalues.push_back(lambda);
        if (residuals != nullptr) {
            residuals->push_back(resid);
        }
        outputs[k] = collect_eigenvalues_only ? std::vector<float>(model.n_embd, 0.0f) : v;

        if (k + 1 < outputs.size()) {
            deflate(params, model, v, lambda);
        }
    }
}

}
