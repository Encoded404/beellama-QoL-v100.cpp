#include "ggml.h"
#include "gguf.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "pca.hpp"
#include "mean.hpp"

#include <clocale>

// streaming SHA-256 lives in the vendored hash library (no external dependency).
// note: hash.h only exposes a one-shot API, which would require loading the whole
// model file into memory, so we use the incremental interface directly.
extern "C" {
#include "hash/sha256/sha256.h"
}

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>


//////////////////////////////////////////////////
// utils

template <class Iter>
static std::string tokens_to_str(llama_context * ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += common_token_to_piece(ctx, *begin);
    }

    return ret;
}

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    CPU only:   %s -m ./llama-3.Q4_K_M.gguf\n", argv[0]);
    printf("\n    with GPU:   %s -m ./llama-3.Q4_K_M.gguf -ngl 99\n", argv[0]);
    printf("\n    advanced:   %s -m ./llama-3.Q4_K_M.gguf -ngl 99 --pca-iter 2000 --pca-batch 100\n", argv[0]);
    printf("\n    using mean: %s -m ./llama-3.Q4_K_M.gguf --method mean\n", argv[0]);
    printf("\n    CAA/ASC semantics (last token only):\n");
    printf("                %s -m ./llama-3.Q4_K_M.gguf --pos-mode last\n", argv[0]);
    printf("\n    top-3 components with a pairing-breaking null:\n");
    printf("                %s -m ./llama-3.Q4_K_M.gguf --pos-mode last --n-components 3 --null-permutations 16 --stats\n", argv[0]);
    printf("\n");
    printf("note: emitted vectors are named `direction.N` and are applied to the residual\n");
    printf("      stream *at the end of layer N*, i.e. direction.N == the activation it was\n");
    printf("      measured from (l_out-N). layer 0 is not steerable and is not emitted.\n");
    printf("\n");
}

static std::string to_string_(int val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}

static std::vector<std::string> ctrlvec_load_prompt_file(std::string path, bool skip_empty_lines) {
    std::vector<std::string> output;
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "error: unable to open file: %s\n", path.c_str());
        exit(1);
    }
    std::string line;
    while (std::getline(file, line)) {
        bool is_skip = skip_empty_lines && line.empty();
        if (!is_skip) {
            string_process_escapes(line);
            output.push_back(line);
        }
    }
    file.close();
    return output;
}

static std::string hex_encode(const unsigned char * digest, size_t len) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(2*len);
    for (size_t i = 0; i < len; ++i) {
        out += hex[digest[i] >> 4];
        out += hex[digest[i] & 0xf];
    }
    return out;
}

static std::string sha256_of_string(const std::string & data) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(digest, (const unsigned char *) data.data(), data.size());
    return hex_encode(digest, SHA256_DIGEST_SIZE);
}

// streaming so that hashing a multi-GB model file does not require loading it into RAM
static std::string sha256_of_file(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return std::string();
    }
    sha256_t ctx;
    sha256_init(&ctx);
    std::vector<char> buf(1u << 20);
    while (file.good()) {
        file.read(buf.data(), buf.size());
        const std::streamsize got = file.gcount();
        if (got > 0) {
            sha256_update(&ctx, (const unsigned char *) buf.data(), (size_t) got);
        }
    }
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&ctx, digest);
    return hex_encode(digest, SHA256_DIGEST_SIZE);
}

//////////////////////////////////////////////////
// token position selection

struct pos_spec {
    cvector_position_mode mode = CVECTOR_POS_ALL;
    int start = 0;
    int end   = -1;
};

// returns the token positions that contribute to the difference, as indices into `len` positions
static std::vector<int> select_positions(const pos_spec & spec, int len) {
    std::vector<int> idx;
    switch (spec.mode) {
        case CVECTOR_POS_ALL:
            for (int i = 0; i < len; ++i) {
                idx.push_back(i);
            }
            break;
        case CVECTOR_POS_LAST:
            if (len > 0) {
                idx.push_back(len - 1);
            }
            break;
        case CVECTOR_POS_RANGE: {
            const int end   = spec.end < 0 ? len : std::min(spec.end, len);
            const int start = std::max(0, std::min(spec.start, end));
            for (int i = start; i < end; ++i) {
                idx.push_back(i);
            }
            break;
        }
    }
    return idx;
}

static const char * pos_mode_name(cvector_position_mode m) {
    switch (m) {
        case CVECTOR_POS_ALL:   return "all";
        case CVECTOR_POS_LAST:  return "last";
        case CVECTOR_POS_RANGE: return "range";
    }
    return "?";
}

struct tokenized_prompt {
    std::vector<llama_token> tokens_pos;
    std::vector<llama_token> tokens_neg;
    std::vector<int> cols_pos;
    std::vector<int> cols_neg;
    int len_pos = 0; // true (unpadded) lengths
    int len_neg = 0;
    int n_cols  = 0;

    bool init(const llama_vocab * vocab, const std::string & pos, const std::string & neg, const pos_spec & spec) {
        const bool add_bos = llama_vocab_get_add_bos(vocab);

        tokens_pos = common_tokenize(vocab, pos, add_bos, true);
        tokens_neg = common_tokenize(vocab, neg, add_bos, true);
        len_pos = (int) tokens_pos.size();
        len_neg = (int) tokens_neg.size();

        // Both sides are decoded unpadded, so no padding token is ever introduced. `all` compares
        // position-by-position over the range where BOTH sequences have real tokens; the previous
        // implementation padded the shorter side with spaces, which compared real tokens against
        // padding and then tried to filter the resulting garbage out afterwards.
        switch (spec.mode) {
            case CVECTOR_POS_ALL: {
                const int overlap = std::min(len_pos, len_neg);
                cols_pos = select_positions(spec, overlap);
                cols_neg = select_positions(spec, overlap);
                break;
            }
            case CVECTOR_POS_LAST:
            case CVECTOR_POS_RANGE:
                cols_pos = select_positions(spec, len_pos);
                cols_neg = select_positions(spec, len_neg);
                break;
        }

        if (cols_pos.empty()) {
            fprintf(stderr, "error: no token positions selected for the positive prompt "
                            "(mode=%s, len=%d, range=[%d,%d))\n",
                    pos_mode_name(spec.mode), len_pos, spec.start, spec.end);
            return false;
        }
        if (cols_pos.size() != cols_neg.size()) {
            fprintf(stderr, "error: positive and negative prompts must yield the same number of selected "
                            "positions (got %zu and %zu); pos has %d tokens, neg has %d\n",
                    cols_pos.size(), cols_neg.size(), len_pos, len_neg);
            return false;
        }

        n_cols = (int) cols_pos.size();
        return true;
    }
};

//////////////////////////////////////////////////
// hidden state capture

struct callback_data {
    int n_embd   = 0;
    int n_layers = 0;
    int n_tokens = 0;                        // token count of the sequence being evaluated
    const std::vector<int> * cols = nullptr; // selected positions for this pass
    // [layer][pair] -> floats, where pair == dest_pair is written for every layer
    std::vector<std::vector<std::vector<float>>> * dest = nullptr;
    int dest_pair = 0;

    // `il` comes from the tensor name (`l_out-<il>`), not from the capture order, so the
    // set of contributing layers is explicit and does not depend on whether the architecture
    // row-selects its final layer.
    void save(int il, struct ggml_tensor * t) {
        // layer 0 is never steerable: llama_adapter_cvec leaves tensors[0] null and apply()
        // starts at layer 1, and the loader rejects a `direction.0` tensor outright.
        // every other layer, including the last, is a candidate -- whether the final layer is
        // actually capturable depends on the architecture (see the capture validation in main).
        if (il < 1 || il >= n_layers) {
            return;
        }
        if (t->type != GGML_TYPE_F32) {
            return;
        }
        if (t->ne[0] != n_embd || t->ne[1] != n_tokens) {
            return;
        }
        if (t->nb[0] != sizeof(float) || t->nb[1] != (size_t) n_embd * sizeof(float)) {
            fprintf(stderr, "%s: warning: l_out-%d is not contiguous, skipping\n", __func__, il);
            return;
        }

        const size_t n_elem = (size_t) t->ne[0] * (size_t) t->ne[1];
        std::vector<float> tmp(n_elem);
        ggml_backend_tensor_get(t, tmp.data(), 0, n_elem * sizeof(float));

        const int n_cols = (int) cols->size();
        std::vector<float> & out = (*dest)[il][dest_pair];
        out.resize((size_t) n_cols * n_embd);
        for (int c = 0; c < n_cols; ++c) {
            const size_t src = (size_t) (*cols)[c] * n_embd;
            memcpy(out.data() + (size_t) c * n_embd, tmp.data() + src, n_embd * sizeof(float));
        }
    }
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    static const char * l_out_prefix = "l_out-";
    const bool is_l_out = strncmp(t->name, l_out_prefix, strlen(l_out_prefix)) == 0;

    if (ask) {
        return is_l_out;
    }

    if (!is_l_out) {
        return true;
    }

    int il = -1;
    if (sscanf(t->name, "l_out-%d", &il) != 1) {
        return true;
    }

    cb_data->save(il, t);
    return true;
}

//////////////////////////////////////////////////
// collected data

struct train_context {
    int n_embd   = 0;
    int n_layers = 0;
    int n_pairs  = 0;

    // number of selected positions per pair. `all` mode selects each pair's positional overlap,
    // whose length generally varies between pairs; `last` / `range` select the same count for all.
    std::vector<int> pair_n_cols;
    int n_samples = 0; // sum of pair_n_cols

    std::vector<std::string> positive_entries;
    std::vector<std::string> negative_entries;

    // [layer][pair] -> pair_n_cols[i] * n_embd floats, sample-major
    // (sample s occupies [s*n_embd, (s+1)*n_embd))
    // pos and neg are stored separately rather than pre-subtracted so that a pairing-breaking
    // null can be built from the same capture without re-running the model
    std::vector<std::vector<std::vector<float>>> pos_data;
    std::vector<std::vector<std::vector<float>>> neg_data;

    size_t block_size(int pair) const { return (size_t) pair_n_cols[pair] * n_embd; }

    size_t bytes() const {
        size_t n = 0;
        for (const auto & v : pos_data) for (const auto & x : v) n += x.size() * sizeof(float);
        for (const auto & v : neg_data) for (const auto & x : v) n += x.size() * sizeof(float);
        return n;
    }
};

// diff blocks for one layer under a given pairing permutation, sample-major
static std::vector<float> build_samples(const train_context & tc, int il, const std::vector<int> & perm) {
    std::vector<float> out((size_t) tc.n_samples * tc.n_embd);

    size_t offset = 0;
    for (int i = 0; i < tc.n_pairs; ++i) {
        const std::vector<float> & a = tc.pos_data[il][i];
        const std::vector<float> & b = tc.neg_data[il][perm[i]];
        GGML_ASSERT(a.size() == b.size());
        float * dst = out.data() + offset;
        for (size_t j = 0; j < a.size(); ++j) {
            dst[j] = a[j] - b[j];
        }
        offset += a.size();
    }
    GGML_ASSERT(offset == out.size());
    return out;
}

// PCA expects [n_samples, n_embd] laid out embedding-major (element (r,c) at c*n_samples + r)
static std::vector<float> transpose_to_embd_major(const std::vector<float> & s, int n_rows, int n_embd) {
    std::vector<float> t((size_t) n_rows * n_embd);
    for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_embd; ++c) {
            t[(size_t) c * n_rows + r] = s[(size_t) r * n_embd + c];
        }
    }
    return t;
}

static std::vector<float> mean_of_samples(const std::vector<float> & s, int n_rows, int n_embd) {
    std::vector<float> m(n_embd, 0.0f);
    for (int r = 0; r < n_rows; ++r) {
        const float * row = s.data() + (size_t) r * n_embd;
        for (int c = 0; c < n_embd; ++c) {
            m[c] += row[c];
        }
    }
    for (int c = 0; c < n_embd; ++c) {
        m[c] /= (float) n_rows;
    }
    return m;
}

// trace of the (uncentered) scatter matrix S = sum_s x_s x_s^T, computed exactly: sum_s ||x_s||^2
static double scatter_trace(const std::vector<float> & s) {
    double t = 0.0;
    for (float v : s) {
        t += (double) v * (double) v;
    }
    return t;
}

//////////////////////////////////////////////////
// export

struct provenance {
    std::string model_path;
    std::string model_sha256;
    std::string model_desc;
    uint64_t    model_size = 0;

    std::string pos_file, pos_sha256;
    std::string neg_file, neg_sha256;

    std::string method;
    std::string pos_mode;
    int pos_start = 0;
    int pos_end   = -1;
    int n_pairs   = 0;
    int n_cols    = 0; // common positions-per-pair for last/range; the first pair's count for all
    int n_samples = 0; // total sample rows that went into the reduction

    int n_components    = 1;
    int component_index = 0;
    int n_pca_batch     = 0;
    int n_pca_iterations = 0;

    int n_layers = 0;
    int n_embd   = 0;
    int layer_first = 1;  // layer index that the first emitted direction corresponds to

    std::vector<float> layer_scale; // per emitted layer: eigenvalue (pca) or pre-normalization norm (mean)
    std::vector<std::string> scale_kind; // unused, kept for clarity of intent
};

static void export_gguf(
        const std::vector<struct ggml_tensor *> & v_ctrl,
        const std::string fname,
        const std::string model_hint,
        const provenance & prov) {
    struct gguf_context * ctx = gguf_init_empty();

    const std::string arch = "controlvector";
    gguf_set_val_str(ctx, "general.architecture", arch.c_str());
    gguf_set_val_str(ctx, (arch + ".model_hint").c_str(), model_hint.c_str());
    gguf_set_val_u32(ctx, (arch + ".layer_count").c_str(), v_ctrl.size());

    // provenance: without this a vector is unidentifiable once it lands in a results table
    gguf_set_val_str(ctx, (arch + ".source_model_path").c_str(),   prov.model_path.c_str());
    gguf_set_val_str(ctx, (arch + ".source_model_sha256").c_str(), prov.model_sha256.c_str());
    gguf_set_val_str(ctx, (arch + ".source_model_desc").c_str(),   prov.model_desc.c_str());
    gguf_set_val_u64(ctx, (arch + ".source_model_size").c_str(),   prov.model_size);

    gguf_set_val_str(ctx, (arch + ".positive_file").c_str(),   prov.pos_file.c_str());
    gguf_set_val_str(ctx, (arch + ".positive_sha256").c_str(), prov.pos_sha256.c_str());
    gguf_set_val_str(ctx, (arch + ".negative_file").c_str(),   prov.neg_file.c_str());
    gguf_set_val_str(ctx, (arch + ".negative_sha256").c_str(), prov.neg_sha256.c_str());

    gguf_set_val_str(ctx, (arch + ".method").c_str(),   prov.method.c_str());
    gguf_set_val_str(ctx, (arch + ".pos_mode").c_str(), prov.pos_mode.c_str());
    gguf_set_val_u32(ctx, (arch + ".pos_start").c_str(), (uint32_t) prov.pos_start);
    // written as i32: the convention is that a negative END means "end of sequence", which would be
    // lost if this were stored as an unsigned value
    gguf_set_val_i32(ctx, (arch + ".pos_end").c_str(),   prov.pos_end);
    gguf_set_val_u32(ctx, (arch + ".n_pairs").c_str(),   (uint32_t) prov.n_pairs);
    gguf_set_val_u32(ctx, (arch + ".n_cols").c_str(),    (uint32_t) prov.n_cols);
    gguf_set_val_u32(ctx, (arch + ".n_samples").c_str(), (uint32_t) prov.n_samples);

    gguf_set_val_u32(ctx, (arch + ".n_components").c_str(),    (uint32_t) prov.n_components);
    gguf_set_val_u32(ctx, (arch + ".component_index").c_str(), (uint32_t) prov.component_index);
    gguf_set_val_u32(ctx, (arch + ".pca_batch").c_str(),       (uint32_t) prov.n_pca_batch);
    gguf_set_val_u32(ctx, (arch + ".pca_iterations").c_str(),  (uint32_t) prov.n_pca_iterations);

    gguf_set_val_u32(ctx, (arch + ".n_embd").c_str(),      (uint32_t) prov.n_embd);
    gguf_set_val_u32(ctx, (arch + ".layer_first").c_str(), (uint32_t) prov.layer_first);

    if (!prov.layer_scale.empty()) {
        gguf_set_arr_data(ctx, (arch + ".layer_scale").c_str(), GGUF_TYPE_FLOAT32,
                          prov.layer_scale.data(), prov.layer_scale.size());
    }

    for (size_t i = 0; i < v_ctrl.size(); ++i) {
        gguf_add_tensor(ctx, v_ctrl[i]);
        print_debug_tensor(v_ctrl[i]);
        printf("Added tensor: %s\n", v_ctrl[i]->name);
    }

    printf("%s: writing file...\n", __func__);
    gguf_write_to_file(ctx, fname.c_str(), false);
    printf("%s: wrote file '%s'\n", __func__, fname.c_str());
    gguf_free(ctx);
}

static std::string component_filename(const std::string & base, int component) {
    if (component == 0) {
        return base;
    }
    const std::string suffix = "_c" + to_string_(component + 1);
    const size_t dot = base.find_last_of('.');
    if (dot == std::string::npos) {
        return base + suffix;
    }
    return base.substr(0, dot) + suffix + base.substr(dot);
}

//////////////////////////////////////////////////

static int prepare_entries(common_params & params, train_context & ctx_train) {
    std::vector<std::string> positive_prompts = ctrlvec_load_prompt_file(params.cvector_positive_file, true);
    std::vector<std::string> negative_prompts = ctrlvec_load_prompt_file(params.cvector_negative_file, true);
    if (positive_prompts.size() != negative_prompts.size()) {
        fprintf(stderr, "number of positive and negative prompts must be equal\n");
        return 1;
    }
    if (positive_prompts.empty()) {
        fprintf(stderr, "must provide at least one prompt pair\n");
        return 1;
    }
    ctx_train.positive_entries = positive_prompts;
    ctx_train.negative_entries = negative_prompts;
    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    params.out_file = "control_vector.gguf";

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CVECTOR_GENERATOR, print_usage)) {
        return 1;
    }

    if (params.n_pca_batch <= 0) {
        fprintf(stderr, "error: --pca-batch must be >= 1\n");
        return 1;
    }

    if (params.n_pca_iterations <= 0) {
        fprintf(stderr, "error: --pca-iter must be >= 1\n");
        return 1;
    }

    if (params.cvector_n_components < 1) {
        fprintf(stderr, "error: --n-components must be >= 1\n");
        return 1;
    }

    if (params.cvector_dimre_method == DIMRE_METHOD_MEAN && params.cvector_n_components > 1) {
        fprintf(stderr, "error: --n-components > 1 requires --method pca\n");
        return 1;
    }

    if (params.cvector_null_perms < 0) {
        fprintf(stderr, "error: --null-permutations must be >= 0\n");
        return 1;
    }

    if (params.cvector_null_perms > 0 && params.cvector_dimre_method == DIMRE_METHOD_MEAN) {
        fprintf(stderr, "error: --null-permutations only applies to --method pca "
                        "(the mean method produces a single deterministic direction)\n");
        return 1;
    }

    if (params.cvector_pos_mode == CVECTOR_POS_RANGE && params.cvector_pos_end == 0) {
        fprintf(stderr, "error: --pos-mode range requires a non-empty --pos-range\n");
        return 1;
    }

    pos_spec spec;
    spec.mode  = params.cvector_pos_mode;
    spec.start = params.cvector_pos_start;
    spec.end   = params.cvector_pos_end;

    callback_data cb_data;

    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    llama_print_build_info(llama_version());
    llama_backend_init();
    llama_numa_init(params.numa);

    // --- 1. load the model only, so that we can tokenize before sizing the context ---
    auto llama_init = common_init_from_params(params, /*model_only=*/true);

    auto * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "error: failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_layers = llama_model_n_layer(model);
    const int n_embd   = llama_model_n_embd(model);

    // capture everything needed for provenance now: the model is released before export
    char model_hint_buf[128] = {0};
    llama_model_meta_val_str(model, "general.architecture", model_hint_buf, sizeof(model_hint_buf));
    const std::string model_hint = model_hint_buf;

    char model_desc_buf[256] = {0};
    llama_model_desc(model, model_desc_buf, sizeof(model_desc_buf));
    const std::string model_desc = model_desc_buf;

    const uint64_t model_size = llama_model_size(model);

    cb_data.n_embd   = n_embd;
    cb_data.n_layers = n_layers;

    // --- 2. tokenize everything, determine the longest sequence ---
    train_context ctx_train;
    ctx_train.n_embd   = n_embd;
    ctx_train.n_layers = n_layers;

    if (prepare_entries(params, ctx_train) != 0) {
        return 1;
    }
    ctx_train.n_pairs = (int) ctx_train.positive_entries.size();

    std::vector<tokenized_prompt> tokenized_prompts;
    tokenized_prompts.reserve(ctx_train.n_pairs);

    int max_decode_len = 0;
    int n_cols_first  = -1;

    ctx_train.pair_n_cols.resize(ctx_train.n_pairs);

    for (int i = 0; i < ctx_train.n_pairs; ++i) {
        tokenized_prompt t;
        if (!t.init(vocab, ctx_train.positive_entries[i], ctx_train.negative_entries[i], spec)) {
            return 1;
        }
        // `all` mode selects each pair's overlap, whose length generally varies between pairs;
        // `last` / `range` must agree or the position-by-position diff would be meaningless
        if (i == 0) {
            n_cols_first = t.n_cols;
        } else if (spec.mode != CVECTOR_POS_ALL && t.n_cols != n_cols_first) {
            fprintf(stderr, "error: pair %d selects %d positions but earlier pairs selected %d; "
                            "with --pos-mode %s all pairs must select the same number of positions\n",
                    i + 1, t.n_cols, n_cols_first, pos_mode_name(spec.mode));
            return 1;
        }
        ctx_train.pair_n_cols[i] = t.n_cols;
        ctx_train.n_samples += t.n_cols;

        max_decode_len = std::max(max_decode_len, (int) t.tokens_pos.size());
        max_decode_len = std::max(max_decode_len, (int) t.tokens_neg.size());
        tokenized_prompts.push_back(std::move(t));
    }

    // a pairing-breaking null swaps which negative member each positive member is differenced
    // against, which is only well defined when every pair contributes the same number of rows
    if (params.cvector_null_perms > 0) {
        if (ctx_train.n_pairs < 2) {
            fprintf(stderr, "error: --null-permutations requires at least 2 prompt pairs "
                            "(with 1 pair the only permutation is the identity, so the null would be "
                            "identical to the real run)\n");
            return 1;
        }
        for (int i = 1; i < ctx_train.n_pairs; ++i) {
            if (ctx_train.pair_n_cols[i] != ctx_train.pair_n_cols[0]) {
                fprintf(stderr,
                        "error: --null-permutations requires every pair to contribute the same number of "
                        "positions, but pair %d contributes %d and pair 1 contributes %d.\n"
                        "       Use --pos-mode last, or --pos-mode range with a fixed --pos-range."
                        "  (--pos-mode all selects each pair's own overlap length, which generally varies.)\n",
                        i + 1, ctx_train.pair_n_cols[i], ctx_train.pair_n_cols[0]);
                return 1;
            }
        }
    }

    printf("model:            %d layers, n_embd = %d\n", n_layers, n_embd);
    printf("prompt pairs:     %d\n", ctx_train.n_pairs);
    printf("pos-mode:         %s%s\n", pos_mode_name(spec.mode),
           spec.mode == CVECTOR_POS_RANGE ? (" (" + to_string_(spec.start) + ", " +
                                             (spec.end < 0 ? std::string("end") : to_string_(spec.end)) + ")").c_str() : "");
    printf("positions/pair:   %d (total samples: %d)\n", n_cols_first, ctx_train.n_samples);
    printf("longest sequence: %d tokens\n", max_decode_len);

    // activations for every pair and every layer are held in host memory until the reduction, so
    // the footprint grows with positions-per-pair. Say so before allocating rather than after.
    {
        const double gib = (double) (n_layers - 2) * ctx_train.n_pairs * ctx_train.n_samples *
                           n_embd * sizeof(float) * 2.0 / (1024.0 * 1024.0 * 1024.0);
        printf("capture estimate: %.2f GiB host RAM for %d layers x %d pairs x %d samples\n",
               gib, n_layers - 2, ctx_train.n_pairs, ctx_train.n_samples);
        if (gib > 8.0) {
            fprintf(stderr,
                    "warning: this run will hold ~%.1f GiB of activations in host RAM.\n"
                    "         Reduce --pos-range, use --pos-mode last, or use fewer pairs.\n"
                    "         (Storing positives and negatives separately is what makes the\n"
                    "          pairing-breaking null possible; --null-permutations 0 would only\n"
                    "          halve this, not fix the scaling.)\n", gib);
        }
    }

    if (n_layers - 2 <= 0) {
        fprintf(stderr, "error: model has too few layers (%d) for control vector extraction\n", n_layers);
        return 1;
    }

    // --- 3. size the context from the measured sequence length ---
    //
    // three separate constraints have to be satisfied, and missing any one of them either
    // aborts (n_batch) or silently captures nothing (n_ubatch, because the graph is built
    // per-ubatch and the capture callback matches on the full sequence length):
    //   n_ctx    >= L  (KV cache capacity)
    //   n_batch  >= L  (GGML_ASSERT in llama_context::decode)
    //   n_ubatch >= L  (otherwise l_out never has ne[1] == L)
    {
        const int need = std::max(max_decode_len, 32);
        // n_ctx is rounded up to a multiple of 256 internally; mirror that so the
        // min(n_ctx, n_batch) clamp in the context constructor cannot bite
        const int need_ctx = ((need + 255) / 256) * 256;

        if (params.n_ctx == 0 || params.n_ctx < need_ctx) {
            params.n_ctx = need_ctx;
        }
        if (params.n_batch < need) {
            params.n_batch = need;
        }
        if (params.n_ubatch < need) {
            params.n_ubatch = need;
        }

        printf("context:          n_ctx = %d, n_batch = %d, n_ubatch = %d\n",
               params.n_ctx, params.n_batch, params.n_ubatch);
    }

    // --- 4. create the context with the capture callback installed ---
    struct llama_context * ctx = nullptr;
    {
        struct llama_context_params cparams = common_context_params_to_llama(params);
        cparams.cb_eval           = cb_eval;
        cparams.cb_eval_user_data = &cb_data;

        ctx = llama_init_from_model(model, cparams);
        if (ctx == nullptr) {
            fprintf(stderr, "error: failed to create context\n");
            return 1;
        }
    }

    // --- 5. capture ---
    ctx_train.pos_data.assign(n_layers, {});
    ctx_train.neg_data.assign(n_layers, {});
    for (int il = 0; il < n_layers; ++il) {
        ctx_train.pos_data[il].resize(ctx_train.n_pairs);
        ctx_train.neg_data[il].resize(ctx_train.n_pairs);
    }

    auto run_pass = [&](const std::vector<llama_token> & tokens, const std::vector<int> & cols) -> bool {
        cb_data.n_tokens = (int) tokens.size();
        cb_data.cols     = &cols;

        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(tokens.data()), tokens.size()))) {
            fprintf(stderr, "%s: failed to eval\n", __func__);
            return false;
        }
        return true;
    };

    for (int i = 0; i < ctx_train.n_pairs; ++i) {
        const tokenized_prompt & t = tokenized_prompts[i];

        printf("Evaluating prompt[%d/%d]: \"%s\" - \"%s\" (%d / %d tokens, %d positions)\n",
               i + 1, ctx_train.n_pairs,
               tokens_to_str(ctx, t.tokens_pos.cbegin(), t.tokens_pos.cend()).c_str(),
               tokens_to_str(ctx, t.tokens_neg.cbegin(), t.tokens_neg.cend()).c_str(),
               t.len_pos, t.len_neg, t.n_cols);

        cb_data.dest      = &ctx_train.pos_data;
        cb_data.dest_pair = i;
        if (!run_pass(t.tokens_pos, t.cols_pos)) {
            return 1;
        }

        cb_data.dest      = &ctx_train.neg_data;
        cb_data.dest_pair = i;
        if (!run_pass(t.tokens_neg, t.cols_neg)) {
            return 1;
        }

        for (int il = 1; il < n_layers; ++il) {
            const size_t expect = ctx_train.block_size(i);
            const size_t got_pos = ctx_train.pos_data[il][i].size();
            const size_t got_neg = ctx_train.neg_data[il][i].size();

            if (got_pos == expect && got_neg == expect) {
                continue;
            }

            // the final layer is OPTIONAL. architectures that row-select their last layer down to
            // the output positions (via inp_out_ids, so ne[1] becomes 1 instead of n_tokens) cannot
            // contribute it; that is an architecture property, not an error.
            if (il == n_layers - 1 && got_pos == 0 && got_neg == 0) {
                continue;
            }

            fprintf(stderr, "error: layer %d was not captured for pair %d "
                            "(got %zu/%zu floats, expected %zu).\n"
                            "The capture callback matches `l_out-<il>` tensors of shape [%d, %d]; "
                            "this model may name or shape its layer outputs differently.\n",
                    il, i + 1, got_pos, got_neg, expect,
                    n_embd, (int) t.tokens_pos.size());
            return 1;
        }
    }

    printf("Done evaluating prompts, unload model...\n");

    llama_free(ctx);
    ctx = nullptr;
    llama_init.reset(); // releases the model

    printf("captured data: %.1f MiB\n", ctx_train.bytes() / (1024.0 * 1024.0));

    // the highest layer index that produced activations. everything from layer 1 up to here is a
    // candidate; the final layer is present only for architectures that do not row-select it.
    int il_last = 0;
    for (int il = 1; il < n_layers; ++il) {
        if (!ctx_train.pos_data[il][0].empty()) {
            il_last = il;
        }
    }

    if (il_last < 1) {
        fprintf(stderr, "error: no layers were captured\n");
        return 1;
    }

    // --- 6. reduce ---
    const bool use_pca = params.cvector_dimre_method == DIMRE_METHOD_PCA;

    const int n_components = params.cvector_n_components;
    const int n_out_layers = il_last; // layers 1 .. il_last -> direction.1 .. direction.il_last

    printf("layers emitted:   %d (direction.1 .. direction.%d)\n", n_out_layers, n_out_layers);
    if (il_last < n_layers - 1) {
        printf("note:              the final layer (l_out-%d) was row-selected by this architecture "
               "and is not emitted\n", n_layers - 1);
    }

    // per layer, per component directions + the scale we report for that layer
    std::vector<std::vector<std::vector<float>>> dirs(n_out_layers); // [layer][component][n_embd]
    std::vector<std::vector<float>> eigenvalues(n_out_layers);       // [layer][component]
    std::vector<float> prenorms(n_out_layers, 0.0f);                 // mean method only

    // null calibration
    std::vector<std::vector<float>> null_frac(n_out_layers);  // lambda1 / trace, per permutation
    std::vector<std::vector<float>> null_ratio(n_out_layers); // lambda1 / lambda2, per permutation

    // exact trace(S) = sum_s ||x_s||^2 per layer, so the real and null fractions are comparable
    std::vector<double> real_trace(n_out_layers, 0.0);

    // fraction of the scatter mass along the mean difference direction, and the terminal power
    // iteration residual per layer (so unconverged spectra are visible rather than silent)
    std::vector<double> real_mean_frac(n_out_layers, 0.0);
    std::vector<std::vector<float>> real_residual(n_out_layers);

    // reusable host-side tensors. the ggml context has a fixed pool, so we allocate one tensor
    // per shape up front and just repoint `->data` at the current buffer.
    const int n_samples_total = ctx_train.n_samples;

    struct ggml_init_params host_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) (16 + n_out_layers),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx_host = ggml_init(host_params);

    // PCA input: [n_samples, n_embd]; mean input: [n_embd, n_samples]
    struct ggml_tensor * t_in_pca  = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F32, n_samples_total, n_embd);
    struct ggml_tensor * t_in_mean = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F32, n_embd, n_samples_total);
    struct ggml_tensor * t_mean    = ggml_new_tensor_1d(ctx_host, GGML_TYPE_F32, n_embd);

    // one output tensor per emitted layer, reused for every component
    std::vector<struct ggml_tensor *> t_dirs(n_out_layers);
    for (int out = 0; out < n_out_layers; ++out) {
        t_dirs[out] = ggml_new_tensor_1d(ctx_host, GGML_TYPE_F32, n_embd);
        ggml_format_name(t_dirs[out], "direction.%d", out + 1);
    }

    PCA::pca_params pca_params;
    pca_params.n_threads    = params.cpuparams.n_threads;
    pca_params.n_batch      = params.n_pca_batch;
    pca_params.n_iterations = params.n_pca_iterations;
    pca_params.n_layers     = n_out_layers;
    pca_params.verbose      = false;

    PCA::pca_params null_params = pca_params;
    // NOTE: the null deliberately uses the SAME iteration budget as the real run. Comparing a
    // statistic computed under one budget against the same statistic under a different budget is
    // not a valid comparison, even though a smaller budget would be much cheaper.
    null_params.verbose = false;

    std::mt19937 rng(1234);

    for (int il = 1; il <= il_last; ++il) {
        const int out = il - 1;

        std::vector<int> perm_id(ctx_train.n_pairs);
        std::iota(perm_id.begin(), perm_id.end(), 0);

        std::vector<float> samples = build_samples(ctx_train, il, perm_id);
        const int n_samples = ctx_train.n_samples;
        const double trace = scatter_trace(samples);
        real_trace[out] = trace;
        const std::vector<float> mean_diff = mean_of_samples(samples, n_samples, n_embd);

        // Fraction of the (uncentered) scatter mass that lies along the mean difference direction.
        //
        // This is the quantity that actually matters for a steering vector: it says how much of the
        // pos-neg difference is a single shared direction. It is also permutation-INVARIANT, which
        // is why the pairing-broken null below cannot see it -- the null answers a different,
        // narrower question (does the identity pairing add structure beyond arbitrary pairings).
        {
            double mean_sqr = 0.0;
            for (float f : mean_diff) {
                mean_sqr += (double) f * (double) f;
            }
            real_mean_frac[out] = trace > 0.0 ? (double) n_samples * mean_sqr / trace : 0.0;
        }

        if (use_pca) {
            std::vector<float> embd_major = transpose_to_embd_major(samples, n_samples, n_embd);
            GGML_ASSERT((size_t) ggml_nelements(t_in_pca) == embd_major.size());
            t_in_pca->data = embd_major.data();

            pca_params.i_layer = out;
            PCA::run_layer(pca_params, t_in_pca, n_components, false, dirs[out], eigenvalues[out],
                           &real_residual[out]);

            // an eigenvector is only defined up to sign; orient it against mean(pos) - mean(neg)
            bool oriented = true;
            for (auto & d : dirs[out]) {
                oriented = PCA::orient_direction(d, mean_diff) && oriented;
            }
            if (!oriented) {
                fprintf(stderr, "warning: layer %d: mean(pos) - mean(neg) is numerically zero, "
                                "so the sign of direction.%d could not be determined and is arbitrary\n",
                        il, il);
            }

            if (params.cvector_stats) {
                printf("layer %d: ", il);
                for (size_t k = 0; k < eigenvalues[out].size(); ++k) {
                    const double frac = trace > 0.0 ? eigenvalues[out][k] / trace : 0.0;
                    printf("lambda%zu=%.6g (%.4f%%) ", k + 1, eigenvalues[out][k], 100.0 * frac);
                }
                printf("\n");
            }

            if (params.cvector_null_perms > 0) {
                for (int p = 0; p < params.cvector_null_perms; ++p) {
                    std::vector<int> perm(ctx_train.n_pairs);
                    std::iota(perm.begin(), perm.end(), 0);
                    if (ctx_train.n_pairs > 1) {
                        do {
                            std::shuffle(perm.begin(), perm.end(), rng);
                        } while (perm == perm_id);
                    }

                    std::vector<float> ns = build_samples(ctx_train, il, perm);
                    std::vector<float> nm = transpose_to_embd_major(ns, n_samples, n_embd);
                    GGML_ASSERT((size_t) ggml_nelements(t_in_pca) == nm.size());
                    t_in_pca->data = nm.data();

                    std::vector<std::vector<float>> ndirs;
                    std::vector<float> nev;
                    null_params.i_layer = out;
                    PCA::run_layer(null_params, t_in_pca, 2, true, ndirs, nev);

                    const double ntrace = scatter_trace(ns);
                    if (ntrace > 0.0 && !nev.empty()) {
                        null_frac[out].push_back((float) (nev[0] / ntrace));
                        if (nev.size() > 1 && nev[1] > 0.0f) {
                            null_ratio[out].push_back(nev[0] / nev[1]);
                        }
                    }
                    printf("\r  null %d/%d", p + 1, params.cvector_null_perms);
                    fflush(stdout);
                }
                printf("\r");
            }
        } else {
            std::vector<float> out_vec(n_embd, 0.0f);
            GGML_ASSERT((size_t) ggml_nelements(t_in_mean) == samples.size());
            t_in_mean->data = samples.data();
            t_mean->data    = out_vec.data();

            mean::run_layer(t_in_mean, t_mean, &prenorms[out]);

            dirs[out].push_back(out_vec);

            if (params.cvector_stats) {
                printf("layer %d: prenorm=%.6g\n", il, prenorms[out]);
            }
        }

        // release this layer's captured activations
        ctx_train.pos_data[il].clear();
        ctx_train.pos_data[il].shrink_to_fit();
        ctx_train.neg_data[il].clear();
        ctx_train.neg_data[il].shrink_to_fit();
    }

    // --- 7. stats / null summary ---
    if (params.cvector_stats) {
        auto mean_sd = [](const std::vector<float> & v) {
            double m = 0.0;
            for (float x : v) m += x;
            m /= (double) v.size();
            double s = 0.0;
            for (float x : v) s += (x - m) * (x - m);
            s = std::sqrt(s / (double) std::max<size_t>(1, v.size() - 1));
            return std::make_pair(m, s);
        };

        printf("\n=== summary ===\n");
        if (use_pca) {
            printf("layer   mean_frac   real l1/trace");
            if (params.cvector_null_perms > 0) {
                printf("   null l1/trace (pairing-only)");
            }
            printf("   real l1/l2");
            if (params.cvector_null_perms > 0) {
                printf("   null l1/l2   max_resid");
            }
            printf("\n");
        }
        for (int out = 0; out < n_out_layers; ++out) {
            const int il = out + 1;
            if (use_pca) {
                const double real_frac = real_trace[out] > 0.0 ? eigenvalues[out][0] / real_trace[out] : 0.0;
                const bool have_ratio = eigenvalues[out].size() > 1 && eigenvalues[out][1] > 0.0f;
                const double real_ratio = have_ratio ? eigenvalues[out][0] / eigenvalues[out][1] : 0.0;
                float max_resid = 0.0f;
                for (float r : real_residual[out]) {
                    max_resid = std::max(max_resid, r);
                }

                printf("layer %-3d  %.4f     %.4f", il, real_mean_frac[out], real_frac);
                if (params.cvector_null_perms > 0) {
                    if (!null_frac[out].empty()) {
                        const auto f = mean_sd(null_frac[out]);
                        printf("            %.4f +/- %.4f", f.first, f.second);
                    } else {
                        printf("            n/a");
                    }
                }
                if (have_ratio) {
                    printf("        %.4f", real_ratio);
                } else {
                    // do NOT print 0.0 here: with a single extracted component lambda2 does not
                    // exist, and a bare "0.0000" reads as a catastrophic measurement rather than
                    // as "not computed"
                    printf("        n/a (need --n-components 2)");
                }
                if (params.cvector_null_perms > 0) {
                    if (!null_ratio[out].empty()) {
                        const auto r = mean_sd(null_ratio[out]);
                        printf("      %.4f +/- %.4f", r.first, r.second);
                    } else {
                        printf("      n/a");
                    }
                    printf("   %.2e", max_resid);
                }
                printf("\n");
            } else {
                printf("layer %-3d  prenorm=%.6g\n", il, prenorms[out]);
            }
        }
        printf("\n");

        // an unconverged power iteration silently under-reports the eigenvalue, so surface it
        if (use_pca) {
            float worst = 0.0f;
            int worst_layer = 0;
            for (int out = 0; out < n_out_layers; ++out) {
                for (float r : real_residual[out]) {
                    if (r > worst) {
                        worst = r;
                        worst_layer = out + 1;
                    }
                }
            }
            if (worst > pca_params.tolerance) {
                printf("warning: power iteration did not reach the tolerance of %.1e (worst residual "
                       "%.3e at layer %d). The reported eigenvalues are lower bounds; raise --pca-iter.\n",
                       pca_params.tolerance, worst, worst_layer);
            }
        }

        if (params.cvector_null_perms > 0) {
            printf("how to read this:\n");
            printf("  mean_frac is the fraction of the scatter mass along the mean difference direction.\n");
            printf("  It is the quantity that matters for a steering vector, and it is permutation-INVARIANT,\n");
            printf("  so the null below cannot see it by construction: a strong SHARED direction produces\n");
            printf("  real == null. The null only answers 'does the identity pairing add structure beyond\n");
            printf("  arbitrary pairings of the same negatives'. real >> null means pairing-specific\n");
            printf("  structure; real ~= null does NOT mean 'no structure', only 'no pairing-specific\n");
            printf("  structure'. With few pairs the null also has little power (4 pairs => 23 distinct\n");
            printf("  permutations, minimum one-sided p ~ 0.04).\n\n");
        }
    }

    // --- 8. export ---
    provenance prov;
    prov.model_path  = params.model.path;
    prov.model_desc  = model_desc;
    prov.model_size  = model_size;
    prov.pos_file    = params.cvector_positive_file;
    prov.neg_file    = params.cvector_negative_file;
    prov.method      = use_pca ? "pca" : "mean";
    prov.pos_mode    = pos_mode_name(spec.mode);
    prov.pos_start   = spec.start;
    prov.pos_end     = spec.end;
    prov.n_pairs     = ctx_train.n_pairs;
    prov.n_cols      = n_cols_first; // common count for last/range; the first pair's count for all
    prov.n_samples   = ctx_train.n_samples;
    prov.n_components     = n_components;
    prov.n_pca_batch      = params.n_pca_batch;
    prov.n_pca_iterations = params.n_pca_iterations;
    prov.n_layers    = n_layers;
    prov.n_embd      = n_embd;
    prov.layer_first = 1;

    // hashes of the *effective* prompt sets (after escape processing and blank-line skipping)
    prov.pos_sha256 = sha256_of_string([&] {
        std::string s;
        for (const auto & e : ctx_train.positive_entries) s += e + "\n";
        return s;
    }());
    prov.neg_sha256 = sha256_of_string([&] {
        std::string s;
        for (const auto & e : ctx_train.negative_entries) s += e + "\n";
        return s;
    }());
    if (!params.cvector_no_hash) {
        printf("hashing model file (use --no-hash to skip)...\n");
        fflush(stdout);
        prov.model_sha256 = sha256_of_file(params.model.path);
        if (prov.model_sha256.empty()) {
            fprintf(stderr, "warning: could not hash model file '%s'\n", params.model.path.c_str());
        }
    }

    // build the per-component output files
    for (int c = 0; c < n_components; ++c) {
        std::vector<std::vector<float>> storage;
        storage.resize(n_out_layers);

        bool missing = false;
        for (int out = 0; out < n_out_layers; ++out) {
            if (c >= (int) dirs[out].size()) {
                missing = true;
                break;
            }
            storage[out] = dirs[out][c];
        }
        if (missing) {
            if (c == 0) {
                fprintf(stderr, "error: no components extracted\n");
                return 1;
            }
            printf("component %d not available for all layers, stopping\n", c + 1);
            break;
        }

        std::vector<struct ggml_tensor *> v_final(n_out_layers);
        for (int out = 0; out < n_out_layers; ++out) {
            GGML_ASSERT(storage[out].size() == (size_t) n_embd);
            t_dirs[out]->data = storage[out].data();
            v_final[out] = t_dirs[out];
        }

        provenance prov_c = prov;
        prov_c.component_index = c;
        prov_c.layer_scale.clear();
        for (int out = 0; out < n_out_layers; ++out) {
            prov_c.layer_scale.push_back(use_pca ? eigenvalues[out][c] : prenorms[out]);
        }

        export_gguf(v_final, component_filename(params.out_file, c), model_hint, prov_c);
    }

    ggml_free(ctx_host);

    llama_backend_free();

    return 0;
}
