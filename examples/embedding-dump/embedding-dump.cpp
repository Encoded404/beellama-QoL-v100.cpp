#include "arg.h"
#include "chat.h"
#include "common.h"
#include "json.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn

#include <algorithm>
#include <cmath>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

//
// Minimal NPZ (zip with stored entries) writer. Each numpy array is stored as a
// single 'key.npy' entry in the zip archive, serialized in the .npy v1.0 format.
//

static void write_le32(std::vector<uint8_t> & out, uint32_t v) {
    out.push_back((uint8_t) (v & 0xFF));
    out.push_back((uint8_t) ((v >>  8) & 0xFF));
    out.push_back((uint8_t) ((v >> 16) & 0xFF));
    out.push_back((uint8_t) ((v >> 24) & 0xFF));
}

static void write_le16(std::vector<uint8_t> & out, uint16_t v) {
    out.push_back((uint8_t) (v & 0xFF));
    out.push_back((uint8_t) ((v >>  8) & 0xFF));
}

static std::vector<uint8_t> make_npy(
        const std::string & dtype,           // e.g. "<f4" or "<i4"
        const std::vector<int64_t> & shape,
        const void * data,
        size_t nbytes) {
    std::string header = std::string("{'descr': '" + dtype + "', 'fortran_order': False, 'shape': (");
    for (size_t i = 0; i < shape.size(); ++i) {
        header += std::to_string(shape[i]);
        if (i + 1 < shape.size()) {
            header += ", ";
        }
    }
    // numpy always emits a trailing comma inside the tuple and a space before
    // the closing brace, regardless of dimensionality
    header += shape.size() == 1 ? ",), }" : ",), }";

    // pad header to a 64-byte aligned total (npy v1.0 requires 16-byte alignment
    // of the data section; header + 10 bytes must be a multiple of 64 for the
    // standard numpy writer, numpy reads any alignment as long as the length is
    // recorded correctly)
    const size_t hdr_len_unaligned = header.size() + 1;
    const size_t total_unaligned   = 10 + hdr_len_unaligned + nbytes;
    const size_t pad = (64 - (total_unaligned % 64)) % 64;
    for (size_t i = 0; i < pad; ++i) {
        header += '\x20'; // spaces
    }
    header += '\n';

    std::vector<uint8_t> out;
    out.reserve(10 + header.size() + nbytes);

    const uint8_t magic[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
    out.insert(out.end(), magic, magic + 6);

    // .npy v1.0: version is a single 2-byte field { major, minor }
    out.push_back(1);
    out.push_back(0);

    const uint16_t hdr_len = (uint16_t) header.size();
    write_le16(out, hdr_len);

    out.insert(out.end(), header.begin(), header.end());

    const uint8_t * src = (const uint8_t *) data;
    out.insert(out.end(), src, src + nbytes);

    return out;
}

// Returns a complete .npz archive containing the given entries.
// Each entry is {name, array bytes}. Uses the STORE (uncompressed) zip method.
static std::vector<uint8_t> make_npz(
        const std::vector<std::pair<std::string, std::vector<uint8_t>>> & entries) {
    std::vector<uint8_t> out;
    std::vector<uint32_t> central_offsets;

    for (const auto & [name, payload] : entries) {
        const std::string fname = name + ".npy";

        // compute CRC32 of the payload
        uint32_t crc32 = 0xFFFFFFFFu;
        for (uint8_t b : payload) {
            crc32 ^= b;
            for (int k = 0; k < 8; ++k) {
                crc32 = (crc32 >> 1) ^ (0xEDB88320u & -(crc32 & 1));
            }
        }
        crc32 = ~crc32;

        central_offsets.push_back((uint32_t) out.size());

        // local file header
        write_le32(out, 0x04034b50u);          // local header signature
        write_le16(out, 20);                    // version needed to extract
        write_le16(out, 0);                     // general purpose bit flag
        write_le16(out, 0);                     // compression method: STORE
        write_le16(out, 0);                     // last mod file time
        write_le16(out, 0);                     // last mod file date
        write_le32(out, crc32);                 // crc-32
        write_le32(out, (uint32_t) payload.size()); // compressed size
        write_le32(out, (uint32_t) payload.size()); // uncompressed size
        write_le16(out, (uint16_t) fname.size());
        write_le16(out, 0);                     // extra field length
        out.insert(out.end(), fname.begin(), fname.end());
        out.insert(out.end(), payload.begin(), payload.end());
    }

    const uint32_t cd_offset = (uint32_t) out.size();

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto & [name, payload] = entries[i];
        const std::string fname = name + ".npy";

        uint32_t crc32 = 0xFFFFFFFFu;
        for (uint8_t b : payload) {
            crc32 ^= b;
            for (int k = 0; k < 8; ++k) {
                crc32 = (crc32 >> 1) ^ (0xEDB88320u & -(crc32 & 1));
            }
        }
        crc32 = ~crc32;

        write_le32(out, 0x02014b50u);           // central directory signature
        write_le16(out, 20);                    // version made by
        write_le16(out, 20);                    // version needed to extract
        write_le16(out, 0);                     // general purpose bit flag
        write_le16(out, 0);                     // compression method: STORE
        write_le16(out, 0);                     // last mod file time
        write_le16(out, 0);                     // last mod file date
        write_le32(out, crc32);
        write_le32(out, (uint32_t) payload.size());
        write_le32(out, (uint32_t) payload.size());
        write_le16(out, (uint16_t) fname.size());
        write_le16(out, 0);                     // extra field length
        write_le16(out, 0);                     // file comment length
        write_le16(out, 0);                     // disk number start
        write_le16(out, 0);                     // internal file attributes
        write_le32(out, 0);                     // external file attributes
        write_le32(out, central_offsets[i]);    // relative offset of local header
        out.insert(out.end(), fname.begin(), fname.end());
    }

    const uint32_t cd_size = (uint32_t) (out.size() - cd_offset);

    // end of central directory
    write_le32(out, 0x06054b50u);
    write_le16(out, 0);      // disk number
    write_le16(out, 0);      // disk with central directory
    write_le16(out, (uint16_t) entries.size());
    write_le16(out, (uint16_t) entries.size());
    write_le32(out, cd_size);
    write_le32(out, cd_offset);
    write_le16(out, 0);      // comment length

    return out;
}

static bool write_file(const std::string & path, const std::vector<uint8_t> & data) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    const size_t written = fwrite(data.data(), 1, data.size(), f);
    fclose(f);
    return written == data.size();
}

// Serialize a float array using the requested dump dtype. The dump targets a
// training corpus that is typically much larger than the model, so the dtype is
// selectable: f16 halves the on-disk size at a negligible cost for activations.
static std::vector<uint8_t> make_npy_floats(
        const std::string & dtype,
        const std::vector<int64_t> & shape,
        const float * data,
        size_t n) {
    if (dtype == "f16") {
        std::vector<ggml_fp16_t> half(n);
        ggml_fp32_to_fp16_row(data, half.data(), (int64_t) n);
        return make_npy("<f2", shape, half.data(), n * sizeof(ggml_fp16_t));
    }

    return make_npy("<f4", shape, data, n * sizeof(float));
}

// Parse a --dump-kv-layers spec: "" or "none" (nothing), "all", or a
// comma-separated list of layer indices. Returns false on a malformed list or an
// out-of-range index. Layer selection is explicit rather than derived from any
// one architecture, so the same flag works for any model.
static bool parse_layer_list(
        const std::string & spec,
        int32_t n_layer,
        std::vector<int32_t> & out) {
    out.clear();

    if (spec.empty() || spec == "none") {
        return true;
    }

    if (spec == "all") {
        out.resize(n_layer);
        for (int32_t il = 0; il < n_layer; ++il) {
            out[il] = il;
        }
        return true;
    }

    std::stringstream ss(spec);
    std::string item;

    while (std::getline(ss, item, ',')) {
        const size_t first = item.find_first_not_of(" \t");
        const size_t last  = item.find_last_not_of(" \t");

        if (first == std::string::npos) {
            LOG_ERR("%s: empty entry in --dump-kv-layers '%s'\n", __func__, spec.c_str());
            return false;
        }

        item = item.substr(first, last - first + 1);

        try {
            size_t pos = 0;
            const long il = std::stol(item, &pos);

            if (pos != item.size() || il < 0 || il >= n_layer) {
                LOG_ERR("%s: layer '%s' out of range in --dump-kv-layers '%s' (model has %d layers)\n",
                        __func__, item.c_str(), spec.c_str(), n_layer);
                return false;
            }

            out.push_back((int32_t) il);
        } catch (const std::exception &) {
            LOG_ERR("%s: invalid layer '%s' in --dump-kv-layers '%s'\n", __func__, item.c_str(), spec.c_str());
            return false;
        }
    }

    if (out.empty()) {
        LOG_ERR("%s: --dump-kv-layers '%s' selected no layers\n", __func__, spec.c_str());
        return false;
    }

    return true;
}

// reads a JSONL corpus: one chat document (OpenAI messages array) per line
static bool read_jsonl_corpus(
        const std::string & path,
        std::vector<std::vector<common_chat_msg>> & docs) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        LOG_ERR("%s: failed to open corpus file %s\n", __func__, path.c_str());
        return false;
    }

    char * line = nullptr;
    size_t cap  = 0;
    ssize_t len;

    int line_no = 0;
    while ((len = getline(&line, &cap, f)) != -1) {
        ++line_no;
        std::string text(line, (size_t) len);
        // strip trailing newline
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
            text.pop_back();
        }
        if (text.empty()) {
            continue;
        }

        try {
            auto obj = common_json::parse(text);
            if (!obj.contains("messages")) {
                LOG_ERR("%s: line %d: missing 'messages'\n", __func__, line_no);
                free(line);
                fclose(f);
                return false;
            }
            auto msgs = common_chat_msgs_parse_oaicompat(obj.at("messages"));
            docs.push_back(std::move(msgs));
        } catch (const std::exception & e) {
            LOG_ERR("%s: line %d: failed to parse JSON: %s\n", __func__, line_no, e.what());
            free(line);
            fclose(f);
            return false;
        }
    }

    free(line);
    fclose(f);
    return true;
}

// sanity check: hidden-state row norm statistics (post-final-norm RMS should be
// close to 1 for RMSNorm models; deviations from a bf16 reference are expected
// with IQ4_XS weights)
static void log_norm_stats(const std::vector<float> & hidden, int64_t n_rows, int64_t n_embd, const char * tag) {
    if (n_rows == 0) {
        return;
    }
    double mean_rms = 0.0, min_rms = 1e30, max_rms = 0.0;
    for (int64_t i = 0; i < n_rows; ++i) {
        double sse = 0.0;
        const float * row = hidden.data() + i * n_embd;
        for (int64_t j = 0; j < n_embd; ++j) {
            sse += (double) row[j] * (double) row[j];
        }
        const double rms = std::sqrt(sse / (double) n_embd);
        mean_rms += rms;
        min_rms = std::min(min_rms, rms);
        max_rms = std::max(max_rms, rms);
    }
    mean_rms /= (double) n_rows;
    LOG_INF("%s: hidden %s: rows=%lld mean_rms=%.4f min_rms=%.4f max_rms=%.4f\n",
            __func__, tag, (long long) n_rows, mean_rms, min_rms, max_rms);
}

// per-token role encoding: 0=user, 1=assistant, 2=system, 3=tool, 4=unknown
static constexpr int32_t ROLE_UNKNOWN = 4;

static void build_role_labels(
        const llama_tokens & tokens,
        const common_chat_msg_delimiters & delimiters,
        const llama_vocab * vocab,
        std::vector<int32_t> & roles,
        std::vector<int32_t> & turn_id) {
    roles.assign(tokens.size(), ROLE_UNKNOWN);
    turn_id.assign(tokens.size(), -1);

    if (delimiters.delimiters.empty()) {
        return;
    }

    auto delims = delimiters;
    delims.tokenize(vocab);

    const auto spans = delims.split(tokens);

    int32_t turn = 0;
    for (const auto & span : spans.spans) {
        if (!span.valid() || span.len == 0) {
            continue;
        }
        for (size_t i = span.pos; i < span.pos + span.len && i < roles.size(); ++i) {
            switch (span.role) {
                case COMMON_CHAT_ROLE_USER:      roles[i] = 0; break;
                case COMMON_CHAT_ROLE_ASSISTANT: roles[i] = 1; break;
                case COMMON_CHAT_ROLE_SYSTEM:    roles[i] = 2; break;
                case COMMON_CHAT_ROLE_TOOL:      roles[i] = 3; break;
                default:                         roles[i] = ROLE_UNKNOWN; break;
            }
        }
        // each user turn starts a new turn id; assistant/system/tool tokens carry it over
        if (span.role == COMMON_CHAT_ROLE_USER) {
            ++turn;
        }
        for (size_t i = span.pos; i < span.pos + span.len && i < turn_id.size(); ++i) {
            turn_id[i] = turn;
        }
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING)) {
        return 1;
    }

    if (params.model.path.empty()) {
        LOG_ERR("%s: --model is required\n", __func__);
        return 1;
    }
    if (params.jsonl_path.empty()) {
        LOG_ERR("%s: --jsonl is required\n", __func__);
        return 1;
    }
    if (params.dump_format != "plain" && params.dump_format != "roles" && params.dump_format != "turns") {
        LOG_ERR("%s: invalid --dump-format '%s' (expected plain, roles or turns)\n", __func__, params.dump_format.c_str());
        return 1;
    }
    if (params.dump_dtype != "f32" && params.dump_dtype != "f16") {
        LOG_ERR("%s: invalid --dump-dtype '%s' (expected f32 or f16)\n", __func__, params.dump_dtype.c_str());
        return 1;
    }

    // nextn hidden states require per-token (unpooled) output
    params.embedding   = false;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    params.embd_normalize = -1;

    llama_backend_init();
    llama_numa_init(params.numa);

    // load model and context (IQ4_XS GGUFs and -ngl all are handled here as usual)
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        LOG_ERR("%s: dumping hidden states of encoder-decoder models is not supported\n", __func__);
        return 1;
    }
    if (!llama_model_has_decoder(model)) {
        LOG_ERR("%s: model has no decoder; hidden-state dump requires a generative model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_ctx       = llama_n_ctx(ctx);
    const int32_t n_batch     = llama_n_batch(ctx);
    const int32_t n_embd_out  = llama_model_n_embd_out(model);

    LOG_INF("%s: n_ctx = %d, n_batch = %d, n_embd_out = %d\n", __func__, n_ctx, n_batch, n_embd_out);

    // per-layer K/V capture. the layers are named explicitly instead of being
    // derived from the architecture, so any model whose attention goes through
    // the standard attention builders (src/llama-graph.cpp) can be recorded.
    // this is what a shared-KV draft head reads - for a Gemma 4 MTP assistant,
    // the two target layers it borrows.
    std::vector<int32_t> kv_layers;
    if (!parse_layer_list(params.dump_kv_layers, llama_model_n_layer(model), kv_layers)) {
        return 1;
    }

    if (!kv_layers.empty()) {
        try {
            llama_set_kv_dump_layers(ctx, kv_layers.data(), kv_layers.size());
        } catch (const std::exception & e) {
            LOG_ERR("%s: --dump-kv-layers failed: %s\n", __func__, e.what());
            return 1;
        }

        for (size_t i = 0; i < llama_get_kv_dump_n_layers(ctx); ++i) {
            LOG_INF("%s: dumping layer %d: %u K + %u V floats per token\n", __func__,
                    llama_get_kv_dump_layer(ctx, i),
                    llama_get_kv_dump_n_embd_k(ctx, i),
                    llama_get_kv_dump_n_embd_v(ctx, i));
        }
    }

    // enable extraction of the post-final-norm hidden state for every token in
    // the batch (unmasked). gemma4 sets res->t_h_nextn to exactly this tensor
    // (src/models/gemma4.cpp), which is what the MTP drafter consumes as inp_h.
    if (params.dump_hidden) {
        llama_set_embeddings_nextn(ctx, true, /*masked=*/ false);
    }

    // chat templates (jinja) from the model's GGUF metadata
    common_chat_templates_ptr tmpls;
    try {
        tmpls = common_chat_templates_init(model, params.chat_template);
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to initialize chat template: %s\n", __func__, e.what());
        return 1;
    }

    // read corpus
    std::vector<std::vector<common_chat_msg>> docs;
    if (!read_jsonl_corpus(params.jsonl_path, docs)) {
        return 1;
    }
    LOG_INF("%s: loaded %zu documents from %s\n", __func__, docs.size(), params.jsonl_path.c_str());

    // ensure output dir exists
    if (!fs::exists(params.dump_outdir)) {
        std::error_code ec;
        fs::create_directories(params.dump_outdir, ec);
        if (ec) {
            LOG_ERR("%s: failed to create output dir %s: %s\n", __func__, params.dump_outdir.c_str(), ec.message().c_str());
            return 1;
        }
    }

    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    const llama_token eos_token = llama_vocab_eos(vocab);

    for (size_t doc_idx = 0; doc_idx < docs.size(); ++doc_idx) {
        const auto & msgs = docs[doc_idx];

        // apply the chat template (adds generation prompt; tokenization happens next)
        common_chat_templates_inputs inputs;
        inputs.messages          = msgs;
        inputs.add_generation_prompt = true;
        inputs.use_jinja         = params.use_jinja;

        std::string prompt;
        common_chat_msg_delimiters message_delimiters;
        try {
            auto chat_params = common_chat_templates_apply(tmpls.get(), inputs);
            prompt              = chat_params.prompt;
            message_delimiters  = chat_params.message_delimiters;
        } catch (const std::exception & e) {
            LOG_ERR("%s: doc %zu: chat template failed: %s\n", __func__, doc_idx, e.what());
            continue;
        }

        // tokenize with the chat template applied. parse_special must match the
        // server's tokenization (add_special=true, parse_special=true) so that
        // (a) the ids dumped are the real model inputs and (b) the message
        // delimiter tokens match for role/turn extraction.
        std::vector<llama_token> tokens = common_tokenize(vocab, prompt, /*add_special=*/true, /*parse_special=*/true);

        if (tokens.empty()) {
            LOG_WRN("%s: doc %zu: empty after tokenization, skipping\n", __func__, doc_idx);
            continue;
        }

        if ((int32_t) tokens.size() > n_ctx) {
            LOG_ERR("%s: doc %zu: %zu tokens exceed context size %d; increase -c\n",
                    __func__, doc_idx, tokens.size(), n_ctx);
            continue;
        }

        // run the model over the whole sequence in chunks of n_batch
        std::vector<float> hidden;
        hidden.reserve(tokens.size() * (size_t) n_embd_out);

        // per selected layer, the K/V rows accumulated across chunks as
        // [n_total, n_embd_k/v]. a layer that does not produce a tensor (it
        // caches K only, or reuses an earlier layer's K/V) stays empty.
        const size_t n_kv = llama_get_kv_dump_n_layers(ctx);

        std::vector<std::vector<float>> kv_k(n_kv);
        std::vector<std::vector<float>> kv_v(n_kv);

        const int64_t n_total = (int64_t) tokens.size();
        int64_t offset = 0;

        // clear the KV cache so each document starts from a fresh sequence
        llama_memory_clear(llama_get_memory(ctx), true);

        while (offset < n_total) {
            const int64_t n_chunk = std::min<int64_t>(n_batch, n_total - offset);

            common_batch_clear(batch);
            for (int64_t i = 0; i < n_chunk; ++i) {
                // logits only needed on the last token of the whole sequence;
                // hidden states are extracted for every token regardless
                const bool want_logits = (offset + i == n_total - 1);
                common_batch_add(batch, tokens[offset + i], offset + i, { 0 }, want_logits);
            }

            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("%s: doc %zu: llama_decode failed at offset %lld\n", __func__, doc_idx, (long long) offset);
                return 1;
            }

            // read all hidden rows for this chunk (unmasked -> dense per token)
            if (params.dump_hidden) {
                const float * h = llama_get_embeddings_nextn(ctx);
                if (h == nullptr) {
                    LOG_ERR("%s: doc %zu: no nextn embeddings; is the model graph emitting t_h_nextn?\n",
                            __func__, doc_idx);
                    return 1;
                }

                hidden.insert(hidden.end(), h, h + n_chunk * n_embd_out);

                // the nextn output buffer is sized up front but only filled when the
                // model graph sets t_h_nextn (gemma4, qwen3next/qwen35, deepseek2/4/32,
                // step35, hy-v3, ...). If the arch never sets it, the rows stay zero -
                // fail loudly instead of dumping garbage.
                const size_t n_new = (size_t) n_chunk * n_embd_out;
                bool all_zero = true;
                for (size_t i = hidden.size() - n_new; i < hidden.size(); ++i) {
                    if (hidden[i] != 0.0f) {
                        all_zero = false;
                        break;
                    }
                }
                if (all_zero) {
                    LOG_ERR("%s: doc %zu: hidden states are all zero; this model architecture does "
                            "not expose the post-final-norm hidden state (t_h_nextn)\n",
                            __func__, doc_idx);
                    return 1;
                }
            }

            // read the K/V rows of every selected layer for this chunk. each
            // llama_decode() starts filling the host buffers at row 0, so this
            // chunk occupies the first n_chunk rows.
            for (size_t i = 0; i < n_kv; ++i) {
                const size_t nk = llama_get_kv_dump_n_embd_k(ctx, i);
                const size_t nv = llama_get_kv_dump_n_embd_v(ctx, i);

                if (const float * k = llama_get_kv_dump_k(ctx, i)) {
                    kv_k[i].insert(kv_k[i].end(), k, k + (size_t) n_chunk * nk);
                }
                if (const float * v = llama_get_kv_dump_v(ctx, i)) {
                    kv_v[i].insert(kv_v[i].end(), v, v + (size_t) n_chunk * nv);
                }
            }

            offset += n_chunk;
        }

        // sanity check: post-final-norm hidden-state norms
        if (params.dump_hidden) {
            log_norm_stats(hidden, n_total, n_embd_out, "dump");
        }

        // build aligned labels: label[i] = next token id
        std::vector<int32_t> labels(tokens.size());
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            labels[i] = tokens[i + 1];
        }
        if (!tokens.empty()) {
            labels[tokens.size() - 1] = eos_token;
        }

        std::vector<int32_t> input_ids(tokens.begin(), tokens.end());

        // optional per-token role / turn metadata derived from the chat template's
        // message delimiters (e.g. gemma4: "<|turn>user" / "<|turn>model")
        std::vector<int32_t> roles;
        std::vector<int32_t> turn_id;
        if (params.dump_format == "roles" || params.dump_format == "turns") {
            build_role_labels(tokens, message_delimiters, vocab, roles, turn_id);
            if (params.dump_format == "roles") {
                turn_id.clear();
            }
        }

        // serialize to .npz. float arrays are [seq, dim]; input_ids/labels are [seq] i32.
        // the K/V arrays are named k_l<il> / v_l<il>, and "kv_layers" records which
        // layer each pair belongs to so a reader does not have to parse names.
        // V is absent for layers that cache K only.
        std::vector<std::pair<std::string, std::vector<uint8_t>>> entries;

        if (params.dump_hidden) {
            entries.push_back({ "hidden",
                    make_npy_floats(params.dump_dtype, { n_total, n_embd_out }, hidden.data(), hidden.size()) });
        }

        if (n_kv > 0) {
            std::vector<int32_t> kv_meta;
            kv_meta.reserve(n_kv);

            for (size_t i = 0; i < n_kv; ++i) {
                const int32_t il = llama_get_kv_dump_layer(ctx, i);
                const int64_t nk = llama_get_kv_dump_n_embd_k(ctx, i);
                const int64_t nv = llama_get_kv_dump_n_embd_v(ctx, i);

                kv_meta.push_back(il);

                if (!kv_k[i].empty()) {
                    GGML_ASSERT(kv_k[i].size() == (size_t) n_total*nk);
                    entries.push_back({ "k_l" + std::to_string(il),
                            make_npy_floats(params.dump_dtype, { n_total, nk }, kv_k[i].data(), kv_k[i].size()) });
                }
                if (!kv_v[i].empty()) {
                    GGML_ASSERT(kv_v[i].size() == (size_t) n_total*nv);
                    entries.push_back({ "v_l" + std::to_string(il),
                            make_npy_floats(params.dump_dtype, { n_total, nv }, kv_v[i].data(), kv_v[i].size()) });
                }
            }

            entries.push_back({ "kv_layers",
                    make_npy("<i4", { (int64_t) kv_meta.size() }, kv_meta.data(), kv_meta.size()*sizeof(int32_t)) });
        }

        entries.push_back({ "input_ids", make_npy("<i4", { n_total }, input_ids.data(), input_ids.size() * sizeof(int32_t)) });
        entries.push_back({ "labels",    make_npy("<i4", { n_total }, labels.data(),    labels.size()    * sizeof(int32_t)) });

        if (!roles.empty()) {
            auto npy_roles = make_npy("<i4", { n_total }, roles.data(), roles.size() * sizeof(int32_t));
            entries.push_back({ "roles", std::move(npy_roles) });
        }
        if (!turn_id.empty()) {
            auto npy_turn = make_npy("<i4", { n_total }, turn_id.data(), turn_id.size() * sizeof(int32_t));
            entries.push_back({ "turn_id", std::move(npy_turn) });
        }

        // note: build the path as a std::string - a fixed-size buffer silently
        // truncates a long --output-dir and writes the archive elsewhere
        char base[32];
        snprintf(base, sizeof(base), "%06zu.npz", doc_idx);

        const std::string fpath = params.dump_outdir + "/" + base;

        auto npz = make_npz(entries);

        if (!write_file(fpath, npz)) {
            LOG_ERR("%s: doc %zu: failed to write %s\n", __func__, doc_idx, fpath.c_str());
            return 1;
        }

        LOG_INF("%s: doc %zu: wrote %s (%lld tokens, %lld KiB)\n",
                __func__, doc_idx, fpath.c_str(), (long long) n_total,
                (long long) (npz.size() / 1024));
    }

    llama_batch_free(batch);

    llama_perf_context_print(ctx);

    // clean up
    llama_backend_free();

    return 0;
}
