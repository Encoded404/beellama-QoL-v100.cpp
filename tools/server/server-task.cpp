#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"
#include "server-prompt-persist.h"

#include <chrono>
#include <sstream>

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_loop_guard",      common_reasoning_loop_guard_mode_name(reasoning_loop_guard.mode)},
            {"reasoning_loop_min_tokens", reasoning_loop_guard.min_reasoning_tokens},
            {"reasoning_loop_window",     reasoning_loop_guard.window_tokens},
            {"reasoning_loop_max_period", reasoning_loop_guard.max_period},
            {"reasoning_loop_min_coverage", reasoning_loop_guard.min_repeated_coverage},
            {"reasoning_loop_check_interval", reasoning_loop_guard.check_interval},
            {"reasoning_loop_interventions", reasoning_loop_guard.interventions_max},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_loop_guard",      common_reasoning_loop_guard_mode_name(reasoning_loop_guard.mode)},
        {"reasoning_loop_min_tokens", reasoning_loop_guard.min_reasoning_tokens},
        {"reasoning_loop_window",     reasoning_loop_guard.window_tokens},
        {"reasoning_loop_max_period", reasoning_loop_guard.max_period},
        {"reasoning_loop_min_coverage", reasoning_loop_guard.min_repeated_coverage},
        {"reasoning_loop_check_interval", reasoning_loop_guard.check_interval},
        {"reasoning_loop_interventions", reasoning_loop_guard.interventions_max},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(const common_chat_parser_params & chat_parser_params)
    : chat_parser_params(chat_parser_params)
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
    if (chat_parser_params.is_continuation && !chat_parser_params.echo) {
        // initialize chat_msg to avoid emitting a delta containing the assistant prefill
        chat_msg = common_chat_parse("", true, chat_parser_params);
    }
}

common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
        generated_text,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // the JSON library converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stop_detail",         stop_detail},
        {"reasoning_tokens",    reasoning_output_tokens},
        {"visible_completion_tokens", visible_output_tokens},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             stats.to_json()},
    };
    if (loop_guard_event.triggered) {
        res["loop_guard"] = json {
            {"triggered", true},
            {"region", loop_guard_event.region},
            {"detector", loop_guard_event.detector},
            {"period", loop_guard_event.period},
            {"coverage", loop_guard_event.coverage},
            {"score", loop_guard_event.score},
            {"interventions", loop_guard_event.interventions},
            {"action", loop_guard_event.action},
            {"decoded_token_index", loop_guard_event.decoded_token_index},
            {"token", loop_guard_event.token},
            {"token_piece", loop_guard_event.token_piece},
            {"reason", loop_guard_event.reason},
        };
    }
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    json usage = json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
    usage["completion_tokens_details"] = json {
        {"reasoning_tokens", reasoning_output_tokens},
        {"visible_tokens", visible_output_tokens},
    };
    return usage;
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (stats.is_set()) {
        deltas.back()["timings"] = stats.to_json();
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (stats.is_set()) {
        server_sent_events.back().at("data")["timings"] = stats.to_json();
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (stats.is_set()) {
            last_json["timings"] = stats.to_json();
        }
        if (is_progress) {
            last_json["prompt_progress"] = progress.to_json();
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (stats.is_set()) {
            data["timings"] = stats.to_json();
        }
        if (is_progress) {
            data["prompt_progress"] = progress.to_json();
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_decision
//
json server_task_result_decision::to_json() {
    return json {
        {"index",            index},
        {"scores",           scores},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_slots::to_json() {
    return slots_data;
}

json server_task_result_metrics::to_json() {
    // not used, /metrics renders prometheus text via to_metrics()
    return json{};
}

// metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
std::string server_task_result_metrics::to_metrics() {
    const std::vector<metric_item> counters = {
        {
            "prompt_tokens_total",
            "Number of prompt tokens processed, excluding cached tokens",
            (double) metrics.prompt.count
        }, {
            "prompt_tokens_cached_total",
            "Number of prompt tokens reused from the cache",
            (double) metrics.n_prompt_cached
        }, {
            "prompt_seconds_total",
            "Total time spent processing prompts",
            metrics.prompt.time / 1.e6
        }, {
            "tokens_predicted_total",
            "Number of generation tokens processed",
            (double) metrics.predict.count
        }, {
            "tokens_predicted_seconds_total",
            "Total time spent generating tokens",
            metrics.predict.time / 1.e6
        }, {
            "n_decode_total",
            "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
            (double) metrics.n_decode
        }, {
            "n_tokens_max",
            "Largest observed sequence length (prompt + generation)",
            (double) metrics.n_tokens_max
        }, {
            "spec_decode_num_draft_tokens_total",
            "Speculative: Total draft tokens generated",
            (double) metrics.n_draft_tokens
        }, {
            "spec_decode_num_accepted_tokens_total",
            "Speculative: Total draft tokens accepted by the target model",
            (double) metrics.n_draft_accepted
        }, {
            "spec_decode_num_drafts_total",
            "Speculative: Total speculative decoding verification steps",
            (double) metrics.n_draft_verif_steps
        }, {
            "prompt_cache_admission_attempts_total",
            "Total immutable RAM prompt-cache admission attempts",
            (double) metrics.prompt_cache_admission_attempts
        }, {
            "prompt_cache_admission_successes_total",
            "Total immutable RAM prompt-cache admissions",
            (double) metrics.prompt_cache_admission_successes
        }, {
            "prompt_cache_admission_failures_total",
            "Total rejected RAM prompt-cache admissions",
            (double) metrics.prompt_cache_admission_failures
        }, {
            "prompt_cache_restore_attempts_total",
            "Total transactional RAM prompt-cache restore attempts",
            (double) metrics.prompt_cache_restore_attempts
        }, {
            "prompt_cache_restore_successes_total",
            "Total committed RAM prompt-cache restores",
            (double) metrics.prompt_cache_restore_successes
        }, {
            "prompt_cache_restore_failures_total",
            "Total aborted RAM prompt-cache restores",
            (double) metrics.prompt_cache_restore_failures
        },
    };

    const std::vector<metric_item> gauges = {
        {
            "prompt_tokens_seconds",
            "Average prompt throughput in tokens/s",
            metrics.prompt_bucket.n_per_second()
        }, {
            "predicted_tokens_seconds",
            "Average generation throughput in tokens/s",
            metrics.predict_bucket.n_per_second()
        }, {
            "requests_processing",
            "Number of requests processing",
            (double) n_processing_slots
        }, {
            "requests_deferred",
            "Number of requests deferred",
            (double) n_tasks_deferred
        }, {
            "n_busy_slots_per_decode",
            "Average number of busy slots per llama_decode() call",
            (double) metrics.n_busy_slots / std::max((double) metrics.n_decode, 1.0)
        }, {
            "kv_tail_requested_tokens",
            "Configured exact-tail tokens currently requested across server slots and cache groups",
            (double) metrics.kv_tail_requested
        }, {
            "kv_tail_exact_tokens",
            "Exact-tail tokens currently covered across server slots and cache groups",
            (double) metrics.kv_tail_exact
        }, {
            "kv_tail_complete_groups",
            "Server slot cache groups with complete exact-tail coverage",
            (double) metrics.kv_tail_complete_groups
        }, {
            "kv_tail_partial_groups",
            "Server slot cache groups with partial exact-tail coverage",
            (double) metrics.kv_tail_partial_groups
        }, {
            "kv_tail_none_groups",
            "Server slot cache groups with no exact-tail coverage",
            (double) metrics.kv_tail_none_groups
        }, {
            "kv_tail_degraded_sequences",
            "Server slots reporting an explicit exact-tail degradation reason",
            (double) metrics.kv_tail_degraded_sequences
        }, {
            "prompt_cache_accounted_bytes",
            "Serialized RAM prompt-cache payload bytes, excluding container and allocator overhead",
            (double) metrics.prompt_cache_accounted_bytes
        }, {
            "prompt_cache_disk_spill_total",
            "Prompt-cache entries successfully spilled to the automatic disk swap tier",
            (double) metrics.prompt_cache_disk_spill_successes
        }, {
            "prompt_cache_disk_restore_total",
            "Prompt-cache restores served from the automatic disk swap tier",
            (double) metrics.prompt_cache_disk_restore_successes
        }, {
            "prompt_cache_disk_entries",
            "Prompt-cache entries currently held on disk",
            (double) metrics.prompt_cache_disk_entries
        }, {
            "prompt_cache_disk_bytes",
            "Serialized prompt-cache payload bytes currently held on disk",
            (double) metrics.prompt_cache_disk_bytes
        }, {
            "prompt_cache_persist_entries",
            "Prompt-cache entries currently held by the persistent prompt-cache tier",
            (double) metrics.prompt_cache_persist_entries
        }, {
            "prompt_cache_persist_bytes",
            "Serialized prompt-cache payload bytes currently held by the persistent prompt-cache tier",
            (double) metrics.prompt_cache_persist_bytes
        },
    };

    std::stringstream prometheus;

    auto add_items = [&prometheus](const char * type, const std::vector<metric_item> & items) {
        for (const auto & item : items) {
            prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                       << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
                       << "llamacpp:"        << item.name << " " << item.value       << "\n";
        }
    };

    add_items("counter", counters);
    add_items("gauge",   gauges);

    // labeled counter: one time series per draft position
    if (!metrics.n_accepted_per_pos.empty()) {
        prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                      " Accepted tokens per draft position\n"
                   << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
        for (size_t i = 0; i < metrics.n_accepted_per_pos.size(); i++) {
            prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""
                       << i << "\"} " << metrics.n_accepted_per_pos[i] << "\n";
        }
    }

    return prometheus.str();
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

server_prompt_cache::server_prompt_cache(
        int32_t limit_size_mib, size_t limit_tokens,
        server_prompt_disk::config disk_config, uint64_t disk_identity,
        server_prompt_disk::config persist_config, uint64_t persist_identity)
    : disk(std::move(disk_config), disk_identity),
      persist(std::move(persist_config), persist_identity) {
    this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
    this->limit_tokens = limit_tokens;
}

server_prompt_cache::~server_prompt_cache() {
    // release every file this cache owns in the session swap tier; the store
    // is session-scoped. Files owned by the persistent tier survive the
    // process and are re-adopted by the next run's identity check.
    for (auto & state : states) {
        if (!state.disk.persist_tier) {
            disk.erase(state.disk);
        }
    }
    states.clear();
    accounted_bytes_cached = 0;
}

size_t server_prompt_cache::accounted_size() const {
    size_t res = 0;
    std::unordered_set<const void *> checkpoint_storage;

    for (const auto & state : states) {
        res += state.data.size();
        for (const auto & checkpoint : state.prompt.checkpoints) {
            if (!checkpoint.data_tgt.empty() &&
                    checkpoint_storage.insert(checkpoint.data_tgt.storage_id()).second) {
                res += checkpoint.data_tgt.size();
            }
            if (!checkpoint.data_dft.empty() &&
                    checkpoint_storage.insert(checkpoint.data_dft.storage_id()).second) {
                res += checkpoint.data_dft.size();
            }
            res += checkpoint.data_spec.size();
        }
    }

    return res;
}

size_t server_prompt_cache::accounted_disk_size() const {
    return disk.bytes_on_disk();
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

size_t server_prompt_cache::n_tokens_ram() const {
    size_t res = 0;

    for (const auto & state : states) {
        if (!state.disk.on_disk()) {
            res += state.prompt.n_tokens();
        }
    }

    return res;
}

void server_prompt_cache::disk_flush() {
    disk.flush();
}

void server_prompt_cache::refresh_accounted_bytes() {
    accounted_bytes_cached = accounted_size();
}

server_prompt_cache_state * server_prompt_cache::admit(server_prompt_cache_state && candidate) {
    ++admission_attempts;
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(candidate.prompt.tokens);

        if (cur_lcp_len == (int) candidate.prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            ++admission_failures;
            return nullptr;
        }
    }

    // Count retained immutable checkpoint buffers once across cache entries.
    // Copying a server_prompt shares these buffers and detaches on mutation.
    std::unordered_set<const void *> retained_checkpoint_storage;
    for (const auto & state : states) {
        for (const auto & checkpoint : state.prompt.checkpoints) {
            retained_checkpoint_storage.insert(checkpoint.data_tgt.storage_id());
            retained_checkpoint_storage.insert(checkpoint.data_dft.storage_id());
        }
    }
    size_t checkpoints_size = 0;
    for (const auto & ckpt : candidate.prompt.checkpoints) {
        if (!ckpt.data_tgt.empty() &&
                retained_checkpoint_storage.insert(ckpt.data_tgt.storage_id()).second) {
            checkpoints_size += ckpt.data_tgt.size();
        }
        if (!ckpt.data_dft.empty() &&
                retained_checkpoint_storage.insert(ckpt.data_dft.storage_id()).second) {
            checkpoints_size += ckpt.data_dft.size();
        }
        checkpoints_size += ckpt.data_spec.size();
    }

    const size_t state_size_new = candidate.data.size() + checkpoints_size;
    const size_t standalone_size = candidate.accounted_size();

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && (state_size_new > limit_size || standalone_size > limit_size)) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                standalone_size / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        ++admission_failures;
        return nullptr;
    }

    // Publish the already-materialized candidate node before touching existing
    // entries.  list node allocation is the last fallible admission step; if it
    // fails, the cache is unchanged.
    try {
        states.push_back(std::move(candidate));
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate prompt cache entry: %s\n", e.what());
        ++admission_failures;
        return nullptr;
    }
    auto admitted = std::prev(states.end());

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        if (it == admitted) {
            ++it;
            continue;
        }
        const int len = it->prompt.tokens.get_common_prefix(admitted->prompt.tokens);

        if (len == (int) it->prompt.tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = erase_iterator(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // The candidate is already present and fully materialized. Evict only
        // older entries until unique payload accounting is within budget.
        while (!states.empty() && accounted_size() > limit_size) {
            demote_oldest_ram();
        }
    }

    enforce_disk_budget();
    refresh_accounted_bytes();

    ++admission_successes;
    return &*admitted;
}

server_prompt_cache_state * server_prompt_cache::insert(
        const server_prompt & prompt,
        server_prompt_data && data) {
    server_prompt_cache_state candidate;
    candidate.prompt = prompt.clone();
    candidate.data = std::move(data);
    return admit(std::move(candidate));
}

server_prompt_cache_state * server_prompt_cache::alloc(
        const server_prompt & prompt,
        size_t state_size_tgt,
        size_t state_size_dft) {
    server_prompt_data data;
    try {
        data.main.resize(state_size_tgt);
        data.drft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        ++admission_attempts;
        ++admission_failures;
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());
        return nullptr;
    }
    return insert(prompt, std::move(data));
}

bool server_prompt_cache::erase(const server_prompt_cache_state * entry) {
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (&*it == entry) {
            erase_iterator(it);
            refresh_accounted_bytes();
            return true;
        }
    }
    return false;
}

std::list<server_prompt_cache_state>::iterator server_prompt_cache::erase_iterator(
        std::list<server_prompt_cache_state>::iterator it) {
    // the file is owned by whichever tier the entry was persisted into
    (it->disk.persist_tier ? persist : disk).erase(it->disk);
    return states.erase(it);
}

void server_prompt_cache::erase_front() {
    erase_iterator(states.begin());
}

bool server_prompt_cache::demote_oldest_ram() {
    if (states.empty()) {
        return false;
    }

    // Disk-resident entries contribute nothing to the RAM budget, so skip them
    // instead of rewriting a file that cannot relieve any pressure.
    auto it = std::find_if(states.begin(), states.end(), [](const server_prompt_cache_state & state) {
        return !state.disk.on_disk();
    });

    if (it == states.end()) {
        // no RAM payload is left to free; drop the oldest disk entry
        SRV_WRN(" - dropping oldest disk entry (size = %.3f MiB)\n", states.front().disk.bytes / (1024.0 * 1024.0));
        erase_front();
        return false;
    }

    server_prompt_data & data = it->data;

    const llama_tokens & toks = it->prompt.tokens.get_tokens();

    if (disk.enabled() && !data.main.empty()) {
        const int64_t t_spill_start = ggml_time_us();

        // take ownership of the payload: the writer frees it, so the spill costs
        // no extra copy and the RAM is released as soon as the job is queued
        const bool spilled = disk.spill(it->disk,
                    std::move(data.main), std::move(data.drft), std::move(data.spec),
                    it->prompt.checkpoints,
                    (uint64_t) toks.size(),
                    server_prompt_disk::hash_bytes(toks.data(), toks.size()*sizeof(llama_token)));

        disk_spill_us += uint64_t(ggml_time_us() - t_spill_start);

        if (spilled) {
            SRV_WRN(" - moving oldest RAM entry to disk (size = %.3f MiB)\n", it->disk.bytes / (1024.0 * 1024.0));
            // the RAM payload is already owned by the disk tier; release the
            // checkpoint buffers too so no serialized state is pinned in RAM
            for (auto & ckpt : it->prompt.checkpoints) {
                ckpt.clear_tgt();
                ckpt.clear_dft();
                std::vector<uint8_t>().swap(ckpt.data_spec);
            }
            return true;
        }

        // the payload was consumed by the failed spill, so the entry can no
        // longer serve either tier
        SRV_WRN("%s", " - dropping oldest entry after a failed disk spill\n");
        erase_iterator(it);
        return false;
    }

    SRV_WRN(" - dropping oldest entry (size = %.3f MiB)\n", it->accounted_size() / (1024.0 * 1024.0));
    erase_iterator(it);
    return false;
}

void server_prompt_cache::enforce_disk_budget() {
    if (!disk.enabled() || disk.bytes_on_disk() <= disk.limit_bytes()) {
        return;
    }

    for (auto it = states.begin(); it != states.end() && disk.bytes_on_disk() > disk.limit_bytes();) {
        if (!it->disk.on_disk()) {
            ++it;
            continue;
        }

        SRV_WRN(" - disk swap budget reached, removing oldest disk entry (size = %.3f MiB)\n",
                it->disk.bytes / (1024.0 * 1024.0));

        it = erase_iterator(it);
    }
}

server_prompt_restore_result server_prompt_restore_transaction_diagnostic(
        server_prompt_state_view target,
        server_prompt_state_view draft,
        server_prompt_state_view speculative,
        const server_prompt_restore_transaction_io & io,
        server_prompt_restore_timings * timings) {
    if (!io.prepare || !io.commit) {
        return { false, false, SERVER_PROMPT_STATE_MAIN, SERVER_PROMPT_RESTORE_INVALID_IO };
    }
    if (io.restore_target && target.size == 0) {
        return { false, true, SERVER_PROMPT_STATE_MAIN, SERVER_PROMPT_RESTORE_MISSING_REQUIRED_STATE };
    }
    if (io.restore_draft && draft.size == 0) {
        return { false, true, SERVER_PROMPT_STATE_DRAFT, SERVER_PROMPT_RESTORE_MISSING_REQUIRED_STATE };
    }

    const auto prepare = [&](bool enabled, server_prompt_state_kind kind, server_prompt_state_view state) {
        if (enabled) {
            const auto t_start = std::chrono::steady_clock::now();
            const bool prepared = io.prepare(kind, state);
            if (timings) {
                timings->prepare_us += std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t_start).count();
            }
            if (!prepared) {
                return server_prompt_restore_result {
                    false, true, kind, SERVER_PROMPT_RESTORE_PREPARE_REJECTED
                };
            }
        }
        return server_prompt_restore_result {
            true, false, SERVER_PROMPT_STATE_MAIN, SERVER_PROMPT_RESTORE_NONE
        };
    };
    for (const auto & step : {
            std::pair { io.restore_target, SERVER_PROMPT_STATE_MAIN },
            std::pair { io.restore_draft, SERVER_PROMPT_STATE_DRAFT },
            std::pair { io.restore_speculative, SERVER_PROMPT_STATE_SPECULATIVE } }) {
        const server_prompt_state_view state = step.second == SERVER_PROMPT_STATE_MAIN ? target :
                step.second == SERVER_PROMPT_STATE_DRAFT ? draft : speculative;
        const auto result = prepare(step.first, step.second, state);
        if (!result.success) {
            return result;
        }
    }

    // Speculative apply is prepared and no-fail. Memory commits likewise only
    // publish already-validated backend writes and metadata.
    const auto commit = [&](bool enabled, server_prompt_state_kind kind) {
        if (!enabled) {
            return;
        }
        const auto t_start = std::chrono::steady_clock::now();
        io.commit(kind);
        if (timings) {
            timings->commit_us += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_start).count();
        }
    };
    commit(io.restore_speculative, SERVER_PROMPT_STATE_SPECULATIVE);
    commit(io.restore_target, SERVER_PROMPT_STATE_MAIN);
    commit(io.restore_draft, SERVER_PROMPT_STATE_DRAFT);
    return { true, false, SERVER_PROMPT_STATE_MAIN, SERVER_PROMPT_RESTORE_NONE };
}

bool server_prompt_restore_transaction(
        server_prompt_state_view target,
        server_prompt_state_view draft,
        server_prompt_state_view speculative,
        const server_prompt_restore_transaction_io & io) {
    return server_prompt_restore_transaction_diagnostic(target, draft, speculative, io).success;
}

server_prompt_restore_result server_prompt_restore_transaction_diagnostic(
        llama_context * target,
        llama_context * draft,
        common_speculative * speculative,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        server_prompt_state_view target_state,
        server_prompt_state_view draft_state,
        server_prompt_state_view speculative_state,
        bool restore_target,
        bool restore_draft,
        bool restore_speculative,
        server_prompt_restore_timings * timings) {
    using memory_plan_ptr = std::unique_ptr<
            llama_state_seq_restore_plan,
            decltype(&llama_state_seq_restore_plan_free)>;
    using speculative_plan_ptr = std::unique_ptr<
            common_speculative_state_restore_plan,
            decltype(&common_speculative_state_restore_plan_free)>;

    memory_plan_ptr target_plan(nullptr, llama_state_seq_restore_plan_free);
    memory_plan_ptr draft_plan(nullptr, llama_state_seq_restore_plan_free);
    speculative_plan_ptr speculative_plan(nullptr, common_speculative_state_restore_plan_free);

    server_prompt_restore_transaction_io io {
        /*.restore_target =*/ restore_target,
        /*.restore_draft =*/ restore_draft,
        /*.restore_speculative =*/ restore_speculative,
        /*.prepare =*/ [&](server_prompt_state_kind kind, server_prompt_state_view state) {
            if (kind == SERVER_PROMPT_STATE_SPECULATIVE) {
                speculative_plan.reset(common_speculative_prepare_state(
                        speculative, seq_id, state.data, state.size));
                return speculative_plan != nullptr;
            }

            llama_context * ctx = kind == SERVER_PROMPT_STATE_MAIN ? target : draft;
            memory_plan_ptr & plan = kind == SERVER_PROMPT_STATE_MAIN ? target_plan : draft_plan;
            if (ctx == nullptr) {
                return false;
            }
            plan.reset(llama_state_seq_prepare_data_ext(
                    ctx, state.data, state.size, seq_id, flags));
            return plan != nullptr;
        },
        /*.commit =*/ [&](server_prompt_state_kind kind) {
            if (kind == SERVER_PROMPT_STATE_SPECULATIVE) {
                common_speculative_state_restore_plan_commit(speculative_plan.get());
                return;
            }
            memory_plan_ptr & plan = kind == SERVER_PROMPT_STATE_MAIN ? target_plan : draft_plan;
            const size_t expected = kind == SERVER_PROMPT_STATE_MAIN ? target_state.size : draft_state.size;
            GGML_ASSERT(llama_state_seq_restore_plan_commit(plan.get()) == expected);
        },
    };
    return server_prompt_restore_transaction_diagnostic(
            target_state, draft_state, speculative_state, io, timings);
}

bool server_prompt_restore_transaction(
        llama_context * target,
        llama_context * draft,
        common_speculative * speculative,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        server_prompt_state_view target_state,
        server_prompt_state_view draft_state,
        server_prompt_state_view speculative_state,
        bool restore_target,
        bool restore_draft,
        bool restore_speculative) {
    return server_prompt_restore_transaction_diagnostic(
            target, draft, speculative, seq_id, flags,
            target_state, draft_state, speculative_state,
            restore_target, restore_draft, restore_speculative).success;
}

bool server_prompt_cache::load(
        server_prompt & prompt,
        const server_tokens & tokens_new,
        size_t live_native_restorable_tokens,
        int32_t reuse_alignment,
        const server_prompt_cache_state_io & io,
        const server_prompt_cache_state * excluded) {
    const auto live_plan = server_prompt_plan_reuse(
            prompt, tokens_new, reuse_alignment, live_native_restorable_tokens, false);

    size_t restorable_best = live_plan.restorable_tokens;
    float f_keep_best = prompt.tokens.size() > 0 ?
            float(restorable_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(restorable_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (&*it == excluded) {
            continue;
        }
        const auto plan_cur = server_prompt_plan_reuse(
                it->prompt, tokens_new, reuse_alignment, 0, true);
        const size_t lcp_cur = plan_cur.lexical_tokens;
        const size_t restorable_cur = plan_cur.restorable_tokens;

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7zu, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (restorable_cur > restorable_best ||
                (restorable_cur == restorable_best && f_keep_cur > f_keep_best)) {
            restorable_best = restorable_cur;
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }
    }

    if (it_best != states.end()) {
        ++restore_attempts;
        SRV_TRC(" - found better restorable prompt with n = %zu, f_keep = %.3f, f_sim = %.3f\n",
                restorable_best, f_keep_best, f_sim_best);

        auto & data = it_best->data;

        // A demoted entry owns no RAM payload; materialize it from disk. The
        // mapping must outlive the restore transaction because the transaction
        // reads tensor data straight out of it.
        std::unique_ptr<server_prompt_disk::mapping> mapping;
        const uint8_t * p_main = data.main.data(); size_t n_main = data.main.size();
        const uint8_t * p_drft = data.drft.data(); size_t n_drft = data.drft.size();
        const uint8_t * p_spec = data.spec.data(); size_t n_spec = data.spec.size();

        if (it_best->disk.on_disk()) {
            const int64_t t_read_start = ggml_time_us();
            // materialize from whichever tier owns the entry's file; the
            // persistent tier's read path validates identity the same way
            mapping = it_best->disk.persist_tier
                    ? persist.open(it_best->disk)
                    : disk.open(it_best->disk);
            disk_read_us += uint64_t(ggml_time_us() - t_read_start);
            if (!mapping) {
                // open() already invalidated the entry: it has no RAM payload and
                // no readable file, so it must not stay selectable.
                ++restore_failures;
                erase_iterator(it_best);
                return false;
            }
            p_main = mapping->main(); n_main = mapping->main_size();
            p_drft = mapping->drft(); n_drft = mapping->drft_size();
            p_spec = mapping->spec(); n_spec = mapping->spec_size();
        }

        if (n_main == 0 ||
                io.has_draft != (n_drft > 0) ||
                (!io.has_speculative && n_spec > 0) ||
                !io.restore_transaction) {
            ++restore_failures;
            return false;
        }

        if (!io.restore_transaction(p_main, n_main, p_drft, n_drft, p_spec, n_spec)) {
            ++restore_failures;
            return false;
        }

        if (mapping) {
            // Rehydrate the checkpoint appendix only after the transaction has
            // committed. A payload-less checkpoint must never enter the slot, so
            // a failed read drops the metadata instead.
            std::list<common_prompt_checkpoint> checkpoints;
            size_t n_ckpt_consumed = 0;
            const bool ok = server_prompt_disk::read_checkpoints(
                    mapping->ckpt(), mapping->ckpt_size(), checkpoints, SIZE_MAX, &n_ckpt_consumed) &&
                    n_ckpt_consumed == mapping->ckpt_size() &&
                    server_prompt_disk::checkpoints_match_prompt(
                            checkpoints, it_best->prompt.checkpoints.size(),
                            it_best->prompt.n_tokens(), it_best->prompt.tokens.pos_next());

            if (ok) {
                it_best->prompt.checkpoints = std::move(checkpoints);
            } else {
                SRV_WRN("%s", " - disk swap entry has an unusable checkpoint appendix, dropping checkpoints\n");
                it_best->prompt.checkpoints.clear();
            }
        }

        prompt = it_best->prompt.clone();
        if (mapping) {
            ++restore_disk_successes;
        }
        ++restore_successes;
    }

    return true;
}

bool server_prompt_cache::load(
        server_prompt & prompt,
        const server_tokens & tokens_new,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        common_speculative * spec,
        int32_t id_slot,
        size_t live_native_restorable_tokens,
        int32_t reuse_alignment,
        const server_prompt_cache_state * excluded,
        server_prompt_restore_timings * timings) {
    constexpr llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED;
    server_prompt_cache_state_io io {
        /*.has_draft =*/ ctx_dft != nullptr,
        /*.has_speculative =*/ spec != nullptr,
        /*.restore_transaction =*/ [&](const uint8_t * main, size_t main_size,
                                       const uint8_t * drft, size_t drft_size,
                                       const uint8_t * speculative_state, size_t speculative_size) {
            return server_prompt_restore_transaction_diagnostic(
                    ctx_tgt, ctx_dft, spec, id_slot, flags,
                    { main, main_size }, { drft, drft_size },
                    { speculative_state, speculative_size },
                    true, ctx_dft != nullptr, spec != nullptr, timings).success;
        },
    };

    return load(prompt, tokens_new, live_native_restorable_tokens, reuse_alignment, io, excluded);
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && accounted_size() > limit_size) {
            demote_oldest_ram();
        }
    }

    // average size per token over RAM-resident payloads: the derived token
    // budget tracks RAM capacity, so disk-resident entries must not count
    const float size_per_token = std::max<float>(1.0f, float(accounted_size()) / (std::max<size_t>(1, n_tokens_ram())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens_ram() > limit_tokens_cur) {
            demote_oldest_ram();
        }
    }

    enforce_disk_budget();
    refresh_accounted_bytes();

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB RAM + %.3f MiB disk (limits: %.3f MiB, %.3f MiB disk, %zu tokens, %zu est)\n",
            states.size(), accounted_size() / (1024.0 * 1024.0), accounted_disk_size() / (1024.0 * 1024.0),
            limit_size / (1024.0 * 1024.0), disk.limit_bytes() / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB%s\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(),
                state.accounted_size() / (1024.0 * 1024.0), state.disk.on_disk() ? " [disk]" : "");
    }
}

//
// persistent prompt-cache tier
//

namespace {

std::string persist_file_name(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

} // namespace

bool server_prompt_cache::persist_load(bool with_mmproj) {
    if (!persist.enabled()) {
        return false;
    }

    const std::string dir = persist.path();

    std::vector<server_prompt_persist::record> records;
    if (!server_prompt_persist::index_read(dir, records)) {
        // no usable index: the directory is cleared instead of trusting files
        // whose token metadata nobody can validate
        const size_t removed = server_prompt_persist::remove_stale_files(dir, {});
        SRV_WRN("persisted prompt cache: no usable index in '%s', removed %zu stale file(s)\n",
                dir.c_str(), removed);
        return false;
    }

    std::vector<server_prompt_persist::record> adopted;
    adopted.reserve(records.size());

    for (auto & rec : records) {
        if (rec.tokens.size() % sizeof(llama_token) != 0 || rec.tokens.empty()) {
            server_prompt_persist::remove_file(dir + rec.name);
            continue;
        }

        server_tokens toks;
        try {
            const llama_tokens packed((const llama_token *) rec.tokens.data(),
                    (const llama_token *) (rec.tokens.data() + rec.tokens.size()));
            toks = server_tokens::deserialize(packed, rec.has_media && with_mmproj);
        } catch (const std::exception & err) {
            // media tokens without an mmproj, or a corrupt token payload
            SRV_WRN("persisted prompt cache: unusable token payload for '%s', dropping: %s\n",
                    rec.name.c_str(), err.what());
            server_prompt_persist::remove_file(dir + rec.name);
            continue;
        }

        const llama_tokens & raw = toks.get_tokens();
        if (raw.empty()) {
            server_prompt_persist::remove_file(dir + rec.name);
            continue;
        }

        // capture the correlation values now: raw points into toks, which is
        // moved into the state below
        const uint64_t n_tok = (uint64_t) raw.size();
        const uint64_t tokens_hash = server_prompt_disk::hash_bytes(
                raw.data(), raw.size()*sizeof(llama_token));

        server_prompt_cache_state state;
        state.prompt.tokens = std::move(toks);

        // checkpoint appendix metadata only; the buffers live in the entry
        // file, exactly like a runtime entry that was demoted to disk
        for (const auto & meta : rec.checkpoints) {
            common_prompt_checkpoint ckpt;
            ckpt.id_task = -1;
            ckpt.n_tokens = meta.n_tokens;
            ckpt.pos_min  = meta.pos_min;
            ckpt.pos_max  = meta.pos_max;
            state.prompt.checkpoints.push_back(std::move(ckpt));
        }

        server_prompt_disk::entry & e = state.disk;
        e.status      = server_prompt_disk::entry::status::present;
        e.path        = dir + rec.name;
        e.n_tokens    = n_tok;
        e.tokens_hash = tokens_hash;

        // adopt() validates the file header against the identity and the
        // metadata above; on any mismatch the file is dropped
        if (!persist.adopt(e)) {
            SRV_WRN("persisted prompt cache: entry '%s' failed validation, dropping\n", rec.name.c_str());
            server_prompt_persist::remove_file(dir + rec.name);
            continue;
        }

        // keep the save order: records are oldest first, and the list expects
        // the same order so eviction semantics carry over the restart
        states.push_back(std::move(state));
        adopted.push_back(std::move(rec));
    }

    // files present but not referenced by adopted records are stale
    const size_t removed = server_prompt_persist::remove_stale_files(dir, adopted);
    refresh_accounted_bytes();

    SRV_INF("persisted prompt cache loaded: %zu entr%s (%.3f MiB), %zu stale file(s) removed\n",
            adopted.size(), adopted.size() == 1 ? "y" : "ies",
            persist.bytes_on_disk() / (1024.0 * 1024.0), removed);

    return !states.empty();
}

void server_prompt_cache::persist_save() {
    if (!persist.enabled()) {
        return;
    }

    // every published session spill must have landed before its file moves
    disk.flush();

    std::vector<server_prompt_persist::record> records;
    records.reserve(states.size());

    for (auto it = states.begin(); it != states.end();) {
        server_prompt_cache_state & state = *it;
        server_prompt_disk::entry & e = state.disk;

        if (e.persist_tier && e.on_disk()) {
            // already persisted (rehydrated earlier or saved by an earlier
            // call): nothing to move, only the index record is refreshed
        } else if (e.on_disk() && !e.path.empty()) {
            // a session spill: promote by rename plus a 128-byte in-place
            // header identity patch. The payload is never read or rewritten.
            const std::string src = e.path;
            const std::string dst = persist.next_name();
            if (dst.empty() || !server_prompt_persist::move_file(src, dst)) {
                SRV_WRN("persisted prompt cache: failed to move '%s', entry not persisted\n", src.c_str());
                ++it;
                continue;
            }
            // the swap tier wrote its own identity into the header; re-stamp it
            // before adoption so the persistent tier's read path accepts it
            if (!server_prompt_persist::patch_header_identity(dst, persist.identity())) {
                SRV_WRN("persisted prompt cache: failed to re-stamp '%s', entry not persisted\n", dst.c_str());
                server_prompt_persist::remove_file(dst);
                ++it;
                continue;
            }
            disk.release(e);
            e.path = dst;
            if (!persist.adopt(e)) {
                SRV_WRN("persisted prompt cache: failed to adopt '%s', removing it\n", dst.c_str());
                server_prompt_persist::remove_file(dst);
                e = server_prompt_disk::entry {};
                ++it;
                continue;
            }
        } else if (!state.data.main.empty()) {
            // a RAM-resident entry with no file: one write from already
            // serialized buffers, never a disk read
            const llama_tokens & raw = state.prompt.tokens.get_tokens();
            if (raw.empty()) {
                ++it;
                continue;
            }

            const uint64_t tokens_hash = server_prompt_disk::hash_bytes(
                    raw.data(), raw.size()*sizeof(llama_token));

            if (!persist.spill(e,
                        std::move(state.data.main), std::move(state.data.drft), std::move(state.data.spec),
                        state.prompt.checkpoints,
                        (uint64_t) raw.size(), tokens_hash)) {
                // the payload was consumed by the failed spill, so the entry
                // can no longer serve either tier and must not stay cached
                SRV_WRN("%s", "persisted prompt cache: a RAM entry failed to persist, dropping it\n");
                it = erase_iterator(it);
                continue;
            }
            e.persist_tier = true;
        } else {
            // no file, no payload: nothing to persist
            ++it;
            continue;
        }

        server_prompt_persist::record rec;
        rec.name      = persist_file_name(e.path);
        rec.tokens    = state.prompt.tokens.serialize();
        rec.has_media = state.prompt.tokens.has_mtmd;

        for (const auto & ckpt : state.prompt.checkpoints) {
            rec.checkpoints.push_back({ ckpt.n_tokens, ckpt.pos_min, ckpt.pos_max });
        }

        records.push_back(std::move(rec));
        ++it;
    }

    // every entry file must be complete before the index names it
    persist.flush();

    // the index is published last and atomically: an interruption between the
    // promotions and this point loses the save, never leaves a stale partial one
    if (!server_prompt_persist::index_write(persist.path(), persist.identity(), records)) {
        SRV_WRN("persisted prompt cache: failed to write the index in '%s'\n", persist.path().c_str());
    }

    const size_t removed = server_prompt_persist::remove_stale_files(persist.path(), records);
    refresh_accounted_bytes();

    SRV_INF("persisted prompt cache saved: %zu entr%s (%.3f MiB), %zu stale file(s) removed\n",
            records.size(), records.size() == 1 ? "y" : "ies",
            persist.bytes_on_disk() / (1024.0 * 1024.0), removed);
}
