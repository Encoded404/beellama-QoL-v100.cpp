#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

static std::string read_file(const std::string & path) {
    std::ifstream file(path);
    if (!file.good()) {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(1);
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static bool expect(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
    }
    return ok;
}

static std::string slice_between(const std::string & text, const std::string & begin, const std::string & end) {
    const size_t b = text.find(begin);
    if (b == std::string::npos) {
        return {};
    }
    const size_t e = text.find(end, b);
    if (e == std::string::npos) {
        return text.substr(b);
    }
    return text.substr(b, e - b);
}

static size_t count_occurrences(const std::string & text, const std::string & needle) {
    size_t count = 0;
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

int main(int argc, char ** argv) {
    bool ok = true;

    ok &= expect(argc == 2, "expected repo root argument");
    if (!ok) {
        return 1;
    }

    const std::string root = argv[1];
    const std::string vec = read_file(root + "/ggml/src/ggml-cuda/fattn-vec.cuh");
    const std::string cmake = read_file(root + "/ggml/CMakeLists.txt");
    const std::string instances = read_file(root + "/ggml/cmake/common.cmake");
    const std::string fattn = read_file(root + "/ggml/src/ggml-cuda/fattn.cu");

    // v0.4.8 owns the FlashAttention vector pair mechanism: the pair list is built by
    // ggml_cuda_fattn_vec_instances() in ggml/cmake/common.cmake and the runtime dispatch
    // is ggml_cuda_get_fattn_vec_case() in fattn.cu. The fork contributes
    // GGML_CUDA_FA_NO_BF16, the SM70 D256 route, and the Q2_0S cache type.
    const std::string dispatch = slice_between(fattn,
            "static fattn_vec_case_t ggml_cuda_get_fattn_vec_case",
            "static ggml_type ggml_cuda_fattn_canonical_kv_type");
    ok &= expect(!dispatch.empty() && count_occurrences(dispatch, "FATTN_VEC_CASES_ALL_D(") == 169,
        "the runtime vector dispatch must enumerate all 169 ordered pairs of the 13 retained types");
    ok &= expect(dispatch.find("FATTN_VEC_CASES_ALL_D(F16   , F16)") != std::string::npos &&
                 dispatch.find("FATTN_VEC_CASES_ALL_D(BF16  , BF16)") != std::string::npos,
        "the full runtime vector dispatch must retain homogeneous F16 and BF16 instances");
    ok &= expect(dispatch.find("Q2_0S") != std::string::npos && dispatch.find("GGML_TYPE_Q2_0,") == std::string::npos,
        "the fork low-bit vector type must be Q2_0S, distinct from upstream Q2_0");
    ok &= expect(fattn.find("ggml_cuda_get_fattn_vec_case(128, type_K, type_V)") != std::string::npos,
        "the compiled-pair gate must consult the same runtime dispatch as the kernel selector");
    ok &= expect(fattn.find("GGML_CUDA_FA_HALF_QUANTS") == std::string::npos,
        "the removed HALF build tier must not survive in the vector dispatch");
    ok &= expect(fattn.find("BEST_FATTN_KERNEL_SM70_D256") != std::string::npos,
        "the fork Volta D256 prefill route must survive the merge");
    ok &= expect(fattn.find("GGML_CUDA_FATTN_OP_PARAM_FORCE_VEC = 9") != std::string::npos,
        "the fork force-vec op param must not collide with upstream public op param slots 3..8");

    ok &= expect(cmake.find("set(GGML_CUDA_FA_QUANTS") != std::string::npos &&
                 cmake.find("option(GGML_CUDA_FA_ALL_QUANTS") != std::string::npos,
        "ggml/CMakeLists.txt must expose the FA quants selector alongside the legacy ALL option");
    ok &= expect(cmake.find("option(GGML_CUDA_FA_NO_BF16") != std::string::npos,
        "the fork GGML_CUDA_FA_NO_BF16 option must survive the merge");
    ok &= expect(cmake.find("function(ggml_cuda_get_fattn_vec_default_pairs") != std::string::npos,
        "ggml/CMakeLists.txt must own the default pair policy");
    ok &= expect(instances.find("ggml_cuda_get_fattn_vec_default_pairs") != std::string::npos &&
                 instances.find("GGML_CUDA_FA_ALL_QUANTS") != std::string::npos,
        "the pair selector must call the default policy and keep honoring the legacy ALL option");
    ok &= expect(instances.find("GGML_CUDA_FA_NO_BF16") != std::string::npos &&
                 instances.find("add_compile_definitions(GGML_CUDA_FA_NO_BF16)") != std::string::npos &&
                 cmake.find("NO_BF16") != std::string::npos,
        "the pair selector and the default policy must both honor GGML_CUDA_FA_NO_BF16");


    ok &= expect(vec.find("static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device()") != std::string::npos,
        "the vector kernel helper section must remain present");
    ok &= expect(vec.find("GGML_TYPE_TURBO") == std::string::npos &&
                 vec.find("TCQ") == std::string::npos,
        "the vector kernel must not retain TurboQuant or TCQ cache handling");
    ok &= expect(vec.find("GGML_TYPE_Q2_0S") != std::string::npos &&
                 vec.find("GGML_TYPE_Q6_1") != std::string::npos,
        "the vector kernel must retain declarations for the fork low-bit cache types");

    return ok ? 0 : 1;
}
