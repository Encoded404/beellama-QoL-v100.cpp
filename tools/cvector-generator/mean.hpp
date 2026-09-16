#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <string>
#include <vector>
#include <math.h>

namespace mean {

// reduce a single layer to one direction by taking the mean of the sample rows.
//
// input  : shape [n_embd, n_samples], host-side F32 data
// output : one vector of length n_embd, L2-normalized
// prenorm: optional out-param, receives the norm of the mean vector *before* normalization.
//          because the emitted vector is unit-length this is the only informative magnitude,
//          and it is what a caller would need in order to compare across layers.
static void run_layer(
        struct ggml_tensor * input,
        struct ggml_tensor * output,
        float * prenorm = nullptr) {
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    GGML_ASSERT(output->type == GGML_TYPE_F32);

    // calculate mean vector
    GGML_ASSERT(input->ne[0] == output->ne[0]); // == n_embd
    for (int ic = 0; ic < input->ne[0]; ic++) {
        float f = 0.0;
        for (int ir = 0; ir < input->ne[1]; ir++) {
            f += ggml_get_f32_nd(input, ic, ir, 0, 0);
        }
        f /= input->ne[1];
        ggml_set_f32_1d(output, ic, f);
    }

    // normalize output vector
    float norm = 0.0;
    for (int i = 0; i < ggml_nelements(output); i++) {
        float f = ggml_get_f32_1d(output, i);
        norm += f*f;
    }
    norm = sqrt(norm);

    if (prenorm != nullptr) {
        *prenorm = norm;
    }

    if (norm > 0.0f) {
        for (int i = 0; i < ggml_nelements(output); i++) {
            float f = ggml_get_f32_1d(output, i);
            ggml_set_f32_1d(output, i, f / norm);
        }
    }
}

}
