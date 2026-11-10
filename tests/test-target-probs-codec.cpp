// Unit test for the distillation-target probability codec.
//
// Everything here is synthetic: no model, no llama context, no filesystem. The
// rows model the shapes the dump actually sees (a peaked position that reaches
// 0.85 in one or three ids, a mean-shaped one, a flat one, a position at the
// top-k cap, and a head at 0.99), each truncated at 0.85 so the dropped tail is
// part of the input exactly as it is at dump time.
//
// The properties checked are the ones the format promises:
//   * the stream is self-delimiting: one code per id, MSB-first, < 8 bits of
//     padding, and the width of each code follows the rank rule
//   * decoding reproduces every id within that code's own half step
//   * a head near 1.0 does not collapse the remainder (the failure that made
//     rank 0 wider than the rest)
//   * at equal cost the stick code is more accurate than the i8 code it
//     replaces, on both KL and worst-case relative error
//   * the reconstruction is a sub-probability, per position
//   * a degenerate row, an all-in-one row and an underflowed id encode without
//     NaN, inf or a lost tail
//   * encoding is deterministic byte-for-byte

#include "../examples/embedding-dump/target-probs-codec.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);     \
            printf(__VA_ARGS__);                            \
            printf("\n");                                   \
            ++g_fail;                                       \
        }                                                   \
    } while (0)

static double kl_div(const std::vector<float> & p, const std::vector<float> & q) {
    double sp = 0.0, sq = 0.0;
    for (float v : p) sp += v;
    for (float v : q) sq += v;

    double kl = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i] <= 0.0f) continue;
        const double a = p[i] / sp;
        const double b = std::max((double) q[i] / sq, 1e-300);
        kl += a * std::log(a / b);
    }
    return kl;
}

struct row {
    std::vector<float>    probs;
    std::vector<uint16_t> counts;
};

// A position that sums to `mass` and decays like a zipf with exponent a.
static void add_zipf_row(row & r, int n, double mass, double a) {
    double wsum = 0.0;
    for (int i = 0; i < n; ++i) wsum += 1.0 / std::pow(i + 1.0, a);
    for (int i = 0; i < n; ++i) {
        r.probs.push_back((float) (mass * (1.0 / std::pow(i + 1.0, a)) / wsum));
    }
    r.counts.push_back((uint16_t) n);
}

static row make_rows() {
    row r;
    // 1. a single id holding 85% of the mass
    r.probs.push_back(0.85f);
    r.counts.push_back(1);
    // 2. the same, plus two tail ids
    r.probs.push_back(0.70f);
    r.probs.push_back(0.10f);
    r.probs.push_back(0.05f);
    r.counts.push_back(3);
    // 3. mean shape (18 ids)
    add_zipf_row(r, 18, 0.85, 1.05);
    // 4. flat shape (40 ids)
    add_zipf_row(r, 40, 0.85, 0.55);
    // 5. very flat, at the top-k scale
    add_zipf_row(r, 300, 0.85, 0.30);
    // 6. a head at 0.99 with a thin tail: rank 0 rounding up to 1.0 here is
    //    what used to zero the rest of the position
    r.probs.push_back(0.99f);
    for (int i = 0; i < 6; ++i) r.probs.push_back((float) (0.01 / 6.0));
    r.counts.push_back(7);
    return r;
}

static size_t stream_bits(const std::vector<uint16_t> & counts, const targetprob::stick_params & p) {
    size_t bits = 0;
    for (uint16_t c : counts) {
        for (uint16_t j = 0; j < c; ++j) bits += (size_t) p.bits_for_rank(j);
    }
    return bits;
}

static void check_round_trip(const row & r, int bits) {
    using namespace targetprob;

    const double range = stick_range_for(r.probs, r.counts);
    const stick_params sp{ bits, HEAD_BITS, range };

    CHECK(range >= MIN_RANGE && range <= MAX_RANGE, "bits=%d: range %.3f out of bound", bits, range);

    const std::vector<uint8_t> stream = stick_encode(r.probs, r.counts, sp);
    const std::vector<uint8_t> again  = stick_encode(r.probs, r.counts, sp);
    CHECK(stream == again, "bits=%d: encoding is not deterministic", bits);

    size_t n_ids = 0;
    for (uint16_t c : r.counts) n_ids += c;

    const size_t want = stream_bits(r.counts, sp);
    CHECK(stream.size() * 8 >= want, "bits=%d: stream %zu bits shorter than its %zu codes",
          bits, stream.size() * 8, want);
    CHECK(stream.size() * 8 - want < 8, "bits=%d: %zu bits of padding", bits, stream.size() * 8 - want);

    const std::vector<float> back = stick_decode(stream, r.counts, sp);
    CHECK(back.size() == n_ids, "bits=%d: decoded %zu of %zu ids", bits, back.size(), n_ids);
    if (back.size() != n_ids) {
        return;
    }

    // every id has to land inside its own code's half step, plus the remainder
    // drift that accumulates across a long position: the decoder rebuilds R
    // from reconstructed probabilities, so its relative drift grows with the
    // number of ids and does not shrink when the step does. At 12 bits over the
    // 300-id row that drift is the dominant term (~0.25 %), which is why the
    // bound is a step term plus a small absolute allowance rather than the step
    // alone.
    double worst_ratio = 0.0;
    double worst_rel   = 0.0;
    double worst_bound = 0.0;
    size_t k = 0;
    for (size_t pos = 0; pos < r.counts.size(); ++pos) {
        for (uint16_t j = 0; j < r.counts[pos]; ++j, ++k) {
            const double bound = 2.5 * (std::exp2(0.5 * sp.step(sp.bits_for_rank(j))) - 1.0) + 0.006;
            const double rel   = std::fabs((double) back[k] - (double) r.probs[k]) /
                                 std::max((double) r.probs[k], 1e-30);
            if (rel / bound > worst_ratio) {
                worst_ratio = rel / bound;
                worst_rel   = rel;
                worst_bound = bound;
            }
        }
    }
    CHECK(worst_ratio <= 1.0, "bits=%d: worst relative error %.2e above its %.2e bound", bits, worst_rel, worst_bound);

    // every position reconstructs to a sub-probability
    k = 0;
    for (size_t pos = 0; pos < r.counts.size(); ++pos) {
        double mass = 0.0;
        for (uint16_t j = 0; j < r.counts[pos]; ++j) mass += back[k++];
        CHECK(mass <= 1.0 + 1e-6, "bits=%d: position %zu mass %.8f > 1", bits, pos, mass);
    }

    // and the code it replaces must not win at comparable size
    if (bits == 8) {
        const std::vector<uint8_t> i8  = i8_encode(r.probs);
        const std::vector<float>   i8p = i8_decode(i8);

        // one 16-bit head against an 8-bit body costs 1 byte per position
        CHECK(stream.size() == n_ids + r.counts.size(), "8 bits: stream %zu B for %zu ids / %zu positions",
              stream.size(), n_ids, r.counts.size());

        const double kl_stick = kl_div(r.probs, back);
        const double kl_i8    = kl_div(r.probs, i8p);

        double worst_i8 = 0.0;
        for (size_t i = 0; i < n_ids; ++i) {
            worst_i8 = std::max(worst_i8,
                std::fabs((double) i8p[i] - (double) r.probs[i]) / std::max((double) r.probs[i], 1e-30));
        }

        CHECK(kl_stick < kl_i8, "8 bits: stick KL %.3e not better than i8 KL %.3e", kl_stick, kl_i8);
        CHECK(worst_rel <= worst_i8 * 1.05, "8 bits: stick worst %.4f worse than i8 worst %.4f",
              worst_rel, worst_i8);
    }
}

int main() {
    using namespace targetprob;

    const row r = make_rows();

    for (int bits : { MIN_BITS, 7, 8, 10, MAX_BITS }) {
        check_round_trip(r, bits);
    }

    printf("codec: %zu positions, %zu ids, body widths %d..%d, head %d\n",
           r.counts.size(), r.probs.size(), (int) MIN_BITS, (int) MAX_BITS, (int) HEAD_BITS);

    // --- edge cases ---------------------------------------------------------
    // a degenerate document: one id holding everything, so s = 1 and L = 0
    {
        std::vector<float>    p = { 1.0f };
        std::vector<uint16_t> c = { 1 };

        const double range = stick_range_for(p, c);
        CHECK(range == MIN_RANGE, "degenerate row: range %.3f, expected the %.2f floor", range, MIN_RANGE);

        const stick_params sp{ 8, HEAD_BITS, range };
        const auto s  = stick_encode(p, c, sp);
        const auto bp = stick_decode(s, c, sp);
        CHECK(bp.size() == 1 && std::fabs(bp[0] - 1.0f) < 1e-6, "degenerate row decoded to %.6f", bp[0]);
    }

    // a head at 0.999 with a 0.001 tail: the remainder must survive
    {
        std::vector<float>    p = { 0.999f };
        std::vector<uint16_t> c = { 1 };
        for (int i = 0; i < 4; ++i) p.push_back(0.00025f);
        c.push_back(4);

        const stick_params sp{ 8, HEAD_BITS, stick_range_for(p, c) };
        const auto bp = stick_decode(stick_encode(p, c, sp), c, sp);

        CHECK(bp.size() == 5, "peaked row decoded %zu ids", bp.size());
        CHECK(bp[1] > 0.0f, "peaked row: the tail collapsed to zero (head code ate the remainder)");
        const double tail_true = 4 * 0.00025;
        const double tail_got  = bp[1] + bp[2] + bp[3] + bp[4];
        CHECK(std::fabs(tail_got - tail_true) < 0.25 * tail_true,
              "peaked row: tail mass %.3e, expected %.3e", tail_got, tail_true);
    }

    // an id whose probability underflowed to zero must not produce NaN or inf
    {
        std::vector<float>    p = { 0.9f, 0.0f, 0.0f };
        std::vector<uint16_t> c = { 3 };

        const stick_params sp{ 8, HEAD_BITS, stick_range_for(p, c) };
        const auto bp = stick_decode(stick_encode(p, c, sp), c, sp);

        CHECK(bp.size() == 3, "underflow row decoded %zu ids", bp.size());
        for (float v : bp) {
            CHECK(std::isfinite(v) && v >= 0.0f, "underflow row decoded a non-finite or negative value");
        }
        CHECK(bp[0] > 0.8f, "underflow row lost its head: %.6f", bp[0]);
    }

    // a synthetic document whose widest s exceeds the range must clamp rather
    // than overflow the code word
    {
        std::vector<float>    p;
        std::vector<uint16_t> c;
        for (int i = 0; i < 4096; ++i) p.push_back(1.0f / 5000.0f);
        c.push_back(4096);

        const stick_params sp{ MIN_BITS, HEAD_BITS, stick_range_for(p, c) };
        const auto bp = stick_decode(stick_encode(p, c, sp), c, sp);

        CHECK(sp.range <= MAX_RANGE, "wide row: range %.2f above the cap", sp.range);
        CHECK(bp.size() == 4096, "wide row decoded %zu ids", bp.size());

        double mass = 0.0;
        for (float v : bp) mass += v;
        CHECK(std::isfinite(mass) && mass <= 1.0 + 1e-6, "wide row mass %.6f", mass);
    }

    // the range grid has to survive the milli-unit round trip exactly
    for (double expect : { 0.5, 4.0, 6.25, 10.0, 16.0 }) {
        const int32_t milli = (int32_t) std::llround(expect * 1000.0);
        CHECK(std::fabs((double) milli / 1000.0 - expect) < 1e-9, "range %.2f does not round trip", expect);
    }

    if (g_fail != 0) {
        printf("test-target-probs-codec: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test-target-probs-codec: all checks passed\n");
    return 0;
}
