#ifndef TARGET_PROBS_CODEC_H
#define TARGET_PROBS_CODEC_H

// Probability-stream codec for the distillation-target dump (--dump-targets).
//
// The stream stores, for every kept id, a quantisation of the *stick-breaking*
// share of the mass that was still unallocated when the id was reached:
//
//     R_0 = 1                       (the full distribution, truncated tail included)
//     s_i = p_i / R_i               (in (0, 1])
//     R_{i+1} = R_i - p_i
//
// Rank 0 is coded at HEAD_BITS and every later rank at `bits`, because the
// remainder after rank 0 is what all the later ids are measured against. See
// stick_params::bits_for_rank.
//
// s is coded in the log domain, L_i = -log2(s_i), as a fixed-width integer over
// the per-document range [0, range]. Two properties motivate that:
//
//   * the reconstruction p_i = s_i * R_i is automatically a sub-probability, so
//     the kept mass can never exceed 1 no matter how the codes round;
//   * the relative error of p_i is the relative error of s_i, which is uniform
//     across ranks, and -log2(s) spans only a few bits per row (measured 3.8 to
//     4.8 bits on realistic rows) instead of the ~7+ bits that -log(p) needs.
//     A fixed-width code spends its resolution covering the range, so the
//     smaller range buys accuracy at the same byte count.
//
// This header deliberately depends on nothing but the C++ standard library, so
// the codec can be tested (tests/test-target-probs-codec.cpp) without a model
// or a llama context.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace targetprob {

// Values stored in target_prob_meta[0]. MODE_I8 is the version-1 absolute log
// code, kept byte-identical for compatibility; MODE_STICK is the log-domain
// stick code above.
enum mode : int32_t {
    MODE_F16   = 0, // target_probs holds p as f16, one value per id
    MODE_I8    = 1, // target_probs holds clamp(round(-16*ln p), 0, 255)
    MODE_STICK = 2, // target_probs holds a bit-packed L stream, codes per id
};

constexpr int32_t MIN_BITS   = 6;
constexpr int32_t MAX_BITS   = 12;
constexpr int32_t HEAD_BITS  = 16;    // rank 0, see stick_params::bits_for_rank
constexpr double  MIN_RANGE  = 0.5;   // bits; keeps a degenerate row's step finite
constexpr double  MAX_RANGE  = 16.0;  // bits; anything wider is clamped per code
constexpr double  RANGE_GRID = 0.25;  // range is stored on this grid, so it is exact

// MSB-first bit packer, identical in behaviour to the id-stream writer.
struct bit_writer {
    std::vector<uint8_t> buf;
    uint64_t acc   = 0;
    int      nbits = 0;

    void put(uint32_t value, int bits) {
        acc = (acc << bits) | (uint64_t) (value & ((bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u)));
        nbits += bits;
        while (nbits >= 8) {
            nbits -= 8;
            buf.push_back((uint8_t) ((acc >> nbits) & 0xFFu));
        }
        acc &= (nbits > 0) ? ((1ull << nbits) - 1ull) : 0ull;
    }

    void flush() {
        if (nbits > 0) {
            buf.push_back((uint8_t) ((acc << (8 - nbits)) & 0xFFu));
            acc   = 0;
            nbits = 0;
        }
    }
};

// MSB-first reader; `overrun` remembers whether the stream was shorter than the
// codes asked for, so a caller can report a truncated archive instead of looping
// forever on zeros.
struct bit_reader {
    const uint8_t * data = nullptr;
    size_t          size = 0;
    size_t          bit  = 0;
    bool            overrun = false;

    bit_reader() = default;
    explicit bit_reader(const std::vector<uint8_t> & v) : data(v.data()), size(v.size()) {}

    uint32_t take(int n) {
        uint32_t v = 0;
        for (int i = 0; i < n; ++i) {
            const size_t byte = bit >> 3;
            if (byte >= size) {
                overrun = true;
                v <<= 1;
                ++bit;
                continue;
            }
            v = (v << 1) | uint32_t((data[byte] >> (7 - (bit & 7))) & 1u);
            ++bit;
        }
        return v;
    }

    size_t bits_consumed() const { return bit; }
};

struct stick_params {
    int32_t bits      = 8;  // code width for ranks >= 1
    int32_t head_bits = 16; // code width for rank 0
    double  range     = 8.0;

    uint32_t max_code(int32_t b) const { return (1u << b) - 1u; }
    double   step(int32_t b)     const { return range / (double) max_code(b); }

    // Rank 0 is coded wider than the rest because everything after it is
    // measured against the remainder it leaves behind: a head at 0.99 rounded
    // to 8 bits can round up to exactly 1.0, which sets the decoder's remainder
    // to zero and turns the whole position into zeros. A 16-bit head puts that
    // error at a few thousandths of a bit, which is what makes the rest stable.
    int32_t bits_for_rank(size_t rank) const { return rank == 0 ? head_bits : bits; }
};

// s_i for one id, guarded against a p that underflowed to 0 (which would make
// log2() infinite) and against R having drifted above the true remainder.
inline double stick_s(float p, double R) {
    double s = (R > 0.0) ? (double) p / R : 1.0;
    if (s > 1.0) {
        s = 1.0;
    }
    if (s < 1e-18) {
        s = 1e-18;
    }
    return s;
}

// The per-document scale: the smallest grid point that covers the widest s in
// the document. A plain max over the exact probabilities, so it is deterministic
// and reproducible; the grid makes it round-trip exactly through milli-units.
//
// Ids whose probability underflowed to zero are skipped: their L would be
// infinite, they contribute nothing to the remainder, and one of them would
// otherwise pin the whole document to MAX_RANGE and cost resolution everywhere.
// They clamp to the largest code instead, which hands them the smallest mass the
// format can express.
inline double stick_range_for(const std::vector<float> & probs, const std::vector<uint16_t> & counts) {
    size_t k = 0;
    double max_l = 0.0;

    for (uint16_t c : counts) {
        double R = 1.0;
        for (uint16_t j = 0; j < c && k < probs.size(); ++j, ++k) {
            if (probs[k] > 0.0f) {
                const double l = -std::log2(stick_s(probs[k], R));
                if (l > max_l) {
                    max_l = l;
                }
            }
            R -= (double) probs[k];
            if (R < 0.0) {
                R = 0.0;
            }
        }
    }

    double r = std::ceil(max_l / RANGE_GRID) * RANGE_GRID;
    if (r < MIN_RANGE) {
        r = MIN_RANGE;
    }
    if (r > MAX_RANGE) {
        r = MAX_RANGE;
    }
    return r;
}

// Pack one document's kept probabilities, positions in order and ids inside a
// position in descending probability (the order target_finish emits).
//
// The code is chosen against the exact remainder, so the encoder can never ask
// for s > 1 and a stream can never collapse. (Choosing it against the decoder's
// *reconstructed* remainder tracks the decoder slightly better - measured max
// error 1.5 % against 2.7 % at 8 bits - but feeds rounding back into the
// remainder, and a peaked head that rounds to 1.0 zeroes every later id. The
// 16-bit head below removes the cause of that failure without the feedback.)
inline std::vector<uint8_t> stick_encode(
        const std::vector<float> & probs, const std::vector<uint16_t> & counts, const stick_params & p) {
    bit_writer bw;

    size_t n_bits = 0;
    for (uint16_t c : counts) {
        n_bits += c > 0 ? (size_t) p.head_bits + (size_t) (c - 1) * p.bits : 0;
    }
    bw.buf.reserve(n_bits / 8 + 8);

    size_t k = 0;
    for (uint16_t c : counts) {
        double R = 1.0;
        for (uint16_t j = 0; j < c && k < probs.size(); ++j, ++k) {
            const int32_t b     = p.bits_for_rank(j);
            const long    maxc  = (long) p.max_code(b);
            const double  step  = p.step(b);

            const double s = stick_s(probs[k], R);
            long         code = std::lround(-std::log2(s) / step);
            if (code < 0) {
                code = 0;
            } else if (code > maxc) {
                code = maxc;
            }
            bw.put((uint32_t) code, b);

            R -= (double) probs[k];
            if (R < 0.0) {
                R = 0.0;
            }
        }
    }
    bw.flush();
    return std::move(bw.buf);
}

// Inverse of stick_encode. Used by the round-trip test and by anything that
// needs the codec in-process; the on-disk reader is the Python decoder in
// docs/dump-targets.md.
inline std::vector<float> stick_decode(
        const std::vector<uint8_t> & stream, const std::vector<uint16_t> & counts, const stick_params & p) {
    bit_reader br(stream);

    size_t n = 0;
    for (uint16_t c : counts) {
        n += c;
    }

    std::vector<float> out;
    out.reserve(n);

    for (uint16_t c : counts) {
        double R = 1.0;
        for (uint16_t j = 0; j < c; ++j) {
            const int32_t  b    = p.bits_for_rank(j);
            const uint32_t code = br.take(b);
            double s = std::exp2(-(double) code * p.step(b));
            if (s > 1.0) {
                s = 1.0;
            }
            const double pi = s * R;
            out.push_back((float) pi);
            R -= pi;
            if (R < 0.0) {
                R = 0.0;
            }
        }
    }
    return n == 0 ? std::vector<float>{} : out;
}

// Version-1 absolute log code, unchanged so f16/i8 dumps stay byte-identical.
inline std::vector<uint8_t> i8_encode(const std::vector<float> & probs) {
    std::vector<uint8_t> q(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        const double p = std::max((double) probs[i], 1e-300);
        long v = std::lround(-16.0 * std::log(p));
        q[i] = (uint8_t) std::min<long>(255, std::max<long>(0, v));
    }
    return q;
}

inline std::vector<float> i8_decode(const std::vector<uint8_t> & q) {
    std::vector<float> out(q.size());
    for (size_t i = 0; i < q.size(); ++i) {
        out[i] = (float) std::exp(-(double) q[i] / 16.0);
    }
    return out;
}

} // namespace targetprob

#endif // TARGET_PROBS_CODEC_H
