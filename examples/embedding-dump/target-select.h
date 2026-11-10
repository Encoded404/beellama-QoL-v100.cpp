#ifndef TARGET_SELECT_H
#define TARGET_SELECT_H

// How many ids of one position to keep.
//
// Four rules compose, and every one of them only ever *extends* the set, so the
// result is a max over independent counts rather than a chain of filters:
//
//   nucleus   keep until the cumulative mass reaches top_p (capped at top_k)
//   min-p     keep every id whose own probability is still >= min_p
//   min-ids   keep at least this many
//   extra     once the count is in [extra_min, extra_max], keep extra_ids more
//
// min-p is absolute, not relative to the head. A relative floor (p >= m*p_max)
// raises the bar on exactly the peaked positions that are already keeping few
// ids, which is the opposite of the intent; absolute is also what bounds the
// cost, since at most 1/min_p ids can clear it.
//
// The extra rule runs last, on the final count, so with min_ids = 4 a position
// that the nucleus would have cut at 1 id ends up at 5.

#include <algorithm>
#include <cstdint>

namespace targetsel {

struct rules {
    float   top_p     = 0.85f;
    int32_t top_k     = 256;
    int32_t min_ids   = 4;
    float   min_p     = 0.005f;
    int32_t extra_ids = 1;
    int32_t extra_min = 4;
    int32_t extra_max = 48;
};

// p[] must be descending and n is the number of candidates available (which the
// caller sizes from top_k plus the tail window). Returns the kept count in
// [1, min(n, top_k)], or 0 when there is nothing to keep.
inline int32_t select_count(const float * p, int32_t n, const rules & r) {
    if (p == nullptr || n <= 0) {
        return 0;
    }

    const int32_t cap = std::min(n, std::max(1, r.top_k));
    const double  top = (double) r.top_p;

    // nucleus
    int32_t k = 0;
    double  cum = 0.0;
    while (k < cap) {
        cum += (double) p[k];
        ++k;
        if (cum >= top) {
            break;
        }
    }

    // min-p: ids are descending, so this is a prefix scan that stops once the
    // floor is crossed. p == 0 never clears a positive floor, so an underflowed
    // tail cannot extend anything.
    if (r.min_p > 0.0f) {
        int32_t kmp = 0;
        while (kmp < cap && p[kmp] >= r.min_p) {
            ++kmp;
        }
        if (kmp > k) {
            k = kmp;
        }
    }

    // minimum count
    if (r.min_ids > 0) {
        const int32_t want = std::min(cap, r.min_ids);
        if (k < want) {
            k = want;
        }
    }

    // flat addition, applied to the final count
    if (r.extra_ids > 0 && k >= r.extra_min && k <= r.extra_max) {
        k = std::min(cap, k + r.extra_ids);
    }

    return k;
}

} // namespace targetsel

#endif // TARGET_SELECT_H
