#ifndef TARGET_TAIL_H
#define TARGET_TAIL_H

// Shape of the distribution beyond the last kept id.
//
// The kept ids describe the head exactly; this describes how the remaining mass
// is spread over the next N ranks, which is a property no id/probability list
// can carry cheaply. It is fitted in log-log space because a next-token
// distribution is close to a power law there: a cubic in raw rank spans four
// orders of magnitude and is numerically useless, while a cubic in log rank is
// well conditioned and captures the bend where the power law runs out of mass.
//
//     p(r) = exp(c0 + c1*t + c2*t^2 + c3*t^3),   t = (ln r - ln r0)/(ln r1 - ln r0)
//
// r0 = k+1 (first rank after the kept ids) and r1 = min(k+N, n_vocab), so the
// window is rebuilt by the reader from the position's count, N and the
// vocabulary size; nothing per position is spent on the bounds.
//
// The fit is a plain least squares on (ln r, ln p) over the integer ranks, so it
// carries the same cross-libm caveat as target_lse: the coefficients are
// deterministic within a build, not guaranteed identical across libm versions.
//
// A cubic is not monotone in general. A fit that comes out non-monotone or
// negative inside its own window is refitted as a power law (degree 1), whose
// higher coefficients are zero, so the reader always evaluates the same
// expression and no per-position degree has to be stored.

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace targettail {

constexpr int MAX_DEG = 3;

struct fit_result {
    double c[MAX_DEG + 1] = { 0.0, 0.0, 0.0, 0.0 };
    double band_mass = 0.0; // exact mass over the fitted window (not the curve's)
    bool   ok = false;
};

inline double eval(const double c[MAX_DEG + 1], int32_t r0, int32_t r1, int32_t r) {
    if (r < r0 || r > r1 || r0 <= 0 || r1 <= r0) {
        return 0.0;
    }
    const double lr0  = std::log((double) r0);
    const double lr1  = std::log((double) r1);
    const double span = lr1 > lr0 ? (lr1 - lr0) : 1.0;
    const double t    = (std::log((double) r) - lr0) / span;

    return std::exp(c[0] + t * (c[1] + t * (c[2] + t * c[3])));
}

// Small dense solve with partial pivoting. n <= 4, so this is cheaper and more
// predictable than pulling in anything else, and it keeps the output independent
// of a library's blocking strategy.
inline bool solve(double A[MAX_DEG + 1][MAX_DEG + 1], double b[MAX_DEG + 1], int n, double x[MAX_DEG + 1]) {
    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int r = col + 1; r < n; ++r) {
            if (std::fabs(A[r][col]) > std::fabs(A[piv][col])) {
                piv = r;
            }
        }
        if (std::fabs(A[piv][col]) < 1e-14) {
            return false;
        }
        if (piv != col) {
            for (int c = 0; c < n; ++c) {
                std::swap(A[piv][c], A[col][c]);
            }
            std::swap(b[piv], b[col]);
        }
        const double inv = 1.0 / A[col][col];
        for (int r = col + 1; r < n; ++r) {
            const double f = A[r][col] * inv;
            if (f == 0.0) {
                continue;
            }
            for (int c = col; c < n; ++c) {
                A[r][c] -= f * A[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) {
            s -= A[i][j] * x[j];
        }
        x[i] = s / A[i][i];
    }
    return true;
}

// tail[i] is p(r0 + i), descending, m values.
inline fit_result fit_degrees(const float * tail, int32_t m, int32_t r0, int32_t degree) {
    fit_result out;
    if (tail == nullptr || m <= 0) {
        return out;
    }

    // a probability that underflowed to zero means the tail is over; the
    // points past it carry no information and would make ln(p) infinite
    int32_t me = 0;
    while (me < m && tail[me] > 0.0f && std::isfinite(tail[me])) {
        ++me;
    }
    if (me < 2 || r0 < 1) {
        return out;
    }

    const int32_t r1 = r0 + me - 1;

    out.band_mass = 0.0;
    for (int32_t i = 0; i < m; ++i) {
        out.band_mass += (double) tail[i];
    }

    int deg = std::min(degree, me - 1);
    deg = std::min(deg, MAX_DEG);
    if (deg < 1) {
        deg = 1;
    }

    for (int attempt = 0; attempt < 2; ++attempt) {
        const int d = attempt == 0 ? deg : 1;

        double A[MAX_DEG + 1][MAX_DEG + 1] = {};
        double b[MAX_DEG + 1] = {};
        double x[MAX_DEG + 1] = {};

        const double lr0  = std::log((double) r0);
        const double lr1  = std::log((double) r1);
        const double span = lr1 > lr0 ? (lr1 - lr0) : 1.0;

        double pw[MAX_DEG + 1];
        for (int32_t i = 0; i < me; ++i) {
            const int32_t r = r0 + i;
            const double  t = (std::log((double) r) - lr0) / span;
            const double  y = std::log((double) tail[i]);

            pw[0] = 1.0;
            for (int j = 1; j <= d; ++j) {
                pw[j] = pw[j - 1] * t;
            }
            for (int j = 0; j <= d; ++j) {
                for (int q = 0; q <= d; ++q) {
                    A[j][q] += pw[j] * pw[q];
                }
                b[j] += pw[j] * y;
            }
        }

        if (!solve(A, b, d + 1, x)) {
            continue; // try the power law
        }

        for (int j = 0; j <= d; ++j) {
            out.c[j] = x[j];
        }
        for (int j = d + 1; j <= MAX_DEG; ++j) {
            out.c[j] = 0.0;
        }

        if (attempt == 1 && out.c[1] > 0.0) {
            // a rising power law cannot describe a descending tail; flatten it
            out.c[1] = 0.0;
        }

        // monotone and finite on the fitted window
        bool good = true;
        double prev = 1e300;
        for (int32_t s = 0; s < 17 && good; ++s) {
            const int32_t r = r0 + (int32_t) ((int64_t) (me - 1) * s / 16);
            const double  v = eval(out.c, r0, r1, r);
            if (!std::isfinite(v) || v < 0.0 || v > prev * 1.0000001) {
                good = false;
            }
            prev = v;
        }

        if (good) {
            out.ok = true;
            return out;
        }
    }

    out.c[0] = out.c[1] = out.c[2] = out.c[3] = 0.0;
    return out;
}

inline fit_result fit(const float * tail, int32_t m, int32_t r0, int32_t degree) {
    return fit_degrees(tail, m, r0, degree);
}

} // namespace targettail

#endif // TARGET_TAIL_H
