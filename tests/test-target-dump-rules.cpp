// Unit tests for the two pure pieces the target dump grew in version 3:
// the selection rules (target-select.h) and the log-log tail fit
// (target-tail.h). Synthetic data only: no model, no context, no filesystem.

#include "../examples/embedding-dump/target-select.h"
#include "../examples/embedding-dump/target-tail.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...)                                \
    do {                                                \
        if (!(cond)) {                                  \
            printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                        \
            printf("\n");                               \
            ++g_fail;                                   \
        }                                               \
    } while (0)

static std::vector<float> make_probs(int n, double a, double mass = 1.0) {
    std::vector<float> p(n);
    double w = 0.0;
    for (int i = 0; i < n; ++i) w += 1.0 / std::pow(i + 1.0, a);
    for (int i = 0; i < n; ++i) p[i] = (float) (mass * (1.0 / std::pow(i + 1.0, a)) / w);
    return p;
}

// ---------------------------------------------------------------- selection

static void test_nucleus_matches_old_behaviour() {
    using namespace targetsel;

    const auto p = make_probs(200, 1.0);
    rules r;
    r.min_ids = 0;
    r.min_p = 0.0f;
    r.extra_ids = 0;

    for (float top_p : { 0.5f, 0.85f, 0.95f, 1.0f }) {
        r.top_p = top_p;

        // reference: the loop this replaced
        int expected = 0;
        double cum = 0.0;
        for (int i = 0; i < 200; ++i) {
            cum += p[i];
            ++expected;
            if (cum >= (double) top_p) break;
        }
        CHECK(select_count(p.data(), 200, r) == expected,
              "top_p %.2f: got %d, expected %d", (double) top_p, select_count(p.data(), 200, r), expected);
    }
}

static void test_min_ids() {
    using namespace targetsel;

    // a head that swallows 0.9 on its own, so the nucleus stops at one id
    const std::vector<float> p = { 0.90f, 0.04f, 0.02f, 0.01f, 0.005f, 0.004f, 0.003f, 0.002f, 0.001f, 0.0005f };

    rules r;
    r.min_p = 0.0f;
    r.extra_ids = 0;

    r.min_ids = 0;
    CHECK(select_count(p.data(), (int32_t) p.size(), r) == 1, "nucleus should stop at one id");
    r.min_ids = 4;
    CHECK(select_count(p.data(), (int32_t) p.size(), r) == 4, "min-ids 4 gave %d",
          select_count(p.data(), (int32_t) p.size(), r));
    r.min_ids = 9;
    CHECK(select_count(p.data(), (int32_t) p.size(), r) == 9, "min-ids 9 gave %d",
          select_count(p.data(), (int32_t) p.size(), r));

    // never past the cap, never past the candidates
    r.min_ids = 40;
    r.top_k = 6;
    CHECK(select_count(p.data(), (int32_t) p.size(), r) == 6, "min-ids clamped to top_k failed");
    CHECK(select_count(p.data(), 3, r) == 3, "min-ids clamped to candidates failed");
}

static void test_min_p() {
    using namespace targetsel;

    const std::vector<float> p = { 0.80f, 0.05f, 0.04f, 0.03f, 0.02f, 0.01f, 0.005f, 0.004f, 0.002f, 0.001f };
    const int32_t n = (int32_t) p.size();

    rules r;
    r.min_ids = 0;
    r.extra_ids = 0;

    // nucleus alone: 0.80 then 0.85, and float(0.85f) sits a hair above the
    // double sum of 0.8f+0.05f, so the third id is included
    r.min_p = 0.0f;
    CHECK(select_count(p.data(), n, r) == 3, "nucleus gave %d", select_count(p.data(), n, r));

    // absolute floor extends to every id at or above it
    r.min_p = 0.02f;
    CHECK(select_count(p.data(), n, r) == 5, "min-p 0.02 gave %d", select_count(p.data(), n, r));
    r.min_p = 0.005f;
    CHECK(select_count(p.data(), n, r) == 7, "min-p 0.005 gave %d", select_count(p.data(), n, r));

    // a floor under the whole row keeps everything the cap allows
    r.min_p = 1e-6f;
    CHECK(select_count(p.data(), n, r) == n, "min-p below the tail kept %d of %d",
          select_count(p.data(), n, r), n);

    // a floor above everything the nucleus left behind extends nothing: the
    // head reaches 0.85 in two ids and the third is far below the floor
    const std::vector<float> cliff = { 0.80f, 0.10f, 0.001f, 0.001f, 0.001f, 0.001f };
    rules rc;
    rc.min_ids = 0;
    rc.extra_ids = 0;
    rc.min_p = 0.005f;
    CHECK(select_count(cliff.data(), (int32_t) cliff.size(), rc) == 2,
          "min-p 0.005 kept %d, expected the 2-id nucleus",
          select_count(cliff.data(), (int32_t) cliff.size(), rc));
    rc.min_p = 0.0005f;
    CHECK(select_count(cliff.data(), (int32_t) cliff.size(), rc) == 6,
          "min-p 0.0005 kept %d, expected all 6", select_count(cliff.data(), (int32_t) cliff.size(), rc));

    // p == 0 never clears a positive floor
    std::vector<float> z = { 0.9f, 0.0f, 0.0f };
    r.min_p = 0.001f;
    CHECK(select_count(z.data(), 3, r) == 1, "zero probabilities extended the set");
}

static void test_extra() {
    using namespace targetsel;

    const auto p = make_probs(200, 1.0);
    rules r;
    r.min_ids = 0;
    r.min_p = 0.0f;
    r.extra_ids = 1;
    r.extra_min = 4;
    r.extra_max = 48;

    for (int32_t k0 = 1; k0 <= 60; ++k0) {
        // force a nucleus that stops at exactly k0 ids
        double cum = 0.0;
        for (int i = 0; i < k0; ++i) cum += p[i];
        r.top_p = (float) (cum - 1e-6);

        const int32_t got = select_count(p.data(), 200, r);
        const int32_t want = (k0 >= 4 && k0 <= 48) ? k0 + 1 : k0;
        CHECK(got == want, "nucleus %d -> %d, expected %d", k0, got, want);
    }

    // the documented composition: min-ids 4 lands in [4,48] and is widened to 5
    const std::vector<float> head = { 0.90f, 0.04f, 0.02f, 0.01f, 0.005f, 0.004f, 0.003f, 0.002f };
    r.top_p = 0.85f;
    r.min_ids = 4;
    CHECK(select_count(head.data(), (int32_t) head.size(), r) == 5,
          "min-ids 4 + extra should be 5, got %d", select_count(head.data(), (int32_t) head.size(), r));

    // and the cap still wins
    r.top_k = 4;
    CHECK(select_count(head.data(), (int32_t) head.size(), r) == 4, "extra went past top_k");
}

static void test_select_edges() {
    using namespace targetsel;
    rules r;

    CHECK(select_count(nullptr, 10, r) == 0, "null input");
    CHECK(select_count((const float *) "", 0, r) == 0, "empty input");

    const std::vector<float> one = { 1.0f };
    CHECK(select_count(one.data(), 1, r) == 1, "single id");

    // all the rules at once, on a row where each one binds at a different rank
    const std::vector<float> p = { 0.70f, 0.08f, 0.06f, 0.05f, 0.03f, 0.02f, 0.01f, 0.008f, 0.006f, 0.004f };
    r.top_p = 0.85f;
    r.min_p = 0.02f;
    r.min_ids = 2;
    r.extra_ids = 2;
    r.extra_min = 4;
    r.extra_max = 48;
    r.top_k = 10;
    // nucleus 3 (0.70+0.08+0.06=0.84 <0.85, +0.05=0.89) -> 4; min-p 0.02 -> 6;
    // min-ids no-op; 6 in [4,48] -> +2 = 8
    CHECK(select_count(p.data(), 10, r) == 8, "combined rules gave %d, expected 8",
          select_count(p.data(), 10, r));
}

// ---------------------------------------------------------------- tail fit

static void test_power_law_recovery() {
    using namespace targettail;

    // p(r) = A * r^-b exactly: the fit must recover it and the higher terms
    // must come out empty
    const double b = 1.2, A = 0.9;
    const int32_t r0 = 101, nwin = 1024;
    std::vector<float> tail(nwin);
    for (int32_t i = 0; i < nwin; ++i) {
        tail[i] = (float) (A * std::pow((double) (r0 + i), -b));
    }

    const fit_result f = fit(tail.data(), nwin, r0, 3);
    CHECK(f.ok, "power law fit failed");

    const int32_t r1 = r0 + nwin - 1;
    double worst = 0.0;
    for (int32_t i = 0; i < nwin; i += 37) {
        const int32_t r = r0 + i;
        const double want = A * std::pow((double) r, -b);
        const double got = eval(f.c, r0, r1, r);
        worst = std::max(worst, std::fabs(got - want) / want);
    }
    CHECK(worst < 0.01, "power law recovered to %.3f%%", worst * 100.0);

    // the exact log-log slope, expressed in the normalised basis
    const double span = std::log((double) r1) - std::log((double) r0);
    CHECK(std::fabs(f.c[1] - (-b * span)) < 0.05 * std::fabs(b * span),
          "slope c1 %.3f, expected %.3f", f.c[1], -b * span);
    CHECK(std::fabs(f.c[2]) < 1e-3 && std::fabs(f.c[3]) < 1e-3, "a power law needed higher terms");

    // band mass is the exact sum of the inputs
    double sum = 0.0;
    for (float v : tail) sum += v;
    CHECK(std::fabs(f.band_mass - sum) <= 1e-6 * sum, "band mass %.9f, expected %.9f", f.band_mass, sum);
}

static void test_tail_curvature() {
    using namespace targettail;

    // a tail that runs out of mass partway: power law then a hard floor. the
    // cubic should track it better than a straight line would
    const int32_t r0 = 51, nwin = 512;
    std::vector<float> tail(nwin);
    for (int32_t i = 0; i < nwin; ++i) {
        const double r = (double) (r0 + i);
        tail[i] = (float) std::max(1e-9, 0.05 * std::pow(r, -1.6) - 2e-7);
    }

    const fit_result f = fit(tail.data(), nwin, r0, 3);
    CHECK(f.ok, "curved tail fit failed");

    const int32_t r1 = r0 + nwin - 1;
    double worst = 0.0;
    for (int32_t i = 0; i < nwin; i += 23) {
        const double want = tail[i];
        const double got = eval(f.c, r0, r1, r0 + i);
        worst = std::max(worst, std::fabs(got - want) / want);
    }
    CHECK(worst < 0.15, "curved tail recovered to %.1f%%", worst * 100.0);
}

static void test_tail_monotone_guard() {
    using namespace targettail;

    // a bump in the middle makes the cubic overshoot; the guard has to fall
    // back rather than hand out a rising "distribution"
    const int32_t r0 = 11, nwin = 256;
    std::vector<float> tail(nwin);
    for (int32_t i = 0; i < nwin; ++i) tail[i] = (float) (1e-3 * std::pow((double) (r0 + i), -1.0));
    for (int32_t i = 100; i < 140; ++i) tail[i] *= 40.0f;

    const fit_result f = fit(tail.data(), nwin, r0, 3);
    CHECK(f.ok, "bumped tail produced no fit at all");

    const int32_t r1 = r0 + nwin - 1;
    double prev = 1e300;
    for (int32_t i = 0; i < nwin; ++i) {
        const double v = eval(f.c, r0, r1, r0 + i);
        CHECK(std::isfinite(v) && v >= 0.0, "non-finite or negative value in the fitted tail");
        CHECK(v <= prev * 1.0000001, "fitted tail rises at rank %d", r0 + i);
        prev = v;
    }
}

static void test_tail_edges() {
    using namespace targettail;

    CHECK(!fit(nullptr, 0, 5, 3).ok, "empty tail reported ok");

    const float one_pt[1] = { 1e-3f };
    CHECK(!fit(one_pt, 1, 5, 3).ok, "one point reported ok");

    // underflowed tail: everything past the first zero is dropped from the fit
    std::vector<float> z = { 1e-3f, 1e-4f, 0.0f, 0.0f, 0.0f, 0.0f };
    const fit_result f = fit(z.data(), 6, 1, 3);
    CHECK(f.ok, "truncated tail failed");
    CHECK(std::fabs(f.band_mass - 1.1e-3) < 1e-9, "band mass %.9f", f.band_mass);

    // a request for degree 3 with only two usable points falls back instead of
    // indexing out of the window
    CHECK(fit(z.data(), 2, 1, 3).ok, "two-point tail failed");

    // determinism
    std::vector<float> t(300);
    for (int i = 0; i < 300; ++i) t[i] = (float) (0.02 * std::pow((double) (i + 6), -1.3));
    const fit_result a = fit(t.data(), 300, 6, 3);
    const fit_result b = fit(t.data(), 300, 6, 3);
    CHECK(a.ok && b.ok, "determinism fit failed");
    for (int j = 0; j < 4; ++j) CHECK(a.c[j] == b.c[j], "coefficient %d is not deterministic", j);
    CHECK(a.band_mass == b.band_mass, "band mass is not deterministic");

    // evaluation outside the window is zero, not garbage
    CHECK(eval(a.c, 6, 305, 5) == 0.0, "eval below the window");
    CHECK(eval(a.c, 6, 305, 306) == 0.0, "eval above the window");
}

int main() {
    test_nucleus_matches_old_behaviour();
    test_min_ids();
    test_min_p();
    test_extra();
    test_select_edges();
    test_power_law_recovery();
    test_tail_curvature();
    test_tail_monotone_guard();
    test_tail_edges();

    if (g_fail != 0) {
        printf("test-target-dump-rules: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test-target-dump-rules: all checks passed\n");
    return 0;
}
