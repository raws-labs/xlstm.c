/* The gate transcendentals, in whichever build this was compiled for.
 *
 * Two things need a gate that the golden vectors cannot provide.
 *
 *   1. The zero-exponent shortcut in slstm_s8.c and mlstm_s8.c. It claims to
 *      be an identity, not an approximation, so the check is bit equality
 *      against the expression it replaced - over the non-finite inputs too,
 *      which is where an identity of this shape usually stops being one.
 *      Checked in BOTH builds, because the shortcut is in both.
 *
 *   2. The accuracy of XLSTM_APPROX_GATES=1 against libm. The golden suites
 *      cannot substitute: they assert a whole kernel through INT8
 *      quantization, where an activation error of 1e-7 is four orders below
 *      one output LSB and would pass whatever it did. Only a direct sweep can
 *      say what the approximation actually costs, and only a recorded bound
 *      turns that into something that can fail.
 *
 * In the exact build 2 becomes its own assertion: each wrapper must be bit
 * identical to the libm expression it forwards to, which is what makes
 * "the default numerics are unchanged" a checked claim rather than a comment.
 *
 * Build: make test   (both variants: make test XLSTM_GATES=approx)
 * =========================================================================*/

#include "xlstm_util.h"
#include "test_util.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

// ============================================================================
// Sweep: every float bit pattern at a stride, plus a fine walk over the range
// the gate pre-activations actually occupy, plus the values that break things.
// ============================================================================

/* Two strides, because the two checks cost wildly different amounts per
 * point: the accuracy sweep is four libm calls, the shortcut sweep is 23
 * paired evaluations. Both land on every exponent and every representable
 * sign; kAccStride also lands on every 4096th mantissa step. */
static const uint32_t kAccStride = 1024;
static const uint32_t kShortcutStride = 0x40000;

static float FromBits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static uint32_t ToBits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

/* Bit equality, with all NaNs treated as one value: the kernels never
 * distinguish payloads and neither libm nor the polynomials promise to
 * preserve them. */
static bool SameFloat(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return ToBits(a) == ToBits(b);
}

/* Calls fn on the sweep. Returns the number of points visited. */
template <typename F>
static long Sweep(uint32_t stride, int band_step, F fn) {
    long n = 0;
    for (uint64_t u = 0; u < 0x100000000ull; u += stride) {
        fn(FromBits((uint32_t)u));
        ++n;
    }
    /* The band the pre-activations live in, at a step far finer than the
     * stride above reaches: it crosses both exp limits, both tanh
     * saturations and the log-sigmoid kink at zero. */
    for (int i = -100000; i <= 100000; i += band_step) {
        fn((float)i * 1e-3f);
        ++n;
    }
    const float kSpecial[] = {
        0.0f, -0.0f, 1.0f, -1.0f,
        HUGE_VALF, -HUGE_VALF, NAN, -NAN,
        FLT_MIN, -FLT_MIN, FLT_TRUE_MIN, -FLT_TRUE_MIN,
        88.722839f, 88.7228394f, 88.7228317f,      /* exp overflow edge */
        -103.972084f, -103.9720764f, -103.972092f, /* exp underflow edge */
        -87.336544f, -87.336548f,                  /* exp denormal edge */
        2.44140625e-4f, -2.44140625e-4f,           /* tanh linear-arm edge */
        2.4414061e-4f, 2.4414065e-4f,
    };
    for (float v : kSpecial) { fn(v); ++n; }
    return n;
}

// ============================================================================
// 1. The zero-exponent shortcut is an identity
// ============================================================================

/* slstm_s8.c's gate expression, both ways round. `m` is always xlstm_maxf of
 * the two operands (or i_raw itself on the first timestep), which is what
 * makes one of the two subtractions a zero.
 *
 * The clamp is not decoration here, it is what makes the shortcut exact. At
 * a == m == +inf the subtraction is NaN and the exponential is NaN, and it
 * is xlstm_minf(NaN, 1.0f) returning 1.0f that lands both arms on the same
 * value. This check found that the hard way, on the unclamped spelling
 * mlstm_s8.c would have used - which is why that file does not use one. */
static bool ShortcutPair(float a, float m) {
    float ref = xlstm_minf(xlstm_gate_expf(a - m), 1.0f);
    float shortcut = (a == m) ? 1.0f
                              : xlstm_minf(xlstm_gate_expf(a - m), 1.0f);
    if (!SameFloat(ref, shortcut)) {
        std::printf("  FAIL zero-exponent shortcut: a=%.9g m=%.9g -> "
                    "clamped exp %.9g, shortcut %.9g\n", a, m, ref, shortcut);
        return false;
    }
    return true;
}

static bool TestZeroExponentShortcutIsExact() {
    /* The case that matters is a == m, so drive it directly for every value
     * on the sweep as well as against a second operand. */
    const float kOther[] = {0.0f, -0.0f, 1.0f, -1.0f, 5.0f, -5.0f, 700.0f,
                            -700.0f, HUGE_VALF, -HUGE_VALF, NAN};
    bool ok = true;
    long n = Sweep(kShortcutStride, 10, [&](float v) {
        if (!ok) return;
        if (!ShortcutPair(v, v)) { ok = false; return; }        /* a == m */
        for (float o : kOther) {
            float m = xlstm_maxf(v, o);
            if (!ShortcutPair(v, m) || !ShortcutPair(o, m)) { ok = false; return; }
        }
    });
    if (ok) {
        std::printf("  %ld values x 11 partners, shortcut and clamped "
                    "exponential bit identical\n", n);
    }
    return ok;
}

// ============================================================================
// 2 / 3. The four wrappers, against the correctly rounded answer
// ============================================================================

/* The libm spellings the exact build forwards to, for the side-by-side. */
static float LibmExp(float x) { return std::exp(x); }
static float LibmSigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static float LibmTanh(float x) { return std::tanh(x); }
static float LibmLogSigmoid(float x) {
    if (x >= 0.0f) return -std::log(1.0f + std::exp(-x));
    return x - std::log(1.0f + std::exp(x));
}

#if XLSTM_APPROX_GATES

/* The reference is computed in double and rounded once to float, not taken
 * from float libm. That matters: at x = 15.537 the exact build's own
 * -logf(1.0f + expf(-x)) is 33% wrong, because floats near 1.0 are spaced
 * 1.19e-07 apart and the addend there is 1.79e-07. Measuring against float
 * libm would have scored the approximation as catastrophically inaccurate at
 * exactly the points where it is the more accurate of the two. */
static double TrueExp(double x) { return std::exp(x); }
static double TrueSigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static double TrueTanh(double x) { return std::tanh(x); }
static double TrueLogSigmoid(double x) {
    return (x < 0.0 ? x : 0.0) - std::log1p(std::exp(-std::fabs(x)));
}

/* Error in units of the last place of the correctly rounded result - the one
 * scale-free measure that stays meaningful across seven decades of gate
 * output and into the denormals, where a relative error of 1e-2 can still be
 * the best a float can do and an absolute error of 1e-40 says nothing. */
static double UlpErr(float got, double want) {
    float w = (float)want;
    if (!std::isfinite(w) || !std::isfinite(got)) {
        return SameFloat(got, w) ? 0.0 : 1e9;
    }
    double a = std::fabs((double)w);
    double ulp = (double)std::nextafterf((float)a, HUGE_VALF) - a;
    if (ulp <= 0.0) ulp = (double)FLT_TRUE_MIN;
    return std::fabs((double)got - want) / ulp;
}

/* Normal and subnormal results are scored separately, and not because
 * subnormals deserve a softer bound. A gate value below FLT_MIN has already
 * lost most of its mantissa, so `ulp` there measures a quantity the format
 * itself cannot hold - and both builds land in the same place for the same
 * reason: 1/(1 + exp(-x)) overflows its denominator below x = -88.7 and
 * returns zero where the answer is 3.6e-39. Folding the two together would
 * report that shared property of the formula as a cost of the
 * approximation. Split, each column says what it means. */
struct ErrStat {
    double normal = 0.0;      /* max ulp error where the answer is normal */
    double sub = 0.0;         /* ... where it is subnormal or zero */
    float at = 0.0f;
    long nonfinite_bad = 0;
};

static void Accumulate(ErrStat* s, float x, float got, double want) {
    float w = (float)want;
    if (!std::isfinite(w) || !std::isfinite(got)) {
        if (!SameFloat(got, w)) {
            ++s->nonfinite_bad;
            if (s->nonfinite_bad <= 4) {
                std::printf("  FAIL non-finite disagreement at x=%.9g: "
                            "want %.9g, got %.9g\n", x, (double)w, (double)got);
            }
        }
        return;
    }
    double u = UlpErr(got, want);
    if (std::fabs((double)w) >= (double)FLT_MIN) {
        if (u > s->normal) { s->normal = u; s->at = x; }
    } else if (u > s->sub) {
        s->sub = u;
    }
}

/* Recorded ULP bounds for XLSTM_APPROX_GATES=1.
 *
 * These ARE the accuracy claim. Each is about 1.5x what this sweep measures,
 * which is margin for a target whose double reference or whose rounding of
 * the polynomial differs in the last bit, and nothing more. Raising one to
 * make a run pass is the one edit that would empty this check out; a
 * polynomial that has drifted needs refitting, not a wider bound.
 *
 * For scale: the libm the exact build calls measures 0.5 to 2.1 ulp on this
 * same sweep, which the run prints beside each figure, and the activation
 * drift the INT8 bounds are built to tolerate is 1e-3 relative - about 8000
 * ulp. These bounds are three orders of magnitude inside that. */
static const double kExpUlpBound        = 2.0;
static const double kLogSigmoidUlpBound = 4.0;
static const double kSigmoidUlpBound    = 5.0;
static const double kTanhUlpBound       = 3.0;

static bool Report(const char* name, const ErrStat& a, const ErrStat& l,
                   double bound) {
    bool ok = true;
    if (a.nonfinite_bad) {
        std::printf("  FAIL %s: %ld disagreements on a non-finite result\n",
                    name, a.nonfinite_bad);
        ok = false;
    }
    std::printf("  %-24s %5.2f ulp at %-12.6g (subnormal %.3g)   "
                "libm %5.2f / %.3g   [<= %.1f]\n",
                name, a.normal, a.at, a.sub, l.normal, l.sub, bound);
    if (a.normal > bound) {
        std::printf("  FAIL %s: %.4f ulp exceeds the recorded bound %.1f. Do not "
                    "raise the bound to make this pass - it is the accuracy "
                    "claim, and something moved under it.\n",
                    name, a.normal, bound);
        ok = false;
    }
    /* In the subnormal tail the requirement is not a number, it is that the
     * approximation is either correctly rounded anyway or no worse than the
     * formula's existing saturation. Nothing below one ulp is worth a bound:
     * that is already the best the format can carry. */
    if (a.sub > 1.0 && a.sub > l.sub) {
        std::printf("  FAIL %s: %.4g ulp on subnormal results, worse than the "
                    "libm path's %.4g\n", name, a.sub, l.sub);
        ok = false;
    }
    return ok;
}

static bool TestApproxGatesAgainstTruth() {
    ErrStat e, l, s, t, le, ll, ls, lt;
    long n = Sweep(kAccStride, 1, [&](float x) {
        if (std::isnan(x)) {
            /* NaN in, NaN out - the one property to check here, since a NaN
             * has no correctly rounded neighbour to be some number of ulp
             * away from. */
            return;
        }
        double d = (double)x;
        Accumulate(&e,  x, xlstm_gate_expf(x),          TrueExp(d));
        Accumulate(&l,  x, xlstm_gate_log_sigmoidf(x),  TrueLogSigmoid(d));
        Accumulate(&s,  x, xlstm_gate_sigmoidf(x),      TrueSigmoid(d));
        Accumulate(&t,  x, xlstm_gate_tanhf(x),         TrueTanh(d));
        Accumulate(&le, x, LibmExp(x),                  TrueExp(d));
        Accumulate(&ll, x, LibmLogSigmoid(x),           TrueLogSigmoid(d));
        Accumulate(&ls, x, LibmSigmoid(x),              TrueSigmoid(d));
        Accumulate(&lt, x, LibmTanh(x),                 TrueTanh(d));
    });
    bool ok = true;
    /* NaN propagation, which the ulp sweep above steps over. */
    const float kNan[] = {NAN, -NAN};
    for (float v : kNan) {
        if (!std::isnan(xlstm_gate_expf(v)) ||
            !std::isnan(xlstm_gate_log_sigmoidf(v)) ||
            !std::isnan(xlstm_gate_sigmoidf(v)) ||
            !std::isnan(xlstm_gate_tanhf(v))) {
            std::printf("  FAIL a NaN input did not produce a NaN result\n");
            ok = false;
        }
    }
    std::printf("  %ld inputs, error in ulp of the correctly rounded result\n", n);
    ok &= Report("xlstm_gate_expf", e, le, kExpUlpBound);
    ok &= Report("xlstm_gate_log_sigmoidf", l, ll, kLogSigmoidUlpBound);
    ok &= Report("xlstm_gate_sigmoidf", s, ls, kSigmoidUlpBound);
    ok &= Report("xlstm_gate_tanhf", t, lt, kTanhUlpBound);
    return ok;
}

#else /* exact build */

static bool TestExactGatesAreLibm() {
    long bad = 0;
    long n = Sweep(kAccStride, 1, [&](float x) {
        if (bad > 4) return;
        struct { const char* name; float got, want; } c[] = {
            {"xlstm_gate_expf", xlstm_gate_expf(x), LibmExp(x)},
            {"xlstm_gate_log_sigmoidf", xlstm_gate_log_sigmoidf(x), LibmLogSigmoid(x)},
            {"xlstm_gate_sigmoidf", xlstm_gate_sigmoidf(x), LibmSigmoid(x)},
            {"xlstm_gate_tanhf", xlstm_gate_tanhf(x), LibmTanh(x)},
        };
        for (auto& v : c) {
            if (!SameFloat(v.got, v.want)) {
                std::printf("  FAIL %s(%.9g): %.9g, libm %.9g - the default build "
                            "must be bit identical to libm\n",
                            v.name, x, (double)v.got, (double)v.want);
                ++bad;
            }
        }
    });
    if (!bad) std::printf("  %ld inputs, all four wrappers bit identical to libm\n", n);
    return bad == 0;
}

#endif /* XLSTM_APPROX_GATES */

int main() {
    std::printf("[==========] Running gate-math checks (XLSTM_APPROX_GATES=%d)\n",
                XLSTM_APPROX_GATES);
    RUN_TEST(TestZeroExponentShortcutIsExact);
#if XLSTM_APPROX_GATES
    RUN_TEST(TestApproxGatesAgainstTruth);
#else
    RUN_TEST(TestExactGatesAreLibm);
#endif
    std::printf("[==========] %d/%d tests passed\n", g_tests_passed, g_tests_run);
    return g_tests_passed == g_tests_run ? 0 : 1;
}
