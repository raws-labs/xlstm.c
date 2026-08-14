/* Test-harness sizing knob - shared by all four *_test.cc runners.
 *
 * XLSTM_TEST_MAX_H bounds the STATIC BUFFERS declared in the test runners
 * themselves (test/slstm_test.cc, test/mlstm_test.cc, test/slstm_s8_test.cc,
 * test/mlstm_s8_test.cc) - the [XLSTM_TEST_MAX_H]/[XLSTM_TEST_MAX_H *
 * XLSTM_TEST_MAX_H]-shaped state/scratch/output arrays each runner sizes for
 * the largest case in test/reference_data.h.
 *
 * This is deliberately a DIFFERENT knob from XLSTM_MAX_HIDDEN
 * (include/xlstm_simd.h), which bounds the SIMD kernels' own internal stack
 * temporaries - a different buffer class entirely. Shrinking
 * XLSTM_TEST_MAX_H shrinks only the host-side test harness's static
 * buffers; it has no effect on what the kernels under test can handle, and
 * does not touch XLSTM_MAX_HIDDEN. Do not conflate the two.
 *
 * Default 256, matching the literal these buffers were hardcoded to before
 * this knob existed. Override at build time with -DXLSTM_TEST_MAX_H=<n> to
 * shrink the harness's BSS footprint (e.g. for a memory-constrained target)
 * - 64 comfortably covers every case in reference_data.h today (the
 * largest is H=64, I=64: SweepS64/SweepM64).
 */
#ifndef XLSTM_TEST_CONFIG_H_
#define XLSTM_TEST_CONFIG_H_

#ifndef XLSTM_TEST_MAX_H
#define XLSTM_TEST_MAX_H 256
#endif

#endif /* XLSTM_TEST_CONFIG_H_ */
