#!/usr/bin/env python3
"""Mutation battery: proof that the correctness gate can still fail.

Each entry below injects one known defect into the kernels, rebuilds, and
asserts the suites FAIL - and that the assertion which fails is the one
recorded for that defect. A defect the suites do not notice is an ESCAPE; a
defect some other check happens to catch while its own has gone blind is a
WRONG CHECK, and both fail this run. "The suite failed" alone would let a
check quietly stop firing behind a neighbour that still does, which is the
same silent loosening this battery exists to detect. Run it after any change
to tolerances, bounds, or test/generate_reference.py - that change class fails
by making a gate stop failing, which nothing else here detects.

A1 must PASS, and is as load-bearing as the rest. It is the 0.1% activation
drift generate_reference.py folds into every bound, standing in for a backend
whose sigmoid/tanh are approximations rather than this one's libm (a CMSIS-NN
LUT, say). Bounds tight enough to "catch" A1 reject that whole class of
legitimate backend with a false failure. A2 is the same defect at 0.2% and
must fail: the margin is deliberate, not unlimited.

Backends differ in which code they compile, so several defects apply to one
and are absent from the other; those report n/a, which is not an escape.

SIX backends, not two. ref and sse2 run on the build host; neon, cortexm, esp
and helium are cross-compiled and executed under emulation by their own make
targets, which is why each of those takes a SUBSET of the shared table (A1 to
B2, one mutation per assertion family plus the pass case) rather than all of
it, and adds mutations for what only it compiles: loop tails, alignment
instances, zero-point folding, lane order, the accelerated-path guards, and
the out-of-bounds checks in test/{simd,esp,cortexm,helium}_gate.cc. A backend
whose toolchain is not installed is reported as NOT COVERED at the end rather
than passed over in silence.

One class of mutation is here for a reason worth stating: forcing an
accelerated body unreachable while leaving every answer intact. The scalar
remainder under each vector loop computes the whole row when the loop runs no
passes, so the suites go green with no vector instruction executed - which is
how the esp backend once ran its accelerated matvec 6 times in 76 suite calls
without any gate noticing. Only a counter can see it, and the entries that
inject it (S1 to S4, N7 to N10, M6, M7, P1 to P4, H1 to H4) are the proof that
those counters are wired to an assertion.

Deliberately not in CI: this edits files in the working tree. Sources are
copied to .mutants-backup/ and restored on exit, on failure and on SIGINT.
A run killed outright leaves that directory behind, and the next run restores
from it before touching anything.

Usage: make mutants   (or: python3 test/mutants.py [backend ...])
"""

import os
import re
import shutil
import signal
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUP = os.path.join(REPO, ".mutants-backup")

S8, M8 = "src/slstm_s8.c", "src/mlstm_s8.c"
SCALAR, SSE2 = "src/xlstm_simd_scalar.h", "src/xlstm_simd_sse2.c"
NEON, CORTEXM = "src/xlstm_simd_neon.c", "src/xlstm_simd_cortexm.c"
ESP, HELIUM = "src/xlstm_simd_esp.c", "src/xlstm_simd_helium.c"

# HOST is the pair `make test` can run here; CROSS each need a toolchain and an
# emulator and are driven through their own make target.
HOST = ("ref", "sse2")
CROSS = ("neon", "cortexm", "esp", "helium")
EVERY = HOST + CROSS

# What each cross backend's gate needs on PATH. Used only to decide the default
# backend list: a backend named explicitly is attempted whatever is installed,
# so a missing tool fails loudly rather than reducing the run.
CROSS_TOOLS = {
    "neon": ("aarch64-linux-gnu-gcc", "qemu-aarch64"),
    "cortexm": ("arm-linux-gnueabihf-gcc", "qemu-arm"),
    "esp": ("xtensa-esp32s3-elf-gcc", "qemu-system-xtensa"),
    "helium": ("arm-none-eabi-gcc", "qemu-system-arm"),
}

# --- anchors: exact text in the current tree, replaced literally ------------

SY = "        float y_new = o_gate * (c_new / xlstm_maxf(n_new, 1e-6f));"
MY = "        float y_new = xlstm_gate_sigmoidf(o_raw[j]) * (qC_j / denom);"
SCQ = "        float c_q = c_new / params->c_quant.scale;"
MCQ = "            float C_q = C_new / params->C_quant.scale;"
NQ = "        float n_q = n_new / params->n_quant.scale;"
YQ = ("        float y_q = y_new / params->y_quant.scale"
      " + (float)params->y_quant.zero_point;")
TAIL = """            /* Copy hidden state to output */
            for (i = 0; i < H; ++i) {
                output[(batch * T + t) * H + i] = y[batch * H + i];
            }
        }
    }
"""
S_MATVEC = """        for (j = 0; j < cols; ++j) {
            acc += M[i * cols + j] * v[j];"""
S_MATVEC_S8 = """        for (j = 0; j < cols; ++j) {
            acc += (int32_t)M[i * cols + j] * ((int32_t)v[j] - v_zp);"""
SSE_TAIL_F32 = """        for (; j < cols; ++j) {
            s += row[j] * v[j];"""
SSE_TAIL_S8 = """        for (; j < cols; ++j) {
            s += (int32_t)row[j] * ((int32_t)v[j] - v_zp);"""

# sse2 and neon: the four vector loops. A bound of zero runs no pass at all
# and leaves every answer intact - the scalar remainder below each loop then
# computes the whole row - so nothing but the counters can see it.
SSE_VEC_F32 = """        for (j = 0; j < cols4; j += 4) {
            __m128 m = _mm_loadu_ps(row + j);"""
SSE_VEC_S8 = "        for (j = 0; j < cols8; j += 8) {"
SSE_VEC_RANK1 = "        for (c = 0; c < H4; c += 4) {"
SSE_VEC_VECMAT = """        for (j = 0; j < cols4; j += 4) {
            __m128 mv = _mm_loadu_ps(Mrow + j);"""

# neon: one tail per kernel, the vector zero point, and the lane pairing.
NE_TAIL_F32 = """        float s = vaddvq_f32(acc);
        for (; j < cols; ++j) {
            s += row[j] * v[j];"""
NE_TAIL_S8 = """        int32_t s = vaddvq_s32(acc);
        for (; j < cols; ++j) {
            s += (int32_t)row[j] * ((int32_t)v[j] - v_zp);"""
NE_TAIL_RANK1 = """        for (; c < H; ++c) {
            Crow[c] = f_gate * Crow[c] + i_gate * k[r] * v[c];"""
NE_TAIL_VECMAT = """        for (; j < cols; ++j) {
            out[j] += q[i] * Mrow[j];"""
NE_ZP = "            int16x8_t v16 = vsubq_s16(vmovl_s8(vr), vzp);"
NE_LANES = """            acc = vmlal_s16(acc, vget_low_s16(m16), vget_low_s16(v16));
            acc = vmlal_s16(acc, vget_high_s16(m16), vget_high_s16(v16));"""
NE_VEC_F32 = """        for (j = 0; j < cols4; j += 4) {
            float32x4_t m = vld1q_f32(row + j);"""
NE_VEC_S8 = "        for (j = 0; j < cols8; j += 8) {"
NE_VEC_RANK1 = "        for (c = 0; c < H4; c += 4) {"
NE_VEC_VECMAT = """        for (j = 0; j < cols4; j += 4) {
            float32x4_t mv = vld1q_f32(Mrow + j);"""

# cortexm: the two group-load instances, the zero-point fold, the lane pairing,
# the f32 accumulator seed, and a group load that reads one word ahead.
CM_UNALIGNED = "        xlstm_cm_matvec_s8(M, v, out, rows, cols, v_zp, 0);"
CM_ZP = """    const int16x2_t nzp = (int16x2_t)(((uint32_t)(-v_zp) & 0xFFFFu) |
                                      ((uint32_t)(-v_zp) << 16));"""
CM_LANES = """            a0 = __smlad(__sxtb16(m0), ve, a0);
            a0 = __smlad(xlstm_cm_sxtb16_ror8(m0), vo, a0);"""
CM_SEED = """            float a0 = out[i], a1 = out[i + 1];
            float a2 = out[i + 2], a3 = out[i + 3];
            float a4 = out[i + 4], a5 = out[i + 5];
            float a6 = out[i + 6], a7 = out[i + 7];"""
CM_LD4 = """    if (aligned) {
        w = *(const xlstm_cm_word*)(const void*)p;
    } else {"""
# The two accelerated bodies, made unreachable. Both leave every answer
# unchanged: the scalar body computes the same integers, and the two-row tier
# uses the same fmaf as the eight-row block it replaces.
CM_S8_GUARD = ("    if (cols <= 0 || v_zp > XLSTM_CM_ZP_MAX"
               " || v_zp < -XLSTM_CM_ZP_MAX) {")
CM_BLOCK8 = "        for (; i + 7 < rows; i += 8) {"

# esp: the four accelerated-path guards, and the partial group's upper block.
EP_F32 = "    const int fast = cols >= 7 && rows >= tile;"
EP_S8 = """    const int fast = cols > 0 &&
                     v_zp >= -XLSTM_ESP_ZP_MAX && v_zp <= XLSTM_ESP_ZP_MAX;"""
EP_RANK1 = "    const int fast = H >= 7;"
EP_VECMAT = "    const int wide = (cols % 4 == 0) && cols >= 7;"
EP_UPPER = ("        mh = (const int8_t*)((uintptr_t)ml"
            " + ((sm + t > 16) ? 16u : 0u));")

# helium: the four vector-body guards, and the gather clamp.
HL_F32 = """    if (cols <= 0 || rows <= 0) {
        XLSTM_HL_COUNT(matvec_f32, 0, 0);"""
HL_S8 = """    if (cols <= 0 || rows <= 0) {
        XLSTM_HL_COUNT(matvec_s8, 0, 0);"""
HL_RANK1 = """    if (H <= 0) {
        XLSTM_HL_COUNT(rank1_f32, 0, 0);"""
HL_VECMAT = """    if (rows <= 0 || cols <= 0) {
        XLSTM_HL_COUNT(vecmat_f32, 0, 0);"""
HL_CLAMP = """        const uint32x4_t off =
            vmulq_n_u32(vminq_u32(lane, vdupq_n_u32((uint32_t)(rows - i - 1))),
                        (uint32_t)cols);"""

# Zero every state element under half its tensor's maximum. The defect an
# earlier per-tensor bound could not see: each element individually sat under
# a bound derived from the whole tensor's largest.
SUBMAX = """    { int16_t* tt[2]; int len[2]; int e, kk; int32_t v, mx;
      %s
      for (kk = 0; kk < 2; ++kk) {
          mx = 0;
          for (e = 0; e < len[kk]; ++e) { v = tt[kk][e]; if (v < 0) v = -v;
              if (v > mx) mx = v; }
          for (e = 0; e < len[kk]; ++e) { v = tt[kk][e]; if (v < 0) v = -v;
              if (v < mx / 2) tt[kk][e] = 0; }
      } }
"""


def drift(f):
    """Persistent factor on y before requantization, both cells - the same
    place and the same form as generate_reference.py's `perturb`."""
    return [(S8, SY, SY + "\n        y_new *= %sf;" % f),
            (M8, MY, MY + "\n        y_new *= %sf;" % f)]


def after_eval(s_snippet, m_snippet):
    """Corrupt state after the timestep loop, where it can no longer reach
    output[] - only the exit-state assertions can see it."""
    return [(S8, TAIL, TAIL + s_snippet), (M8, TAIL, TAIL + m_snippet)]


# --- signatures: WHICH assertion has to fire ------------------------------
#
# "The suite failed" is a weaker claim than it looks. A mutation can stay
# caught while the check it was written to exercise has gone blind, because
# some other check happens to fire first - and a check that quietly stopped
# firing is the exact defect this battery exists to detect. So every mutation
# records the assertion expected to catch it, matched against the first FAIL
# line the suites print, and a caught-by-the-wrong-check run fails.
#
# Each pattern names a check and the tensor it fires on, and deliberately not
# the element index: which element trips first is a property of the golden
# data, not of the check. The same defect turns out to have the same catcher
# on every backend that compiles it, including the ones whose f32 rounding
# differs, so one pattern per mutation is the whole of it.
def near(t):    # ExpectNear, the f32 runners' only comparison (test_util.h).
    return r"FAIL %s\[\d+\]: expected .*\(diff [^,)]*\)$" % t


def elem(t):    # ExpectStatePerElem - per-element INT8 exit-state bound.
    return r"FAIL %s\[\d+\]: .*element magnitude" % t


def sfloor(t):  # ExpectStateFloorConsistent - exit-state drift against the
    return r"FAIL state-floor-consistency %s\[" % t   # numpy replica's floor.


def chan(t):    # Per-channel INT8 output bound, open-coded in the s8 runners.
    return r"FAIL %s\[\d+\]: .*, tol " % t


# Per-channel floor consistency, the binding bound on the INT8 output path.
CHFLOOR = r"FAIL floor-consistency ch\["

# The gate binaries' own assertions, one per kernel and not one per gate: each
# gate compares values as well as counting bodies, so a signature that accepted
# any FAIL from esp_gate would let a path mutation be "caught" by a value
# mismatch while the counter check itself had gone blind.
ESP_PATH_F32 = r"FAIL rows=\d+ cols=\d+ M\+\d+ v\+\d+: expected the \w+ path"
ESP_PATH_S8 = r"FAIL s8 rows=\d+ .*: expected the \w+ path"
ESP_PATH_RANK1 = r"FAIL rank1 H=\d+ .*: expected fast\+"
ESP_PATH_VECMAT = r"FAIL vecmat rows=\d+ .*: expected wide\+"


def vecpath(k):  # the CheckSplit in test/simd_gate.cc and test/helium_gate.cc,
    return r"FAIL %s .*expected vector\+" % k   # per contract function.


# cortexm's two splits: three arms for the INT8 dispatch, and whether the f32
# kernel's eight-row block ran.
CM_PATH_S8 = r"FAIL matvec_s8 .*expected aligned\+"
CM_PATH_F32 = r"FAIL matvec_f32 .*expected blocked\+"


# What an out-of-bounds READ looks like, which is a fault and not a comparison:
# the cortexm gate catches its own SIGSEGV and names the operand, the helium
# boot code reports through semihosting, and on xtensa the toolchain's handler
# prints PANIC.
CM_OOB = r"FAIL edge \S+ \S+: the kernel accessed memory past"
HL_OOB = r"FAIL fault: the image faulted"
EP_OOB = r"PANIC: Unhandled exception!"

# id, what, expect, signature, backends, edits
#
# Read the signature column as the honest record of what catches what, not as
# what one might wish caught it. Every state-corrupting mutation is caught by a
# state assertion, which is unremarkable. The two that are worth knowing:
# F1/F2 zero a single output channel and are caught by the exit-state checks,
# not by the per-channel output bound they were written for - a zeroed channel
# feeds back through c and n before the output path ever sees it. B1 is the
# mutation that does isolate the output path, by corrupting y after the state
# is final. Recording the real catcher is the point; a signature written to
# flatter the design would enforce nothing.
MUTANTS = [
    ("A1", "activation drift y * 1.001", "pass", None, EVERY, drift("1.001")),
    ("A2", "activation drift y * 1.002", "fail", sfloor("m"), EVERY,
     drift("1.002")),
    ("A3", "activation drift y * 1.05", "fail", CHFLOOR, EVERY, drift("1.05")),
    ("B1", "final y zeroed at H=8, output[] correct", "fail", chan("y"), EVERY,
     after_eval("    if (H == 8) { for (i = 0; i < B * H; ++i) y[i] = 0; }\n",
                "    if (H == 8) { for (i = 0; i < B * H; ++i) y[i] = 0; }\n")),
    ("B2", "exit state zeroed", "fail", elem("m"), EVERY,
     after_eval("    for (i = 0; i < B * H; ++i) { c[i] = 0; n[i] = 0;"
                " m[i] = 0.0f; }\n",
                "    for (i = 0; i < B * H * H; ++i) C[i] = 0;\n"
                "    for (i = 0; i < B * H; ++i) n[i] = 0;\n"
                "    for (i = 0; i < B; ++i) m[i] = 0.0f;\n")),
    ("B3", "sub-maximum state elements zeroed", "fail", elem("c"), HOST,
     after_eval(SUBMAX % "tt[0] = c; len[0] = B * H; tt[1] = n; len[1] = B * H;",
                SUBMAX % "tt[0] = C; len[0] = B * H * H; tt[1] = n; len[1] = B * H;")),
    ("C1", "state requantization drift 1.05x", "fail", sfloor("n"), HOST,
     [(S8, SCQ, SCQ.replace("c_new /", "1.05f * c_new /")),
      (S8, NQ, NQ.replace("n_new /", "1.05f * n_new /")),
      (M8, MCQ, MCQ.replace("C_new /", "1.05f * C_new /")),
      (M8, NQ, NQ.replace("n_new /", "1.05f * n_new /"))]),
    ("D1", "n requantized with the c/C scale", "fail", elem("n"), HOST,
     [(S8, NQ, NQ.replace("n_quant", "c_quant")),
      (M8, NQ, NQ.replace("n_quant", "C_quant"))]),
    ("D2", "dropped y zero point", "fail", chan("y"), HOST,
     [(S8, YQ, "        float y_q = y_new / params->y_quant.scale;"),
      (M8, YQ, "        float y_q = y_new / params->y_quant.scale;")]),
    ("D3", "dropped matvec input zero point", "fail", elem("m"), ("ref",),
     [(SCALAR, "((int32_t)v[j] - v_zp)", "((int32_t)v[j] - 0 * v_zp)")]),
    ("D4", "sse2 matvec drops its vector zero point", "fail", elem("m"),
     ("sse2",),
     [(SSE2, "        v16 = _mm_sub_epi16(v16, vzp);",
       "        v16 = _mm_sub_epi16(v16, _mm_setzero_si128()); (void)vzp;")]),
    ("E1", "INT8 matvec drops its last column", "fail", elem("m"), ("ref",),
     [(SCALAR, S_MATVEC_S8, S_MATVEC_S8.replace("j < cols;", "j < cols - 1;"))]),
    ("E2", "f32 matvec drops its last column", "fail", near("y"), ("ref",),
     [(SCALAR, S_MATVEC, S_MATVEC.replace("j < cols;", "j < cols - 1;"))]),
    ("E3", "sse2 INT8 matvec skips its scalar tail", "fail", elem("m"),
     ("sse2",),
     [(SSE2, SSE_TAIL_S8, SSE_TAIL_S8.replace("j < cols;", "j < cols8;"))]),
    ("E4", "sse2 f32 matvec skips its scalar tail", "fail", near("y"),
     ("sse2",),
     [(SSE2, SSE_TAIL_F32, SSE_TAIL_F32.replace("j < cols;", "j < cols4;"))]),

    # --- sse2: the four vector loops ---------------------------------------
    #
    # S1 is the one that is NOT caught by its gate, and the signature says so.
    # Losing the vector body of a f32 matvec also changes the summation order
    # - four lane accumulators become one running sum - and the f32 goldens
    # turn out to be tight enough to see that, by 4.6e-05 against their own
    # bound. So the suites fail first and this defect never reaches the gate.
    # That is luck rather than design, and it holds for exactly one of these
    # eight vector bodies: S2 to S4 and N8 to N10 are bit-identical either
    # way, which is why nothing but a counter could ever have seen them.
    ("S1", "sse2 f32 matvec never enters its vector body", "fail", near("y"),
     ("sse2",),
     [(SSE2, SSE_VEC_F32, SSE_VEC_F32.replace("j < cols4", "j < 0"))]),
    ("S2", "sse2 INT8 matvec never enters its vector body", "fail",
     vecpath("matvec_s8"), ("sse2",),
     [(SSE2, SSE_VEC_S8, SSE_VEC_S8.replace("j < cols8", "j < 0"))]),
    ("S3", "sse2 rank-1 update never enters its vector body", "fail",
     vecpath("rank1_f32"), ("sse2",),
     [(SSE2, SSE_VEC_RANK1, SSE_VEC_RANK1.replace("c < H4", "c < 0"))]),
    ("S4", "sse2 vecmat never enters its vector body", "fail",
     vecpath("vecmat_f32"), ("sse2",),
     [(SSE2, SSE_VEC_VECMAT, SSE_VEC_VECMAT.replace("j < cols4", "j < 0"))]),
    # SweepS17 ch5 and ch6 are the narrowest windows in the whole table: the
    # bound sits at ~0.96x the channel's own range, so these two are where a
    # single corrupted channel comes closest to fitting underneath it.
    ("F1", "sLSTM H=17 channel 5 zeroed", "fail", sfloor("m"), HOST,
     [(S8, SY, SY + "\n        if (H == 17 && i == 5) y_new = 0.0f;")]),
    ("F2", "sLSTM H=17 channel 6 zeroed", "fail", elem("m"), HOST,
     [(S8, SY, SY + "\n        if (H == 17 && i == 6) y_new = 0.0f;")]),

    # --- neon: four vectorized kernels, so four scalar tails ---------------
    ("N1", "neon f32 matvec skips its scalar tail", "fail", near("y"),
     ("neon",),
     [(NEON, NE_TAIL_F32, NE_TAIL_F32.replace("j < cols;", "j < cols4;"))]),
    ("N2", "neon INT8 matvec skips its scalar tail", "fail", elem("m"),
     ("neon",),
     [(NEON, NE_TAIL_S8, NE_TAIL_S8.replace("j < cols;", "j < cols8;"))]),
    ("N3", "neon rank-1 update skips its scalar tail", "fail", near("y"),
     ("neon",),
     [(NEON, NE_TAIL_RANK1, NE_TAIL_RANK1.replace("c < H;", "c < H4;"))]),
    ("N4", "neon vecmat skips its scalar tail", "fail", near("y"), ("neon",),
     [(NEON, NE_TAIL_VECMAT, NE_TAIL_VECMAT.replace("j < cols;",
                                                    "j < cols4;"))]),
    ("N5", "neon INT8 matvec drops its vector zero point", "fail", elem("m"),
     ("neon",),
     [(NEON, NE_ZP, "            int16x8_t v16 = vmovl_s8(vr); (void)vzp;")]),
    ("N6", "neon INT8 matvec pairs the wrong halves", "fail", elem("m"),
     ("neon",),
     [(NEON, NE_LANES,
       "            acc = vmlal_s16(acc, vget_low_s16(m16),"
       " vget_high_s16(v16));\n"
       "            acc = vmlal_s16(acc, vget_high_s16(m16),"
       " vget_low_s16(v16));")]),
    # N7 is S1 on the other backend, and is caught the same way - by the f32
    # goldens rather than by the gate. See the note above S1.
    ("N7", "neon f32 matvec never enters its vector body", "fail", near("y"),
     ("neon",),
     [(NEON, NE_VEC_F32, NE_VEC_F32.replace("j < cols4", "j < 0"))]),
    ("N8", "neon INT8 matvec never enters its vector body", "fail",
     vecpath("matvec_s8"), ("neon",),
     [(NEON, NE_VEC_S8, NE_VEC_S8.replace("j < cols8", "j < 0"))]),
    ("N9", "neon rank-1 update never enters its vector body", "fail",
     vecpath("rank1_f32"), ("neon",),
     [(NEON, NE_VEC_RANK1, NE_VEC_RANK1.replace("c < H4", "c < 0"))]),
    ("N10", "neon vecmat never enters its vector body", "fail",
     vecpath("vecmat_f32"), ("neon",),
     [(NEON, NE_VEC_VECMAT, NE_VEC_VECMAT.replace("j < cols4", "j < 0"))]),

    # --- cortexm: two group-load instances, and a read that faults ---------
    ("M1", "cortexm INT8 matvec always takes the aligned instance", "fail",
     elem("m"), ("cortexm",),
     [(CORTEXM, CM_UNALIGNED,
       CM_UNALIGNED.replace("v_zp, 0);", "v_zp, 1);"))]),
    ("M2", "cortexm folds +v_zp instead of -v_zp", "fail", elem("m"),
     ("cortexm",),
     [(CORTEXM, CM_ZP, CM_ZP.replace("(-v_zp)", "(v_zp)"))]),
    ("M3", "cortexm INT8 matvec pairs the wrong halfword lanes", "fail",
     elem("m"), ("cortexm",),
     [(CORTEXM, CM_LANES,
       "            a0 = __smlad(__sxtb16(m0), vo, a0);\n"
       "            a0 = __smlad(xlstm_cm_sxtb16_ror8(m0), ve, a0);")]),
    ("M4", "cortexm f32 matvec drops its accumulator seed", "fail", near("y"),
     ("cortexm",),
     [(CORTEXM, CM_SEED,
       "            float a0 = 0.0f, a1 = 0.0f;\n"
       "            float a2 = 0.0f, a3 = 0.0f;\n"
       "            float a4 = 0.0f, a5 = 0.0f;\n"
       "            float a6 = 0.0f, a7 = 0.0f;")]),
    # The one defect no value comparison can reach: four bytes read past a row
    # and discarded. Only the guard page in test/cortexm_gate.cc sees it.
    ("M5", "cortexm group load reads one word ahead", "fail", CM_OOB,
     ("cortexm",),
     [(CORTEXM, CM_LD4,
       "    if (aligned) {\n"
       "        { volatile uint32_t ahead_ ="
       " ((const xlstm_cm_word*)(const void*)p)[1];\n"
       "          (void)ahead_; }\n"
       "        w = *(const xlstm_cm_word*)(const void*)p;\n"
       "    } else {")]),
    ("M6", "cortexm INT8 matvec never enters its DSP body", "fail",
     CM_PATH_S8, ("cortexm",),
     [(CORTEXM, CM_S8_GUARD, CM_S8_GUARD.replace("if (cols", "if (1 || cols"))]),
    ("M7", "cortexm f32 matvec never enters its eight-row block", "fail",
     CM_PATH_F32, ("cortexm",),
     [(CORTEXM, CM_BLOCK8, CM_BLOCK8.replace("i + 7 < rows", "0"))]),

    # --- esp: the four accelerated-path guards, and an over-read -----------
    ("P1", "esp f32 matvec never enters its blocked body", "fail",
     ESP_PATH_F32, ("esp",),
     [(ESP, EP_F32, EP_F32.replace("= cols", "= 0 && cols"))]),
    ("P2", "esp INT8 matvec never enters its vector body", "fail",
     ESP_PATH_S8, ("esp",),
     [(ESP, EP_S8, EP_S8.replace("= cols", "= 0 && cols"))]),
    ("P3", "esp rank-1 update never enters its 128-bit body", "fail",
     ESP_PATH_RANK1, ("esp",),
     [(ESP, EP_RANK1, EP_RANK1.replace("= H", "= 0 && H"))]),
    ("P4", "esp vecmat never takes its 128-bit load of M", "fail",
     ESP_PATH_VECMAT, ("esp",),
     [(ESP, EP_VECMAT, EP_VECMAT.replace("= (cols", "= 0 && (cols"))]),
    # Loading the partial group's upper block unconditionally is invisible to
    # every value check - the lanes above the row meet a zero-padded constant -
    # and reads up to 16 bytes past the operand. Only the edge placement in
    # test/esp_gate.cc sees it, as a fault.
    ("P5", "esp INT8 tail always loads the block after its row", "fail",
     EP_OOB, ("esp",),
     [(ESP, EP_UPPER,
       "        mh = (const int8_t*)((uintptr_t)ml + 16u);")]),

    # --- helium: the four vector-body guards, and the gather clamp ---------
    ("H1", "helium f32 matvec never enters its vector body", "fail",
     vecpath("matvec_f32"), ("helium",),
     [(HELIUM, HL_F32, HL_F32.replace("if (cols", "if (1 || cols"))]),
    ("H2", "helium INT8 matvec never enters its vector body", "fail",
     vecpath("matvec_s8"), ("helium",),
     [(HELIUM, HL_S8, HL_S8.replace("if (cols", "if (1 || cols"))]),
    ("H3", "helium rank-1 update never enters its vector body", "fail",
     vecpath("rank1_f32"), ("helium",),
     [(HELIUM, HL_RANK1, HL_RANK1.replace("if (H", "if (1 || H"))]),
    ("H4", "helium vecmat never enters its vector body", "fail",
     vecpath("vecmat_f32"), ("helium",),
     [(HELIUM, HL_VECMAT, HL_VECMAT.replace("if (rows", "if (1 || rows"))]),
    # The gather's spare lanes are clamped onto the last row and discarded by
    # a predicated store. Unclamped, every answer is still right and the
    # kernel reads up to three rows past M.
    ("H5", "helium f32 matvec gathers past its last row", "fail", HL_OOB,
     ("helium",),
     [(HELIUM, HL_CLAMP,
       "        const uint32x4_t off = vmulq_n_u32(lane, (uint32_t)cols);")]),
]

FILES = sorted({f for m in MUTANTS for (f, _, _) in m[5]})

# The three ways a gate here reports a failure: FAIL is an assertion in a
# runner or a gate binary, FATAL is a gate refusing to run at all, and PANIC is
# the xtensa toolchain's handler after an unhandled exception - which is how an
# out-of-bounds read fails on esp. \b so that the "[  FAILED  ]" summary line
# can never stand in for the assertion above it.
MARKER = re.compile(r"\b(FAIL|FATAL|PANIC)\b")


# --- tree handling ---------------------------------------------------------

def restore():
    if not os.path.isdir(BACKUP):
        return
    for rel in FILES:
        saved = os.path.join(BACKUP, rel.replace("/", "__"))
        if os.path.exists(saved):
            shutil.copyfile(saved, os.path.join(REPO, rel))
    shutil.rmtree(BACKUP)


def snapshot():
    os.mkdir(BACKUP)
    for rel in FILES:
        shutil.copyfile(os.path.join(REPO, rel),
                        os.path.join(BACKUP, rel.replace("/", "__")))


def apply(edits):
    for rel, old, new in edits:
        path = os.path.join(REPO, rel)
        with open(path) as fh:
            text = fh.read()
        if text.count(old) != 1:
            raise SystemExit("mutants: %s: anchor matches %d times, not 1:\n%s"
                             % (rel, text.count(old), old))
        with open(path, "w") as fh:
            fh.write(text.replace(old, new))


def unmutate(edits):
    for rel in {rel for rel, _, _ in edits}:
        shutil.copyfile(os.path.join(BACKUP, rel.replace("/", "__")),
                        os.path.join(REPO, rel))


BINS = ["build/%s_test" % t for t in
        ("slstm", "mlstm", "slstm_s8", "mlstm_s8")]


def make(args, backend=None):
    cmd = ["make"] + args
    if backend is not None:
        cmd.append("XLSTM_SIMD=" + backend)
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


def first_failure(text):
    return next((l.strip() for l in text.splitlines() if MARKER.search(l)), "")


def build_and_run(backend):
    """(built, suite_result, first failing assertion). Kept apart on purpose:
    a mutation that does not compile is a broken battery entry, not a caught
    defect, and both show up as a non-zero make."""
    if backend in CROSS:
        # One target, because these cross-compile and then invoke an emulator;
        # splitting build from run would mean reimplementing the target here.
        # A build that failed never reaches a runner, so the banner every
        # binary prints first is what separates the two outcomes.
        r = make(["test-" + backend])
        out = r.stdout + r.stderr
        return ("[==========]" in out,
                "fail" if r.returncode else "pass", first_failure(out))
    b = make(BINS, backend)
    if b.returncode != 0:
        return False, "-", b.stderr.strip().splitlines()[0] if b.stderr else ""
    r = make(["test"], backend)
    return True, ("fail" if r.returncode else "pass"), first_failure(r.stdout)


def have(backend):
    return all(shutil.which(t) for t in CROSS_TOOLS.get(backend, ()))


def missing(backend):
    return [t for t in CROSS_TOOLS.get(backend, ()) if not shutil.which(t)]


# --- run -------------------------------------------------------------------

def main(argv):
    chosen = argv or [b for b in EVERY if have(b)]
    skipped = [b for b in EVERY if b not in chosen]
    for b in chosen:
        if b not in EVERY:
            raise SystemExit("mutants: no such backend: %s" % b)

    # Before the anchor check, not after: a run killed outright leaves mutated
    # sources whose anchors no longer match, and aborting on that would leave
    # them mutated forever.
    if os.path.isdir(BACKUP):
        print("mutants: %s left by an earlier run - restoring it first"
              % os.path.basename(BACKUP))
        restore()

    for rel, old, _ in [(f, o, n) for m in MUTANTS for (f, o, n) in m[5]]:
        with open(os.path.join(REPO, rel)) as fh:
            if fh.read().count(old) != 1:
                raise SystemExit(
                    "mutants: %s no longer contains this text exactly once, so "
                    "the battery cannot be trusted. Re-derive the anchor:\n%s"
                    % (rel, old))

    snapshot()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: (restore(), sys.exit(130)))

    start = time.time()
    results, bad = {}, 0
    try:
        for backend in chosen:
            # Between backends and not between mutations: only xlstm_simd.o
            # differs, and make would otherwise find the previous backend's
            # object up to date and test it under the new backend's name.
            make(["clean"])
            built, suite, _ = build_and_run(backend)
            if not built or suite != "pass":
                print("mutants: %s does not pass unmutated - fix that first"
                      % backend)
                if backend in CROSS:
                    print("  %s needs %s on PATH"
                          % (backend, " and ".join(CROSS_TOOLS[backend])))
                return 1
            print("\n[%s] baseline green" % backend)
            for mid, what, expect, sig, on, edits in MUTANTS:
                if backend not in on:
                    results[(mid, backend)] = "n/a"
                    continue
                apply(edits)
                built, suite, first = build_and_run(backend)
                unmutate(edits)
                if not built:
                    verdict = "BUILD FAIL"
                elif expect == "fail":
                    verdict = "caught" if suite == "fail" else "ESCAPED"
                    # Caught by the recorded assertion, or not caught in the
                    # sense this battery means. A different check firing says
                    # the recorded one no longer sees this defect.
                    if verdict == "caught" and not re.search(sig, first):
                        verdict = "WRONG CHECK"
                else:
                    verdict = "pass" if suite == "pass" else "FALSE FAIL"
                results[(mid, backend)] = verdict
                if verdict != ("pass" if expect == "pass" else "caught"):
                    bad += 1
                print("  %-3s %-45s built=%-3s suite=%-5s %s"
                      % (mid, what, "yes" if built else "NO", suite, verdict))
                if verdict == "WRONG CHECK":
                    print("      expected an assertion matching: %s" % sig)
                    print("      got: %s" % (first or "(no FAIL line at all)"))
                elif first:
                    print("      %s" % first[:72])
    finally:
        restore()
        make(["clean"])

    width = max(len(m[1]) for m in MUTANTS)
    print("\n%-3s %-*s %-7s %s"
          % ("id", width, "mutation", "expect", " ".join("%-10s" % b
                                                         for b in chosen)))
    for mid, what, expect, _, _, _ in MUTANTS:
        print("%-3s %-*s %-7s %s"
              % (mid, width, what, expect,
                 " ".join("%-10s" % results[(mid, b)] for b in chosen)))
    print("\n%d mutations across %s in %.0fs. n/a means the mutation does not "
          "apply there: it edits code that backend does not compile, or it is "
          "one of the shared defects the slower emulated gates do not repeat."
          % (len(MUTANTS), ", ".join(chosen), time.time() - start))
    for b in skipped:
        lack = missing(b)
        print("NOT COVERED: %s - %s"
              % (b, " and ".join(lack) + " not on PATH" if lack
                 else "not named on this run"))
    print("mutants: " + ("OK - every injected defect is caught by the assertion "
                         "recorded for it, and the portability margin holds"
                         if not bad else
                         "FAILED - %d mutation(s) escaped, false-failed, or were "
                         "caught by the wrong check" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
