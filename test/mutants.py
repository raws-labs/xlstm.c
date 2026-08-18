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
and are absent from the other; those report n/a, which is not an escape. Only
the backends that run on the build host are covered: neon, cortexm, esp and
helium have tails and zero-point handling of their own that nothing here
mutates. Each of those is gated instead by its own emulated target.

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
BOTH = ("ref", "sse2")

# --- anchors: exact text in the current tree, replaced literally ------------

SY = "        float y_new = o_gate * (c_new / xlstm_maxf(n_new, 1e-6f));"
MY = "        float y_new = sigmoid_f32(o_raw[j]) * (qC_j / denom);"
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
# data, not of the check. The five spellings below are all the assertions
# these four suites have.
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
    ("A1", "activation drift y * 1.001", "pass", None, BOTH, drift("1.001")),
    ("A2", "activation drift y * 1.002", "fail", sfloor("m"), BOTH,
     drift("1.002")),
    ("A3", "activation drift y * 1.05", "fail", CHFLOOR, BOTH, drift("1.05")),
    ("B1", "final y zeroed at H=8, output[] correct", "fail", chan("y"), BOTH,
     after_eval("    if (H == 8) { for (i = 0; i < B * H; ++i) y[i] = 0; }\n",
                "    if (H == 8) { for (i = 0; i < B * H; ++i) y[i] = 0; }\n")),
    ("B2", "exit state zeroed", "fail", elem("m"), BOTH,
     after_eval("    for (i = 0; i < B * H; ++i) { c[i] = 0; n[i] = 0;"
                " m[i] = 0.0f; }\n",
                "    for (i = 0; i < B * H * H; ++i) C[i] = 0;\n"
                "    for (i = 0; i < B * H; ++i) n[i] = 0;\n"
                "    for (i = 0; i < B; ++i) m[i] = 0.0f;\n")),
    ("B3", "sub-maximum state elements zeroed", "fail", elem("c"), BOTH,
     after_eval(SUBMAX % "tt[0] = c; len[0] = B * H; tt[1] = n; len[1] = B * H;",
                SUBMAX % "tt[0] = C; len[0] = B * H * H; tt[1] = n; len[1] = B * H;")),
    ("C1", "state requantization drift 1.05x", "fail", sfloor("n"), BOTH,
     [(S8, SCQ, SCQ.replace("c_new /", "1.05f * c_new /")),
      (S8, NQ, NQ.replace("n_new /", "1.05f * n_new /")),
      (M8, MCQ, MCQ.replace("C_new /", "1.05f * C_new /")),
      (M8, NQ, NQ.replace("n_new /", "1.05f * n_new /"))]),
    ("D1", "n requantized with the c/C scale", "fail", elem("n"), BOTH,
     [(S8, NQ, NQ.replace("n_quant", "c_quant")),
      (M8, NQ, NQ.replace("n_quant", "C_quant"))]),
    ("D2", "dropped y zero point", "fail", chan("y"), BOTH,
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
    # SweepS17 ch5 and ch6 are the narrowest windows in the whole table: the
    # bound sits at ~0.96x the channel's own range, so these two are where a
    # single corrupted channel comes closest to fitting underneath it.
    ("F1", "sLSTM H=17 channel 5 zeroed", "fail", sfloor("m"), BOTH,
     [(S8, SY, SY + "\n        if (H == 17 && i == 5) y_new = 0.0f;")]),
    ("F2", "sLSTM H=17 channel 6 zeroed", "fail", elem("m"), BOTH,
     [(S8, SY, SY + "\n        if (H == 17 && i == 6) y_new = 0.0f;")]),
]

FILES = sorted({f for m in MUTANTS for (f, _, _) in m[5]})


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


def make(args, backend):
    return subprocess.run(["make"] + args + ["XLSTM_SIMD=" + backend],
                          cwd=REPO, capture_output=True, text=True)


def build_and_run(backend):
    """(built, suite_result, first failing assertion). Kept apart on purpose:
    a mutation that does not compile is a broken battery entry, not a caught
    defect, and both show up as a non-zero make."""
    b = make(BINS, backend)
    if b.returncode != 0:
        return False, "-", b.stderr.strip().splitlines()[0] if b.stderr else ""
    r = make(["test"], backend)
    first = next((l.strip() for l in r.stdout.splitlines() if "FAIL" in l), "")
    return True, ("fail" if r.returncode else "pass"), first


# --- run -------------------------------------------------------------------

def main(backends):
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
        for backend in backends:
            # Between backends and not between mutations: only xlstm_simd.o
            # differs, and make would otherwise find the previous backend's
            # object up to date and test it under the new backend's name.
            make(["clean"], backend)
            built, suite, _ = build_and_run(backend)
            if not built or suite != "pass":
                print("mutants: %s does not pass unmutated - fix that first"
                      % backend)
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
                print("  %-3s %-42s built=%-3s suite=%-5s %s"
                      % (mid, what, "yes" if built else "NO", suite, verdict))
                if verdict == "WRONG CHECK":
                    print("      expected an assertion matching: %s" % sig)
                    print("      got: %s" % (first or "(no FAIL line at all)"))
                elif first:
                    print("      %s" % first[:72])
    finally:
        restore()
        make(["clean"], backends[0])

    width = max(len(m[1]) for m in MUTANTS)
    print("\n%-3s %-*s %-7s %s"
          % ("id", width, "mutation", "expect", " ".join("%-10s" % b
                                                         for b in backends)))
    for mid, what, expect, _, _, _ in MUTANTS:
        print("%-3s %-*s %-7s %s"
              % (mid, width, what, expect,
                 " ".join("%-10s" % results[(mid, b)] for b in backends)))
    print("\n%d mutations across %s in %.0fs. n/a means that backend does not "
          "compile the mutated code."
          % (len(MUTANTS), " and ".join(backends), time.time() - start))
    print("mutants: " + ("OK - every injected defect is caught by the assertion "
                         "recorded for it, and the portability margin holds"
                         if not bad else
                         "FAILED - %d mutation(s) escaped, false-failed, or were "
                         "caught by the wrong check" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["ref", "sse2"]))
