#!/usr/bin/env python3
"""Does this configuration fit the part? State and weight bytes, before you
write any code.

    python3 tools/footprint.py            # self-check (stdlib only)
    python3 tools/footprint.py 64 32 8    # hidden_size, input_size, heads

hidden_size is the PER-HEAD width (DH in the reference), so heads multiply
everything below. The number that usually decides the answer is mLSTM state:
its cell state is a hidden_size x hidden_size matrix PER HEAD, so it grows
quadratically while everything else grows linearly.

What is counted, and why the two cells differ by one tensor: sLSTM's y is
recurrent (slstm.h marks it in/out) and is carried state; mLSTM's y is an
output (mlstm.h marks it out) and is not. Weights are per head - a fused
model's weights divided by the head count, not multiplied by it. Scratch is
caller-provided and reused across heads, so it is listed once.

The formulas are checked against the tensor lengths in
test/reference_data.json, which are the shapes the kernels are gated on, and
against the three figures README.md quotes.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(os.path.dirname(HERE), "test")


def state_bytes(cell, h, s8):
    """Carried state, one head. m is float32 in both precisions."""
    if cell == "slstm":
        return (h * (1 if s8 else 4)      # y  int8  / float
                + h * (2 if s8 else 4)    # c  int16 / float
                + h * (2 if s8 else 4)    # n  int16 / float
                + h * 4)                  # m  float, always
    return (h * h * (2 if s8 else 4)      # C  [h*h] int16 / float
            + h * (2 if s8 else 4)        # n  int16 / float
            + 4)                          # m  one float per head


def weight_bytes(cell, h, i, s8):
    """Weights, one head. The INT8 bias is int32, not int8: it is added into
    the accumulator at W_scale * x_scale and never fits a byte."""
    if cell == "slstm":
        return (4 * h * i * (1 if s8 else 4)    # W [4h, i]
                + 4 * h * h * (1 if s8 else 4)  # R [4h, h]
                + 4 * h * 4)                    # b [4h] int32 or float
    rows = 4 * h + 2                            # q,k,v, i,f scalars, o
    return rows * i * (1 if s8 else 4) + rows * 4


def scratch_bytes(cell, h):
    """Caller-provided accumulators, reused across heads. int32 and float are
    the same width, so this does not depend on the precision."""
    return 4 * h * 4 if cell == "slstm" else (4 * h + 2) * 4


def report(h, i, heads):
    print("xlstm.c footprint: hidden_size %d (per head), input_size %d, "
          "%d head(s)\n" % (h, i, heads))
    print("  cell   prec   state/head  weights/head    total state  "
          "total weights")
    for cell in ("slstm", "mlstm"):
        for s8 in (False, True):
            s, w = state_bytes(cell, h, s8), weight_bytes(cell, h, i, s8)
            print("  %-6s %-5s %11s %13s %14s %14s"
                  % ("sLSTM" if cell == "slstm" else "mLSTM",
                     "int8" if s8 else "f32", "{:,}".format(s),
                     "{:,}".format(w), "{:,}".format(s * heads),
                     "{:,}".format(w * heads)))
    print("\n  scratch (once, not per head): sLSTM %s B, mLSTM %s B"
          % ("{:,}".format(scratch_bytes("slstm", h)),
             "{:,}".format(scratch_bytes("mlstm", h))))
    print("  all figures in bytes")


# ---------------------------------------------------------------------------
# Self-check: the formulas above against the tensor lengths the correctness
# gate actually runs on. A formula that has drifted from the kernels cannot
# survive this, because these are the same arrays the kernels are handed.
# ---------------------------------------------------------------------------

def _measured(cell, tc, s8):
    """Bytes implied by one case's stored tensors."""
    r, t = tc["s8"], tc
    if not s8:
        state = [(len(t["expected_n"]), 4), (len(t["expected_m"]), 4)]
        state += ([(len(t["expected_y"]), 4), (len(t["expected_c"]), 4)]
                  if cell == "slstm" else [(len(t["expected_C"]), 4)])
        weights = [(len(t["W"]), 4), (len(t["b"]), 4)]
        if cell == "slstm":
            weights.append((len(t["R"]), 4))
    else:
        state = [(len(r["expected_n_q"]), 2), (len(r["expected_m"]), 4)]
        state += ([(len(r["expected_y_q"]), 1), (len(r["expected_c_q"]), 2)]
                  if cell == "slstm" else [(len(r["expected_C_q"]), 2)])
        weights = [(len(r["W_q"]), 1), (len(r["b_q"]), 4)]
        if cell == "slstm":
            weights.append((len(r["R_q"]), 1))
    return (sum(n * w for n, w in state), sum(n * w for n, w in weights))


def self_check():
    with open(os.path.join(TEST, "reference_data.json")) as f:
        data = json.load(f)
    fails = n = 0
    for cell in ("slstm", "mlstm"):
        for name, tc in sorted(data[cell].items()):
            h, i = tc["H"], tc["I"]
            for s8 in (False, True):
                got = (state_bytes(cell, h, s8), weight_bytes(cell, h, i, s8))
                want = _measured(cell, tc, s8)
                n += 1
                if got != want:
                    fails += 1
                    print("FAIL %s %s %s: state/weights %s, tensors say %s"
                          % (cell, name, "int8" if s8 else "f32", got, want))
    print("%d configurations agree with the tensor shapes in "
          "test/reference_data.json" % n)

    # The three mLSTM f32 state figures README.md quotes, which reach a width
    # the reference data does not: 16.3 KB and 64.5 KB per head, and 130 KB
    # for an 8-head layer that therefore does not fit a 128 KB part.
    for label, got, want in (("H=64 per head", state_bytes("mlstm", 64, False), 16644),
                             ("H=128 per head", state_bytes("mlstm", 128, False), 66052),
                             ("H=64, 8 heads", state_bytes("mlstm", 64, False) * 8, 133152)):
        if got != want:
            fails += 1
            print("FAIL: mLSTM f32 state %s is %d, README.md says %d"
                  % (label, got, want))
    if fails:
        return 1
    print("  and with the three mLSTM state figures in README.md")

    # Negative controls: the two ways a footprint tool is usually wrong. If
    # either passed, the check above would not be measuring anything.
    tc = data["mlstm"]["test8"]
    h = tc["H"]
    for label, wrong, measured in (
            ("mLSTM state linear in hidden_size",     # C treated as a vector
             h * 4 + h * 4 + 4, _measured("mlstm", tc, False)[0]),
            ("INT8 state the same size as f32",
             state_bytes("mlstm", h, False), _measured("mlstm", tc, True)[0])):
        if wrong == measured:
            print("FAIL: %s is NOT caught, so the check above is blind" % label)
            return 1
        print("  caught: %s" % label)
    return 0


def main(argv):
    rc = self_check()
    if rc or not argv:
        if not argv:
            print("\nreport a configuration with: python3 tools/footprint.py"
                  " <hidden_size> <input_size> [heads]")
        return rc
    print()
    report(int(argv[0]), int(argv[1]), int(argv[2]) if len(argv) > 2 else 1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
