#!/usr/bin/env python3
"""Does this configuration fit the part? State and weight bytes, before you
write any code.

    python3 tools/footprint.py               # self-check (stdlib only)
    python3 tools/footprint.py 64 32 8       # hidden_size, input_size, heads
    python3 tools/footprint.py 64 32 8 256   # ... and a 256 KB SRAM budget

hidden_size is the PER-HEAD width (DH in the reference), so heads multiply
everything below. The number that usually decides the answer is mLSTM state:
its cell state is a hidden_size x hidden_size matrix PER HEAD, so it grows
quadratically while everything else grows linearly.

What is counted, and why the two cells differ by one tensor: sLSTM's y is
recurrent (slstm.h marks it in/out) and is carried state; mLSTM's y is an
output (mlstm.h marks it out) and is not. Weights are per head - a fused
model's weights divided by the head count, not multiplied by it. Scratch is
caller-provided and reused across heads, so it is listed once.

It also prices what recurrence is usually weighed against: a full attention
KV cache, which grows with the sequence while the cell state does not. Read
the caveats compare() prints before quoting that half of it.

The cell formulas are checked against the tensor lengths in
test/reference_data.json, which are the shapes the kernels are gated on, and
against the three figures README.md quotes. The attention formula cannot be:
nothing in this repo produces a K or a V. It is a definition, and the checks
only hold it to itself and to the crossover and ceiling derived from it.
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


def mlstm_at(t, h, heads=1, s8=False):
    """mLSTM carried state after t steps. t is accepted and ignored - that
    independence is the claim, and it stays in the signature to be tested."""
    del t
    return state_bytes("mlstm", h, s8) * heads


def kv_at(t, h, heads=1, s8=False):
    """Full-attention KV cache after t steps: K and V are each a [t, h]
    matrix per head, 4 bytes per element in f32 and 1 in INT8. Per-token
    dequant scales are not counted, which flatters attention slightly."""
    return 2 * t * h * heads * (1 if s8 else 4)


def crossover(h, s8=False, cell=mlstm_at, kv=kv_at):
    """First t at which the KV cache exceeds everything the mLSTM cell will
    ever hold, or None if it never does. Head count cancels - both sides are
    per head. The costs are parameters so the controls can inject wrong ones."""
    for t in range(1, 1 << 16):
        if kv(t, h, 1, s8) > cell(t, h, 1, s8):
            return t
    return None


def ceiling(budget, h, heads=1, s8=False, kv=kv_at):
    """Longest sequence whose KV cache alone fits budget bytes. The cell has
    no equivalent: it fits at t = 1 or never, and then for any t."""
    return budget // kv(1, h, heads, s8)


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


def compare(h, heads, budgets):
    print("attention KV cache vs mLSTM state: hidden_size %d (per head), "
          "%d head(s)\n" % (h, heads))
    print("  prec    mLSTM state   KV per step   KV passes the cell at")
    for s8 in (False, True):
        print("  %-5s %13s %13s   t = %s"
              % ("int8" if s8 else "f32",
                 "{:,}".format(mlstm_at(0, h, heads, s8)),
                 "{:,}".format(kv_at(1, h, heads, s8)), crossover(h, s8)))
    print("\n  SRAM budget   longest t, f32 KV   int8 KV   mLSTM cell state")
    fixed = mlstm_at(0, h, heads, False)
    for kb in budgets:
        print("  %-13s %17s %9s   %s"
              % ("%d KB" % kb, "{:,}".format(ceiling(kb << 10, h, heads)),
                 "{:,}".format(ceiling(kb << 10, h, heads, True)),
                 "fits, any t" if fixed <= (kb << 10) else "does not fit"))
    # The four objections a reader should raise, answered here rather than
    # left for them to find. Each one narrows the claim.
    print("""
  Full attention only: sliding-window and chunked attention bound the cache
  too, and linear attention has constant state. The claim is against a cache
  that grows with t, not against every alternative to recurrence.
  An INT8 KV cache shrinks that growth 4x - it does not remove it.
  Below the crossover attention is cheaper and carries no recurrent state at
  all: at t = 1 its cache is %s B against the cell's %s B.
  State only - no weights, activations, scratch, code or stack, which
  compete for the same SRAM - so every longest-t above is a best case for
  attention, and the real ceiling is lower.""" % (
        "{:,}".format(kv_at(1, h, heads, False)),
        "{:,}".format(mlstm_at(0, h, heads, False))))


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

    # 16,644 is the per-head figure README.md quotes, so this keeps the two in
    # step. The other two pin the formula at widths the reference data does not
    # reach, including an 8-head layer that does not fit a 128 KB part.
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

    # The mLSTM side of the comparison is state_bytes, so it inherits the
    # grounding above. The attention side has none to inherit - no test here
    # produces a K or a V - so all that can be checked is that crossover and
    # ceiling fall out of the costs compare() prints, not typed in by hand.
    #
    # First kv_at against a tensor list, in the shape _measured() uses. That
    # is the definition written a second way, not evidence for it: it pins
    # the two tensors and their element widths, and nothing in this repo can
    # confirm an attention implementation stores exactly those two.
    for t, hh, n, s8 in ((1, 8, 1, False), (7, 17, 3, True), (64, 64, 2, False)):
        kv = [(t * hh * n, 1 if s8 else 4),      # K [t, h] per head
              (t * hh * n, 1 if s8 else 4)]      # V [t, h] per head
        if kv_at(t, hh, n, s8) != sum(a * b for a, b in kv):
            print("FAIL: kv_at(t=%d, H=%d, %d heads, %s) is not K plus V"
                  % (t, hh, n, "int8" if s8 else "f32"))
            return 1

    for h in (1, 8, 17, 64, 128):
        for s8 in (False, True):
            t = crossover(h, s8)
            ok = t is not None and (kv_at(t - 1, h, 1, s8)
                                    <= mlstm_at(t, h, 1, s8) < kv_at(t, h, 1, s8))
            for kb in (128, 512, 520):
                n = ceiling(kb << 10, h, 3, s8)
                ok = ok and kv_at(n, h, 3, s8) <= kb << 10 < kv_at(n + 1, h, 3, s8)
            if not ok:
                print("FAIL: crossover t=%s / ceiling at H=%d %s do not fall "
                      "out of the costs" % (t, h, "int8" if s8 else "f32"))
                return 1
    print("kv_at is K plus V, and crossover and ceiling fall out of the two"
          "\n  cost functions the report prints, at H = 1, 8, 17, 64, 128 "
          "and 128/512/520 KB")

    # Negative controls: three wrong cost functions, each of which must move
    # or destroy the crossover, or the check above is measuring nothing.
    truth = (crossover(64), crossover(64, True))
    for label, cell, kv in (
            ("an mLSTM state that grows with t",
             lambda t, h, n=1, s8=False: state_bytes("mlstm", h, s8) * n * t,
             kv_at),
            ("a KV cache that does not depend on t", mlstm_at,
             lambda t, h, n=1, s8=False: 2 * h * n * (1 if s8 else 4)),
            ("an INT8 KV cache the same size as f32", mlstm_at,
             lambda t, h, n=1, s8=False: 2 * t * h * n * 4)):
        if (crossover(64, False, cell, kv), crossover(64, True, cell, kv)) == truth:
            print("FAIL: %s is NOT caught, so the crossover is blind" % label)
            return 1
        print("  caught: %s" % label)
    return 0


def main(argv):
    rc = self_check()
    if rc or not argv:
        if not argv:
            print("\nreport a configuration with: python3 tools/footprint.py"
                  " <hidden_size> <input_size> [heads] [budget_kb ...]")
        return rc
    h, i = int(argv[0]), int(argv[1])
    heads = int(argv[2]) if len(argv) > 2 else 1
    print()
    report(h, i, heads)
    print()
    compare(h, heads, [int(a) for a in argv[3:]] or [128, 512, 520])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
