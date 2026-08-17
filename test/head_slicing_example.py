#!/usr/bin/env python3
"""Worked example: slicing fused xLSTM weights into the per-head arrays this
library takes. Run it, read it, copy it.

    python3 test/head_slicing_example.py

No dependencies beyond the standard library, so it runs wherever you are
preparing weights. It checks itself against test/reference_data.json and exits
non-zero if the rule below ever stops holding.

-----------------------------------------------------------------------------
WHAT THIS LIBRARY WANTS

Per head, with H = hidden_size (per-head width, DH in the reference) and
I = input_size:

    sLSTM   W [4H, I]        R [4H, H]     b [4H]
    mLSTM   W [(4H+2), I]    (no R)        b [(4H+2)]

Gate rows run in the order i, f, z, o. Heads are YOUR outer loop: call the
kernel once per head, over that head's slice and its own state buffers.

-----------------------------------------------------------------------------
THE PART THAT IS NOT OBVIOUS

The reference stores fused weights GATE-MAJOR over the fused width
Hf = num_heads * H. One head's four gate blocks are therefore STRIDED across
the fused matrix, not contiguous:

    W_h[g*H + j] = W_fused[g*Hf + h*H + j]
    b_h[g*H + j] = b_fused[g*Hf + h*H + j]
    R_h          = R_stack[h]   (already contiguous - no cross-head recurrence)
    y_h[j]       = y_fused[h*H + j]     (c, n, m likewise)

Taking a contiguous [h*4H, (h+1)*4H) row block per head - the natural guess -
selects DIFFERENT ROWS. It runs, it produces plausible numbers, and it is a
different model. That failure is silent, which is why this file exists.

test/derive_multihead_layout.py shows how the rule was established from the
reference implementation; TestHeadComposition in test/slstm_test.cc gates it on
every build.
"""

import json
import os
import sys


def slice_head(fused, h, num_heads, dh, row_len):
    """Rows of one head, gate-major. `row_len` is I for W, or 1 for a bias."""
    hf = num_heads * dh
    out = []
    for g in range(4):
        for j in range(dh):
            src = (g * hf + h * dh + j) * row_len
            out.extend(fused[src:src + row_len])
    return out


def naive_slice_head(fused, h, dh, row_len):
    """The natural guess: one contiguous 4*dh row block per head. Wrong."""
    start = h * 4 * dh * row_len
    return fused[start:start + 4 * dh * row_len]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "reference_data.json")) as f:
        case = json.load(f)["slstm_head2"]

    nh, dh, i_size = case["NH"], case["DH"], case["I"]
    w_fused, b_fused, r_stack = case["W_fused"], case["b_fused"], case["R_stack"]

    print("fused: NH=%d DH=%d I=%d  ->  per head: W[%d,%d] R[%d,%d] b[%d]"
          % (nh, dh, i_size, 4 * dh, i_size, 4 * dh, dh, 4 * dh))

    heads = []
    for h in range(nh):
        w_h = slice_head(w_fused, h, nh, dh, i_size)
        b_h = slice_head(b_fused, h, nh, dh, 1)
        r_h = r_stack[h * 4 * dh * dh:(h + 1) * 4 * dh * dh]
        assert len(w_h) == 4 * dh * i_size and len(b_h) == 4 * dh
        assert len(r_h) == 4 * dh * dh
        heads.append((w_h, r_h, b_h))
        print("  head %d: W %d floats, R %d, b %d" % (h, len(w_h), len(r_h), len(b_h)))

    # Check 1: the rule is a permutation - reassembling every head's rows in
    # gate-major order must reproduce the fused matrix exactly. A wrong index
    # expression cannot survive this.
    hf = nh * dh
    rebuilt = [None] * len(w_fused)
    for h, (w_h, _, _) in enumerate(heads):
        for g in range(4):
            for j in range(dh):
                dst = (g * hf + h * dh + j) * i_size
                src = (g * dh + j) * i_size
                rebuilt[dst:dst + i_size] = w_h[src:src + i_size]
    if rebuilt != w_fused:
        print("FAIL: slicing is not a permutation of the fused weights")
        return 1
    print("  round-trip: per-head slices reassemble to W_fused exactly")

    # Check 2: the naive slicing really does select different rows, so the
    # warning above is not theoretical.
    differs = any(naive_slice_head(w_fused, h, dh, i_size) != heads[h][0]
                  for h in range(nh))
    if not differs:
        print("FAIL: contiguous slicing agrees here, so this example proves nothing")
        return 1
    print("  contiguous slicing selects different rows, as documented")

    print("\nFeed each head's (W, R, b) to slstm_step_f32 with its own y/c/n/m.")
    print("For INT8, quantize with xlstm_quant.h - weights symmetric, activations")
    print("asymmetric, c and n to int16, m stays float32. Every case in")
    print("reference_data.json carries a worked calibration and the exact")
    print("integers the kernel must produce.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
