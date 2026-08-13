#!/usr/bin/env python3
"""Establish the NX-AI sLSTM multi-head weight layout empirically.

Success criterion: running one num_heads=2 cell of head width DH must equal
the concatenation of two independent num_heads=1 cells of width DH, given a
slicing rule. This script searches the small set of plausible slicings and
prints the one that matches. Run it before adding any 2-head reference
vector, so the C-side composition test rests on a verified rule.

Why this is not obvious from the declared shapes: sLSTMCellBase declares
_recurrent_kernel_ as (num_heads, head_dim, num_gates, head_dim) and _bias_
as (num_heads, num_gates, head_dim), but sLSTMCell_vanilla immediately
rewrites both through _recurrent_kernel_ext2int / _bias_ext2int, so the
tensors that actually exist on the module have DIFFERENT shapes and a
DIFFERENT axis order. On top of that, generate_reference.py's run_slstm
assigns cell._recurrent_kernel_.data / cell._bias_.data directly, which
bypasses the ParameterProxy (and therefore bypasses ext2int entirely), so
what it writes must already be in the internal layout. Neither the declared
shapes nor the proxy's conversion functions describe what run_slstm sees.
Hence: measure, do not read off.

Run:  .venv/bin/python3 test/derive_multihead_layout.py
"""

import itertools
import os
import sys

import torch

from xlstm.blocks.slstm.cell import (  # type: ignore[import-untyped]
    sLSTMCell,
    sLSTMCellConfig,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_reference import (  # noqa: E402
    _pack_gate_major,
    run_slstm,
    run_slstm_multihead,
)

DH = 4          # per-head width (this library's `hidden_size`)
NH = 2          # heads
H = DH * NH     # fused width
I = 3
T = 2
NG = 4          # gates: i, f, z, o

TOL = 1e-6      # both paths are the same vanilla float32 code; a correct
                # slicing matches to roughly machine epsilon. Do not loosen.


def make_cell(hidden_size, num_heads):
    cfg = sLSTMCellConfig(
        hidden_size=hidden_size, num_heads=num_heads, backend="vanilla",
        function="slstm", bias_init="zeros", recurrent_weight_init="zeros",
        dtype="float32")
    cell = sLSTMCell(cfg)
    cell.eval()
    return cell


# ---------------------------------------------------------------------------
# Runners. Both are the generator's own, so this script verifies the exact
# code path that produces test/reference_data.h, not a lookalike.
#   run_multi_head_reference:  one num_heads=NH cell -> ([B,NH,T,DH], states
#                              [B,NH*DH] in whatever the fused ordering is)
#   run_single_head_reference: one num_heads=1 cell of width DH ->
#                              ([B,T,DH], states [B,DH])
# ---------------------------------------------------------------------------

run_multi_head_reference = run_slstm_multihead
run_single_head_reference = run_slstm


# ---------------------------------------------------------------------------
# Candidate packings. Each takes the per-head pieces and builds the fused
# tensor a num_heads=NH cell would be fed.
# ---------------------------------------------------------------------------

def pack_rows_gate_major(parts):
    """[gate][head][dim]: gate block g of the fused tensor is the
    concatenation over heads of each head's gate-g block."""
    return torch.cat(
        [torch.cat([p[g * DH:(g + 1) * DH] for p in parts], dim=0)
         for g in range(NG)], dim=0)


def pack_rows_head_major(parts):
    """[head][gate][dim]: each head's whole [NG*DH] block laid end to end."""
    return torch.cat(list(parts), dim=0)


ROW_PACKINGS = {
    "gate_major [gate][head][dim]": pack_rows_gate_major,
    "head_major [head][gate][dim]": pack_rows_head_major,
}


def stack_R_as_is(parts):
    """R_stack[h] is this library's per-head [NG*DH, DH] matrix unchanged."""
    return torch.stack(list(parts), dim=0)


def stack_R_dim_major(parts):
    """R_stack[h] has its NG*DH rows reordered to [dim][gate] instead of
    this library's [gate][dim]. Negative control: if this also matched, the
    row order inside a head would be unconstrained by the data."""
    out = []
    for p in parts:
        out.append(p.reshape(NG, DH, DH).permute(1, 0, 2).reshape(NG * DH, DH))
    return torch.stack(out, dim=0)


R_STACKINGS = {
    "R[h] = per-head [NG*DH, DH] as-is": stack_R_as_is,
    "R[h] rows reordered [dim][gate]":   stack_R_dim_major,
}


def unpack_state_head_major(v):
    """Split a fused-width state vector [NH*DH] into per-head [DH] pieces,
    assuming the fused width is ordered [head][dim]."""
    return [v[..., h * DH:(h + 1) * DH] for h in range(NH)]


def unpack_state_dim_major(v):
    """Negative control: fused width ordered [dim][head] (interleaved)."""
    return [v[..., h::NH] for h in range(NH)]


STATE_UNPACKINGS = {
    "state [head][dim]": unpack_state_head_major,
    "state [dim][head]": unpack_state_dim_major,
}


def main():
    torch.manual_seed(7)

    print("=" * 72)
    print("Step 1/2: declared vs. actual parameter shapes")
    print("=" * 72)
    cell2 = make_cell(H, NH)
    cell1 = make_cell(DH, 1)
    print(f"NH={NH}, DH={DH}, fused hidden_size={H}, I={I}, T={T}")
    print("  NH=2 cell class            :", type(cell2).__name__)
    print("  NH=2 _recurrent_kernel_    :", tuple(cell2._recurrent_kernel_.shape))
    print("  NH=2 _bias_                :", tuple(cell2._bias_.shape))
    print("  NH=1 (width DH) _recurrent_kernel_:",
          tuple(cell1._recurrent_kernel_.shape))
    print("  NH=1 (width DH) _bias_            :", tuple(cell1._bias_.shape))
    print()
    print("  Declared in sLSTMCellBase.__init__:")
    print("    _recurrent_kernel_ (num_heads, head_dim, num_gates, head_dim)"
          f" = {(NH, DH, NG, DH)}")
    print(f"    _bias_             (num_heads, num_gates, head_dim) = {(NH, NG, DH)}")
    print("  -> the vanilla cell's ext2int rewrites both, so the DECLARED")
    print("     shapes are not what .data assignment must match.")
    print()

    # Per-head parameters, in this library's flat packing.
    W_parts = [torch.randn(NG * DH, I) * 0.5 for _ in range(NH)]
    R_parts = [torch.randn(NG * DH, DH) * 0.5 for _ in range(NH)]
    b_parts = [torch.randn(NG * DH) * 0.1 for _ in range(NH)]
    x = torch.randn(1, T, I)

    # Independent single-head runs: the thing the fused cell must reproduce.
    singles = [run_single_head_reference(W_parts[h], R_parts[h], b_parts[h], x)
               for h in range(NH)]
    y_joined = torch.cat([s[1] for s in singles], dim=-1)          # [B, NH*DH]

    print("=" * 72)
    print("Step 2/2: candidate slicings, measured")
    print("=" * 72)
    print(f"{'row packing (W and b)':<30} {'R stacking':<36} "
          f"{'state order':<20} {'max abs err':>12}")
    print("-" * 102)

    verified = []
    for (pname, pack), (rname, rstack), (sname, unpack) in itertools.product(
            ROW_PACKINGS.items(), R_STACKINGS.items(), STATE_UNPACKINGS.items()):
        W_fused = pack(W_parts)
        b_fused = pack(b_parts)
        R_stack = rstack(R_parts)
        try:
            out2, y2, _, _, _ = run_multi_head_reference(
                W_fused, R_stack, b_fused, x, NH)
        except Exception as exc:  # shape mismatch etc. is a legitimate answer
            print(f"{pname:<30} {rname:<36} {sname:<20} "
                  f"{'raised ' + type(exc).__name__:>12}")
            continue
        # Metric: final y only. The state ordering candidate is what `unpack`
        # varies, and out2 already carries an explicit head axis, so folding
        # the output in here would just add a constant to every row of the
        # table. The unique survivor is re-checked against y, c, n, m AND
        # every output timestep below.
        y_cat = torch.cat(unpack(y2), dim=-1)
        err = (y_cat - y_joined).abs().max().item()
        print(f"{pname:<30} {rname:<36} {sname:<20} {err:>12.3e}")
        if err < TOL:
            verified.append((pname, rname, sname, err))

    print()
    if len(verified) != 1:
        print(f"FAILED: expected exactly one candidate under {TOL:g}, "
              f"got {len(verified)}: {[v[:3] for v in verified]}")
        raise SystemExit(1)

    pname, rname, sname, err = verified[0]
    print(f"max |NH=2 - concat(NH=1, NH=1)| = {err:.3e}")
    print()
    print("Unique match:")
    print(f"  W and b rows : {pname}")
    print(f"  R            : {rname}")
    print(f"  y/c/n/m      : {sname}")
    print()
    print("Slicing rule (fused width Hf = NH*DH, gate g in 0..3, head h,")
    print("dim j in 0..DH-1), to recover this library's per-head arrays:")
    print("  W_h[g*DH + j][:] = W_fused[g*Hf + h*DH + j][:]")
    print("  b_h[g*DH + j]    = b_fused[g*Hf + h*DH + j]")
    print("  R_h              = R_stack[h]           (already [4*DH, DH])")
    print("  y_h[j]           = y_fused[h*DH + j]    (same for c, n, m)")
    print()

    # Belt and braces: assert the rule directly, in the form the C test and
    # the generator will use, rather than trusting the search loop's bookkeeping.
    W_fused = pack_rows_gate_major(W_parts)
    b_fused = pack_rows_gate_major(b_parts)
    R_stack = stack_R_as_is(R_parts)

    # ...and pin the generator to the winning candidate, so the packing that
    # actually writes reference_data.h cannot drift away from the one measured
    # above without this script failing.
    for label, mine, theirs in (
            ("W", W_fused, _pack_gate_major(W_parts, NH, DH)),
            ("b", b_fused, _pack_gate_major(b_parts, NH, DH))):
        assert torch.equal(mine, theirs), (
            f"generate_reference._pack_gate_major disagrees with this script's "
            f"winning candidate on {label}")

    out2, y2, c2s, n2, m2 = run_multi_head_reference(
        W_fused, R_stack, b_fused, x, NH)

    worst = 0.0
    for h in range(NH):
        W_h = torch.stack(
            [W_fused[g * H + h * DH + j] for g in range(NG) for j in range(DH)])
        b_h = torch.stack(
            [b_fused[g * H + h * DH + j] for g in range(NG) for j in range(DH)])
        R_h = R_stack[h]
        out1, y1, c1s, n1, m1 = run_single_head_reference(W_h, R_h, b_h, x)
        sl = slice(h * DH, (h + 1) * DH)
        for label, a, bb in (
                ("W_h", W_h, W_parts[h]), ("b_h", b_h, b_parts[h]),
                ("R_h", R_h, R_parts[h]),
                ("y", y1, y2[:, sl]), ("c", c1s, c2s[:, sl]),
                ("n", n1, n2[:, sl]), ("m", m1, m2[:, sl]),
                ("output", out1, out2[:, h])):
            d = (a - bb).abs().max().item()
            worst = max(worst, d)
            if d >= TOL:
                print(f"  head {h} {label}: max abs diff {d:.3e}")

    err = worst
    print(f"max |NH=2 - concat(NH=1, NH=1)| over y, c, n, m and every "
          f"output timestep = {err:.3e}")
    assert err < TOL, "slicing rule is wrong; do not encode a vector yet"
    print("SLICING RULE VERIFIED")


if __name__ == "__main__":
    main()
