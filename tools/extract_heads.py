#!/usr/bin/env python3
"""Worked example: trained sLSTM weights -> this library's per-head arrays.

    python3 tools/extract_heads.py                # self-check (stdlib only)
    python3 tools/extract_heads.py --emit         # ... and print C arrays
    python3 tools/extract_heads.py --check-torch  # ... and check the torch
                                                  #     reader (torch + xlstm)

Not an exporter, not a format, and nothing here is promised to keep working.
It is a worked example that checks itself against test/reference_data.json,
so the index expressions below are verified rather than merely asserted.

For your own model:

    from extract_heads import emit_array, fused_from_slstm_layer, heads_from_fused
    heads = heads_from_fused(**fused_from_slstm_layer(model.blocks[0].xlstm))
    open("w.h", "w").write(emit_array("float", "head0_W", heads[0]["W"]))

Only the first of those needs torch, to read tensors out of a live module.
Everything after it is plain lists, which is why the part that can be wrong
is the part CI runs.

The slicing rule is not restated here: it lives in
test/head_slicing_example.py, which shows why the natural contiguous guess is
wrong, and this file imports it. One copy, one place to fix.

sLSTM only. mlstm.h states that the gate-major rule is proven for sLSTM and
does not carry over on its own to mLSTM's fused q/k/v/i/f/o projection, so
there is nothing verified to ship for mLSTM extraction.
"""

import json
import os
import struct
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(os.path.dirname(HERE), "test")
sys.path.insert(0, TEST)

from head_slicing_example import naive_slice_head, slice_head  # noqa: E402


def heads_from_fused(w_fused, r_stack, b_fused, num_heads, dh, input_size):
    """Fused reference weights -> one (W, R, b) triple per head, flat lists.

    W is [4*dh, input_size], R is [4*dh, dh], b is [4*dh], gate rows i,f,z,o:
    exactly what slstm_step_f32 takes, with hidden_size = dh."""
    per_head_r = 4 * dh * dh
    return [{"W": slice_head(w_fused, h, num_heads, dh, input_size),
             "R": r_stack[h * per_head_r:(h + 1) * per_head_r],
             "b": slice_head(b_fused, h, num_heads, dh, 1)}
            for h in range(num_heads)]


def emit_array(ctype, name, values, fmt="%.9g"):
    """A C definition of one array. %.9g because 9 significant digits is what
    a float32 needs to survive the round trip through decimal; %f or the
    default %g silently rounds weights on the way out."""
    suffix = "f" if ctype == "float" else ""
    body = ", ".join(fmt % v + suffix for v in values)
    return "static const %s %s[%d] = {\n%s\n};\n" % (
        ctype, name, len(values),
        textwrap.fill(body, 76, initial_indent="    ", subsequent_indent="    "))


def parse_array(text):
    """The numbers back out of emit_array, for the round-trip check."""
    body = text[text.index("{") + 1:text.rindex("}")]
    return [float(tok.strip().rstrip("f")) for tok in body.split(",")]


def _f32(values):
    return struct.pack("<%df" % len(values), *values)


# ---------------------------------------------------------------------------
# The torch layer: the only part that needs a framework, and the only part CI
# cannot check. Kept to reading tensors out of a live module.
# ---------------------------------------------------------------------------

def fused_from_slstm_layer(layer):
    """A reference sLSTMLayer -> the fused tensors heads_from_fused slices.

    Three traps, all silent if you guess:

    1. The layer feeds its cell cat([fgate(x), igate(x), zgate(x), ogate(x)]),
       and the cell reads those blocks as i, f, z, o. The module NAMED fgate
       supplies the i gate. Collecting the modules in gate order i, f, z, o
       swaps input and forget, which runs and is a different model.
    2. LinearHeadwiseExpand.weight is (heads, out_per_head, in_per_head): the
       input projection is block diagonal, so head h reads only its own
       in_per_head slice of the layer input. That width is this library's
       input_size, and the head's x is x[h*in:(h+1)*in].
    3. A raw state_dict is not enough: _recurrent_kernel_ and _bias_ are
       stored in the cell backend's INTERNAL layout, and vanilla and cuda
       disagree. Load the checkpoint into a model and convert with the
       backend's own int2ext, as below.
    """
    cell = layer.slstm_cell
    nh, dh = cell.config.num_heads, cell.config.head_dim
    if getattr(layer, "conv1d", None) is not None:
        raise ValueError(
            "this layer has a causal conv1d, so its i and f gates see the "
            "convolved input while z and o see the raw input; slstm_step_f32 "
            "takes one x and cannot express that. Either build the layer with "
            "conv1d_kernel_size=0, or feed the kernel x' = [conv(x), x] with "
            "the unused half of each gate's rows zeroed, which costs twice "
            "the input-projection work.")

    w_fused = []
    for gate in (layer.fgate, layer.igate, layer.zgate, layer.ogate):
        wt = gate.weight.detach().cpu()          # (nh, dh, in_per_head)
        for h in range(nh):
            for j in range(dh):
                w_fused.extend(float(v) for v in wt[h, j])
    input_size = layer.fgate.weight.shape[2]

    b_ext = cell._bias_int2ext(cell._bias_.detach().cpu())          # (nh,4,dh)
    b_fused = [float(b_ext[h, g, j])
               for g in range(4) for h in range(nh) for j in range(dh)]

    # ext is (heads, dh, gates, dh); this library wants R[g*dh + a][bb], which
    # is the backend-independent external tensor transposed back the way the
    # vanilla cell's own ext2int does it.
    r_ext = cell._recurrent_kernel_int2ext(
        cell._recurrent_kernel_.detach().cpu())
    r_stack = [float(r_ext[h, bb, g, a])
               for h in range(nh) for g in range(4)
               for a in range(dh) for bb in range(dh)]

    return {"w_fused": w_fused, "r_stack": r_stack, "b_fused": b_fused,
            "num_heads": nh, "dh": dh, "input_size": input_size}


def check_torch():
    """Each extracted head must reproduce its slice of a real 2-head layer's
    cell output. Needs torch and xlstm, so CI cannot run it."""
    import torch
    from xlstm.blocks.slstm.layer import sLSTMLayer, sLSTMLayerConfig
    from generate_reference import run_slstm

    torch.manual_seed(3)
    nh, dh, t = 2, 4, 3
    layer = sLSTMLayer(sLSTMLayerConfig(
        embedding_dim=nh * dh, num_heads=nh, conv1d_kernel_size=0, dropout=0.0,
        backend="vanilla", function="slstm", bias_init="standard",
        recurrent_weight_init="standard", dtype="float32"))
    layer.eval()

    x = torch.randn(1, t, nh * dh)
    with torch.no_grad():
        gates = torch.cat([layer.fgate(x), layer.igate(x),
                           layer.zgate(x), layer.ogate(x)], dim=-1)
        ref_out, _ = layer.slstm_cell(gates)     # [B, NH, T, DH]

    fused = fused_from_slstm_layer(layer)
    heads = heads_from_fused(**fused)
    isz = fused["input_size"]
    worst = 0.0
    for h, head in enumerate(heads):
        out, _, _, _, _ = run_slstm(
            torch.tensor(head["W"]).reshape(4 * dh, isz),
            torch.tensor(head["R"]).reshape(4 * dh, dh),
            torch.tensor(head["b"]),
            x[..., h * isz:(h + 1) * isz])
        worst = max(worst, float((out - ref_out[:, h]).abs().max()))
    print("  torch reader: max |per-head kernel - layer cell| = %.3e" % worst)
    if worst >= 1e-6:
        print("FAIL: the extracted heads do not reproduce the layer")
        return 1
    return 0


def main(argv):
    with open(os.path.join(TEST, "reference_data.json")) as f:
        case = json.load(f)["slstm_head2"]
    nh, dh, isz = case["NH"], case["DH"], case["I"]
    heads = heads_from_fused(case["W_fused"], case["R_stack"], case["b_fused"],
                             nh, dh, isz)

    for h, head in enumerate(heads):
        if (len(head["W"]), len(head["R"]), len(head["b"])) != (
                4 * dh * isz, 4 * dh * dh, 4 * dh):
            print("FAIL: head %d has the wrong shape" % h)
            return 1
    print("fused NH=%d DH=%d I=%d -> %d heads of W[%d,%d] R[%d,%d] b[%d]"
          % (nh, dh, isz, nh, 4 * dh, isz, 4 * dh, dh, 4 * dh))

    # Check 1: the extraction is a permutation of the fused rows. Slicing a
    # vector of row INDICES with the same rule says which fused row each
    # per-head row came from; every row must be claimed exactly once, so
    # nothing is dropped, duplicated or invented.
    rows = list(range(4 * nh * dh))
    claimed = sorted(i for h in range(nh) for i in slice_head(rows, h, nh, dh, 1))
    if claimed != rows:
        print("FAIL: slicing does not claim every fused row exactly once")
        return 1
    print("  rows: all %d fused rows claimed exactly once" % len(rows))

    # Check 2: the contiguous per-head block really is a different set of
    # rows, so the warning in head_slicing_example.py is not theoretical.
    if not any(naive_slice_head(case["W_fused"], h, dh, isz) != heads[h]["W"]
               for h in range(nh)):
        print("FAIL: contiguous slicing agrees here, so this proves nothing")
        return 1
    print("  contiguous slicing selects different rows, as documented")

    # Check 3: emitting and reading back must not move a single float32 bit.
    # The negative control is the same emit at %.6g, which must lose data -
    # otherwise this check would pass for a lossy formatter too.
    w0 = heads[0]["W"]
    if _f32(parse_array(emit_array("float", "w", w0))) != _f32(w0):
        print("FAIL: emit_array does not round-trip in float32")
        return 1
    if _f32(parse_array(emit_array("float", "w", w0, "%.6g"))) == _f32(w0):
        print("FAIL: %.6g round-trips too, so check 3 cannot fail")
        return 1
    print("  emit: %d floats round-trip exactly, %%.6g does not" % len(w0))

    if "--emit" in argv:
        print()
        for h, head in enumerate(heads):
            for key in ("W", "R", "b"):
                print(emit_array("float", "slstm_head%d_%s" % (h, key),
                                 head[key]))
    if "--check-torch" in argv:
        return check_torch()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
