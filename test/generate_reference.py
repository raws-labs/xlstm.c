#!/usr/bin/env python3
"""Generate reference test values for sLSTM and mLSTM kernel tests.

Writes test/reference_data.h from the NX-AI/xlstm reference (vanilla backend).
https://github.com/NX-AI/xlstm

Requires: pip install torch xlstm
Usage:    make reference
"""

import json
import math
import os
import torch
from xlstm.blocks.slstm.cell import (  # type: ignore[import-untyped]
    sLSTMCell,
    sLSTMCellConfig,
)
from xlstm.blocks.mlstm.backends import (  # type: ignore[import-untyped]
    recurrent_step_stabilized_simple,
)


# ============================================================================
# sLSTM helpers
# ============================================================================

def make_slstm_cell(hidden_size):
    """Create a vanilla sLSTM cell (num_heads=1, float32)."""
    config = sLSTMCellConfig(
        hidden_size=hidden_size,
        num_heads=1,
        backend="vanilla",
        function="slstm",
        bias_init="zeros",
        recurrent_weight_init="zeros",
        dtype="float32",
    )
    cell = sLSTMCell(config)
    cell.eval()
    return cell


def run_slstm(W, R, b, x_seq):
    """
    Run sLSTM on a sequence using the NX-AI xlstm reference.

    Args:
        W: [4*H, I] input weight matrix
        R: [4*H, H] recurrent weight matrix
        b: [4*H] bias vector
        x_seq: [B, T, I] input sequence

    Returns:
        output: [B, T, H] hidden outputs per timestep
        y, c, n, m: [B, H] final states
    """
    H = R.shape[1]
    B = x_seq.shape[0]

    cell = make_slstm_cell(H)

    with torch.no_grad():
        cell._recurrent_kernel_.data = R.unsqueeze(0)
        cell._bias_.data = b

        Wx = torch.matmul(x_seq, W.T)
        state = torch.zeros(4, B, H, dtype=torch.float32)
        output, final_state = cell(Wx, state=state)

    output = output.squeeze(1)
    y, c, n, m = final_state[0], final_state[1], final_state[2], final_state[3]
    return output, y, c, n, m


# ============================================================================
# mLSTM helpers
# ============================================================================

def run_mlstm(W, b, x_seq):
    """
    Run mLSTM on a sequence with sigmoid output gate.

    Our kernel weight layout: W[(4*H+2), I], b[4*H+2]
      Rows 0..H-1:     W_q (query)
      Rows H..2H-1:    W_k (key)
      Rows 2H..3H-1:   W_v (value)
      Row  3H:          w_i (scalar input gate)
      Row  3H+1:        w_f (scalar forget gate)
      Rows 3H+2..4H+1: W_o (output gate)

    Cross-validates core state update against recurrent_step_stabilized_simple.

    Args:
        W: [(4*H+2), I] weight matrix
        b: [4*H+2] bias vector
        x_seq: [B, T, I] input sequence

    Returns:
        output: [B, T, H] hidden outputs per timestep
        y: [B, H] final hidden state
        C: [B, H*H] final cell state (flattened)
        n: [B, H] final normalizer
        m: [B, 1] final stabilizer
    """
    total_rows = W.shape[0]
    I = W.shape[1]
    H = (total_rows - 2) // 4
    B, T = x_seq.shape[0], x_seq.shape[1]

    # Initialize states
    C = torch.zeros(B, H, H, dtype=torch.float32)
    n = torch.zeros(B, H, 1, dtype=torch.float32)
    m = torch.zeros(B, 1, 1, dtype=torch.float32)
    outputs = []

    with torch.no_grad():
        for t in range(T):
            x_t = x_seq[:, t, :]  # [B, I]

            # Compute projections: [B, 4*H+2]
            proj = x_t @ W.T + b  # [B, 4*H+2]

            q = proj[:, :H]              # [B, H]
            k = proj[:, H:2*H]           # [B, H]
            v = proj[:, 2*H:3*H]         # [B, H]
            i_raw = proj[:, 3*H:3*H+1]   # [B, 1]
            f_raw = proj[:, 3*H+1:3*H+2] # [B, 1]
            o_raw = proj[:, 3*H+2:]       # [B, H]

            # Cross-validate with NX-AI reference (NH=1)
            q_ref = q.unsqueeze(1).unsqueeze(2)    # [B, 1, 1, H]
            k_ref = k.unsqueeze(1).unsqueeze(2)    # [B, 1, 1, H]
            v_ref = v.unsqueeze(1).unsqueeze(2)    # [B, 1, 1, H]
            i_ref = i_raw.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 1]
            f_ref = f_raw.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 1]
            C_ref = C.unsqueeze(1)                 # [B, 1, H, H]
            n_ref = n.unsqueeze(1)                 # [B, 1, H, 1]
            m_ref = m.unsqueeze(1)                 # [B, 1, 1, 1]

            h_ref, (C_new_ref, n_new_ref, m_new_ref) = \
                recurrent_step_stabilized_simple(
                    C_ref, n_ref, m_ref,
                    q_ref, k_ref, v_ref,
                    i_ref, f_ref)

            # h_ref: [B, 1, 1, H] -> squeeze to [B, H]
            h_ref = h_ref.squeeze(1).squeeze(1)
            C = C_new_ref.squeeze(1)      # [B, H, H]
            n = n_new_ref.squeeze(1)      # [B, H, 1]
            m = m_new_ref.squeeze(1)      # [B, 1, 1]

            # Apply sigmoid output gate (our kernel includes this)
            o_gate = torch.sigmoid(o_raw)  # [B, H]
            y = o_gate * h_ref             # [B, H]
            outputs.append(y)

    output = torch.stack(outputs, dim=1)  # [B, T, H]
    C_flat = C.reshape(B, H * H)          # [B, H*H]
    n_flat = n.squeeze(-1)                # [B, H]
    m_flat = m.squeeze(-1)                # [B, 1]
    return output, y, C_flat, n_flat, m_flat


def fmt(tensor):
    """Format tensor values as C float initializer list."""
    return ", ".join(f"{v:.8f}f" for v in tensor.flatten().tolist())


def build_cases():
    """Build every reference case once. Both emitters consume this.

    Returns (slstm_cases, mlstm_cases). Each case is a dict of torch tensors
    plus its dimensions. Keeping this as the single source of truth is what
    lets the .h and .json emitters stay in sync.

    Required keys: name, B, T, I, H, W, b, input, y, c, n, m, output
    (output may be None). sLSTM cases also need R.

    Optional keys, all read via .get() with safe defaults in _emit_case, so
    a new case can omit them entirely:
      - comment: one-line description shown in the header comment.
        Defaults to "" if omitted.
      - label: display text for the header comment, e.g. "Test 1" or
        "mLSTM Test 1". Only the six legacy cases below need this (their
        display text has a space the C identifier prefix doesn't); new
        cases can skip it and it defaults to the case's `name`.
      - combined: set True only to reuse the previous case's W/R/b unchanged
        (compact single-line header, no new k<name>_W/R/b arrays emitted).
        New cases should leave this unset/False and supply their own W/R/b.
      - note: extra comment line (e.g. an overflow-prevention explainer)
        emitted after the B=..., T=... line. Omit if not needed.
    """
    slstm_cases = []
    mlstm_cases = []

    # --- sLSTM 1: single timestep, zero initial state ---
    torch.manual_seed(42)
    W = torch.randn(8, 2) * 0.5
    R = torch.randn(8, 2) * 0.5
    b = torch.zeros(8)
    x1 = torch.tensor([[[1.0, 0.5]]])
    _, y, c, n, m = run_slstm(W, R, b, x1)
    slstm_cases.append(dict(
        name="Test1", label="Test 1", comment="Single timestep, zero initial state",
        B=1, T=1, I=2, H=2,
        W=W, R=R, b=b, input=x1,
        y=y, c=c, n=n, m=m, output=None))

    # --- sLSTM 2: 3 timesteps, state propagation (reuses Test1 weights) ---
    x2 = torch.tensor([[[1.0, 0.5], [0.3, -0.2], [-0.5, 1.0]]])
    output, y, c, n, m = run_slstm(W, R, b, x2)
    slstm_cases.append(dict(
        name="Test2", label="Test 2", comment="3 timesteps, state propagation",
        B=1, T=3, I=2, H=2, combined=True,
        W=W, R=R, b=b, input=x2,
        y=y, c=c, n=n, m=m, output=output))

    # --- sLSTM 3: large inputs, overflow prevention ---
    W3 = torch.tensor([
        [5.0, 5.0], [5.0, 5.0],
        [5.0, 5.0], [5.0, 5.0],
        [0.5, 0.5], [0.5, 0.5],
        [0.5, 0.5], [0.5, 0.5],
    ])
    R3 = torch.zeros(8, 2)
    b3 = torch.zeros(8)
    x3 = torch.tensor([[[10.0, 10.0]]])
    _, y, c, n, m = run_slstm(W3, R3, b3, x3)
    slstm_cases.append(dict(
        name="Test3", label="Test 3", comment="Large inputs, overflow prevention",
        note="i_raw = 100 - would overflow without m-stabilizer",
        B=1, T=1, I=2, H=2,
        W=W3, R=R3, b=b3, input=x3,
        y=y, c=c, n=n, m=m, output=None))

    # --- mLSTM 1: single timestep, zero initial state ---
    torch.manual_seed(123)
    mW = torch.randn(10, 3) * 0.5
    mb = torch.zeros(10)
    mx1 = torch.tensor([[[1.0, 0.5, -0.3]]])
    _, y, C, n, m = run_mlstm(mW, mb, mx1)
    mlstm_cases.append(dict(
        name="MTest1", label="mLSTM Test 1", comment="Single timestep, zero initial state",
        B=1, T=1, I=3, H=2,
        W=mW, b=mb, input=mx1,
        y=y, c=C, n=n, m=m, output=None))

    # --- mLSTM 2: 3 timesteps, state propagation ---
    mx2 = torch.tensor([[[1.0, 0.5, -0.3], [0.3, -0.2, 0.8], [-0.5, 1.0, 0.1]]])
    output, y, C, n, m = run_mlstm(mW, mb, mx2)
    mlstm_cases.append(dict(
        name="MTest2", label="mLSTM Test 2", comment="3 timesteps, state propagation",
        B=1, T=3, I=3, H=2, combined=True,
        W=mW, b=mb, input=mx2,
        y=y, c=C, n=n, m=m, output=output))

    # --- mLSTM 3: large values, overflow prevention ---
    mW3 = torch.tensor([
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0],
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
    ])
    mb3 = torch.zeros(10)
    mx3 = torch.tensor([[[10.0, 10.0, 10.0]]])
    _, y, C, n, m = run_mlstm(mW3, mb3, mx3)
    mlstm_cases.append(dict(
        name="MTest3", label="mLSTM Test 3", comment="Large values, overflow prevention",
        note="i_raw = 150 - would overflow without m-stabilizer",
        B=1, T=1, I=3, H=2,
        W=mW3, b=mb3, input=mx3,
        y=y, c=C, n=n, m=m, output=None))

    return slstm_cases, mlstm_cases


def _emit_case(f, tc, state_key, has_R):
    """Emit one case as C arrays. state_key is 'c' (sLSTM) or 'C' (mLSTM).

    combined=True cases reuse the previous case's W/R/b, so their header
    folds onto a single comment line and no k<name>_W/R/b arrays are written.

    label, comment, combined, and note are all optional (see build_cases'
    docstring for defaults) so a case built from only the documented
    interface keys still emits without raising.
    """
    n = tc["name"]
    label = tc.get("label", n)
    comment = tc.get("comment", "")
    if tc.get("combined"):
        f.write(
            f"// {label}: {comment} "
            f"(B={tc['B']}, T={tc['T']}, I={tc['I']}, H={tc['H']})\n"
        )
    else:
        f.write(f"// {label}: {comment}\n")
        f.write(f"// B={tc['B']}, T={tc['T']}, I={tc['I']}, H={tc['H']}\n")
        if tc.get("note"):
            f.write(f"// {tc['note']}\n")
        f.write(f"const float k{n}_W[] = {{{fmt(tc['W'])}}};\n")
        if has_R:
            f.write(f"const float k{n}_R[] = {{{fmt(tc['R'])}}};\n")
        f.write(f"const float k{n}_b[] = {{{fmt(tc['b'])}}};\n")
    f.write(f"const float k{n}_input[] = {{{fmt(tc['input'])}}};\n")
    f.write(f"const float k{n}_expected_y[] = {{{fmt(tc['y'])}}};\n")
    f.write(f"const float k{n}_expected_{state_key}[] = {{{fmt(tc['c'])}}};\n")
    f.write(f"const float k{n}_expected_n[] = {{{fmt(tc['n'])}}};\n")
    f.write(f"const float k{n}_expected_m[] = {{{fmt(tc['m'])}}};\n")
    if tc["output"] is not None:
        f.write(f"const float k{n}_expected_output[] = {{{fmt(tc['output'])}}};\n")
    f.write("\n")


def generate(f):
    """Generate all reference data into file handle f."""
    slstm_cases, mlstm_cases = build_cases()

    f.write(
        "/* Auto-generated - do not edit.\n"
        " * Source: NX-AI/xlstm reference (vanilla backend)\n"
        " * Regenerate: make reference\n"
        " */\n\n"
        "#ifndef REFERENCE_DATA_H_\n"
        "#define REFERENCE_DATA_H_\n\n"
    )

    f.write("// " + "=" * 72 + "\n")
    f.write("// sLSTM reference data\n")
    f.write("// " + "=" * 72 + "\n\n")
    for tc in slstm_cases:
        _emit_case(f, tc, state_key="c", has_R=True)

    f.write("// " + "=" * 72 + "\n")
    f.write("// mLSTM reference data\n")
    f.write("// " + "=" * 72 + "\n\n")
    for tc in mlstm_cases:
        _emit_case(f, tc, state_key="C", has_R=False)

    f.write("#endif /* REFERENCE_DATA_H_ */\n")


def to_list(tensor):
    """Convert tensor to a flat Python list of floats."""
    return [round(float(v), 8) for v in tensor.flatten().tolist()]


def generate_json(path):
    """Generate reference_data.json with the same values as the C header."""
    slstm_cases, mlstm_cases = build_cases()
    data = {"slstm": {}, "mlstm": {}}

    for i, tc in enumerate(slstm_cases, start=1):
        entry = {
            "B": tc["B"], "T": tc["T"], "I": tc["I"], "H": tc["H"],
            "W": to_list(tc["W"]), "R": to_list(tc["R"]), "b": to_list(tc["b"]),
            "input": to_list(tc["input"]),
            "expected_y": to_list(tc["y"]), "expected_c": to_list(tc["c"]),
            "expected_n": to_list(tc["n"]), "expected_m": to_list(tc["m"]),
        }
        if tc["output"] is not None:
            entry["expected_output"] = to_list(tc["output"])
        data["slstm"][f"test{i}"] = entry

    for i, tc in enumerate(mlstm_cases, start=1):
        entry = {
            "B": tc["B"], "T": tc["T"], "I": tc["I"], "H": tc["H"],
            "W": to_list(tc["W"]), "b": to_list(tc["b"]),
            "input": to_list(tc["input"]),
            "expected_y": to_list(tc["y"]), "expected_C": to_list(tc["c"]),
            "expected_n": to_list(tc["n"]), "expected_m": to_list(tc["m"]),
        }
        if tc["output"] is not None:
            entry["expected_output"] = to_list(tc["output"])
        data["mlstm"][f"test{i}"] = entry

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {path}")


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "reference_data.h")
    with open(out_path, "w") as f:
        generate(f)
    print(f"Wrote {out_path}")

    json_path = os.path.join(os.path.dirname(__file__), "reference_data.json")
    generate_json(json_path)
