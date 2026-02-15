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


def generate(f):
    """Generate all reference data into file handle f."""
    f.write(
        "/* Auto-generated — do not edit.\n"
        " * Source: NX-AI/xlstm reference (vanilla backend)\n"
        " * Regenerate: make reference\n"
        " */\n\n"
        "#ifndef REFERENCE_DATA_H_\n"
        "#define REFERENCE_DATA_H_\n\n"
    )

    # ========================================================================
    # sLSTM reference data
    # ========================================================================
    f.write("// " + "=" * 72 + "\n")
    f.write("// sLSTM reference data\n")
    f.write("// " + "=" * 72 + "\n\n")

    # --- sLSTM Test 1: Single timestep, zero initial state (B=1, T=1, I=2, H=2)
    torch.manual_seed(42)
    W = torch.randn(8, 2) * 0.5
    R = torch.randn(8, 2) * 0.5
    b = torch.zeros(8)
    x1 = torch.tensor([[[1.0, 0.5]]])

    _, y, c, n, m = run_slstm(W, R, b, x1)

    f.write("// Test 1: Single timestep, zero initial state\n")
    f.write("// B=1, T=1, I=2, H=2\n")
    f.write(f"const float kTest1_W[] = {{{fmt(W)}}};\n")
    f.write(f"const float kTest1_R[] = {{{fmt(R)}}};\n")
    f.write(f"const float kTest1_b[] = {{{fmt(b)}}};\n")
    f.write(f"const float kTest1_input[] = {{{fmt(x1)}}};\n")
    f.write(f"const float kTest1_expected_y[] = {{{fmt(y)}}};\n")
    f.write(f"const float kTest1_expected_c[] = {{{fmt(c)}}};\n")
    f.write(f"const float kTest1_expected_n[] = {{{fmt(n)}}};\n")
    f.write(f"const float kTest1_expected_m[] = {{{fmt(m)}}};\n\n")

    # --- sLSTM Test 2: 3 timesteps, state propagation (B=1, T=3, I=2, H=2)
    x2 = torch.tensor([[[1.0, 0.5], [0.3, -0.2], [-0.5, 1.0]]])
    output, y, c, n, m = run_slstm(W, R, b, x2)

    f.write("// Test 2: 3 timesteps, state propagation (B=1, T=3, I=2, H=2)\n")
    f.write(f"const float kTest2_input[] = {{{fmt(x2)}}};\n")
    f.write(f"const float kTest2_expected_y[] = {{{fmt(y)}}};\n")
    f.write(f"const float kTest2_expected_c[] = {{{fmt(c)}}};\n")
    f.write(f"const float kTest2_expected_n[] = {{{fmt(n)}}};\n")
    f.write(f"const float kTest2_expected_m[] = {{{fmt(m)}}};\n")
    f.write(f"const float kTest2_expected_output[] = {{{fmt(output)}}};\n\n")

    # --- sLSTM Test 3: Large inputs, overflow prevention (B=1, T=1, I=2, H=2)
    W3 = torch.tensor([
        [5.0, 5.0], [5.0, 5.0],   # i gates
        [5.0, 5.0], [5.0, 5.0],   # f gates
        [0.5, 0.5], [0.5, 0.5],   # z gates
        [0.5, 0.5], [0.5, 0.5],   # o gates
    ])
    R3 = torch.zeros(8, 2)
    b3 = torch.zeros(8)
    x3 = torch.tensor([[[10.0, 10.0]]])

    _, y, c, n, m = run_slstm(W3, R3, b3, x3)

    f.write("// Test 3: Large inputs, overflow prevention\n")
    f.write("// B=1, T=1, I=2, H=2\n")
    f.write("// i_raw = 100 — would overflow without m-stabilizer\n")
    f.write(f"const float kTest3_W[] = {{{fmt(W3)}}};\n")
    f.write(f"const float kTest3_R[] = {{{fmt(R3)}}};\n")
    f.write(f"const float kTest3_b[] = {{{fmt(b3)}}};\n")
    f.write(f"const float kTest3_input[] = {{{fmt(x3)}}};\n")
    f.write(f"const float kTest3_expected_y[] = {{{fmt(y)}}};\n")
    f.write(f"const float kTest3_expected_c[] = {{{fmt(c)}}};\n")
    f.write(f"const float kTest3_expected_n[] = {{{fmt(n)}}};\n")
    f.write(f"const float kTest3_expected_m[] = {{{fmt(m)}}};\n\n")

    # ========================================================================
    # mLSTM reference data
    # ========================================================================
    f.write("// " + "=" * 72 + "\n")
    f.write("// mLSTM reference data\n")
    f.write("// " + "=" * 72 + "\n\n")

    # --- mLSTM Test 1: Single timestep, zero initial state (B=1, T=1, I=3, H=2)
    torch.manual_seed(123)
    # W: [(4*H+2), I] = [10, 3]
    mW = torch.randn(10, 3) * 0.5
    mb = torch.zeros(10)
    mx1 = torch.tensor([[[1.0, 0.5, -0.3]]])

    output, y, C, n, m = run_mlstm(mW, mb, mx1)

    f.write("// mLSTM Test 1: Single timestep, zero initial state\n")
    f.write("// B=1, T=1, I=3, H=2\n")
    f.write(f"const float kMTest1_W[] = {{{fmt(mW)}}};\n")
    f.write(f"const float kMTest1_b[] = {{{fmt(mb)}}};\n")
    f.write(f"const float kMTest1_input[] = {{{fmt(mx1)}}};\n")
    f.write(f"const float kMTest1_expected_y[] = {{{fmt(y)}}};\n")
    f.write(f"const float kMTest1_expected_C[] = {{{fmt(C)}}};\n")
    f.write(f"const float kMTest1_expected_n[] = {{{fmt(n)}}};\n")
    f.write(f"const float kMTest1_expected_m[] = {{{fmt(m)}}};\n\n")

    # --- mLSTM Test 2: 3 timesteps, state propagation (B=1, T=3, I=3, H=2)
    mx2 = torch.tensor([[[1.0, 0.5, -0.3], [0.3, -0.2, 0.8], [-0.5, 1.0, 0.1]]])
    output, y, C, n, m = run_mlstm(mW, mb, mx2)

    f.write("// mLSTM Test 2: 3 timesteps, state propagation (B=1, T=3, I=3, H=2)\n")
    f.write(f"const float kMTest2_input[] = {{{fmt(mx2)}}};\n")
    f.write(f"const float kMTest2_expected_y[] = {{{fmt(y)}}};\n")
    f.write(f"const float kMTest2_expected_C[] = {{{fmt(C)}}};\n")
    f.write(f"const float kMTest2_expected_n[] = {{{fmt(n)}}};\n")
    f.write(f"const float kMTest2_expected_m[] = {{{fmt(m)}}};\n")
    f.write(f"const float kMTest2_expected_output[] = {{{fmt(output)}}};\n\n")

    # --- mLSTM Test 3: Large values, overflow prevention (B=1, T=1, I=3, H=2)
    # W: [10, 3] — large weights to produce large gate pre-activations
    mW3 = torch.tensor([
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],   # W_q
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],   # W_k
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],   # W_v
        [5.0, 5.0, 5.0],                      # w_i (scalar)
        [5.0, 5.0, 5.0],                      # w_f (scalar)
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],   # W_o
    ])
    mb3 = torch.zeros(10)
    mx3 = torch.tensor([[[10.0, 10.0, 10.0]]])

    output, y, C, n, m = run_mlstm(mW3, mb3, mx3)

    f.write("// mLSTM Test 3: Large values, overflow prevention\n")
    f.write("// B=1, T=1, I=3, H=2\n")
    f.write("// i_raw = 150 — would overflow without m-stabilizer\n")
    f.write(f"const float kMTest3_W[] = {{{fmt(mW3)}}};\n")
    f.write(f"const float kMTest3_b[] = {{{fmt(mb3)}}};\n")
    f.write(f"const float kMTest3_input[] = {{{fmt(mx3)}}};\n")
    f.write(f"const float kMTest3_expected_y[] = {{{fmt(y)}}};\n")
    f.write(f"const float kMTest3_expected_C[] = {{{fmt(C)}}};\n")
    f.write(f"const float kMTest3_expected_n[] = {{{fmt(n)}}};\n")
    f.write(f"const float kMTest3_expected_m[] = {{{fmt(m)}}};\n\n")

    f.write("#endif /* REFERENCE_DATA_H_ */\n")


def to_list(tensor):
    """Convert tensor to a flat Python list of floats."""
    return [round(float(v), 8) for v in tensor.flatten().tolist()]


def generate_json(path):
    """Generate reference_data.json with the same values as the C header."""
    data = {"slstm": {}, "mlstm": {}}

    # --- sLSTM Test 1 ---
    torch.manual_seed(42)
    W = torch.randn(8, 2) * 0.5
    R = torch.randn(8, 2) * 0.5
    b = torch.zeros(8)
    x1 = torch.tensor([[[1.0, 0.5]]])

    _, y, c, n, m = run_slstm(W, R, b, x1)

    data["slstm"]["test1"] = {
        "B": 1, "T": 1, "I": 2, "H": 2,
        "W": to_list(W), "R": to_list(R), "b": to_list(b),
        "input": to_list(x1),
        "expected_y": to_list(y), "expected_c": to_list(c),
        "expected_n": to_list(n), "expected_m": to_list(m),
    }

    # --- sLSTM Test 2 ---
    x2 = torch.tensor([[[1.0, 0.5], [0.3, -0.2], [-0.5, 1.0]]])
    output, y, c, n, m = run_slstm(W, R, b, x2)

    data["slstm"]["test2"] = {
        "B": 1, "T": 3, "I": 2, "H": 2,
        "W": to_list(W), "R": to_list(R), "b": to_list(b),
        "input": to_list(x2),
        "expected_y": to_list(y), "expected_c": to_list(c),
        "expected_n": to_list(n), "expected_m": to_list(m),
        "expected_output": to_list(output),
    }

    # --- sLSTM Test 3 ---
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

    data["slstm"]["test3"] = {
        "B": 1, "T": 1, "I": 2, "H": 2,
        "W": to_list(W3), "R": to_list(R3), "b": to_list(b3),
        "input": to_list(x3),
        "expected_y": to_list(y), "expected_c": to_list(c),
        "expected_n": to_list(n), "expected_m": to_list(m),
    }

    # --- mLSTM Test 1 ---
    torch.manual_seed(123)
    mW = torch.randn(10, 3) * 0.5
    mb = torch.zeros(10)
    mx1 = torch.tensor([[[1.0, 0.5, -0.3]]])

    output, y, C, n, m = run_mlstm(mW, mb, mx1)

    data["mlstm"]["test1"] = {
        "B": 1, "T": 1, "I": 3, "H": 2,
        "W": to_list(mW), "b": to_list(mb),
        "input": to_list(mx1),
        "expected_y": to_list(y), "expected_C": to_list(C),
        "expected_n": to_list(n), "expected_m": to_list(m),
    }

    # --- mLSTM Test 2 ---
    mx2 = torch.tensor([[[1.0, 0.5, -0.3], [0.3, -0.2, 0.8], [-0.5, 1.0, 0.1]]])
    output, y, C, n, m = run_mlstm(mW, mb, mx2)

    data["mlstm"]["test2"] = {
        "B": 1, "T": 3, "I": 3, "H": 2,
        "W": to_list(mW), "b": to_list(mb),
        "input": to_list(mx2),
        "expected_y": to_list(y), "expected_C": to_list(C),
        "expected_n": to_list(n), "expected_m": to_list(m),
        "expected_output": to_list(output),
    }

    # --- mLSTM Test 3 ---
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

    output, y, C, n, m = run_mlstm(mW3, mb3, mx3)

    data["mlstm"]["test3"] = {
        "B": 1, "T": 1, "I": 3, "H": 2,
        "W": to_list(mW3), "b": to_list(mb3),
        "input": to_list(mx3),
        "expected_y": to_list(y), "expected_C": to_list(C),
        "expected_n": to_list(n), "expected_m": to_list(m),
    }

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
