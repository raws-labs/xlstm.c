#!/usr/bin/env python3
"""ONNX Runtime integration test for sLSTM + mLSTM custom ops.

Builds ONNX graphs with custom op nodes (com.raws.xlstm::SLSTM / MLSTM),
loads the shared library via register_custom_ops_library(), runs inference,
and validates outputs against reference data.
"""

import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..")
REF_PATH = os.path.join(ROOT_DIR, "test", "reference_data.json")
LIB_PATH = os.path.join(ROOT_DIR, "libxlstm_ort.so")

CUSTOM_DOMAIN = "com.raws.xlstm"
ATOL = 1e-5


def make_slstm_model(B, T, I, H):
    """Build an ONNX model with a single sLSTM custom op node.

    Inputs:  X[B,T,I], W[4H,I], R[4H,H], b[4H],
             y_init[B,H], c_init[B,H], n_init[B,H], m_init[B,H]
    Outputs: output[B,T,H], y[B,H], c[B,H], n[B,H], m[B,H]
    """
    inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [B, T, I]),
        helper.make_tensor_value_info("W", TensorProto.FLOAT, [4*H, I]),
        helper.make_tensor_value_info("R", TensorProto.FLOAT, [4*H, H]),
        helper.make_tensor_value_info("b", TensorProto.FLOAT, [4*H]),
        helper.make_tensor_value_info("y_init", TensorProto.FLOAT, [B, H]),
        helper.make_tensor_value_info("c_init", TensorProto.FLOAT, [B, H]),
        helper.make_tensor_value_info("n_init", TensorProto.FLOAT, [B, H]),
        helper.make_tensor_value_info("m_init", TensorProto.FLOAT, [B, H]),
    ]
    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [B, T, H]),
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [B, H]),
        helper.make_tensor_value_info("c", TensorProto.FLOAT, [B, H]),
        helper.make_tensor_value_info("n", TensorProto.FLOAT, [B, H]),
        helper.make_tensor_value_info("m", TensorProto.FLOAT, [B, H]),
    ]
    node = helper.make_node(
        "SLSTM",
        inputs=["X", "W", "R", "b", "y_init", "c_init", "n_init", "m_init"],
        outputs=["output", "y", "c", "n", "m"],
        domain=CUSTOM_DOMAIN,
    )
    graph = helper.make_graph([node], "slstm_test", inputs, outputs)
    opset = [
        helper.make_opsetid("", 17),
        helper.make_opsetid(CUSTOM_DOMAIN, 1),
    ]
    model = helper.make_model(graph, opset_imports=opset)
    return model


def make_mlstm_model(B, T, I, H):
    """Build an ONNX model with a single mLSTM custom op node.

    Inputs:  X[B,T,I], W[4H+2,I], b[4H+2],
             y_init[B,H], C_init[B,H*H], n_init[B,H], m_init[B,1]
    Outputs: output[B,T,H], y[B,H], C[B,H*H], n[B,H], m[B,1]
    """
    inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [B, T, I]),
        helper.make_tensor_value_info("W", TensorProto.FLOAT, [4*H+2, I]),
        helper.make_tensor_value_info("b", TensorProto.FLOAT, [4*H+2]),
        helper.make_tensor_value_info("y_init", TensorProto.FLOAT, [B, H]),
        helper.make_tensor_value_info("C_init", TensorProto.FLOAT, [B, H*H]),
        helper.make_tensor_value_info("n_init", TensorProto.FLOAT, [B, H]),
        helper.make_tensor_value_info("m_init", TensorProto.FLOAT, [B, 1]),
    ]
    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [B, T, H]),
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [B, H]),
        helper.make_tensor_value_info("C", TensorProto.FLOAT, [B, H*H]),
        helper.make_tensor_value_info("n", TensorProto.FLOAT, [B, H]),
        helper.make_tensor_value_info("m", TensorProto.FLOAT, [B, 1]),
    ]
    node = helper.make_node(
        "MLSTM",
        inputs=["X", "W", "b", "y_init", "C_init", "n_init", "m_init"],
        outputs=["output", "y", "C", "n", "m"],
        domain=CUSTOM_DOMAIN,
    )
    graph = helper.make_graph([node], "mlstm_test", inputs, outputs)
    opset = [
        helper.make_opsetid("", 17),
        helper.make_opsetid(CUSTOM_DOMAIN, 1),
    ]
    model = helper.make_model(graph, opset_imports=opset)
    return model


def run_ort_session(model, feeds):
    """Run an ORT inference session with the custom ops library."""
    model_bytes = model.SerializeToString()
    opts = ort.SessionOptions()
    opts.register_custom_ops_library(LIB_PATH)
    sess = ort.InferenceSession(model_bytes, opts, providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


# ---------------------------------------------------------------------------
# INT8 custom ops
#
# Scale and zero-point are scalar tensor inputs sitting next to the tensor
# they describe, the way ONNX's own QLinear* ops carry them.
#
# The assertions below are exact. reference_data.json's "s8" block holds
# both the quantized inputs and the integers the kernel is expected to
# produce, taken from the numpy replica that derives every INT8 tolerance
# in the C++ suite - and that replica reproduces the C kernels' integers
# bit for bit on every case. So there is nothing to tolerance here: a
# wrong scale wired to the wrong parameter, a state tensor not copied back,
# or a change in the gate math all move an integer and fail.
# ---------------------------------------------------------------------------

SLSTM_S8_INPUTS = [
    ("X", TensorProto.INT8), ("x_scale", TensorProto.FLOAT),
    ("x_zero_point", TensorProto.INT8),
    ("W", TensorProto.INT8), ("W_scale", TensorProto.FLOAT),
    ("R", TensorProto.INT8), ("R_scale", TensorProto.FLOAT),
    ("b", TensorProto.INT32),
    ("y_init", TensorProto.INT8), ("y_scale", TensorProto.FLOAT),
    ("y_zero_point", TensorProto.INT8),
    ("c_init", TensorProto.INT16), ("c_scale", TensorProto.FLOAT),
    ("n_init", TensorProto.INT16), ("n_scale", TensorProto.FLOAT),
    ("m_init", TensorProto.FLOAT),
]

MLSTM_S8_INPUTS = [
    ("X", TensorProto.INT8), ("x_scale", TensorProto.FLOAT),
    ("x_zero_point", TensorProto.INT8),
    ("W", TensorProto.INT8), ("W_scale", TensorProto.FLOAT),
    ("b", TensorProto.INT32),
    ("y_init", TensorProto.INT8), ("y_scale", TensorProto.FLOAT),
    ("y_zero_point", TensorProto.INT8),
    ("C_init", TensorProto.INT16), ("C_scale", TensorProto.FLOAT),
    ("n_init", TensorProto.INT16), ("n_scale", TensorProto.FLOAT),
    ("m_init", TensorProto.FLOAT),
]


def make_s8_model(op, in_specs, shapes, out_specs):
    """Build an ONNX model with a single quantized custom op node."""
    inputs = [helper.make_tensor_value_info(n, t, shapes[n])
              for n, t in in_specs]
    outputs = [helper.make_tensor_value_info(n, t, s) for n, t, s in out_specs]
    node = helper.make_node(
        op,
        inputs=[n for n, _ in in_specs],
        outputs=[n for n, _, _ in out_specs],
        domain=CUSTOM_DOMAIN,
    )
    graph = helper.make_graph([node], f"{op.lower()}_test", inputs, outputs)
    opset = [helper.make_opsetid("", 17), helper.make_opsetid(CUSTOM_DOMAIN, 1)]
    return helper.make_model(graph, opset_imports=opset)


def _check_exact(name, pairs):
    """Compare integer tensors elementwise; returns True if all match."""
    ok = True
    for label, got, want in pairs:
        got = np.asarray(got).flatten().astype(np.int64)
        want = np.asarray(want, dtype=np.int64)
        if not np.array_equal(got, want):
            bad = int(np.argmax(got != want))
            print(f"  FAIL {label}[{bad}]: got {got[bad]}, expected {want[bad]} "
                  f"({int((got != want).sum())} of {got.size} elements differ)")
            ok = False
    return ok


def test_slstm_s8(name, tc):
    """Run one sLSTM case through the INT8 custom op."""
    B, T, I, H = tc["B"], tc["T"], tc["I"], tc["H"]
    s = tc["s8"]

    shapes = {
        "X": [B, T, I], "W": [4*H, I], "R": [4*H, H], "b": [4*H],
        "y_init": [B, H], "c_init": [B, H], "n_init": [B, H], "m_init": [B, H],
        "x_scale": [1], "x_zero_point": [1], "W_scale": [1], "R_scale": [1],
        "y_scale": [1], "y_zero_point": [1], "c_scale": [1], "n_scale": [1],
    }
    outs = [("output", TensorProto.INT8, [B, T, H]),
            ("y", TensorProto.INT8, [B, H]),
            ("c", TensorProto.INT16, [B, H]),
            ("n", TensorProto.INT16, [B, H]),
            ("m", TensorProto.FLOAT, [B, H])]
    model = make_s8_model("SLSTM_S8", SLSTM_S8_INPUTS, shapes, outs)

    feeds = {
        "X": np.array(s["x_q"], dtype=np.int8).reshape(B, T, I),
        "W": np.array(s["W_q"], dtype=np.int8).reshape(4*H, I),
        "R": np.array(s["R_q"], dtype=np.int8).reshape(4*H, H),
        "b": np.array(s["b_q"], dtype=np.int32).reshape(4*H),
        "y_init": np.zeros((B, H), dtype=np.int8),
        "c_init": np.zeros((B, H), dtype=np.int16),
        "n_init": np.zeros((B, H), dtype=np.int16),
        "m_init": np.zeros((B, H), dtype=np.float32),
        "x_scale": np.array([s["x_scale"]], dtype=np.float32),
        "x_zero_point": np.array([s["x_zero_point"]], dtype=np.int8),
        "W_scale": np.array([s["W_scale"]], dtype=np.float32),
        "R_scale": np.array([s["R_scale"]], dtype=np.float32),
        "y_scale": np.array([s["y_scale"]], dtype=np.float32),
        "y_zero_point": np.array([s["y_zero_point"]], dtype=np.int8),
        "c_scale": np.array([s["c_scale"]], dtype=np.float32),
        "n_scale": np.array([s["n_scale"]], dtype=np.float32),
    }

    output, y, c, n, m = run_ort_session(model, feeds)

    ok = _check_exact(name, [
        ("output", output, s["expected_output_q"]),
        ("y", y, s["expected_y_q"]),
        ("c", c, s["expected_c_q"]),
        ("n", n, s["expected_n_q"]),
    ])
    # m is the log-space stabilizer: float32 even on the INT8 path.
    if not np.allclose(m.flatten(), np.array(s["expected_m"], dtype=np.float32), atol=1e-5):
        print(f"  FAIL m: got {m.flatten()}, expected {s['expected_m']}")
        ok = False
    ok &= check_dequantized_vs_f32(name, tc, s, output, B, T, H)

    print(f"[{'OK' if ok else 'FAILED'}] sLSTM INT8 {name}")
    return ok


def test_mlstm_s8(name, tc):
    """Run one mLSTM case through the INT8 custom op."""
    B, T, I, H = tc["B"], tc["T"], tc["I"], tc["H"]
    s = tc["s8"]

    shapes = {
        "X": [B, T, I], "W": [4*H+2, I], "b": [4*H+2],
        "y_init": [B, H], "C_init": [B, H*H], "n_init": [B, H], "m_init": [B, 1],
        "x_scale": [1], "x_zero_point": [1], "W_scale": [1],
        "y_scale": [1], "y_zero_point": [1], "C_scale": [1], "n_scale": [1],
    }
    outs = [("output", TensorProto.INT8, [B, T, H]),
            ("y", TensorProto.INT8, [B, H]),
            ("C", TensorProto.INT16, [B, H*H]),
            ("n", TensorProto.INT16, [B, H]),
            ("m", TensorProto.FLOAT, [B, 1])]
    model = make_s8_model("MLSTM_S8", MLSTM_S8_INPUTS, shapes, outs)

    feeds = {
        "X": np.array(s["x_q"], dtype=np.int8).reshape(B, T, I),
        "W": np.array(s["W_q"], dtype=np.int8).reshape(4*H+2, I),
        "b": np.array(s["b_q"], dtype=np.int32).reshape(4*H+2),
        "y_init": np.zeros((B, H), dtype=np.int8),
        "C_init": np.zeros((B, H*H), dtype=np.int16),
        "n_init": np.zeros((B, H), dtype=np.int16),
        "m_init": np.zeros((B, 1), dtype=np.float32),
        "x_scale": np.array([s["x_scale"]], dtype=np.float32),
        "x_zero_point": np.array([s["x_zero_point"]], dtype=np.int8),
        "W_scale": np.array([s["W_scale"]], dtype=np.float32),
        "y_scale": np.array([s["y_scale"]], dtype=np.float32),
        "y_zero_point": np.array([s["y_zero_point"]], dtype=np.int8),
        "C_scale": np.array([s["C_scale"]], dtype=np.float32),
        "n_scale": np.array([s["n_scale"]], dtype=np.float32),
    }

    output, y, C, n, m = run_ort_session(model, feeds)

    ok = _check_exact(name, [
        ("output", output, s["expected_output_q"]),
        ("y", y, s["expected_y_q"]),
        ("C", C, s["expected_C_q"]),
        ("n", n, s["expected_n_q"]),
    ])
    if not np.allclose(m.flatten(), np.array(s["expected_m"], dtype=np.float32), atol=1e-5):
        print(f"  FAIL m: got {m.flatten()}, expected {s['expected_m']}")
        ok = False
    ok &= check_dequantized_vs_f32(name, tc, s, output, B, T, H)

    print(f"[{'OK' if ok else 'FAILED'}] mLSTM INT8 {name}")
    return ok


def check_dequantized_vs_f32(name, tc, s, output, B, T, H):
    """The weaker but more meaningful claim: dequantized INT8 output stays
    within the calibrated per-channel bound of the f32 golden. Same bounds
    test/{slstm,mlstm}_s8_test.cc assert against."""
    golden = tc.get("expected_output") or tc["expected_y"]
    golden = np.array(golden, dtype=np.float32).reshape(-1, H)
    deq = (np.asarray(output).astype(np.float32).reshape(-1, H)
           - s["y_zero_point"]) * s["y_scale"]
    tol = np.array(s["tol_per_channel"], dtype=np.float32)
    bad = np.abs(deq - golden) > tol[None, :]
    if bad.any():
        t, j = np.argwhere(bad)[0]
        print(f"  FAIL dequantized[{t},{j}]: got {deq[t, j]:.6f}, "
              f"f32 golden {golden[t, j]:.6f}, bound {tol[j]:.6f}")
        return False
    return True


def test_slstm(name, tc):
    """Run one sLSTM test case."""
    B, T, I, H = tc["B"], tc["T"], tc["I"], tc["H"]
    model = make_slstm_model(B, T, I, H)

    feeds = {
        "X":      np.array(tc["input"], dtype=np.float32).reshape(B, T, I),
        "W":      np.array(tc["W"], dtype=np.float32).reshape(4*H, I),
        "R":      np.array(tc["R"], dtype=np.float32).reshape(4*H, H),
        "b":      np.array(tc["b"], dtype=np.float32).reshape(4*H),
        "y_init": np.zeros((B, H), dtype=np.float32),
        "c_init": np.zeros((B, H), dtype=np.float32),
        "n_init": np.zeros((B, H), dtype=np.float32),
        "m_init": np.zeros((B, H), dtype=np.float32),
    }

    output, y, c, n, m = run_ort_session(model, feeds)

    expected_y = np.array(tc["expected_y"], dtype=np.float32)
    expected_c = np.array(tc["expected_c"], dtype=np.float32)
    expected_n = np.array(tc["expected_n"], dtype=np.float32)
    expected_m = np.array(tc["expected_m"], dtype=np.float32)

    ok = True
    for label, got, want in [
        ("y", y.flatten(), expected_y),
        ("c", c.flatten(), expected_c),
        ("n", n.flatten(), expected_n),
        ("m", m.flatten(), expected_m),
    ]:
        if not np.allclose(got, want, atol=ATOL):
            print(f"  FAIL {label}: got {got}, expected {want}")
            ok = False

    if "expected_output" in tc:
        expected_out = np.array(tc["expected_output"], dtype=np.float32)
        if not np.allclose(output.flatten(), expected_out, atol=ATOL):
            print(f"  FAIL output: got {output.flatten()}, expected {expected_out}")
            ok = False

    status = "OK" if ok else "FAILED"
    print(f"[{status}] sLSTM {name}")
    return ok


def test_mlstm(name, tc):
    """Run one mLSTM test case."""
    B, T, I, H = tc["B"], tc["T"], tc["I"], tc["H"]
    model = make_mlstm_model(B, T, I, H)

    feeds = {
        "X":      np.array(tc["input"], dtype=np.float32).reshape(B, T, I),
        "W":      np.array(tc["W"], dtype=np.float32).reshape(4*H+2, I),
        "b":      np.array(tc["b"], dtype=np.float32).reshape(4*H+2),
        "y_init": np.zeros((B, H), dtype=np.float32),
        "C_init": np.zeros((B, H*H), dtype=np.float32),
        "n_init": np.zeros((B, H), dtype=np.float32),
        "m_init": np.zeros((B, 1), dtype=np.float32),
    }

    output, y, C, n, m = run_ort_session(model, feeds)

    expected_y = np.array(tc["expected_y"], dtype=np.float32)
    expected_C = np.array(tc["expected_C"], dtype=np.float32)
    expected_n = np.array(tc["expected_n"], dtype=np.float32)
    expected_m = np.array(tc["expected_m"], dtype=np.float32)

    ok = True
    for label, got, want in [
        ("y", y.flatten(), expected_y),
        ("C", C.flatten(), expected_C),
        ("n", n.flatten(), expected_n),
        ("m", m.flatten(), expected_m),
    ]:
        if not np.allclose(got, want, atol=ATOL):
            print(f"  FAIL {label}: got {got}, expected {want}")
            ok = False

    if "expected_output" in tc:
        expected_out = np.array(tc["expected_output"], dtype=np.float32)
        if not np.allclose(output.flatten(), expected_out, atol=ATOL):
            print(f"  FAIL output: got {output.flatten()}, expected {expected_out}")
            ok = False

    status = "OK" if ok else "FAILED"
    print(f"[{status}] mLSTM {name}")
    return ok


def main():
    with open(REF_PATH) as f:
        ref = json.load(f)

    print("=== ONNX Runtime integration tests ===\n")

    all_ok = True
    for name, tc in ref["slstm"].items():
        if not test_slstm(name, tc):
            all_ok = False

    for name, tc in ref["mlstm"].items():
        if not test_mlstm(name, tc):
            all_ok = False

    print()
    for name, tc in ref["slstm"].items():
        if not test_slstm_s8(name, tc):
            all_ok = False

    for name, tc in ref["mlstm"].items():
        if not test_mlstm_s8(name, tc):
            all_ok = False

    print()
    if all_ok:
        print("All tests passed.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
