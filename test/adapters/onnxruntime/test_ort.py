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
    if all_ok:
        print("All tests passed.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
