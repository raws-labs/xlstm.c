#!/usr/bin/env python3
"""microTVM integration test for sLSTM + mLSTM packed functions.

Loads the shared library via tvm.runtime.load_module(), calls registered
packed functions with real tvm.nd.array (DLTensor) objects, and validates
outputs against reference data.
"""

import ctypes
import json
import os
import sys

import numpy as np
import tvm
from tvm import nd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..")
REF_PATH = os.path.join(ROOT_DIR, "test", "reference_data.json")
LIB_PATH = os.path.join(ROOT_DIR, "libxlstm_tvm.so")

ATOL = 1e-5


def test_slstm(name, tc):
    """Run one sLSTM test case via TVM packed function."""
    B, T, I, H = tc["B"], tc["T"], tc["I"], tc["H"]

    x = nd.array(np.array(tc["input"], dtype=np.float32).reshape(B, T, I))
    W = nd.array(np.array(tc["W"], dtype=np.float32).reshape(4*H, I))
    R = nd.array(np.array(tc["R"], dtype=np.float32).reshape(4*H, H))
    b = nd.array(np.array(tc["b"], dtype=np.float32).reshape(4*H))
    y = nd.array(np.zeros((B, H), dtype=np.float32))
    c = nd.array(np.zeros((B, H), dtype=np.float32))
    n = nd.array(np.zeros((B, H), dtype=np.float32))
    m = nd.array(np.zeros((B, H), dtype=np.float32))
    output = nd.array(np.zeros((B, T, H), dtype=np.float32))

    f = tvm.get_global_func("xlstm.slstm_eval")
    f(x, W, R, b, y, c, n, m, output)

    expected_y = np.array(tc["expected_y"], dtype=np.float32)
    expected_c = np.array(tc["expected_c"], dtype=np.float32)
    expected_n = np.array(tc["expected_n"], dtype=np.float32)
    expected_m = np.array(tc["expected_m"], dtype=np.float32)

    ok = True
    for label, got, want in [
        ("y", y.numpy().flatten(), expected_y),
        ("c", c.numpy().flatten(), expected_c),
        ("n", n.numpy().flatten(), expected_n),
        ("m", m.numpy().flatten(), expected_m),
    ]:
        if not np.allclose(got, want, atol=ATOL):
            print(f"  FAIL {label}: got {got}, expected {want}")
            ok = False

    if "expected_output" in tc:
        expected_out = np.array(tc["expected_output"], dtype=np.float32)
        if not np.allclose(output.numpy().flatten(), expected_out, atol=ATOL):
            print(f"  FAIL output: got {output.numpy().flatten()}, expected {expected_out}")
            ok = False

    status = "OK" if ok else "FAILED"
    print(f"[{status}] sLSTM {name}")
    return ok


def test_mlstm(name, tc):
    """Run one mLSTM test case via TVM packed function."""
    B, T, I, H = tc["B"], tc["T"], tc["I"], tc["H"]

    x = nd.array(np.array(tc["input"], dtype=np.float32).reshape(B, T, I))
    W = nd.array(np.array(tc["W"], dtype=np.float32).reshape(4*H+2, I))
    b = nd.array(np.array(tc["b"], dtype=np.float32).reshape(4*H+2))
    y = nd.array(np.zeros((B, H), dtype=np.float32))
    C = nd.array(np.zeros((B, H*H), dtype=np.float32))
    n = nd.array(np.zeros((B, H), dtype=np.float32))
    m = nd.array(np.zeros((B, 1), dtype=np.float32))
    output = nd.array(np.zeros((B, T, H), dtype=np.float32))

    f = tvm.get_global_func("xlstm.mlstm_eval")
    f(x, W, b, y, C, n, m, output)

    expected_y = np.array(tc["expected_y"], dtype=np.float32)
    expected_C = np.array(tc["expected_C"], dtype=np.float32)
    expected_n = np.array(tc["expected_n"], dtype=np.float32)
    expected_m = np.array(tc["expected_m"], dtype=np.float32)

    ok = True
    for label, got, want in [
        ("y", y.numpy().flatten(), expected_y),
        ("C", C.numpy().flatten(), expected_C),
        ("n", n.numpy().flatten(), expected_n),
        ("m", m.numpy().flatten(), expected_m),
    ]:
        if not np.allclose(got, want, atol=ATOL):
            print(f"  FAIL {label}: got {got}, expected {want}")
            ok = False

    if "expected_output" in tc:
        expected_out = np.array(tc["expected_output"], dtype=np.float32)
        if not np.allclose(output.numpy().flatten(), expected_out, atol=ATOL):
            print(f"  FAIL output: got {output.numpy().flatten()}, expected {expected_out}")
            ok = False

    status = "OK" if ok else "FAILED"
    print(f"[{status}] mLSTM {name}")
    return ok


# ---------------------------------------------------------------------------
# INT8 packed functions
#
# Same registered names as the float path: the adapter dispatches on the
# input DLTensor's dtype and reads the quantization from the extra scalar
# args that follow the tensors.
#
# The assertions are exact. reference_data.json's "s8" block holds both the
# quantized inputs and the integers the kernel is expected to produce,
# taken from the numpy replica that derives every INT8 tolerance in the C++
# suite - and that replica reproduces the C kernels' integers bit for bit
# on every case. A scale wired to the wrong parameter, a state tensor not
# written back, or a change in the gate math all move an integer and fail.
# ---------------------------------------------------------------------------


def _check_exact(pairs):
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


def check_dequantized_vs_f32(tc, s, output, H):
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


def test_slstm_s8(name, tc):
    """Run one sLSTM case through the INT8 packed function."""
    B, T, I, H = tc["B"], tc["T"], tc["I"], tc["H"]
    s = tc["s8"]

    x = nd.array(np.array(s["x_q"], dtype=np.int8).reshape(B, T, I))
    W = nd.array(np.array(s["W_q"], dtype=np.int8).reshape(4*H, I))
    R = nd.array(np.array(s["R_q"], dtype=np.int8).reshape(4*H, H))
    b = nd.array(np.array(s["b_q"], dtype=np.int32).reshape(4*H))
    y = nd.array(np.zeros((B, H), dtype=np.int8))
    c = nd.array(np.zeros((B, H), dtype=np.int16))
    n = nd.array(np.zeros((B, H), dtype=np.int16))
    m = nd.array(np.zeros((B, H), dtype=np.float32))
    output = nd.array(np.zeros((B, T, H), dtype=np.int8))

    f = tvm.get_global_func("xlstm.slstm_eval")
    f(x, W, R, b, y, c, n, m, output,
      float(s["x_scale"]), int(s["x_zero_point"]),
      float(s["W_scale"]), float(s["R_scale"]),
      float(s["y_scale"]), int(s["y_zero_point"]),
      float(s["c_scale"]), float(s["n_scale"]))

    ok = _check_exact([
        ("output", output.numpy(), s["expected_output_q"]),
        ("y", y.numpy(), s["expected_y_q"]),
        ("c", c.numpy(), s["expected_c_q"]),
        ("n", n.numpy(), s["expected_n_q"]),
    ])
    # m is the log-space stabilizer: float32 even on the INT8 path.
    if not np.allclose(m.numpy().flatten(),
                       np.array(s["expected_m"], dtype=np.float32), atol=1e-5):
        print(f"  FAIL m: got {m.numpy().flatten()}, expected {s['expected_m']}")
        ok = False
    ok &= check_dequantized_vs_f32(tc, s, output.numpy(), H)

    print(f"[{'OK' if ok else 'FAILED'}] sLSTM INT8 {name}")
    return ok


def test_mlstm_s8(name, tc):
    """Run one mLSTM case through the INT8 packed function."""
    B, T, I, H = tc["B"], tc["T"], tc["I"], tc["H"]
    s = tc["s8"]

    x = nd.array(np.array(s["x_q"], dtype=np.int8).reshape(B, T, I))
    W = nd.array(np.array(s["W_q"], dtype=np.int8).reshape(4*H+2, I))
    b = nd.array(np.array(s["b_q"], dtype=np.int32).reshape(4*H+2))
    y = nd.array(np.zeros((B, H), dtype=np.int8))
    C = nd.array(np.zeros((B, H*H), dtype=np.int16))
    n = nd.array(np.zeros((B, H), dtype=np.int16))
    m = nd.array(np.zeros((B, 1), dtype=np.float32))
    output = nd.array(np.zeros((B, T, H), dtype=np.int8))

    f = tvm.get_global_func("xlstm.mlstm_eval")
    f(x, W, b, y, C, n, m, output,
      float(s["x_scale"]), int(s["x_zero_point"]), float(s["W_scale"]),
      float(s["y_scale"]), int(s["y_zero_point"]),
      float(s["C_scale"]), float(s["n_scale"]))

    ok = _check_exact([
        ("output", output.numpy(), s["expected_output_q"]),
        ("y", y.numpy(), s["expected_y_q"]),
        ("C", C.numpy(), s["expected_C_q"]),
        ("n", n.numpy(), s["expected_n_q"]),
    ])
    if not np.allclose(m.numpy().flatten(),
                       np.array(s["expected_m"], dtype=np.float32), atol=1e-5):
        print(f"  FAIL m: got {m.numpy().flatten()}, expected {s['expected_m']}")
        ok = False
    ok &= check_dequantized_vs_f32(tc, s, output.numpy(), H)

    print(f"[{'OK' if ok else 'FAILED'}] mLSTM INT8 {name}")
    return ok


def main():
    # Load via ctypes with RTLD_GLOBAL so TVM_REGISTER_GLOBAL static
    # initializers can find TVM runtime symbols already in the process.
    # tvm.runtime.load_module() uses RTLD_LOCAL which can cause issues.
    ctypes.CDLL(LIB_PATH, ctypes.RTLD_GLOBAL)

    with open(REF_PATH) as f:
        ref = json.load(f)

    print("=== microTVM integration tests ===\n", flush=True)

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
