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
    if all_ok:
        print("All tests passed.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
