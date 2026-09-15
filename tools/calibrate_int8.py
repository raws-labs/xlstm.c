#!/usr/bin/env python3
"""Worked example: float tensors -> the INT8 scales the kernels expect.

    python3 tools/calibrate_int8.py              # self-check (stdlib only)
    python3 tools/calibrate_int8.py my_case.json # calibrate your own tensors

Not a supported quantizer: a worked example that reproduces, from the float
tensors alone, every scale and every quantized weight recorded in
test/reference_data.json. That file is the contract - if this drifts from it,
this exits non-zero.

my_case.json is one flat object, all float lists:

    {"cell": "slstm",           (or "mlstm", which has no R)
     "W": [...], "R": [...], "b": [...],
     "x": [...],                 calibration inputs, as many as you have
     "y": [...],                 f32 outputs observed over that input
     "c": [...], "n": [...]}     f32 states observed over it ("C" for mLSTM)

The conventions, every one of which matters (xlstm_quant.h):

  weights      symmetric int8, scale = max|w| / 127, zero point 0
  activations  asymmetric int8, scale = range / 255, the range widened to
               include zero so that zero padding is representable
  c, n, C      symmetric INT16 with headroom - never the asymmetric int8
               calibration, which throws away about 7 bits and adds a zero
               point the kernels do not read
  m            stays float32: quantizing the log-space stabilizer costs the
               thing it exists for and buys nothing
  bias         int32 at W_scale * x_scale, the scale the accumulator is
               already in. Quantizing b on its own symmetric scale is the
               natural guess and is wrong.

Headroom is there because states are calibrated from a sample: an exit-state
snapshot can sit several times under the true mid-sequence peak, and a state
that clips is not recoverable. 4.0 is what the correctness gate uses.
"""

import json
import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(os.path.dirname(HERE), "test")
sys.path.insert(0, HERE)

from extract_heads import emit_array  # noqa: E402

STATE_HEADROOM = 4.0


def _check_headroom_matches_generated_data():
    """The same constant lives in four places: here, generate_reference.py's
    GENERATOR_HEADROOM, and the two INT8 test files, which are pinned to the
    generated header by static_assert. This copy was held only by whatever
    the self-check below happened to cover, so pin it explicitly against the
    value the generator emitted."""
    header = os.path.join(TEST, "reference_data.h")
    if not os.path.exists(header):
        return
    with open(header) as f:
        for line in f:
            if "XLSTM_GENERATOR_HEADROOM" in line and "define" in line:
                emitted = float(line.split()[-1].rstrip("f"))
                if emitted != STATE_HEADROOM:
                    raise SystemExit(
                        "STATE_HEADROOM is %g here and XLSTM_GENERATOR_HEADROOM "
                        "is %g in test/reference_data.h; every tol_s8 bound in "
                        "that file was derived assuming they agree"
                        % (STATE_HEADROOM, emitted))
                return


def f32(x):
    """x rounded to float32, which is the only width src/xlstm_quant.c has.

    Every scale below goes through this, and it is not cosmetic. A scale that
    is exact in float64 and rounded in float32 sends a value sitting on a .5
    boundary to a different integer, and one LSB there moves the whole
    trajectory. 1.0f/255.0f is the case that shows it: float32 rounds it up, so
    0.5f / scale is 127.49999237060547 and quantizes to 127, where float64 gives
    exactly 127.5 and quantizes to 128."""
    return struct.unpack("f", struct.pack("f", x))[0]


def roundf(x):
    """C roundf: half away from zero. Python's round() is half to even, and
    the two disagree on exactly the values a quantizer lands on."""
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def quant_symmetric(data):
    """Weights: xlstm_quant_symmetric."""
    m = f32(max(abs(f32(v)) for v in data))
    return (f32(m / 127.0) if m > 0 else 1.0), 0


def quant_asymmetric(data):
    """Activations: xlstm_quant_asymmetric. The range is stretched to include
    zero, so a zero-padded input quantizes to the zero point exactly."""
    lo = f32(min(min(f32(v) for v in data), 0.0))
    hi = f32(max(max(f32(v) for v in data), 0.0))
    rng = f32(hi - lo)
    if rng < f32(1e-10):
        return f32(1.0 / 255.0), 0
    scale = f32(rng / 255.0)
    zp = roundf(f32(-128.0 - f32(lo / scale)))
    return scale, max(-128, min(127, int(zp)))


def quant_symmetric_s16(data, headroom=STATE_HEADROOM):
    """States: xlstm_quant_symmetric_s16. Strictly symmetric - no zero point
    appears anywhere in slstm_s8.c or mlstm_s8.c."""
    m = f32(max(abs(f32(v)) for v in data))
    return (f32(f32(m * headroom) / 32767.0) if m > 0 else 1.0), 0


def quantize_s8(data, scale, zero_point=0):
    """xlstm_quantize_f32_to_s8: float32 divide, roundf, add the zero point,
    clamp. Quantize once, here, and a kernel quantizing at runtime lands on the
    same integers."""
    return [max(-128, min(127, int(roundf(f32(f32(v) / f32(scale)))) + zero_point))
            for v in data]


def quantize_bias(b, w_scale, x_scale):
    """int32, at the accumulator's own scale, and deliberately not clipped:
    a bias that does not fit int32 is a broken calibration, not a saturation."""
    scale = f32(f32(w_scale) * f32(x_scale))
    return [int(roundf(f32(f32(v) / scale))) for v in b]


def calibrate(cell, t, headroom=STATE_HEADROOM):
    """The scales and zero points for one head, from float tensors."""
    w_scale, _ = quant_symmetric(t["W"])
    x_scale, x_zp = quant_asymmetric(t["x"])
    y_scale, y_zp = quant_asymmetric(t["y"])
    out = {"W_scale": w_scale, "x_scale": x_scale, "x_zero_point": x_zp,
           "y_scale": y_scale, "y_zero_point": y_zp,
           "b_scale": f32(w_scale * x_scale),
           "n_scale": quant_symmetric_s16(t["n"], headroom)[0]}
    if cell == "slstm":
        out["R_scale"] = quant_symmetric(t["R"])[0]
        out["c_scale"] = quant_symmetric_s16(t["c"], headroom)[0]
    else:
        out["C_scale"] = quant_symmetric_s16(t["C"], headroom)[0]
    return out


def emit_params(cell, cal):
    """The C initializer for SlstmS8Params / MlstmS8Params."""
    q = lambda s, z: "{%.9gf, %d}" % (s, z)  # noqa: E731
    state = "c" if cell == "slstm" else "C"
    lines = ["    .cell_clip = 0.0f,",
             "    .W_scale = %.9gf," % cal["W_scale"]]
    if cell == "slstm":
        lines.append("    .R_scale = %.9gf," % cal["R_scale"])
    lines += ["    .x_quant = %s," % q(cal["x_scale"], cal["x_zero_point"]),
              "    .y_quant = %s," % q(cal["y_scale"], cal["y_zero_point"]),
              "    .%s_quant = %s," % (state, q(cal[state + "_scale"], 0)),
              "    .n_quant = %s," % q(cal["n_scale"], 0)]
    return "static const %sS8Params params = {\n%s\n};\n" % (
        cell.capitalize(), "\n".join(lines))


# ---------------------------------------------------------------------------
# Self-check. Every case in test/reference_data.json carries the float tensors
# AND the scales and integers the kernel was verified against, so the whole
# calibration can be re-derived and compared.
#
# That file stores round(v, 8), so a re-derived scale differs from the
# recorded one by the rounding of the tensor's largest element. The tolerances
# below are exactly that, pushed through each formula - EPS through max/127,
# range/255, max*headroom/32767 - not a fitted number. They already cover the
# float32 rounding f32() applies on top (138 of the 142 scales come back
# bit-identical, and the worst of the other four sits at 0.75 of its bound), so
# no float32 term is added to them. Zero points and every quantized integer
# must match exactly.
# ---------------------------------------------------------------------------

EPS = 5e-9


def _tensors(cell, tc):
    """One JSON case as calibrate() wants it. y is calibrated on the golden
    output over every timestep, falling back to the exit y for a T=1 case
    that stores no output tensor."""
    t = {"W": tc["W"], "b": tc["b"], "x": tc["input"],
         "y": tc.get("expected_output", tc["expected_y"]),
         "n": tc["expected_n"]}
    if cell == "slstm":
        t["R"], t["c"] = tc["R"], tc["expected_c"]
    else:
        t["C"] = tc["expected_C"]
    return t


def _check_case(cell, name, tc):
    ref, t = tc["s8"], _tensors(cell, tc)
    got = calibrate(cell, t)
    tol = {"W_scale": EPS / 127.0, "R_scale": EPS / 127.0,
           "x_scale": 2 * EPS / 255.0, "y_scale": 2 * EPS / 255.0,
           "c_scale": EPS * STATE_HEADROOM / 32767.0,
           "C_scale": EPS * STATE_HEADROOM / 32767.0,
           "n_scale": EPS * STATE_HEADROOM / 32767.0}
    tol["b_scale"] = (tol["W_scale"] * got["x_scale"]
                      + got["W_scale"] * tol["x_scale"])
    bad = []
    for key, value in sorted(got.items()):
        if key.endswith("zero_point"):
            if value != ref[key]:
                bad.append("%s: %d, recorded %d" % (key, value, ref[key]))
        elif abs(value - ref[key]) > tol[key]:
            bad.append("%s: %.17g, recorded %.17g (bound %.3g)"
                       % (key, value, ref[key], tol[key]))

    ints = [("W_q", quantize_s8(t["W"], got["W_scale"])),
            ("x_q", quantize_s8(t["x"], got["x_scale"], got["x_zero_point"])),
            ("b_q", quantize_bias(t["b"], got["W_scale"], got["x_scale"]))]
    if cell == "slstm":
        ints.append(("R_q", quantize_s8(t["R"], got["R_scale"])))
    n_ints = 0
    for key, value in ints:
        n_ints += len(value)
        if value != ref[key]:
            wrong = sum(1 for a, b in zip(value, ref[key]) if a != b)
            bad.append("%s: %d of %d integers differ" % (key, wrong, len(value)))
    for line in bad:
        print("FAIL %s %s: %s" % (cell, name, line))
    return len(bad), n_ints


def _negative_controls(tc):
    """Three calibrations that are wrong in the ways the header warns about.
    Each must be caught, or the comparison above is not measuring anything."""
    t = _tensors("slstm", tc)
    ref = tc["s8"]
    checks = [
        ("c calibrated asymmetrically, as if it were an int8 activation",
         quant_asymmetric(t["c"])[0] != ref["c_scale"]),
        ("c calibrated with no state headroom",
         quant_symmetric_s16(t["c"], 1.0)[0] != ref["c_scale"]),
        ("b quantized on its own symmetric scale instead of W_scale*x_scale",
         quantize_s8(t["b"], quant_symmetric(t["b"])[0]) != ref["b_q"]),
    ]
    for label, caught in checks:
        if not caught:
            print("FAIL: %s is NOT caught, so the check above is blind" % label)
            return 1
        print("  caught: %s" % label)
    # Stated in the module docstring and nowhere a user of the CLI would meet
    # it, which is where someone reaching for a quantizer meets it.
    print("\n  scope: min/max calibration only - no percentile, entropy or "
          "MSE-optimal\n"
          "  mode, one scale per tensor rather than per channel, and the "
          "state headroom\n"
          "  is the constant above rather than a flag.")
    return 0


def self_check():
    _check_headroom_matches_generated_data()
    with open(os.path.join(TEST, "reference_data.json")) as f:
        data = json.load(f)
    fails = n_cases = n_ints = 0
    for cell in ("slstm", "mlstm"):
        for name, tc in sorted(data[cell].items()):
            bad, ints = _check_case(cell, name, tc)
            fails += bad
            n_ints += ints
            n_cases += 1
    print("reproduced %d calibrations and %d quantized integers from "
          "test/reference_data.json" % (n_cases, n_ints))
    if fails:
        return 1
    return _negative_controls(data["slstm"]["test5"])


def main(argv):
    rc = self_check()
    if rc or not argv:
        return rc
    with open(argv[0]) as f:
        t = json.load(f)
    cell = t.get("cell", "slstm")
    cal = calibrate(cell, t)
    print("\n/* INT8 calibration of %s, cell = %s */" % (argv[0], cell))
    print(emit_params(cell, cal))
    print(emit_array("int8_t", "W_q", quantize_s8(t["W"], cal["W_scale"]), "%d"))
    if cell == "slstm":
        print(emit_array("int8_t", "R_q",
                         quantize_s8(t["R"], cal["R_scale"]), "%d"))
    print(emit_array("int32_t", "b_q",
                     quantize_bias(t["b"], cal["W_scale"], cal["x_scale"]), "%d"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
