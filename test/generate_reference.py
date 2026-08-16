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
import numpy as np
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

def make_slstm_cell(hidden_size, num_heads=1):
    """Create a vanilla sLSTM cell (float32).

    hidden_size here is the reference's FUSED width, i.e. num_heads *
    head_dim. Every case but the head-composition case below uses
    num_heads=1, where fused width and this library's per-head hidden_size
    coincide. See build_head2_composition().
    """
    config = sLSTMCellConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
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


def run_slstm_multihead(W_fused, R_stack, b_fused, x_seq, num_heads):
    """Run a genuine multi-head sLSTM cell through the same vanilla path.

    This is the oracle the caller-side head composition is checked against.
    The layouts below are not the ones sLSTMCellBase declares: the vanilla
    cell rewrites both parameters through _recurrent_kernel_ext2int /
    _bias_ext2int at construction, and assigning `.data` (as run_slstm
    already does) bypasses the ParameterProxy, so what is written must
    already be in the INTERNAL layout. Established empirically, not read
    off the source - see test/derive_multihead_layout.py.

    Args:
        W_fused: [4*NH*DH, I] input projection, rows gate-major:
                 row (g*NH*DH + h*DH + j) is gate g, head h, unit j
        R_stack: [NH, 4*DH, DH] per-head recurrent matrices, each in
                 exactly this library's [4*DH, DH] packing (no cross-head
                 recurrence exists in the reference)
        b_fused: [4*NH*DH] bias, same gate-major row order as W_fused
        x_seq:   [B, T, I]

    Returns:
        output: [B, NH, T, DH] per-head outputs
        y, c, n, m: [B, NH*DH] final states, ordered [head][unit]
    """
    NH, DH = R_stack.shape[0], R_stack.shape[2]
    assert NH == num_heads, (
        f"R_stack has {NH} head slices but num_heads={num_heads}")
    H = NH * DH
    B = x_seq.shape[0]

    cell = make_slstm_cell(H, num_heads=num_heads)

    with torch.no_grad():
        cell._recurrent_kernel_.data = R_stack
        cell._bias_.data = b_fused

        Wx = torch.matmul(x_seq, W_fused.T)
        state = torch.zeros(4, B, H, dtype=torch.float32)
        output, final_state = cell(Wx, state=state)

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


# Hidden sizes for the correctness sweep. 17 is deliberately not a multiple
# of 4: it exercises SIMD loop-tail handling, which is where vectorised
# backends realistically break.
SWEEP_SIZES = [1, 8, 16, 17, 64]


# ============================================================================
# INT8 per-channel tol_s8 derivation
#
# Every INT8 test assertion (RunSlstmS8Case/RunMlstmS8Case in the two
# test/*_s8_test.cc files) checks one channel at a time against one bound,
# so the only invariant that matters is per-channel: a bound is sound iff
#
#     max(baseline_err, perturbed_err)  <  bound  <  own_range
#
# for that specific channel, where own_range is the largest golden value
# that channel ever takes (across y and every stored output timestep).
# A single case-wide bound cannot generally satisfy this for every channel
# at once - real cases have channels spanning multiple orders of magnitude
# - so every channel gets its own bound, computed
# here at generation time from a from-scratch numpy simulation of the
# exact INT8 kernel math (slstm_s8.c / mlstm_s8.c) and calibration
# (PrepareS8 / PrepareMlstmS8 in the two test/*_s8_test.cc files,
# including xlstm_quant_symmetric_s16's headroom=4.0 - kept in sync by hand since
# the calibration lives in C++, not here; see those files' comments).
#
# baseline_err is the whole-trajectory error against the plain kernel.
# perturbed_err is the same, but with a persistent 1.001x multiplicative
# perturbation applied to the pre-requantization y value at every
# timestep - simulating a backend whose activation functions are not
# bit-identical to this task's (e.g. CMSIS-NN's LUT-based sigmoid/tanh on
# Cortex-M). A bound that does not clear both is not portable to that
# class of backend, even though it looks fine against this one.
#
# Selection rule (deliberately maximizing, not minimizing): target a
# bound at HALF of a channel's own range (ratio ~0.5) - the invariant
# above only requires ratio < 1.0, and a bound picked just above the
# measured floor (the earlier, WRONG approach: ~1.5x the measured error)
# leaves near-zero margin for legitimate backend variance, which is
# exactly what a portability check needs room for. If the floor already
# exceeds 0.5x the range (a channel with unusually little headroom), fall
# back to the midpoint of (floor, range), which is the loosest bound that
# still respects both constraints. A future maintainer: do not re-tighten
# these bounds toward the measured error "to make the test stricter" -
# that reintroduces the false-failure risk this rule exists to avoid.
# ============================================================================

def _quant_sym8(a):
    m = float(np.max(np.abs(a)))
    return (m / 127.0 if m > 0 else 1.0), 0


def _quant_asym(a):
    mn = min(float(a.min()), 0.0)
    mx = max(float(a.max()), 0.0)
    rng = mx - mn
    if rng < 1e-10:
        return 1.0 / 255.0, 0
    scale = rng / 255.0
    zp = round(-128.0 - mn / scale)
    zp = max(-128, min(127, zp))
    return scale, zp


# Must match kStateHeadroom in both test/slstm_s8_test.cc and
# test/mlstm_s8_test.cc exactly - it is the headroom those files pass to
# xlstm_quant_symmetric_s16 to calibrate c_quant/n_quant/C_quant, and every
# tol_s8_per_channel bound in reference_data.h assumes that calibration. Emitted
# into reference_data.h as XLSTM_GENERATOR_HEADROOM so both C++ files can
# static_assert against it instead of the two copies silently drifting.
GENERATOR_HEADROOM = 4.0


def _quant_sym16_headroom(a, headroom=GENERATOR_HEADROOM):
    m = float(np.max(np.abs(a)))
    scale = (m * headroom / 32767.0) if m > 0 else 1.0
    return scale, 0


def _qs8(a, scale, zp):
    v = np.round(a / scale) + zp
    return np.clip(v, -128, 127).astype(np.int64)


def _t2n(t):
    return t.detach().cpu().numpy().astype(np.float64)


def _slstm_s8_calib(tc):
    """Scales the sLSTM INT8 path is calibrated with, for one case.

    This is the single definition of PrepareS8 (test/slstm_s8_test.cc) on
    the Python side: _slstm_int8_trace derives every tolerance in
    reference_data.h from it, and generate_json ships it so the framework
    adapter integration tests quantize their inputs identically instead of
    each re-deriving it and drifting."""
    H, T, I = tc["H"], tc["T"], tc["I"]
    W = _t2n(tc["W"]).reshape(4 * H, I)
    R = _t2n(tc["R"]).reshape(4 * H, H)
    y_gold = _t2n(tc["y"]).reshape(H)
    out_gold = _t2n(tc["output"]).reshape(T, H) if tc["output"] is not None else y_gold.reshape(1, H)

    w_scale, _ = _quant_sym8(W.flatten())
    r_scale, _ = _quant_sym8(R.flatten())
    x_scale, x_zp = _quant_asym(_t2n(tc["input"]).flatten())
    y_scale, y_zp = _quant_asym(out_gold.flatten())
    c_scale, _ = _quant_sym16_headroom(_t2n(tc["c"]).reshape(H))
    n_scale, _ = _quant_sym16_headroom(_t2n(tc["n"]).reshape(H))
    return {"W_scale": w_scale, "R_scale": r_scale,
            "x_scale": x_scale, "x_zero_point": x_zp,
            "y_scale": y_scale, "y_zero_point": y_zp,
            "c_scale": c_scale, "n_scale": n_scale}


def _mlstm_s8_calib(tc):
    """Scales the mLSTM INT8 path is calibrated with - the mLSTM twin of
    _slstm_s8_calib above (PrepareMlstmS8 in test/mlstm_s8_test.cc)."""
    H, T, I = tc["H"], tc["T"], tc["I"]
    W = _t2n(tc["W"]).reshape(4 * H + 2, I)
    y_gold = _t2n(tc["y"]).reshape(H)
    out_gold = _t2n(tc["output"]).reshape(T, H) if tc["output"] is not None else y_gold.reshape(1, H)

    w_scale, _ = _quant_sym8(W.flatten())
    x_scale, x_zp = _quant_asym(_t2n(tc["input"]).flatten())
    y_scale, y_zp = _quant_asym(out_gold.flatten())
    n_scale, _ = _quant_sym16_headroom(_t2n(tc["n"]).reshape(H))
    C_scale, _ = _quant_sym16_headroom(_t2n(tc["c"]).flatten())
    return {"W_scale": w_scale,
            "x_scale": x_scale, "x_zero_point": x_zp,
            "y_scale": y_scale, "y_zero_point": y_zp,
            "C_scale": C_scale, "n_scale": n_scale}


def _slstm_int8_trace(tc, perturb=1.0):
    """Numpy replica of PrepareS8 + slstm_eval_s8. Returns (out[T,H],
    golden_output[T,H], golden_y[H], state, quantized), where state is the
    replica's dequantized EXIT state {"state": c[H], "n": n[H], "m": m[H]}
    - the same three buffers the C kernel leaves behind for the caller, in
    the same dequantized units the INT8 test reads them back in. They feed
    compute_state_tol_per_elem below.

    quantized is the same trajectory and exit state as raw integers, the
    exact byte-level values the C kernel writes. Measured across every
    case in this file the two agree to 0 LSB on output/y/c/n, so the
    framework adapter integration tests assert against these directly
    rather than against a tolerance - see generate_json."""
    H, T, I = tc["H"], tc["T"], tc["I"]
    W = _t2n(tc["W"]).reshape(4 * H, I)
    R = _t2n(tc["R"]).reshape(4 * H, H)
    b = _t2n(tc["b"])
    x = _t2n(tc["input"]).reshape(1, T, I)
    y_gold = _t2n(tc["y"]).reshape(H)
    out_gold = _t2n(tc["output"]).reshape(T, H) if tc["output"] is not None else y_gold.reshape(1, H)

    cal = _slstm_s8_calib(tc)
    w_scale, r_scale = cal["W_scale"], cal["R_scale"]
    x_scale, x_zp = cal["x_scale"], cal["x_zero_point"]
    y_scale, y_zp = cal["y_scale"], cal["y_zero_point"]
    c_scale, n_scale = cal["c_scale"], cal["n_scale"]

    Wq = _qs8(W, w_scale, 0)
    Rq = _qs8(R, r_scale, 0)
    xq = _qs8(x, x_scale, x_zp)
    b_scale = w_scale * x_scale
    bq = np.round(b / b_scale).astype(np.int64)

    c_q = np.zeros(H, dtype=np.int64)
    n_q = np.zeros(H, dtype=np.int64)
    m_state = np.zeros(H)
    y_q = np.zeros(H, dtype=np.int64)
    wx_scale = w_scale * x_scale
    ry_scale = r_scale * y_scale
    out_computed = np.zeros((T, H))
    out_q = np.zeros((T, H), dtype=np.int64)
    for t in range(T):
        xt = xq[0, t, :]
        acc_wx = Wq.astype(np.int64) @ (xt - x_zp)
        acc_ry = Rq.astype(np.int64) @ (y_q - y_zp)
        preact = (acc_wx.astype(np.float64) * wx_scale
                  + acc_ry.astype(np.float64) * ry_scale
                  + bq.astype(np.float64) * b_scale)
        i_raw = preact[0:H]; f_raw = preact[H:2*H]
        z_raw = preact[2*H:3*H]; o_raw = preact[3*H:4*H]
        c_prev = c_q.astype(np.float64) * c_scale
        n_prev = n_q.astype(np.float64) * n_scale
        log_f_plus_m = m_state + (-np.logaddexp(0, -f_raw))
        m_new = np.where(n_q == 0, i_raw, np.maximum(i_raw, log_f_plus_m))
        i_gate = np.minimum(np.exp(i_raw - m_new), 1.0)
        f_gate = np.minimum(np.exp(log_f_plus_m - m_new), 1.0)
        o_gate = 1.0 / (1.0 + np.exp(-o_raw))
        c_input = np.tanh(z_raw)
        c_new = f_gate * c_prev + i_gate * c_input
        n_new = f_gate * n_prev + i_gate
        y_new = o_gate * (c_new / np.maximum(n_new, 1e-6))
        y_new = y_new * perturb
        c_q = np.clip(np.round(c_new / c_scale), -32768, 32767).astype(np.int64)
        n_q = np.clip(np.round(n_new / n_scale), -32768, 32767).astype(np.int64)
        m_state = m_new
        y_q = np.clip(np.round(y_new / y_scale + y_zp), -128, 127).astype(np.int64)
        out_computed[t] = y_scale * (y_q - y_zp)
        out_q[t] = y_q
    state = {"state": c_q.astype(np.float64) * c_scale,
             "n": n_q.astype(np.float64) * n_scale,
             "m": m_state.copy()}
    quantized = {"output": out_q.flatten(), "y": y_q, "c": c_q, "n": n_q,
                 "m": m_state.copy()}
    return out_computed, out_gold, y_gold, state, quantized


def _mlstm_int8_trace(tc, perturb=1.0):
    """Numpy replica of PrepareMlstmS8 + mlstm_eval_s8. Returns (out[T,H],
    golden_output[T,H], golden_y[H], state, quantized), where state is the
    replica's dequantized EXIT state {"state": C[H*H], "n": n[H], "m":
    m[1]} and quantized is the raw-integer form of the same - see
    _slstm_int8_trace above."""
    H, T, I = tc["H"], tc["T"], tc["I"]
    W = _t2n(tc["W"]).reshape(4 * H + 2, I)
    b = _t2n(tc["b"])
    x = _t2n(tc["input"]).reshape(1, T, I)
    y_gold = _t2n(tc["y"]).reshape(H)
    out_gold = _t2n(tc["output"]).reshape(T, H) if tc["output"] is not None else y_gold.reshape(1, H)

    cal = _mlstm_s8_calib(tc)
    w_scale = cal["W_scale"]
    x_scale, x_zp = cal["x_scale"], cal["x_zero_point"]
    y_scale, y_zp = cal["y_scale"], cal["y_zero_point"]
    C_scale, n_scale = cal["C_scale"], cal["n_scale"]

    Wq = _qs8(W, w_scale, 0)
    xq = _qs8(x, x_scale, x_zp)
    bq = np.round(b / (w_scale * x_scale)).astype(np.int64)

    C_f = np.zeros((H, H))
    n_f = np.zeros(H)
    m_state = np.zeros(1)
    out_computed = np.zeros((T, H))
    out_q = np.zeros((T, H), dtype=np.int64)
    C_q = np.zeros((H, H))
    n_q = np.zeros(H)
    yq = np.zeros(H)
    for t in range(T):
        xt = xq[0, t, :]
        acc = Wq.astype(np.int64) @ (xt - x_zp)
        preact = acc.astype(np.float64) * w_scale * x_scale + bq.astype(np.float64) * w_scale * x_scale
        q = preact[0:H]; k = preact[H:2*H]; v = preact[2*H:3*H]
        i_raw = preact[3*H]; f_raw = preact[3*H+1]; o_raw = preact[3*H+2:4*H+2]
        k = k / np.sqrt(H)
        m_prev = m_state[0]
        log_f_plus_m = -np.logaddexp(0, -f_raw) + m_prev
        m_new = max(log_f_plus_m, i_raw)
        f_gate = np.exp(log_f_plus_m - m_new)
        i_gate = np.exp(i_raw - m_new)
        C_new = f_gate * C_f + i_gate * np.outer(k, v)
        C_q = np.clip(np.round(C_new / C_scale), -32768, 32767)
        C_f = C_scale * C_q
        n_new = f_gate * n_f + i_gate * k
        n_q = np.clip(np.round(n_new / n_scale), -32768, 32767)
        n_f = n_scale * n_q
        m_state[0] = m_new
        qn = q @ n_f
        denom = max(abs(qn), np.exp(-m_new)) + 1e-6
        qC = q @ C_f
        y_new = (1.0 / (1.0 + np.exp(-o_raw))) * (qC / denom)
        y_new = y_new * perturb
        yq = np.clip(np.round(y_new / y_scale + y_zp), -128, 127)
        out_computed[t] = y_scale * (yq - y_zp)
        out_q[t] = yq.astype(np.int64)
    state = {"state": C_f.flatten().copy(), "n": n_f.copy(), "m": m_state.copy()}
    quantized = {"output": out_q.flatten(), "y": yq.astype(np.int64),
                 "C": C_q.flatten().astype(np.int64),
                 "n": n_q.astype(np.int64), "m": m_state.copy()}
    return out_computed, out_gold, y_gold, state, quantized


def _round_sig(x, sig=2, up=False):
    if x <= 0:
        return x
    mag = 10 ** (sig - 1 - math.floor(math.log10(x)))
    return (math.ceil(x * mag) if up else round(x * mag)) / mag


def _derive_bound(floor, rng):
    """Maximize toward ratio ~0.5 (see module docstring above); never
    below floor (avoids false failures) or at/above rng (avoids vacuity
    - an all-zero corruption of this channel would go undetected).

    floor >= rng is a real, if rare, case (not a bug to round away): it
    means this channel's own measured error already covers its own
    dynamic range, so no interval (floor, rng) exists at all - any bound
    that avoids vacuity (bound < rng) necessarily false-fails on the
    correct kernel (bound < floor), and any bound that avoids a false
    failure (bound > floor) necessarily can't distinguish correct output
    from an all-zero corruption of that one channel (bound >= rng). Two
    known channels are exactly floor == rng (their true value dequantizes
    to precisely the y_quant zero-point - undecidable by any test, not
    merely a hard-to-pick tolerance); sLSTM Test1 channel 0 is floor > rng
    outright. In all these cases this function chooses to avoid the false
    failure (a test that fails on correct, unmodified kernel output is
    worse than one blind to a corruption of a single already-near-zero
    channel) and returns a bound with headroom above floor, accepting
    that this specific channel is not usefully guarded. See
    The exact list and the full reasoning are recorded with the measurements
    in full. """
    if floor >= rng:
        return _round_sig(floor * 1.2, 2, up=True)
    target = 0.5 * rng
    if target > floor * 1.2:
        bound = _round_sig(target, 2, up=False)
        if bound <= floor:
            bound = _round_sig(target, 2, up=True)
    else:
        mid = (floor + rng) / 2.0
        bound = _round_sig(mid, 2, up=False)
        if bound <= floor:
            bound = _round_sig(mid, 2, up=True)
    if bound >= rng:
        bound = _round_sig(rng * 0.98, 2, up=False)
    return bound


def compute_tol_s8_per_channel(tc, cell):
    """Returns (bounds, floor): two lists of H floats each, for one case
    dict (cell is 's' or 'm'). bounds are the tol_s8_per_channel values
    every INT8 assertion checks against. floor is what this function's
    own numpy replica measured as each channel's worst-case error
    (baseline vs. perturbed) - emitted into reference_data.h alongside
    bounds so the real C kernel's measured error can be asserted against
    it at test time (see RunSlstmS8Case/RunMlstmS8Case's floor-
    consistency check). The replica and the real kernel
    (src/slstm_s8.c/src/mlstm_s8.c) are two hand-synchronized
    implementations of the same math; without that assertion, a kernel
    edit that isn't mirrored here would silently invalidate every bound
    in this file rather than failing loudly.

    rng per channel is the max magnitude across that channel's whole
    trajectory (every stored output timestep, plus the final y - which
    for T>1 is redundant with the T-1 output row, but included for T=1
    cases that don't store output at all). Sizing the bound from that
    trajectory-wide max is correct for catching a corruption of any one
    (timestep, channel) cell checked against tol_s8_per_channel[channel]
    - but there is a second, distinct class of corruption this task's
    mutation testing checks for: destroying ONLY the final state (y),
    leaving every stored output timestep - including the kernel's own
    copy of that same final value into output[T-1] - untouched. A bound
    derived from the trajectory-wide max can be safely below that max
    (that is the point) while still being ABOVE this specific channel's
    own final-y magnitude, if that channel's peak happens to occur at an
    earlier timestep - which leaves the case's *entire* channel set
    blind to a final-state-only corruption if it happens to every
    channel. Measured concretely for SweepM8: with the trajectory-wide
    rng, none of its 8 channels had bound < |y_final| - a real,
    demonstrated gap, not a theoretical one.
    Guaranteed here: if no channel's bound already comes in under its own
    |y_final|, the channel with the largest |y_final| (most room to work
    with) gets a second pass, deriving its bound from |y_final| alone
    instead of the trajectory-wide max, so it always catches a
    final-state-only corruption of the whole case."""
    if cell == "s":
        out0, gold, y_gold, _, _ = _slstm_int8_trace(tc, 1.0)
        out1, _, _, _, _ = _slstm_int8_trace(tc, 1.001)
    else:
        out0, gold, y_gold, _, _ = _mlstm_int8_trace(tc, 1.0)
        out1, _, _, _, _ = _mlstm_int8_trace(tc, 1.001)
    err0 = np.max(np.abs(out0 - gold), axis=0)
    err1 = np.max(np.abs(out1 - gold), axis=0)
    floor = np.maximum(err0, err1)
    rng = np.maximum(np.max(np.abs(gold), axis=0), np.abs(y_gold))
    bounds = [_derive_bound(float(floor[i]), float(rng[i])) for i in range(len(rng))]

    if not any(bounds[i] < abs(float(y_gold[i])) for i in range(len(rng))):
        j = int(np.argmax(np.abs(y_gold)))
        bounds[j] = _derive_bound(float(floor[j]), abs(float(y_gold[j])))
    return bounds, [float(v) for v in floor]


# Sentinel written into a state bound array for an element that cannot be
# given a useful bound at all. See compute_state_tol_per_elem below; the
# INT8 runners skip these and print how many they skipped, so an unasserted
# element is visible in the test output rather than silently absent.
STATE_TOL_UNASSERTABLE = -1.0

# Smallest factor by which a state bound must clear that element's own
# measured error to be worth shipping. Matches the 1.2 already used inside
# _derive_bound to decide whether a half-of-range target sits far enough
# above the floor. See compute_state_tol_per_elem.
STATE_MIN_MARGIN = 1.2


def compute_state_tol_per_elem(tc, cell):
    """Per-ELEMENT INT8 bounds and floors for the kernel's exit state.

    Returns (bounds, floors), each {"state": [...], "n": [...],
    "m": [...]} - one value per element of each state tensor (sLSTM
    c[H]/n[H]/m[H], mLSTM C[H*H]/n[H]/m[1]), derived exactly the way
    compute_tol_s8_per_channel derives the output bounds and floors, and
    for the same reasons.

    floors are what this function's own numpy replica measured as each
    element's worst-case error, and they are emitted into
    reference_data.h next to the bounds so the real C kernel's measured
    state error can be asserted against floor*1.5 at test time - the
    exit-state half of the drift detector the output path has had since
    the tol_s8_floor_per_channel round. Without it, a kernel change that
    moves the exit state without moving y is invisible as a replica
    divergence: it either passes (if it stays inside the bound) or fails
    as a bound violation with nothing pointing at the replica. That
    matters most for a new SIMD backend, which is exactly what stresses
    this hand-maintained C-to-Python coupling. Note the floors cover
    EVERY element, including the ones whose bound is
    STATE_TOL_UNASSERTABLE: drift detection does not need a non-vacuous
    bound to exist, so those elements are unguarded for correctness but
    still guarded against divergence.

    Why per element and not one bound per tensor: an earlier round bounded
    each state tensor at a fixed fraction of that tensor's own maximum
    magnitude. That is non-vacuous for the tensor as a whole (zeroing the
    whole tensor fails) but leaves every element whose own magnitude is
    below the fraction individually zeroable with nothing firing - counted
    at the time: 4063 of SweepM64's 4096 C elements (99.2%), 62 of its 64 n
    elements, 56 of SweepS64's 64 c elements. A mutant zeroing exactly
    those elements after the timestep loop, so it never feeds back into y,
    passed both INT8 suites green. The same argument that forced the output
    bounds per channel forces these per element.

    floor is the replica's own measured deviation from the f32 golden for
    that element, taken as the max over the baseline run and a run with the
    same 1.001x pre-requantization perturbation the output floors use.
    Caveat worth stating: in sLSTM that perturbation reaches the state
    through the recurrent y path, but mLSTM has no y recurrence, so its
    state floor is the baseline replica deviation alone. It makes little
    practical difference because the selection rule below lands far above
    the floor on all but a handful of elements.

    rng is that element's own golden magnitude - a state element has one
    golden value, not a trajectory, so there is no trajectory-wide max to
    take. Zeroing the element produces an error of exactly rng, which is
    what makes `bound < rng` the precise non-vacuity condition.

    Bounds must NOT come from the golden magnitude by a fixed relative
    factor: sLSTM Test1's c[0] deviates 3.749x its own golden
    (0.03609742), so any relative bound tight enough to be meaningful
    elsewhere false-fails there. They come from the measured floor,
    maximized toward 0.5 x rng by the same _derive_bound used for outputs.

    An element gets STATE_TOL_UNASSERTABLE when no bound can be both
    non-vacuous and free of false failures - rng == 0 (the golden is
    already zero, so zeroing it is undetectable by construction) or
    floor >= rng (the honest, measured error already covers the element's
    whole dynamic range). Shipping a bound there would mean shipping one
    that either cannot fail or fails on a correct kernel; both are worse
    than an explicit, counted omission.

    STATE_MIN_MARGIN extends that to the near-miss case. When floor is
    only just below rng the surviving window is real but tiny, and
    _derive_bound's midpoint fallback lands a hair above floor: measured
    without this rule, mLSTM SweepM64's C[2071] got bound 0.016 against a
    real-kernel error of 0.0157991, a margin of 1.013x. A bound that
    close to the honest measured error is one a legitimate backend trips,
    which is the failure mode this whole tolerance scheme exists to avoid.
    An element whose best available bound cannot clear its floor by
    STATE_MIN_MARGIN is therefore dropped rather than shipped tight - the
    same 1.2 factor _derive_bound already uses to decide whether a
    half-of-range target has enough room above the floor to be worth
    taking. Measured cost: it moves 32 of 5085 asserted elements (0.6%)
    into the unassertable count; measured benefit: mLSTM's state
    assertions stop being the first thing a LUT-style gate perturbation
    trips (see the report for the sweep)."""
    if cell == "s":
        _, _, _, st0, _ = _slstm_int8_trace(tc, 1.0)
        _, _, _, st1, _ = _slstm_int8_trace(tc, 1.001)
    else:
        _, _, _, st0, _ = _mlstm_int8_trace(tc, 1.0)
        _, _, _, st1, _ = _mlstm_int8_trace(tc, 1.001)

    golds = {"state": _t2n(tc["c"]).flatten(),
             "n": _t2n(tc["n"]).flatten(),
             "m": _t2n(tc["m"]).flatten()}

    out = {}
    out_floor = {}
    for key, gold in golds.items():
        a, b = np.asarray(st0[key]).flatten(), np.asarray(st1[key]).flatten()
        assert a.shape == gold.shape, (
            f"{tc['name']} {key}: replica {a.shape} vs golden {gold.shape}")
        floor = np.maximum(np.abs(a - gold), np.abs(b - gold))
        out_floor[key] = [float(v) for v in floor]
        bounds = []
        for i in range(len(gold)):
            rng = abs(float(gold[i]))
            fl = float(floor[i])
            if rng <= 0.0 or fl >= rng:
                bounds.append(STATE_TOL_UNASSERTABLE)
                continue
            bound = _derive_bound(fl, rng)
            # _derive_bound is written for the output case and can, after
            # 2-significant-digit rounding, land outside (floor, rng) on a
            # narrow window. Rather than trust it, check the properties that
            # actually matter and drop the element if any fails.
            if not (fl * STATE_MIN_MARGIN <= bound < rng):
                bounds.append(STATE_TOL_UNASSERTABLE)
                continue
            bounds.append(bound)
        out[key] = bounds
    return out, out_floor


def slstm_sized_case(H, seed):
    """sLSTM case at hidden size H, 3 timesteps, I = H.

    T=3 matters: a multi-step sequence forces the recurrent path and makes
    state errors observable in the per-timestep output.

    tol_s8 is not set here: build_cases() computes a per-channel tol_s8
    array for every case (see compute_tol_s8_per_channel above) after all
    cases are built, because a single case-wide constant cannot safely
    guard every channel - see that function's module docstring for the
    full derivation.

    H=17's INT8 error is elevated (~0.23 at its worst element) on a few
    channels: with this random weight draw, those channels' pre-activation
    lands near the zero-crossing of tanh/sigmoid, where accumulated
    INT8xINT8 matmul rounding noise (summed over I=17 terms) gets
    amplified by the activation's steep local slope. Verified against a
    from-scratch numpy replica of the quantized math and reproduces
    identically under both the sse2 and ref backends, so it is
    quantization noise particular to this draw, not a kernel bug. This is
    exactly the class of channel the per-channel tol_s8 array exists to
    handle without loosening every other channel of the case - H=17 is
    also the only sweep size with cols % 8 != 0 and cols > 8, i.e. the
    only one that exercises xlstm_matvec_s8's scalar SIMD tail, so a
    blanket loosening here would double as cover for an unrelated tail
    defect.
    """
    torch.manual_seed(seed)
    I = H
    W = torch.randn(4 * H, I) * 0.5
    R = torch.randn(4 * H, H) * 0.5
    b = torch.randn(4 * H) * 0.1
    x = torch.randn(1, 3, I)
    output, y, c, n, m = run_slstm(W, R, b, x)
    return dict(
        name=f"SweepS{H}", comment=f"Size sweep H={H}, 3 timesteps",
        B=1, T=3, I=I, H=H,
        W=W, R=R, b=b, input=x,
        y=y, c=c, n=n, m=m, output=output,
        tol_f32=1e-5)


def mlstm_sized_case(H, seed):
    """mLSTM case at hidden size H, 3 timesteps, I = H.

    The full C matrix is always stored (store_state defaults to True in
    _emit_case/_emit_table), including H=64 (an extra ~4096 floats in
    reference_data.h). An earlier version of this case capped storage at
    H<=17 to bound header size and derived H=64's C_quant scale from an
    empirically-tuned multiple of n's scale instead - a code review showed
    that guess was not actually the binding constraint on H=64's INT8
    error (scaling it across a 128x range left the error unchanged), so
    storing C directly removes a guess without evidence behind it. What
    H=64's residual error actually is remains undiagnosed.

    tol_s8 is not set here, for the same reason as slstm_sized_case: see
    compute_tol_s8_per_channel above.

    H=17's error is elevated (~0.73) on several channels: mLSTM's readout
    is q^T C / max(|q^T n|, exp(-m)), which - unlike sLSTM's gated-tanh
    cell - is not architecturally bounded, so the golden y values
    themselves reach into the double digits for this draw. INT8/INT16
    quantization noise in q, C and n compounds through that dot product
    and division. Verified against a from-scratch numpy replica (matches
    the f32 golden values when run unquantized) and reproduces identically
    under sse2 and ref, so this is quantization noise, not a kernel bug.
    """
    torch.manual_seed(seed)
    I = H
    W = torch.randn(4 * H + 2, I) * 0.5
    b = torch.randn(4 * H + 2) * 0.1
    x = torch.randn(1, 3, I)
    output, y, C, n, m = run_mlstm(W, b, x)
    return dict(
        name=f"SweepM{H}", comment=f"Size sweep H={H}, 3 timesteps",
        B=1, T=3, I=I, H=H,
        W=W, b=b, input=x,
        y=y, c=C, n=n, m=m, output=output,
        tol_f32=1e-5)


# ============================================================================
# Head composition
#
# hidden_size is the PER-HEAD width (DH in the
# NX-AI reference) and multi-head is the caller's outer loop over head-sliced
# weights. Everything else in this file is num_heads=1, where that contract is
# vacuous. This is the only case that is not.
#
# The slicing rule below was established empirically by
# test/derive_multihead_layout.py, which runs one num_heads=2 cell against two
# independent num_heads=1 cells and searches the plausible orderings. Measured
# result: gate-major row packing with per-head R stacked as-is and states
# ordered [head][unit] reproduces the 2-head cell EXACTLY (max abs diff
# 0.000e+00 over y, c, n, m and every output timestep); the seven other
# candidates are wrong by 1.0e-1 to 8.4e-1. Do not change this packing without
# re-running that script.
#
#   Fused width Hf = NH*DH, gate g in 0..3, head h, unit j in 0..DH-1:
#     W_h[g*DH + j][:] = W_fused[g*Hf + h*DH + j][:]
#     b_h[g*DH + j]    = b_fused[g*Hf + h*DH + j]
#     R_h              = R_stack[h]              (already [4*DH, DH])
#     y_h[j]           = y_fused[h*DH + j]       (same for c, n, m)
#
# Note what the gate-major packing means in practice: a head's four gate
# blocks are NOT contiguous in the fused weight matrix. A caller who slices
# rows [h*4*DH, (h+1)*4*DH) - the obvious guess - gets a silently wrong model,
# which is exactly why this vector exists.
# ============================================================================

HEAD2_NH = 2   # heads
HEAD2_DH = 4   # per-head width == this library's hidden_size
HEAD2_I = 3
HEAD2_T = 2


def _pack_gate_major(parts, NH, DH):
    """Interleave per-head [4*DH, ...] tensors into a fused [4*NH*DH, ...]
    one, gate-major: gate block g of the result is the concatenation over
    heads of each head's gate-g block."""
    return torch.cat(
        [torch.cat([p[g * DH:(g + 1) * DH] for p in parts], dim=0)
         for g in range(4)], dim=0)


def build_head2_composition():
    """Build the 2-head composition vector.

    Returns (case_head0, case_head1, fused), where the two cases are ordinary
    single-head sLSTM case dicts holding the per-head SLICES (so they flow
    through the f32 and INT8 table runners like any other case), and `fused`
    holds the unsliced 2-head tensors plus the 2-head cell's own outputs.

    The C test (TestHeadComposition in test/slstm_test.cc) works from `fused`
    only: it does the slicing itself, in C, and checks the concatenation of
    two per-head slstm_eval_f32 calls against the 2-head cell's output. That
    matters - had the C test consumed the pre-sliced per-head arrays instead,
    it would assert nothing about the slicing rule, since Python would have
    done the interesting part.
    """
    NH, DH, I, T = HEAD2_NH, HEAD2_DH, HEAD2_I, HEAD2_T
    Hf = NH * DH

    torch.manual_seed(7)
    W_parts = [torch.randn(4 * DH, I) * 0.5 for _ in range(NH)]
    R_parts = [torch.randn(4 * DH, DH) * 0.5 for _ in range(NH)]
    b_parts = [torch.randn(4 * DH) * 0.1 for _ in range(NH)]
    x = torch.randn(1, T, I)

    W_fused = _pack_gate_major(W_parts, NH, DH)      # [4*Hf, I]
    b_fused = _pack_gate_major(b_parts, NH, DH)      # [4*Hf]
    R_stack = torch.stack(R_parts, dim=0)            # [NH, 4*DH, DH]

    # The oracle: one real 2-head cell.
    out2, y2, c2, n2, m2 = run_slstm_multihead(W_fused, R_stack, b_fused, x, NH)

    cases = []
    for h in range(NH):
        output, y, c, n, m = run_slstm(W_parts[h], R_parts[h], b_parts[h], x)
        # Guard the whole construction: if the packing above ever stops
        # agreeing with the 2-head cell, fail here rather than emitting a
        # header that asserts a false contract.
        sl = slice(h * DH, (h + 1) * DH)
        for label, mine, oracle in (("y", y, y2[:, sl]), ("c", c, c2[:, sl]),
                                    ("n", n, n2[:, sl]), ("m", m, m2[:, sl]),
                                    ("output", output, out2[:, h])):
            err = (mine - oracle).abs().max().item()
            assert err < 1e-6, (
                f"head {h} {label}: 2-head cell and single-head cell differ by "
                f"{err:.3e}; the packing in _pack_gate_major no longer matches "
                f"the reference. Re-run test/derive_multihead_layout.py.")
        cases.append(dict(
            name=f"Head2{'' if h == 0 else 'b'}",
            comment=f"2-head composition, head {h} of {NH} (DH={DH})",
            note=(f"sliced from the fused 2-head weights by "
                  f"W_h[g*{DH}+j] = W_fused[g*{Hf}+{h}*{DH}+j]; see "
                  f"test/derive_multihead_layout.py"),
            B=1, T=T, I=I, H=DH,
            W=W_parts[h], R=R_parts[h], b=b_parts[h], input=x,
            y=y, c=c, n=n, m=m, output=output,
            tol_f32=1e-5))

    fused = dict(
        NH=NH, DH=DH, Hf=Hf, I=I, T=T,
        W=W_fused, R=R_stack, b=b_fused, input=x,
        # [Hf], ordered [head][unit]
        y_joined=y2,
        # [NH, T, DH] -> flattened head-major, which is what the C test builds
        output_joined=out2.squeeze(0),
    )
    return cases[0], cases[1], fused


def build_cases():
    """Build every reference case once. Both emitters consume this.

    Returns (slstm_cases, mlstm_cases, extras). The two case lists hold dicts
    of torch tensors plus their dimensions; `extras` holds data that is not a
    case - today only the fused 2-head tensors from build_head2_composition(),
    which are emitted as loose arrays because they are the INPUT to a
    caller-side slicing test, not something a table runner can drive. Keeping
    all three here is what lets the .h and .json emitters stay in sync.

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

    tol_s8 and tol_s8_per_channel are NOT set by the per-case builders
    below - this function computes them itself, after every case dict
    exists, via compute_tol_s8_per_channel (see that function's module
    docstring for the derivation). A case that skips the INT8 test table
    entirely (there is none today, but the mechanism doesn't assume one
    exists) would simply keep relying on _emit_table's tc.get() defaults.
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

    for idx, H in enumerate(SWEEP_SIZES):
        slstm_cases.append(slstm_sized_case(H, seed=1000 + idx))
        mlstm_cases.append(mlstm_sized_case(H, seed=2000 + idx))

    # The two per-head slices go into the ordinary sLSTM table (they are
    # ordinary single-head cases, and running them through the f32 and INT8
    # runners is free coverage); the fused tensors the C composition test
    # slices from go into extras. See build_head2_composition().
    head0, head1, head2_fused = build_head2_composition()
    slstm_cases.append(head0)
    slstm_cases.append(head1)

    # tol_s8 is computed here, per channel, for every case (both cells) -
    # not hand-picked per case as earlier versions of this file did. A
    # single case-wide constant is unsound in general: the invariant that
    # actually matters is ratio = tol_s8 / max|expected value| evaluated
    # at the granularity the tolerance is actually checked at (one
    # channel at a time - see RunSlstmS8Case/RunMlstmS8Case in the two
    # test/*_s8_test.cc files), and real cases have channels spanning
    # multiple orders of magnitude, so one constant cannot keep every
    # channel's ratio safely below 1.0. Two code reviews found this the
    # hard way: first a case-wide blanket left small-valued channels
    # vacuous (e.g. mLSTM SweepM1's ratio was ~32x under a flat 0.10
    # default), then even a per-case-plus-a-few-hand-picked-overrides
    # scheme kept missing channels (a case with 64 widely-spread channel
    # magnitudes needs closer to 64 individually-derived bounds, not 2 or
    # 3). See compute_tol_s8_per_channel's module docstring above for the
    # derivation.
    #
    # Three individual channels (sLSTM Test1 channel 0; sLSTM SweepS64
    # channel 15; mLSTM SweepM64 channel 63) have a true value close
    # enough to zero that even a bound derived from that channel's own
    # measured error cannot get below a ratio of 1 - this is not a
    # tolerance-tuning problem, it is a representability limit of
    # per-tensor INT8 quantization when one channel's true value is much
    # smaller than the tensor's overall range (two of these, SweepS64
    # ch15 and SweepM64 ch63, dequantize to exactly the y_quant
    # zero-point, so correct and all-zero-corrupted output are
    # bit-identical for that channel - undecidable by any tolerance, not
    # merely a hard-to-choose one). Documented, not hidden.
    # sLSTM SweepS17 channel 5 looked like a fourth
    # case in an earlier round but is not one: its window is narrow
    # (ratio 0.931) but real, and it has a normal, valid bound (0.055) -
    # do not re-add it here.
    #
    # The exit state (c/n/m, C/n/m) gets the same treatment one level down,
    # per ELEMENT rather than per channel - see compute_state_tol_per_elem
    # for why a per-tensor fraction was not enough and what an element with
    # no usable bound gets instead.
    for tc in slstm_cases:
        tc["tol_s8_per_channel"], tc["tol_s8_floor_per_channel"] = \
            compute_tol_s8_per_channel(tc, "s")
        tc["tol_s8"] = max(tc["tol_s8_per_channel"])
        tc["tol_s8_state"], tc["tol_s8_state_floor"] = \
            compute_state_tol_per_elem(tc, "s")
    for tc in mlstm_cases:
        tc["tol_s8_per_channel"], tc["tol_s8_floor_per_channel"] = \
            compute_tol_s8_per_channel(tc, "m")
        tc["tol_s8"] = max(tc["tol_s8_per_channel"])
        tc["tol_s8_state"], tc["tol_s8_state_floor"] = \
            compute_state_tol_per_elem(tc, "m")

    return slstm_cases, mlstm_cases, {"head2": head2_fused}


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
    if tc.get("store_state", True):
        f.write(f"const float k{n}_expected_{state_key}[] = {{{fmt(tc['c'])}}};\n")
    f.write(f"const float k{n}_expected_n[] = {{{fmt(tc['n'])}}};\n")
    f.write(f"const float k{n}_expected_m[] = {{{fmt(tc['m'])}}};\n")
    if tc["output"] is not None:
        f.write(f"const float k{n}_expected_output[] = {{{fmt(tc['output'])}}};\n")
    if "tol_s8_per_channel" in tc:
        vals = ", ".join(f"{v:.8f}f" for v in tc["tol_s8_per_channel"])
        f.write(f"const float k{n}_tol_s8_per_channel[] = {{{vals}}};\n")
    if "tol_s8_floor_per_channel" in tc:
        vals = ", ".join(f"{v:.8f}f" for v in tc["tol_s8_floor_per_channel"])
        f.write(f"const float k{n}_tol_s8_floor_per_channel[] = {{{vals}}};\n")
    # Per-element exit-state bounds, and the replica floors they were
    # derived from (see compute_state_tol_per_elem). The 'state' arrays are
    # skipped for a case that does not store its state golden at all, since
    # there would be nothing to compare against.
    for key, suffix in (("state", state_key), ("n", "n"), ("m", "m")):
        if "tol_s8_state" not in tc:
            break
        if key == "state" and not tc.get("store_state", True):
            continue
        vals = ", ".join(f"{v:.8f}f" for v in tc["tol_s8_state"][key])
        f.write(f"const float k{n}_tol_s8_{suffix}_per_elem[] = {{{vals}}};\n")
        vals = ", ".join(f"{v:.8f}f" for v in tc["tol_s8_state_floor"][key])
        f.write(f"const float k{n}_tol_s8_{suffix}_floor_per_elem[] = {{{vals}}};\n")
    f.write("\n")


CASE_STRUCT = """typedef struct {
    const char* name;
    int B, T, I, H;
    const float* W;
    const float* R;               /* NULL for mLSTM */
    const float* b;
    const float* input;
    const float* expected_y;
    const float* expected_state;  /* sLSTM c, or mLSTM C; NULL if not stored */
    const float* expected_n;
    const float* expected_m;
    const float* expected_output; /* [T*H], NULL if not stored */
    float tol_f32;
    float tol_s8;                 /* max(tol_s8_per_channel); printed alongside the
                                    * measured error in TestS8QuantizationBound /
                                    * TestMlstmS8QuantizationBound - not used for any
                                    * pass/fail check, tol_s8_per_channel is. */
    const float* tol_s8_per_channel;       /* [H], the bound each INT8 assertion actually uses */
    const float* tol_s8_floor_per_channel; /* [H], what generate_reference.py's numpy
                                             * replica measured as each channel's own
                                             * worst-case error - the real kernel's
                                             * measured error is asserted against
                                             * floor*1.5 in RunSlstmS8Case/
                                             * RunMlstmS8Case so a kernel change that
                                             * silently outgrows the replica fails
                                             * loudly instead of just widening a bound. */
    /* Per-ELEMENT bounds on the kernel's INT8 exit state, derived by
     * generate_reference.py's compute_state_tol_per_elem the same way the
     * output bounds above are derived (see that function for the rule and
     * for why a per-tensor bound was not enough). Lengths follow the
     * tensors they bound: state is [H] for sLSTM c and [H*H] for mLSTM C,
     * n is [H], m is [H] for sLSTM and [1] for mLSTM. A negative entry is
     * XLSTM_STATE_TOL_UNASSERTABLE: that element has no bound that is both
     * non-vacuous and free of false failures, so the runners skip it and
     * report how many they skipped. NULL if not emitted. */
    const float* tol_s8_state_per_elem;
    const float* tol_s8_n_per_elem;
    const float* tol_s8_m_per_elem;
    /* What generate_reference.py's numpy replica measured as each state
     * element's own worst-case error - the exit-state twin of
     * tol_s8_floor_per_channel above. The real kernel's measured state
     * error is asserted against floor*1.5 in RunSlstmS8Case/RunMlstmS8Case
     * so a kernel change that moves the exit state without moving y is
     * reported as a replica divergence instead of silently passing or
     * surfacing as an unexplained bound violation. Unlike the bounds,
     * these cover every element, including the ones whose bound is
     * XLSTM_STATE_TOL_UNASSERTABLE: drift detection does not require a
     * non-vacuous bound to exist. Same lengths as the bounds. NULL if not
     * emitted. */
    const float* tol_s8_state_floor_per_elem;
    const float* tol_s8_n_floor_per_elem;
    const float* tol_s8_m_floor_per_elem;
} XlstmRefCase;

"""


def _emit_table(f, cases, table_name, state_key, has_R):
    """Emit a static array of XlstmRefCase referencing the per-case arrays.

    combined=True cases (see build_cases' docstring) did not get their own
    k<name>_W/R/b arrays from _emit_case - they reuse whichever preceding
    case last emitted them. `src` tracks that source case's name so the
    table points at the arrays that actually exist.
    """
    f.write(f"static const XlstmRefCase {table_name}[] = {{\n")
    src = None
    for tc in cases:
        n = tc["name"]
        if not tc.get("combined"):
            src = n
        R = f"k{src}_R" if has_R else "NULL"
        out = f"k{n}_expected_output" if tc["output"] is not None else "NULL"
        state = f"k{n}_expected_{state_key}" if tc.get("store_state", True) else "NULL"
        per_channel = f"k{n}_tol_s8_per_channel" if "tol_s8_per_channel" in tc else "NULL"
        floor = f"k{n}_tol_s8_floor_per_channel" if "tol_s8_floor_per_channel" in tc else "NULL"
        has_state_tol = "tol_s8_state" in tc
        st_tol = (f"k{n}_tol_s8_{state_key}_per_elem"
                  if has_state_tol and tc.get("store_state", True) else "NULL")
        n_tol = f"k{n}_tol_s8_n_per_elem" if has_state_tol else "NULL"
        m_tol = f"k{n}_tol_s8_m_per_elem" if has_state_tol else "NULL"
        st_fl = (f"k{n}_tol_s8_{state_key}_floor_per_elem"
                 if has_state_tol and tc.get("store_state", True) else "NULL")
        n_fl = f"k{n}_tol_s8_n_floor_per_elem" if has_state_tol else "NULL"
        m_fl = f"k{n}_tol_s8_m_floor_per_elem" if has_state_tol else "NULL"
        f.write(
            f'    {{"{n}", {tc["B"]}, {tc["T"]}, {tc["I"]}, {tc["H"]}, '
            f'k{src}_W, {R}, k{src}_b, k{n}_input, k{n}_expected_y, {state}, '
            f'k{n}_expected_n, k{n}_expected_m, {out}, '
            f'{tc.get("tol_f32", 1e-5):.8g}f, {tc.get("tol_s8", 0.10):.8g}f, '
            f'{per_channel}, {floor}, {st_tol}, {n_tol}, {m_tol}, '
            f'{st_fl}, {n_fl}, {m_fl}}},\n'
        )
    f.write("};\n")
    f.write(f"static const int {table_name}Count = "
            f"(int)(sizeof({table_name}) / sizeof({table_name}[0]));\n\n")


def _emit_head2_fused(f, fused):
    """Emit the unsliced 2-head tensors, plus the 2-head cell's own outputs.

    These are deliberately NOT an XlstmRefCase: the table runners drive one
    single-head kernel call per case, and the point here is the caller-side
    slicing loop, which no table entry can express. The per-head slices are
    in the table as the Head2/Head2b cases; these arrays are what
    TestHeadComposition in test/slstm_test.cc slices for itself.
    """
    NH, DH, Hf = fused["NH"], fused["DH"], fused["Hf"]
    f.write("// " + "=" * 72 + "\n")
    f.write(f"// Head composition: one NH={NH} cell of head width DH={DH}\n")
    f.write("//\n")
    f.write("// hidden_size in this library is the PER-HEAD width; multi-head is the\n")
    f.write("// caller's outer loop over head-sliced weights.\n")
    f.write("// The reference's fused weight rows are GATE-major, so head h's four gate\n")
    f.write("// blocks are NOT contiguous:\n")
    f.write("//\n")
    f.write(f"//   W_h[g*{DH} + j][:] = kHead2Fused_W[(g*{Hf} + h*{DH} + j)*I ..]\n")
    f.write(f"//   b_h[g*{DH} + j]    = kHead2Fused_b[g*{Hf} + h*{DH} + j]\n")
    f.write(f"//   R_h                = kHead2Fused_R + h*{4 * DH * DH}"
            f"   (already [4*DH, DH])\n")
    f.write(f"//   y_h[j]             = y_fused[h*{DH} + j]\n")
    f.write("//\n")
    f.write("// Established empirically, not read off the reference source, by\n")
    f.write("// test/derive_multihead_layout.py. Head2/Head2b in the table above are\n")
    f.write("// these same weights already sliced.\n")
    f.write("// " + "=" * 72 + "\n\n")
    f.write(f"#define kHead2_NH {NH}\n")
    f.write(f"#define kHead2_DH {DH}\n")
    f.write(f"#define kHead2_Hf {Hf}\n")
    f.write(f"#define kHead2_I {fused['I']}\n")
    f.write(f"#define kHead2_T {fused['T']}\n")
    f.write(f"const float kHead2Fused_W[] = {{{fmt(fused['W'])}}};\n")
    f.write(f"const float kHead2Fused_R[] = {{{fmt(fused['R'])}}};\n")
    f.write(f"const float kHead2Fused_b[] = {{{fmt(fused['b'])}}};\n")
    f.write("// Final y of the NH=2 cell, [Hf], ordered [head][unit].\n")
    f.write(f"const float kHead2_expected_y_joined[] = {{{fmt(fused['y_joined'])}}};\n")
    f.write("// Per-timestep output of the NH=2 cell, [NH][T][DH].\n")
    f.write(f"const float kHead2_expected_output_joined[] = "
            f"{{{fmt(fused['output_joined'])}}};\n\n")


def generate(f):
    """Generate all reference data into file handle f."""
    slstm_cases, mlstm_cases, extras = build_cases()

    f.write(
        "/* Auto-generated - do not edit.\n"
        " * Source: NX-AI/xlstm reference (vanilla backend)\n"
        " * Regenerate: make reference\n"
        " */\n\n"
        "#ifndef REFERENCE_DATA_H_\n"
        "#define REFERENCE_DATA_H_\n\n"
        "#include <stddef.h>\n\n"
        "/* Must equal kStateHeadroom in slstm_s8_test.cc/mlstm_s8_test.cc -\n"
        " * both files static_assert against this. See GENERATOR_HEADROOM in\n"
        " * generate_reference.py. */\n"
        f"#define XLSTM_GENERATOR_HEADROOM {GENERATOR_HEADROOM:.8f}f\n\n"
        "/* Sentinel in a tol_s8_*_per_elem array: this element has no bound\n"
        " * that is both non-vacuous and free of false failures, so the INT8\n"
        " * runners skip it and report the count. See\n"
        " * compute_state_tol_per_elem in generate_reference.py. */\n"
        f"#define XLSTM_STATE_TOL_UNASSERTABLE {STATE_TOL_UNASSERTABLE:.8f}f\n\n"
    )

    f.write(CASE_STRUCT)

    f.write("// " + "=" * 72 + "\n")
    f.write("// sLSTM reference data\n")
    f.write("// " + "=" * 72 + "\n\n")
    for tc in slstm_cases:
        _emit_case(f, tc, state_key="c", has_R=True)
    _emit_table(f, slstm_cases, "kSlstmCases", "c", True)

    _emit_head2_fused(f, extras["head2"])

    f.write("// " + "=" * 72 + "\n")
    f.write("// mLSTM reference data\n")
    f.write("// " + "=" * 72 + "\n\n")
    for tc in mlstm_cases:
        _emit_case(f, tc, state_key="C", has_R=False)
    _emit_table(f, mlstm_cases, "kMlstmCases", "C", False)

    f.write("#endif /* REFERENCE_DATA_H_ */\n")


def to_list(tensor):
    """Convert tensor to a flat Python list of floats."""
    return [round(float(v), 8) for v in tensor.flatten().tolist()]


def _to_ints(a):
    return [int(v) for v in np.asarray(a).flatten().tolist()]


def _s8_json_block(tc, cell):
    """The INT8 block generate_json attaches to every case.

    Framework adapters reach the INT8 kernels with tensors that are
    already quantized - the framework carries the scale and zero-point,
    the adapter only unpacks them. Their integration tests are therefore
    handed quantized inputs here rather than calibrating their own: it
    keeps four harnesses free of quantizer code, and it removes the one
    way they could disagree with the C kernel without either being wrong
    (numpy's round-half-to-even vs C's roundf rounding a weight the other
    way at an exact .5 tie).

    expected_*_q are what the numpy replica above computes, which is what
    the C kernel computes: measured across every case in this file the
    two agree on every output, y, c/C and n integer exactly, and on m to
    within float32 rounding. A harness asserts the raw integers and needs
    no tolerance at all. tol_per_channel is carried alongside so a harness
    can also state the weaker but more meaningful claim - dequantized INT8
    output stays within the calibrated per-channel bound of the f32 golden
    - using the same bounds test/{slstm,mlstm}_s8_test.cc assert against.
    """
    H, T, I = tc["H"], tc["T"], tc["I"]
    x = _t2n(tc["input"]).reshape(1, T, I)
    W = _t2n(tc["W"])
    b = _t2n(tc["b"])

    if cell == "s":
        cal = _slstm_s8_calib(tc)
        _, _, _, _, q = _slstm_int8_trace(tc)
        R = _t2n(tc["R"]).reshape(4 * H, H)
        blk = {"R_q": _to_ints(_qs8(R, cal["R_scale"], 0)),
               "expected_c_q": _to_ints(q["c"])}
    else:
        cal = _mlstm_s8_calib(tc)
        _, _, _, _, q = _mlstm_int8_trace(tc)
        blk = {"expected_C_q": _to_ints(q["C"])}

    wx = cal["W_scale"] * cal["x_scale"]
    blk.update(cal)
    blk.update({
        "b_scale": wx,
        "x_q": _to_ints(_qs8(x, cal["x_scale"], cal["x_zero_point"])),
        "W_q": _to_ints(_qs8(W.reshape(-1), cal["W_scale"], 0)),
        "b_q": _to_ints(np.round(b / wx)),
        "expected_output_q": _to_ints(q["output"]),
        "expected_y_q": _to_ints(q["y"]),
        "expected_n_q": _to_ints(q["n"]),
        "expected_m": to_list(q["m"]),
        "tol_per_channel": [float(v) for v in tc["tol_s8_per_channel"]],
    })
    return blk


def generate_json(path):
    """Generate reference_data.json with the same values as the C header."""
    slstm_cases, mlstm_cases, extras = build_cases()
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
        entry["s8"] = _s8_json_block(tc, "s")
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
        entry["s8"] = _s8_json_block(tc, "m")
        data["mlstm"][f"test{i}"] = entry

    # The fused 2-head tensors are not a case (see _emit_head2_fused): they
    # live in their own top-level key so the framework adapters, which loop
    # over data["slstm"], keep seeing single-head cases only. An adapter that
    # wants to check the composition contract reads this key explicitly.
    h2 = extras["head2"]
    data["slstm_head2"] = {
        "NH": h2["NH"], "DH": h2["DH"], "Hf": h2["Hf"],
        "I": h2["I"], "T": h2["T"], "B": 1,
        "W_fused": to_list(h2["W"]), "R_stack": to_list(h2["R"]),
        "b_fused": to_list(h2["b"]), "input": to_list(h2["input"]),
        "expected_y_joined": to_list(h2["y_joined"]),
        "expected_output_joined": to_list(h2["output_joined"]),
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
