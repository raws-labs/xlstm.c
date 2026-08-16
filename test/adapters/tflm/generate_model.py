#!/usr/bin/env python3
"""Generate minimal .tflite FlatBuffer models containing sLSTM/mLSTM custom ops.

Outputs C header files with model byte arrays, matching the pattern used by
tflite-micro upstream tests.

These models contain no weights - all tensors are inputs/outputs. The custom
op registration in the test binary provides the kernel implementation.

Usage: python3 generate_model.py
Writes: slstm_model_data.h, mlstm_model_data.h
"""

import json
import os
import struct

import flatbuffers
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..")
REF_PATH = os.path.join(ROOT_DIR, "test", "reference_data.json")


# TFLite schema constants (from tensorflow/lite/schema/schema_generated.h)
# We build the FlatBuffer manually to avoid depending on the full TF build.

class TensorType:
    FLOAT32 = 0
    INT32 = 2
    INT16 = 7
    INT8 = 9

class BuiltinOperator:
    CUSTOM = 32

class Padding:
    SAME = 0

class BuiltinOptions:
    NONE = 0


def build_tflite_model(op_name, tensor_specs, input_indices, output_indices):
    """Build a minimal .tflite FlatBuffer with a single custom op.

    Args:
        op_name: Custom op name string (e.g. "SLSTM")
        tensor_specs: list of (name, shape, type) or
                      (name, shape, type, (scale, zero_point)) tuples.
                      The 4-element form attaches QuantizationParameters to
                      the tensor, which is where the INT8 adapter reads its
                      scale and zero-point from.
        input_indices: list of tensor indices that are op inputs
        output_indices: list of tensor indices that are op outputs

    Returns:
        bytes: Serialized .tflite FlatBuffer
    """
    tensor_specs = [t if len(t) == 4 else (t[0], t[1], t[2], None)
                    for t in tensor_specs]
    builder = flatbuffers.Builder(1024)

    # --- Strings ---
    op_name_off = builder.CreateString(op_name)
    tensor_name_offs = []
    for spec in tensor_specs:
        tensor_name_offs.append(builder.CreateString(spec[0]))

    # --- OperatorCodes ---
    # Table OperatorCode { deprecated_builtin_code:byte, custom_code:string,
    #                      version:int, builtin_code:BuiltinOperator }
    builder.StartObject(4)
    builder.PrependInt8Slot(0, BuiltinOperator.CUSTOM, 0)  # deprecated_builtin_code
    builder.PrependUOffsetTRelativeSlot(1, op_name_off, 0)  # custom_code
    builder.PrependInt32Slot(2, 1, 0)  # version
    builder.PrependInt32Slot(3, BuiltinOperator.CUSTOM, 0)  # builtin_code
    op_code_off = builder.EndObject()

    builder.StartVector(4, 1, 4)
    builder.PrependUOffsetTRelative(op_code_off)
    op_codes_vec = builder.EndVector()

    # --- Quantization parameters (one table per quantized tensor) ---
    # Table QuantizationParameters { min:[float], max:[float], scale:[float],
    #                                zero_point:[long], details_type:ubyte,
    #                                details:..., quantized_dimension:int }
    # Per-tensor quantization: scale and zero_point are 1-element vectors.
    quant_offs = []
    for spec in tensor_specs:
        if spec[3] is None:
            quant_offs.append(None)
            continue
        scale, zero_point = spec[3]
        builder.StartVector(4, 1, 4)
        builder.PrependFloat32(scale)
        scale_vec = builder.EndVector()
        builder.StartVector(8, 1, 8)
        builder.PrependInt64(zero_point)
        zp_vec = builder.EndVector()
        builder.StartObject(7)
        builder.PrependUOffsetTRelativeSlot(2, scale_vec, 0)
        builder.PrependUOffsetTRelativeSlot(3, zp_vec, 0)
        quant_offs.append(builder.EndObject())

    # --- Tensors ---
    tensor_offs = []
    shape_vecs = []
    for _, shape, _, _ in tensor_specs:
        builder.StartVector(4, len(shape), 4)
        for dim in reversed(shape):
            builder.PrependInt32(dim)
        shape_vecs.append(builder.EndVector())

    for i, (name, shape, ttype, _) in enumerate(tensor_specs):
        # Table Tensor { shape:[int], type:TensorType, buffer:uint,
        #                name:string, quantization:QuantizationParameters,
        #                is_variable:bool }
        builder.StartObject(6)
        builder.PrependUOffsetTRelativeSlot(0, shape_vecs[i], 0)  # shape
        builder.PrependInt8Slot(1, ttype, 0)  # type
        builder.PrependUint32Slot(2, i + 1, 0)  # buffer index (0 = sentinel)
        builder.PrependUOffsetTRelativeSlot(3, tensor_name_offs[i], 0)  # name
        if quant_offs[i] is not None:
            builder.PrependUOffsetTRelativeSlot(4, quant_offs[i], 0)
        builder.PrependBoolSlot(5, False, False)  # is_variable
        tensor_offs.append(builder.EndObject())

    builder.StartVector(4, len(tensor_offs), 4)
    for off in reversed(tensor_offs):
        builder.PrependUOffsetTRelative(off)
    tensors_vec = builder.EndVector()

    # --- Operator ---
    # inputs vector
    builder.StartVector(4, len(input_indices), 4)
    for idx in reversed(input_indices):
        builder.PrependInt32(idx)
    inputs_vec = builder.EndVector()

    # outputs vector
    builder.StartVector(4, len(output_indices), 4)
    for idx in reversed(output_indices):
        builder.PrependInt32(idx)
    outputs_vec = builder.EndVector()

    # Table Operator { opcode_index:uint, inputs:[int], outputs:[int],
    #                  builtin_options_type:ubyte, builtin_options:...,
    #                  custom_options:[ubyte] }
    builder.StartObject(6)
    builder.PrependUint32Slot(0, 0, 0)  # opcode_index
    builder.PrependUOffsetTRelativeSlot(1, inputs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(2, outputs_vec, 0)
    builder.PrependUint8Slot(3, BuiltinOptions.NONE, 0)
    # slots 4, 5: skip
    operator_off = builder.EndObject()

    builder.StartVector(4, 1, 4)
    builder.PrependUOffsetTRelative(operator_off)
    operators_vec = builder.EndVector()

    # --- SubGraph ---
    # SubGraph inputs/outputs
    builder.StartVector(4, len(input_indices), 4)
    for idx in reversed(input_indices):
        builder.PrependInt32(idx)
    sg_inputs = builder.EndVector()

    builder.StartVector(4, len(output_indices), 4)
    for idx in reversed(output_indices):
        builder.PrependInt32(idx)
    sg_outputs = builder.EndVector()

    sg_name = builder.CreateString("main")

    # Table SubGraph { tensors:[Tensor], inputs:[int], outputs:[int],
    #                  operators:[Operator], name:string }
    builder.StartObject(5)
    builder.PrependUOffsetTRelativeSlot(0, tensors_vec, 0)
    builder.PrependUOffsetTRelativeSlot(1, sg_inputs, 0)
    builder.PrependUOffsetTRelativeSlot(2, sg_outputs, 0)
    builder.PrependUOffsetTRelativeSlot(3, operators_vec, 0)
    builder.PrependUOffsetTRelativeSlot(4, sg_name, 0)
    subgraph_off = builder.EndObject()

    builder.StartVector(4, 1, 4)
    builder.PrependUOffsetTRelative(subgraph_off)
    subgraphs_vec = builder.EndVector()

    # --- Buffers (empty - all data provided at runtime) ---
    buffer_offs = []
    # Buffer 0: sentinel empty buffer
    for _ in range(len(tensor_specs) + 1):
        builder.StartObject(1)  # Table Buffer { data:[ubyte] }
        buffer_offs.append(builder.EndObject())

    builder.StartVector(4, len(buffer_offs), 4)
    for off in reversed(buffer_offs):
        builder.PrependUOffsetTRelative(off)
    buffers_vec = builder.EndVector()

    # --- Model ---
    desc = builder.CreateString("xlstm-micro integration test")

    # Table Model { version:uint, operator_codes:[OperatorCode],
    #               subgraphs:[SubGraph], description:string,
    #               buffers:[Buffer] }
    builder.StartObject(5)
    builder.PrependUint32Slot(0, 3, 0)  # schema version 3
    builder.PrependUOffsetTRelativeSlot(1, op_codes_vec, 0)
    builder.PrependUOffsetTRelativeSlot(2, subgraphs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(3, desc, 0)
    builder.PrependUOffsetTRelativeSlot(4, buffers_vec, 0)
    model_off = builder.EndObject()

    builder.Finish(model_off, b"TFL3")
    return bytes(builder.Output())


def model_to_c_header(model_bytes, var_name, header_guard):
    """Convert model bytes to a C header with an aligned byte array."""
    lines = []
    lines.append(f"/* Auto-generated - do not edit. */\n")
    lines.append(f"#ifndef {header_guard}")
    lines.append(f"#define {header_guard}\n")
    lines.append(f"alignas(16) const unsigned char {var_name}[] = {{")

    for i in range(0, len(model_bytes), 12):
        chunk = model_bytes[i:i+12]
        hex_vals = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"    {hex_vals},")

    lines.append(f"}};")
    lines.append(f"const unsigned int {var_name}_len = {len(model_bytes)};\n")
    lines.append(f"#endif  /* {header_guard} */\n")
    return "\n".join(lines)


def generate_slstm_model(B, T, I, H):
    """Generate .tflite model for sLSTM custom op."""
    tensors = [
        ("input",  [B, T, I],  TensorType.FLOAT32),  # 0
        ("W",      [4*H, I],   TensorType.FLOAT32),  # 1
        ("R",      [4*H, H],   TensorType.FLOAT32),  # 2
        ("b",      [4*H],      TensorType.FLOAT32),  # 3
        ("y",      [B, H],     TensorType.FLOAT32),  # 4
        ("c",      [B, H],     TensorType.FLOAT32),  # 5
        ("n",      [B, H],     TensorType.FLOAT32),  # 6
        ("m",      [B, H],     TensorType.FLOAT32),  # 7
        ("output", [B, T, H],  TensorType.FLOAT32),  # 8
    ]
    input_indices = list(range(8))
    output_indices = [8]
    return build_tflite_model("SLSTM", tensors, input_indices, output_indices)


def generate_mlstm_model(B, T, I, H):
    """Generate .tflite model for mLSTM custom op."""
    tensors = [
        ("input",  [B, T, I],    TensorType.FLOAT32),  # 0
        ("W",      [4*H+2, I],   TensorType.FLOAT32),  # 1
        ("b",      [4*H+2],      TensorType.FLOAT32),  # 2
        ("y",      [B, H],       TensorType.FLOAT32),  # 3
        ("C",      [B, H*H],     TensorType.FLOAT32),  # 4
        ("n",      [B, H],       TensorType.FLOAT32),  # 5
        ("m",      [B, 1],       TensorType.FLOAT32),  # 6
        ("output", [B, T, H],    TensorType.FLOAT32),  # 7
    ]
    input_indices = list(range(7))
    output_indices = [7]
    return build_tflite_model("MLSTM", tensors, input_indices, output_indices)


def generate_slstm_s8_model(B, T, I, H, s8):
    """Generate a quantized .tflite model for the sLSTM custom op.

    The scales come straight from reference_data.json's s8 block, so the
    adapter reads back exactly the calibration the expected integers were
    produced with. Weights are symmetric (zero_point 0); b carries the
    input*weight scale; m is the log-space stabilizer and stays float32.
    output shares y's quantization, which is what the kernel assumes.
    """
    q = lambda k: (s8[k + "_scale"], s8.get(k + "_zero_point", 0))
    tensors = [
        ("input",  [B, T, I], TensorType.INT8,    q("x")),   # 0
        ("W",      [4*H, I],  TensorType.INT8,    q("W")),   # 1
        ("R",      [4*H, H],  TensorType.INT8,    q("R")),   # 2
        ("b",      [4*H],     TensorType.INT32,   q("b")),   # 3
        ("y",      [B, H],    TensorType.INT8,    q("y")),   # 4
        ("c",      [B, H],    TensorType.INT16,   q("c")),   # 5
        ("n",      [B, H],    TensorType.INT16,   q("n")),   # 6
        ("m",      [B, H],    TensorType.FLOAT32),           # 7
        ("output", [B, T, H], TensorType.INT8,    q("y")),   # 8
    ]
    return build_tflite_model("SLSTM", tensors, list(range(8)), [8])


def generate_mlstm_s8_model(B, T, I, H, s8):
    """Generate a quantized .tflite model for the mLSTM custom op.
    See generate_slstm_s8_model; mLSTM has no recurrent weight."""
    q = lambda k: (s8[k + "_scale"], s8.get(k + "_zero_point", 0))
    tensors = [
        ("input",  [B, T, I],  TensorType.INT8,  q("x")),   # 0
        ("W",      [4*H+2, I], TensorType.INT8,  q("W")),   # 1
        ("b",      [4*H+2],    TensorType.INT32, q("b")),   # 2
        ("y",      [B, H],     TensorType.INT8,  q("y")),   # 3
        ("C",      [B, H*H],   TensorType.INT16, q("C")),   # 4
        ("n",      [B, H],     TensorType.INT16, q("n")),   # 5
        ("m",      [B, 1],     TensorType.FLOAT32),         # 6
        ("output", [B, T, H],  TensorType.INT8,  q("y")),   # 7
    ]
    return build_tflite_model("MLSTM", tensors, list(range(7)), [7])


def write_header(model_bytes, var_name, path):
    guard = var_name.upper() + "_H_"
    with open(path, "w") as f:
        f.write(model_to_c_header(model_bytes, var_name, guard))
    print(f"Wrote {path} ({len(model_bytes)} bytes)")


# Which s8 arrays each cell contributes, and the C type to emit them as.
S8_ARRAYS = {
    "s": [("x_q", "int8_t"), ("W_q", "int8_t"), ("R_q", "int8_t"),
          ("b_q", "int32_t"), ("expected_output_q", "int8_t"),
          ("expected_y_q", "int8_t"), ("expected_c_q", "int16_t"),
          ("expected_n_q", "int16_t"), ("expected_m", "float")],
    "m": [("x_q", "int8_t"), ("W_q", "int8_t"),
          ("b_q", "int32_t"), ("expected_output_q", "int8_t"),
          ("expected_y_q", "int8_t"), ("expected_C_q", "int16_t"),
          ("expected_n_q", "int16_t"), ("expected_m", "float")],
}


def generate_s8_case_header(cases):
    """Emit reference_data.json's s8 arrays as C arrays.

    The TFLM test feeds these into the quantized model's tensors and
    asserts the kernel's integers come back exactly. reference_data.h
    carries the f32 golden values but not the quantized ones, and the
    TFLM test is C++ with no JSON parser, so they come across here.
    """
    lines = ["/* Auto-generated - do not edit. */\n",
             "#ifndef S8_CASE_DATA_H_", "#define S8_CASE_DATA_H_\n",
             "#include <stdint.h>\n"]
    for prefix, cell, s8 in cases:
        for key, ctype in S8_ARRAYS[cell]:
            vals = s8[key]
            fmt = (lambda v: f"{v:.8f}f") if ctype == "float" else (lambda v: str(v))
            body = ", ".join(fmt(v) for v in vals)
            lines.append(f"const {ctype} {prefix}_{key}[] = {{{body}}};")
        lines.append("")
    lines.append("#endif  /* S8_CASE_DATA_H_ */\n")
    return "\n".join(lines)


def main():
    with open(REF_PATH) as f:
        ref = json.load(f)

    # Use test1 dimensions for the model (single timestep)
    st1 = ref["slstm"]["test1"]
    mt1 = ref["mlstm"]["test1"]
    dims_s = (st1["B"], st1["T"], st1["I"], st1["H"])
    dims_m = (mt1["B"], mt1["T"], mt1["I"], mt1["H"])

    for bytes_, var in [
        (generate_slstm_model(*dims_s), "slstm_model_data"),
        (generate_mlstm_model(*dims_m), "mlstm_model_data"),
        (generate_slstm_s8_model(*dims_s, st1["s8"]), "slstm_s8_model_data"),
        (generate_mlstm_s8_model(*dims_m, mt1["s8"]), "mlstm_s8_model_data"),
    ]:
        write_header(bytes_, var, os.path.join(SCRIPT_DIR, var + ".h"))

    path = os.path.join(SCRIPT_DIR, "s8_case_data.h")
    with open(path, "w") as f:
        f.write(generate_s8_case_header([("kS8Test1", "s", st1["s8"]),
                                         ("kMS8Test1", "m", mt1["s8"])]))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
