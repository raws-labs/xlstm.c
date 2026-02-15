#!/usr/bin/env python3
"""Generate minimal .tflite FlatBuffer models containing sLSTM/mLSTM custom ops.

Outputs C header files with model byte arrays, matching the pattern used by
tflite-micro upstream tests.

These models contain no weights — all tensors are inputs/outputs. The custom
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
        tensor_specs: list of (name, shape, type) tuples
        input_indices: list of tensor indices that are op inputs
        output_indices: list of tensor indices that are op outputs

    Returns:
        bytes: Serialized .tflite FlatBuffer
    """
    builder = flatbuffers.Builder(1024)

    # --- Strings ---
    op_name_off = builder.CreateString(op_name)
    tensor_name_offs = []
    for name, _, _ in tensor_specs:
        tensor_name_offs.append(builder.CreateString(name))

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

    # --- Tensors ---
    tensor_offs = []
    shape_vecs = []
    for name, shape, ttype in tensor_specs:
        builder.StartVector(4, len(shape), 4)
        for dim in reversed(shape):
            builder.PrependInt32(dim)
        shape_vecs.append(builder.EndVector())

    for i, (name, shape, ttype) in enumerate(tensor_specs):
        # Table Tensor { shape:[int], type:TensorType, buffer:uint,
        #                name:string, quantization:QuantizationParameters,
        #                is_variable:bool }
        builder.StartObject(6)
        builder.PrependUOffsetTRelativeSlot(0, shape_vecs[i], 0)  # shape
        builder.PrependInt8Slot(1, ttype, 0)  # type
        builder.PrependUint32Slot(2, i + 1, 0)  # buffer index (0 = sentinel)
        builder.PrependUOffsetTRelativeSlot(3, tensor_name_offs[i], 0)  # name
        # quantization: slot 4, skip (default 0)
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

    # --- Buffers (empty — all data provided at runtime) ---
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
    lines.append(f"/* Auto-generated — do not edit. */\n")
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


def main():
    with open(REF_PATH) as f:
        ref = json.load(f)

    # Use test1 dimensions for the model (single timestep)
    st1 = ref["slstm"]["test1"]
    slstm_bytes = generate_slstm_model(st1["B"], st1["T"], st1["I"], st1["H"])
    header = model_to_c_header(slstm_bytes, "slstm_model_data", "SLSTM_MODEL_DATA_H_")
    path = os.path.join(SCRIPT_DIR, "slstm_model_data.h")
    with open(path, "w") as f:
        f.write(header)
    print(f"Wrote {path} ({len(slstm_bytes)} bytes)")

    mt1 = ref["mlstm"]["test1"]
    mlstm_bytes = generate_mlstm_model(mt1["B"], mt1["T"], mt1["I"], mt1["H"])
    header = model_to_c_header(mlstm_bytes, "mlstm_model_data", "MLSTM_MODEL_DATA_H_")
    path = os.path.join(SCRIPT_DIR, "mlstm_model_data.h")
    with open(path, "w") as f:
        f.write(header)
    print(f"Wrote {path} ({len(mlstm_bytes)} bytes)")


if __name__ == "__main__":
    main()
