# Docker integration test for ONNX Runtime custom ops (sLSTM + mLSTM).
#
# Builds libxlstm_ort.so from source, then runs a Python test that creates
# ONNX models with custom op nodes and validates inference via ORT session.
#
# Usage:
#   docker build -f test/adapters/onnxruntime/Dockerfile -t xlstm-test-ort .
#   docker run --rm xlstm-test-ort

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
        python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir onnxruntime onnx numpy

# ORT pip package doesn't ship C++ headers — download from official release.
# Version must match the pip onnxruntime version.
RUN ORT_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)") && \
    echo "Downloading ORT $ORT_VERSION headers..." && \
    curl -sL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
        -o /tmp/ort.tgz && \
    mkdir -p /opt/ort && \
    tar xzf /tmp/ort.tgz -C /opt/ort --strip-components=1 && \
    rm /tmp/ort.tgz

WORKDIR /workspace
COPY include/ include/
COPY src/ src/
COPY adapters/onnxruntime/ adapters/onnxruntime/
COPY test/reference_data.json test/reference_data.json
COPY test/adapters/onnxruntime/test_ort.py test/adapters/onnxruntime/test_ort.py

RUN g++ -std=c++17 -shared -fPIC -O2 \
        -Iinclude \
        -Iadapters/onnxruntime \
        -I/opt/ort/include \
        adapters/onnxruntime/slstm_ort.cc \
        adapters/onnxruntime/mlstm_ort.cc \
        adapters/onnxruntime/xlstm_ort_register.cc \
        src/slstm.c src/mlstm.c \
        -lm \
        -o libxlstm_ort.so

CMD ["python3", "test/adapters/onnxruntime/test_ort.py"]
