# Docker integration test for microTVM adapter (sLSTM + mLSTM).
#
# Builds a shared library that registers packed functions via
# TVM_REGISTER_GLOBAL, then Python test calls them through tvm.runtime
# with real DLTensor objects.
#
# Usage:
#   docker build -f test/adapters/microtvm/Dockerfile -t xlstm-test-tvm .
#   docker run --rm xlstm-test-tvm

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git cmake \
        python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir numpy

# Build TVM runtime from source (pip TVM may lack C++ headers).
# Only init the submodules needed for runtime (dlpack, dmlc-core).
RUN git clone --depth 1 --branch v0.18.0 https://github.com/apache/tvm.git /opt/tvm-src && \
    cd /opt/tvm-src && \
    git submodule update --init --depth 1 3rdparty/dlpack 3rdparty/dmlc-core 3rdparty/libbacktrace && \
    mkdir build && cd build && \
    cp ../cmake/config.cmake . && \
    cmake -DUSE_LLVM=OFF -DUSE_CUDA=OFF -DUSE_OPENCL=OFF \
          -DUSE_LIBBACKTRACE=OFF -DBACKTRACE_ON_SEGFAULT=OFF .. && \
    make tvm_runtime -j$(nproc) && \
    cp libtvm_runtime.so /usr/local/lib/ && \
    ldconfig

# Install TVM Python package
RUN cd /opt/tvm-src/python && pip install --no-cache-dir -e .

ENV TVM_HOME=/opt/tvm-src

WORKDIR /workspace
COPY include/ include/
COPY src/ src/
COPY adapters/microtvm/ adapters/microtvm/
COPY test/reference_data.json test/reference_data.json
COPY test/adapters/microtvm/tvm_register_wrapper.cc test/adapters/microtvm/tvm_register_wrapper.cc
COPY test/adapters/microtvm/test_tvm.py test/adapters/microtvm/test_tvm.py

# Build shared library: adapter + core + TVM registration wrapper
# Build shared library WITHOUT linking -ltvm_runtime.  TVM symbols
# (TVMFuncRegisterGlobal etc.) are resolved at dlopen time from the
# TVM runtime already loaded by the Python process.  Linking a second
# copy would cause duplicate-registration crashes.
RUN g++ -std=c++17 -shared -fPIC -O2 \
        -Iinclude \
        -Iadapters/microtvm \
        -I/opt/tvm-src/include \
        -I/opt/tvm-src/3rdparty/dlpack/include \
        -I/opt/tvm-src/3rdparty/dmlc-core/include \
        test/adapters/microtvm/tvm_register_wrapper.cc \
        adapters/microtvm/slstm_tvm.c \
        adapters/microtvm/mlstm_tvm.c \
        src/slstm.c src/mlstm.c \
        -lm \
        -o libxlstm_tvm.so

CMD ["python3", "test/adapters/microtvm/test_tvm.py"]
