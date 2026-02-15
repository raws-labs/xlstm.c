# Docker integration test for TFLM custom ops (sLSTM + mLSTM).
#
# Clones tflite-micro, uses its Makefile to fetch third-party deps and
# discover the correct source list, then compiles our test against it.
#
# Usage:
#   docker build -f test/adapters/tflm/Dockerfile -t xlstm-test-tflm .
#   docker run --rm xlstm-test-tflm

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git unzip wget curl \
        python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir flatbuffers numpy Pillow

# Clone tflite-micro and fetch third-party deps (flatbuffers etc.)
RUN git clone --depth 1 https://github.com/tensorflow/tflite-micro.git /opt/tflm && \
    cd /opt/tflm && \
    make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads

WORKDIR /workspace
COPY include/ include/
COPY src/ src/
COPY adapters/tflm/ adapters/tflm/
COPY test/reference_data.h test/reference_data.h
COPY test/reference_data.json test/reference_data.json
COPY test/adapters/tflm/ test/adapters/tflm/

# Generate model headers from reference data
RUN python3 test/adapters/tflm/generate_model.py

# Use TFLM's Makefile to build a static library, then link our test against it.
RUN cd /opt/tflm && \
    make -f tensorflow/lite/micro/tools/make/Makefile microlite -j$(nproc)

# Discover include paths from the microlite build and downloaded deps.
RUN TFLM=/opt/tflm && \
    TFLM_DL=$TFLM/tensorflow/lite/micro/tools/make/downloads && \
    TFLM_LIB=$(find $TFLM -name 'libtensorflow-microlite.a' | head -1) && \
    echo "TFLM lib: $TFLM_LIB" && \
    TFLM_GENDIR=$(dirname $(dirname $TFLM_LIB)) && \
    g++ -std=c++17 -O2 -DTF_LITE_STATIC_MEMORY \
        -I$TFLM \
        -I$TFLM_DL/flatbuffers/include \
        -I$TFLM_DL/gemmlowp \
        -I$TFLM_DL/ruy \
        -I$TFLM_GENDIR/genfiles \
        -Iinclude -Iadapters/tflm -Itest -Itest/adapters/tflm \
        test/adapters/tflm/test_tflm.cc \
        adapters/tflm/slstm_tflm.cc \
        adapters/tflm/mlstm_tflm.cc \
        src/slstm.c src/mlstm.c \
        $TFLM_LIB \
        -lm -o tflm_integration_test

CMD ["./tflm_integration_test"]
