CC      := gcc
CXX     := g++
CFLAGS  := -std=c99 -O2 -Wall -Wextra
CXXFLAGS:= -std=c++17 -O2 -Wall -Wextra

VENV    := .venv/bin/python3

.PHONY: all test reference clean \
        test-docker-ort test-docker-tvm test-docker-tflm test-docker-espdl

all: slstm.o mlstm.o

# --- Core objects ---

slstm.o: src/slstm.c include/slstm.h include/xlstm_util.h
	@$(CC) $(CFLAGS) -Iinclude -c $< -o $@

mlstm.o: src/mlstm.c include/mlstm.h include/xlstm_util.h
	@$(CC) $(CFLAGS) -Iinclude -c $< -o $@

# --- Core tests ---

slstm_test: test/slstm_test.cc slstm.o include/slstm.h test/reference_data.h
	@$(CXX) $(CXXFLAGS) -Iinclude -Itest -o $@ $< slstm.o -lm

mlstm_test: test/mlstm_test.cc mlstm.o include/mlstm.h test/reference_data.h
	@$(CXX) $(CXXFLAGS) -Iinclude -Itest -o $@ $< mlstm.o -lm

test: slstm_test mlstm_test
	@./slstm_test
	@./mlstm_test

# --- Docker integration tests ---

test-docker-ort:
	docker build -f test/adapters/onnxruntime/Dockerfile -t xlstm-test-ort .
	docker run --rm xlstm-test-ort

test-docker-tvm:
	docker build -f test/adapters/microtvm/Dockerfile -t xlstm-test-tvm .
	docker run --rm xlstm-test-tvm

test-docker-tflm:
	docker build -f test/adapters/tflm/Dockerfile -t xlstm-test-tflm .
	docker run --rm xlstm-test-tflm

test-docker-espdl:
	docker build -f test/adapters/esp-dl/Dockerfile -t xlstm-test-espdl .
	docker run --rm xlstm-test-espdl

# --- Reference data ---

test/reference_data.h: test/generate_reference.py
	@$(VENV) $<

reference: test/generate_reference.py
	@$(VENV) $<

# --- Cleanup ---

clean:
	@rm -f *.o slstm_test mlstm_test
