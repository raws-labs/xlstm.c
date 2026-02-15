# Contributing

Contributions are welcome. This project is Apache-2.0 licensed — by submitting
a pull request you agree that your contribution will be licensed under the same
terms.

## Getting started

```bash
make test              # core sLSTM + mLSTM tests (requires gcc, g++)
make test-docker-ort   # ONNX Runtime integration test
make test-docker-tvm   # Apache TVM integration test
make test-docker-tflm  # TensorFlow Lite Micro integration test
make test-docker-espdl # ESP-DL integration test (QEMU)
```

`make test` is fast (seconds). Docker integration tests are slower and require
Docker. CI runs all of them on every PR.

## Workflow

1. Fork and create a feature branch
2. Make your changes
3. Run `make test` locally — all core tests must pass
4. Run the relevant `make test-docker-*` if you touched an adapter
5. Open a PR against `main`

## Code style

- Core library: **C99**, no dependencies beyond `math.h`
- Adapters: match the target framework's conventions (C++ for TFLM/ORT/ESP-DL,
  C for microTVM)
- No dynamic allocation in the core — callers provide scratch buffers
- Keep adapters thin: unpack tensors, call core, return

## Regenerating reference data

If you change the core math:

```bash
make reference         # requires Python with torch + xlstm
make test              # verify against new golden values
```

This regenerates both `test/reference_data.h` (C tests) and
`test/reference_data.json` (Python/Docker tests) from the NX-AI/xlstm
reference implementation.

## Reporting issues

Open an issue on GitHub. Include:
- What you expected vs what happened
- Minimal reproduction steps
- Compiler/OS/framework versions if relevant
