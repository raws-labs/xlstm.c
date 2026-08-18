## What and why

<!-- What changes, and what problem it solves. If it fixes something measured,
     give the number. -->

## How it was verified

<!-- Which gates ran, and what they said. Tick what applies; delete what does not. -->

- [ ] `make test` and `make test-ref`
- [ ] `make test-neon` / `make test-cortexm` / `make test-esp` (if a backend changed)
- [ ] `make perf` (instruction counts unchanged, or re-recorded deliberately)
- [ ] `make mutants` (**required** if a tolerance, a bound, or `test/generate_reference.py` changed)
- [ ] `make test-docker-*` (if an adapter changed)

## If this is a performance change

<!-- Which board or backend, at which sizes, measured how. Instruction counts are
     a proxy for time and are blind to cache behaviour, so say which you have.
     A change with identical instruction counts has cost 10 percent on one part
     here before. -->

## Anything that got worse

<!-- Regressions, coverage given up, assumptions that are unverified. Say it here
     rather than leaving it to be found. -->
