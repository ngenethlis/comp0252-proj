# Stashed Bloom Filter

Header-only C++20 implementations of:
1. A standard Bloom filter.
2. A partitioned/blocked Bloom filter baseline.
3. A stash-augmented Bloom filter with tri-valued query semantics.

The project includes experiments (`exp1`-`exp8`), benchmarks, and an interactive
breached-password demo.

## Features

- Header-only API in `include/` (no library `.cpp` files).
- `BloomFilter<Key, HashPolicy>` with double hashing.
- `BlockedBloomFilter<Key, HashPolicy>` partitioned baseline.
- `StashedBloomFilter<Key, HashPolicy, Stash>` with:
  - `StashMode::Positive` (stash as positive evidence),
  - `StashMode::Negative` (stash as negative override).
- Two stash backends:
  - `BloomFilterStash` (probabilistic),
  - `LinearProbingStash` (deterministic, capacity-limited).
- `ProbBool` query outputs: `True`, `Maybe`, `False`.

## Requirements

- CMake >= 3.16
- C++20 compiler
- Python 3 (for plotting/docs scripts)
- Optional for docs: Doxygen + Sphinx
- Optional for developer checks: clang-format + clang-tidy

## Build and run

```bash
cmake -B build -S .
cmake --build build

./build/tests
./build/bench
./build/main
```

### Run specific experiment modes

```bash
./build/main exp1
./build/main exp5 data/hibp-100k.txt --seed=42
./build/main exp7 data/hibp-100k.txt --seed=42
./build/main exp8 data/hibp-100k.txt --seed=42
```

Supported modes: `all`, `exp1`, `exp2`, `exp3`, `exp4`, `exp5`, `exp6`, `exp7`,
`exp8`, `demo`.

### Interactive demo

```bash
./build/main demo
# or custom file:
./build/main demo data/breached_passwords.txt
```

## Reproducing experiment outputs

```bash
./scripts/run_experiments.sh
```

This builds the project, runs `exp1`-`exp8` across multiple seeds, and writes
aggregated CSVs to `results/`.

To generate plots:

```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy matplotlib
.venv/bin/python3 scripts/plot.py
```

Plots are written to `results/plots/`.

## Using this library in your code

Because this is header-only, include from `include/` and compile with C++20.

```cpp
#include "bloom_filter.h"
#include "linear_probing_stash.h"
#include "stashed_bloom_filter.h"
```

If consuming this repository via CMake, link against the `bloom_filter`
interface target to inherit include paths:

```cmake
add_subdirectory(path/to/stashed_bloom_filter)
target_link_libraries(your_target PRIVATE bloom_filter)
```

## API documentation (Sphinx + Doxygen)

API docs are generated from Doxygen-style comments in `include/*.h` and rendered
with Sphinx/Breathe.

```bash
# one-time system dependency (Ubuntu/Debian):
sudo apt-get update && sudo apt-get install -y doxygen

python3 -m venv .venv
.venv/bin/pip install -r docs/requirements.txt
.venv/bin/sphinx-build -b html -W docs docs/_build/html
```

Open `docs/_build/html/index.html`.

If Sphinx and Doxygen are available, you can also use:

```bash
cmake -B build -S .
cmake --build build --target docs
```

## Repository layout

```text
include/   Header-only library API
src/       Experiment and demo runner
tests/     Unit tests
bench/     Throughput benchmark binary
scripts/   Repro and plotting scripts
data/      Input datasets for demo/experiments
docs/      Sphinx + Doxygen configuration
results/   Generated CSVs and plots
```

## Contributing

1. Fork the repo and create a feature branch.
2. Keep changes focused and update docs/tests with code changes.
3. Run local checks before opening a PR:

```bash
cmake -B build -S . -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
./build/tests

find include src tests bench \( -name '*.h' -o -name '*.cpp' \) | xargs clang-format --dry-run --Werror
find include -name '*.h' | xargs clang-tidy -p build
sphinx-build -b html -W docs docs/_build/html
```

4. Open a pull request with a clear description of the motivation and changes.

## License

No license file is currently included in this repository.
