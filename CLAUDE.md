# Stashed Bloom Filter

A header-only C++20 library implementing a standard Bloom filter and a stash-augmented variant that aims to reduce false positive rates by diverting high-collision inserts to a secondary structure.

## Build & test

```bash
cmake -B build && cmake --build build
./build/tests                  # run tests (33 tests)
./build/bench                  # run benchmarks (insert/query throughput)
./build/main                   # run all experiments (exp1-exp6)
./build/main exp6              # run password experiment only
./build/main demo              # interactive breached-password querier
./build/main demo path/to/pw   # use custom password file
```

## Project layout

```
include/                   Header-only library (all templates, no .cpp files)
  bloom_filter.h             BloomFilter<Key, HashPolicy> + DefaultHashPolicy
  partitioned_bloom_filter.h PartitionedBloomFilter<Key, HashPolicy>
  prob_bool.h                ProbBool enum {True, Maybe, False}
  stash_set.h                StashSet CRTP interface
  bloom_filter_stash.h       BloomFilterStash (secondary BF as stash)
  linear_probing_stash.h     LinearProbingStash (deterministic hash table stash)
  stashed_bloom_filter.h     StashedBloomFilter<Key, HashPolicy, Stash>
  experiment_utils.h         Key generation, FPR measurement, file I/O
src/
  main.cpp                   Experiment runner (exp1-exp6 + demo mode)
tests/
  test_bloom_filter.cpp      All unit tests (33 tests)
bench/
  bench_bloom_filter.cpp     Insert/query throughput benchmarks
data/
  breached_passwords.txt     Password data (replace with real breach data)
```

## Architecture

### BloomFilter<Key, HashPolicy>
Standard Bloom filter using double hashing: `h_i(key) = (h1 + i * h2) mod m`. h2 is forced odd for coprimality with power-of-2 table sizes. Provides `insert`, `query`, `count_collisions`, and `bits_set`.

### StashSet (CRTP interface)
Abstract interface for stash implementations. Concrete types must implement:
- `do_insert(key) -> bool` (false if stash is full)
- `do_query(key) -> bool`
- `do_size_bits() -> size_t`

Two implementations:
- **BloomFilterStash** — secondary Bloom filter (never rejects inserts, has its own FPR)
- **LinearProbingStash** — fixed-capacity hash table storing 64-bit fingerprints (no false positives, but can fill up)

### StashedBloomFilter<Key, HashPolicy, Stash>
Wraps a primary BloomFilter + a StashSet. On insert, counts how many of the k bit positions are already set; if `count >= collision_threshold`, the key is diverted to the stash. If the stash is full, falls back to the primary filter.

Supports two stash modes (`StashMode`):
- **Positive** — stash stores "definitely in set" keys. Query returns `True` for stash hits, `Maybe` for primary hits, `False` otherwise.
- **Negative** — stash stores "definitely not in set" keys. Query returns `False` for stash hits, `Maybe` for primary hits.

### ProbBool
Three-valued query result: `True` (certain), `Maybe` (Bloom filter positive), `False` (certain negative). Helper `is_positive()` returns true for True/Maybe.

## Coding conventions

- C++20, Google style base with 4-space indent, 100 column limit (see `.clang-format`)
- Header-only library: all classes are fully templated in `include/`
- Linting: `.clang-tidy` enables bugprone, clang-analyzer, modernize, performance, readability checks
- Tests use simple assert macros (no external test framework); add new TEST() blocks to `tests/test_bloom_filter.cpp`
- Template parameters: `Key` (element type), `HashPolicy` (must provide `static hash_pair(const Key&) -> pair<uint64_t, uint64_t>`), `Stash` (a StashSet CRTP implementation)

## Adding a new stash set implementation

1. Create a class in `include/stash_set.h` inheriting `StashSet<YourClass, Key>` via CRTP
2. Implement `do_insert`, `do_query`, `do_size_bits`
3. Add tests in `tests/test_bloom_filter.cpp`
4. Use as template parameter: `StashedBloomFilter<Key, HashPolicy, YourClass>`

## Key design decisions

- **Double hashing** avoids needing k independent hash functions.
- **CRTP over virtual dispatch** for the StashSet interface — zero overhead at runtime since all types are resolved at compile time.
- **Collision threshold** is the core parameter to experiment with: how many of k bits already set should trigger stashing.
- **sizeof(stashed_bf) = sizeof(bf)**: the total bit budget is split between primary and stash, not increased.
