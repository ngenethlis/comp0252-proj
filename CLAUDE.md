# Stashed Bloom Filter

A header-only C++20 library implementing a standard Bloom filter and a stash-based variant that aims to reduce false positive rates by diverting high-collision inserts to a secondary filter.

## Build

```bash
cmake -B build && cmake --build build
./build/tests   # run tests
./build/bench   # run benchmarks
./build/main    # main executable
```

## Project layout

```
include/           Header-only library
  bloom_filter.h           BloomFilter<Key, HashPolicy>
  stashed_bloom_filter.h   StashedBloomFilter<Key, HashPolicy>
src/               Executables
  main.cpp                 Experiment runner
tests/             Tests
bench/             Benchmarks
data/              Experiment datasets
```

## Usage

### Basic Bloom filter

```cpp
#include "bloom_filter.h"

BloomFilter bf(10000, 7);  // 10000 bits, 7 hash functions
bf.insert(42);
bf.query(42);  // true
```

### Stashed Bloom filter

```cpp
#include "stashed_bloom_filter.h"

// 10000 total bits, 20% to stash, 7 primary hashes, 5 stash hashes,
// stash if >= 5 of 7 bit positions already set
StashedBloomFilter sbf(10000, 0.2, 7, 5, 5);
sbf.insert(42);
sbf.query(42);  // true
```

### Custom key types

Both filters are templated as `BloomFilter<Key, HashPolicy>`. To use a custom key type, define a hash policy struct with a static `hash_pair` method:

```cpp
struct MyHashPolicy {
    static std::pair<uint64_t, uint64_t> hash_pair(const MyType& key) {
        uint64_t h1 = /* ... */;
        uint64_t h2 = /* ... */ | 1;  // h2 must be odd
        return {h1, h2};
    }
};

BloomFilter<MyType, MyHashPolicy> bf(10000, 7);
```

`DefaultHashPolicy` provides `hash_pair` overloads for `uint64_t` and `std::string`.

## Key design decisions

- **Double hashing**: h_i(key) = (h1 + i * h2) mod m. h2 is forced odd so it's coprime with power-of-2 table sizes.
- **Collision threshold**: `StashedBloomFilter` counts how many of the k primary bit positions are already set. If count >= threshold, the element goes to the stash. The optimal threshold is an open research question for this project.
- **Stash is a Bloom filter**: the stash is itself a Bloom filter (not exact storage), so it also has a false positive rate. The hypothesis is that splitting the bit budget can still yield a lower combined FPR.
- **Header-only**: both filters are fully templated, so there are no .cpp files for the library — just link the INTERFACE CMake target.
