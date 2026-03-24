#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bloom_filter.h"
#include "stash_set.h"

// LinearProbingStash: a fixed-capacity hash set using linear probing.
// Stores hashed fingerprints of keys (deterministic, no false positives on
// stored keys, but has a capacity limit).
template <typename Key = uint64_t, typename HashPolicy = DefaultHashPolicy>
class LinearProbingStash : public StashSet<LinearProbingStash<Key, HashPolicy>, Key> {
   public:
    // capacity: max number of elements that can be stored.
    // Each slot stores a 64-bit fingerprint, so total bits = capacity * 64.
    explicit LinearProbingStash(size_t capacity)
        : _slots(capacity, kEmpty), _capacity(capacity) {}

    bool do_insert(const Key& key) {
        if (_count >= _capacity) {
            return false;
        }

        uint64_t fp = fingerprint(key);
        size_t idx = fp % _capacity;

        for (size_t i = 0; i < _capacity; ++i) {
            size_t pos = (idx + i) % _capacity;
            if (_slots[pos] == kEmpty || _slots[pos] == fp) {
                _slots[pos] = fp;
                ++_count;
                return true;
            }
        }
        return false;  // table full (shouldn't happen if count < capacity)
    }

    bool do_query(const Key& key) const {
        uint64_t fp = fingerprint(key);
        size_t idx = fp % _capacity;

        for (size_t i = 0; i < _capacity; ++i) {
            size_t pos = (idx + i) % _capacity;
            if (_slots[pos] == kEmpty) {
                return false;
            }
            if (_slots[pos] == fp) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] size_t do_size_bits() const { return _capacity * 64; }

    [[nodiscard]] size_t count() const { return _count; }
    [[nodiscard]] size_t capacity() const { return _capacity; }

   private:
    static constexpr uint64_t kEmpty = 0;

    static uint64_t fingerprint(const Key& key) {
        auto [h1, h2] = HashPolicy::hash_pair(key);
        // Combine both hashes; ensure non-zero so it's distinguishable from kEmpty.
        uint64_t fp = h1 ^ h2;
        return fp == kEmpty ? 1 : fp;
    }

    std::vector<uint64_t> _slots;
    size_t _count = 0;
    size_t _capacity;
};
