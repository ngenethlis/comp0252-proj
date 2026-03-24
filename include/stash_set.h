#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "bloom_filter.h"

// StashSet interface (CRTP for static polymorphism).
//
// Implementations must provide:
//   bool do_insert(const Key& key)  — returns true if stored, false if full
//   bool do_query(const Key& key) const
//   size_t do_size_bits() const     — number of bits consumed
template <typename Derived, typename Key>
class StashSet {
   public:
    // Insert a key into the stash. Returns true if successful, false if full.
    bool insert(const Key& key) { return static_cast<Derived*>(this)->do_insert(key); }

    // Query whether a key is in the stash.
    bool query(const Key& key) const {
        return static_cast<const Derived*>(this)->do_query(key);
    }

    // Number of bits this stash occupies.
    size_t size_bits() const { return static_cast<const Derived*>(this)->do_size_bits(); }
};

// ---------------------------------------------------------------------------
// BloomFilterStash: uses a secondary Bloom filter as the stash.
// ---------------------------------------------------------------------------
template <typename Key = uint64_t, typename HashPolicy = DefaultHashPolicy>
class BloomFilterStash : public StashSet<BloomFilterStash<Key, HashPolicy>, Key> {
   public:
    BloomFilterStash(size_t num_bits, size_t num_hashes) : _bf(num_bits, num_hashes) {}

    bool do_insert(const Key& key) {
        _bf.insert(key);
        return true;  // Bloom filter insert never fails
    }

    bool do_query(const Key& key) const { return _bf.query(key); }

    size_t do_size_bits() const { return _bf.num_bits(); }

    const BloomFilter<Key, HashPolicy>& bloom_filter() const { return _bf; }

   private:
    BloomFilter<Key, HashPolicy> _bf;
};

// ---------------------------------------------------------------------------
// LinearProbingStash: a fixed-capacity hash set using linear probing.
// Stores hashed fingerprints of keys (deterministic, no false positives on
// stored keys, but has a capacity limit).
// ---------------------------------------------------------------------------
template <typename Key = uint64_t, typename HashPolicy = DefaultHashPolicy>
class LinearProbingStash : public StashSet<LinearProbingStash<Key, HashPolicy>, Key> {
   public:
    // capacity: max number of elements that can be stored.
    // Each slot stores a 64-bit fingerprint, so total bits = capacity * 64.
    explicit LinearProbingStash(size_t capacity)
        : _slots(capacity, kEmpty), _count(0), _capacity(capacity) {}

    bool do_insert(const Key& key) {
        if (_count >= _capacity) return false;

        uint64_t fp = fingerprint(key);
        size_t idx = fp % _capacity;

        for (size_t i = 0; i < _capacity; ++i) {
            size_t pos = (idx + i) % _capacity;
            if (_slots[pos] == kEmpty || _slots[pos] == fp) {
                _slots[pos] = fp;
                if (_slots[pos] == fp && i == 0) {
                    // might be re-insert of same key; only bump count for new
                }
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
            if (_slots[pos] == kEmpty) return false;
            if (_slots[pos] == fp) return true;
        }
        return false;
    }

    size_t do_size_bits() const { return _capacity * 64; }

    size_t count() const { return _count; }
    size_t capacity() const { return _capacity; }

   private:
    static constexpr uint64_t kEmpty = 0;

    static uint64_t fingerprint(const Key& key) {
        auto [h1, h2] = HashPolicy::hash_pair(key);
        // Combine both hashes; ensure non-zero so it's distinguishable from kEmpty.
        uint64_t fp = h1 ^ h2;
        return fp == kEmpty ? 1 : fp;
    }

    std::vector<uint64_t> _slots;
    size_t _count;
    size_t _capacity;
};
