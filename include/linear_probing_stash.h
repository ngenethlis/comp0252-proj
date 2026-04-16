#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bloom_filter.h"
#include "stash_set.h"

/**
 * @file linear_probing_stash.h
 * @brief Deterministic fixed-capacity stash using linear probing.
 */

/**
 * @brief Stash backend implemented as a linear-probing hash table.
 *
 * Stores 64-bit key fingerprints and therefore has a fixed capacity. Query
 * semantics are deterministic with respect to stored fingerprints.
 *
 * @tparam Key Key type accepted by stash operations.
 * @tparam HashPolicy Hash policy used to derive fingerprints.
 */
template <typename Key = uint64_t, typename HashPolicy = DefaultHashPolicy>
class LinearProbingStash : public StashSet<LinearProbingStash<Key, HashPolicy>, Key> {
   public:
    /**
     * @brief Constructs a linear-probing stash.
     * @param capacity Maximum number of slots (and fingerprints) available.
     */
    explicit LinearProbingStash(size_t capacity) : _slots(capacity, kEmpty), _capacity(capacity) {}

    /**
     * @brief Inserts a key fingerprint into the stash.
     * @return `true` if inserted or already present, `false` if full.
     */
    bool do_insert(const Key& key) {
        if (_count >= _capacity) {
            return false;
        }

        uint64_t fp = fingerprint(key);
        size_t idx = fp % _capacity;

        for (size_t i = 0; i < _capacity; ++i) {
            size_t pos = (idx + i) % _capacity;
            if (_slots[pos] == fp) {
                return true;
            }
            if (_slots[pos] == kEmpty) {
                _slots[pos] = fp;
                ++_count;
                return true;
            }
        }
        return false;  // table full (shouldn't happen if count < capacity)
    }

    /**
     * @brief Queries whether a key fingerprint is present.
     */
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

    /** @brief Returns stash size in bits (`capacity * 64`). */
    [[nodiscard]] size_t do_size_bits() const { return _capacity * 64; }

    /** @brief Indicates deterministic stash behavior. */
    [[nodiscard]] bool do_is_probabilistic() const { return false; }

    /** @brief Returns number of occupied slots. */
    [[nodiscard]] size_t count() const { return _count; }
    /** @brief Returns total slot capacity. */
    [[nodiscard]] size_t capacity() const { return _capacity; }

   private:
    static constexpr uint64_t kEmpty = 0;

    /**
     * @brief Computes a non-zero 64-bit fingerprint for a key.
     */
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
