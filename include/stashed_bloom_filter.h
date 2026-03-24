#pragma once

#include <cstddef>
#include <cstdint>

#include "bloom_filter.h"
#include "bloom_filter_stash.h"
#include "prob_bool.h"
#include "stash_set.h"

// StashMode controls the semantics of the stash:
//   Positive — stash stores "definitely yes" keys (high-collision keys go here)
//   Negative — stash stores "definitely no" keys (keys that didn't collide)
enum class StashMode { Positive, Negative };

// StashedBloomFilter<Key, HashPolicy, Stash>
//
// A Bloom filter augmented with a secondary stash structure. Keys whose
// insertion would cause >= collision_threshold bit collisions in the primary
// filter are diverted to the stash instead.
//
// Template parameters:
//   Key        — element type
//   HashPolicy — must provide static hash_pair(const Key&)
//   Stash      — a StashSet implementation (e.g. BloomFilterStash, LinearProbingStash)
template <typename Key = uint64_t, typename HashPolicy = DefaultHashPolicy,
          typename Stash = BloomFilterStash<Key, HashPolicy>>
class StashedBloomFilter {
   public:
    // primary_bits:          bits allocated to the primary Bloom filter
    // num_hashes:            hash function count for the primary filter
    // stash:                 pre-constructed stash (caller controls its size)
    // collision_threshold:   number of already-set bit positions that triggers stashing
    // mode:                  Positive or Negative stash semantics
    StashedBloomFilter(size_t primary_bits, size_t num_hashes, Stash stash,
                       size_t collision_threshold, StashMode mode = StashMode::Positive)
        : _primary(primary_bits, num_hashes),
          _stash(std::move(stash)),
          _collision_threshold(collision_threshold),
          _mode(mode) {}

    void insert(const Key& key) {
        size_t collisions = _primary.count_collisions(key);
        if (collisions >= _collision_threshold) {
            // High collision — try stash first
            if (!_stash.insert(key)) {
                // Stash full, fall back to primary
                _primary.insert(key);
            } else {
                ++_stash_count;
            }
        } else {
            _primary.insert(key);
        }
    }

    [[nodiscard]] ProbBool query(const Key& key) const {
        bool in_stash = _stash.query(key);
        bool in_primary = _primary.query(key);

        if (_mode == StashMode::Positive) {
            // Stash holds "definitely yes" keys.
            if (in_stash) {
                return ProbBool::True;
            }
            if (in_primary) {
                return ProbBool::Maybe;
            }
            return ProbBool::False;
        }
        // Stash holds "definitely no" keys.
        if (in_stash) {
            return ProbBool::False;
        }
        if (in_primary) {
            return ProbBool::Maybe;
        }
        return ProbBool::False;
    }

    // Convenience: returns true if query is True or Maybe.
    [[nodiscard]] bool query_bool(const Key& key) const { return is_positive(query(key)); }

    [[nodiscard]] size_t primary_bits() const { return _primary.num_bits(); }
    [[nodiscard]] size_t stash_bits() const { return _stash.size_bits(); }
    [[nodiscard]] size_t total_bits() const { return primary_bits() + stash_bits(); }
    [[nodiscard]] size_t collision_threshold() const { return _collision_threshold; }
    [[nodiscard]] size_t stash_count() const { return _stash_count; }
    [[nodiscard]] StashMode mode() const { return _mode; }

    [[nodiscard]] const BloomFilter<Key, HashPolicy>& primary() const { return _primary; }
    [[nodiscard]] const Stash& stash() const { return _stash; }

   private:
    BloomFilter<Key, HashPolicy> _primary;
    Stash _stash;
    size_t _collision_threshold;
    StashMode _mode;
    size_t _stash_count = 0;
};
