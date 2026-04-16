#pragma once

#include <cstddef>
#include <cstdint>

#include "bloom_filter.h"
#include "bloom_filter_stash.h"
#include "prob_bool.h"
#include "stash_set.h"

/**
 * @file stashed_bloom_filter.h
 * @brief Stash-augmented Bloom filter with tri-valued query semantics.
 */

/**
 * @brief Defines how stash entries are interpreted at query time.
 */
enum class StashMode {
    Positive, /**< Stash hit indicates positive evidence. */
    Negative, /**< Stash hit indicates known negative override. */
};

/**
 * @brief Bloom filter augmented by a secondary stash structure.
 *
 * In positive mode, inserts are routed by collision count: keys that would
 * collide with at least `collision_threshold` set bits in the primary filter
 * are diverted to the stash. If stash insertion fails, insertion falls back to
 * the primary Bloom filter.
 *
 * In negative mode, regular inserts go to the primary filter; the stash can be
 * populated with known negatives that were false positives in the primary.
 *
 * @tparam Key Element type.
 * @tparam HashPolicy Hash policy used by the primary Bloom filter.
 * @tparam Stash A `StashSet` implementation.
 */
template <typename Key = uint64_t, typename HashPolicy = DefaultHashPolicy,
          typename Stash = BloomFilterStash<Key, HashPolicy>>
class StashedBloomFilter {
   public:
    /**
     * @brief Constructs a stashed Bloom filter.
     * @param primary_bits Bit budget for the primary Bloom filter.
     * @param num_hashes Number of hash probes in the primary filter.
     * @param stash Pre-constructed stash instance (controls stash budget/type).
     * @param collision_threshold Stashing threshold on primary collision count.
     * @param mode Stash semantics mode.
     */
    StashedBloomFilter(size_t primary_bits, size_t num_hashes, Stash stash,
                       size_t collision_threshold, StashMode mode = StashMode::Positive)
        : _primary(primary_bits, num_hashes),
          _stash(std::move(stash)),
          _collision_threshold(collision_threshold),
          _mode(mode) {}

    /**
     * @brief Inserts a known-positive key.
     *
     * In `StashMode::Positive`, high-collision keys are diverted to the stash.
     * In `StashMode::Negative`, all inserts go to the primary Bloom filter.
     */
    void insert(const Key& key) {
        if (_mode == StashMode::Negative) {
            _primary.insert(key);
            return;
        }
        // Positive mode: collision-threshold routing
        size_t collisions = _primary.count_collisions(key);
        if (collisions >= _collision_threshold) {
            if (!_stash.insert(key)) {
                _primary.insert(key);
            } else {
                ++_stash_count;
            }
        } else {
            _primary.insert(key);
        }
    }

    /**
     * @brief Attempts to stash a known-negative key in negative mode workflows.
     *
     * The key is inserted into the stash only when it is currently a positive
     * in the primary Bloom filter.
     *
     * @return `true` if the key was a primary false positive and was stashed.
     */
    bool insert_negative(const Key& key) {
        if (_primary.query(key)) {
            if (_stash.insert(key)) {
                ++_stash_count;
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Bulk-populates negative stash entries from an iterator range.
     *
     * @tparam InputIt Iterator over keys.
     * @param begin Start of key range.
     * @param end End of key range.
     * @return Number of keys successfully stashed via `insert_negative`.
     */
    template <typename InputIt>
    size_t populate_negative_stash(InputIt begin, InputIt end) {
        size_t count = 0;
        for (auto it = begin; it != end; ++it) {
            if (insert_negative(*it)) {
                ++count;
            }
        }
        return count;
    }

    /**
     * @brief Queries key membership with tri-valued semantics.
     *
     * Positive mode:
     * - deterministic stash hit => `ProbBool::True`
     * - probabilistic stash hit => `ProbBool::Maybe`
     * - primary hit => `ProbBool::Maybe`
     * - otherwise => `ProbBool::False`
     *
     * Negative mode:
     * - stash hit => `ProbBool::False`
     * - primary hit => `ProbBool::Maybe`
     * - otherwise => `ProbBool::False`
     */
    [[nodiscard]] ProbBool query(const Key& key) const {
        bool in_stash = _stash.query(key);
        bool in_primary = _primary.query(key);

        if (_mode == StashMode::Positive) {
            // Positive mode semantics depend on stash determinism.
            // Deterministic stash hit => True, probabilistic stash hit => Maybe.
            if (in_stash) {
                return _stash.is_probabilistic() ? ProbBool::Maybe : ProbBool::True;
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

    /** @brief Convenience wrapper equivalent to `is_positive(query(key))`. */
    [[nodiscard]] bool query_bool(const Key& key) const { return is_positive(query(key)); }

    /** @brief Returns primary Bloom filter bit budget. */
    [[nodiscard]] size_t primary_bits() const { return _primary.num_bits(); }
    /** @brief Returns stash bit budget. */
    [[nodiscard]] size_t stash_bits() const { return _stash.size_bits(); }
    /** @brief Returns total bit budget (`primary_bits + stash_bits`). */
    [[nodiscard]] size_t total_bits() const { return primary_bits() + stash_bits(); }
    /** @brief Returns collision threshold used in positive mode routing. */
    [[nodiscard]] size_t collision_threshold() const { return _collision_threshold; }
    /** @brief Returns number of successful stash insertions. */
    [[nodiscard]] size_t stash_count() const { return _stash_count; }
    /** @brief Returns current stash mode. */
    [[nodiscard]] StashMode mode() const { return _mode; }

    /** @brief Accesses the primary Bloom filter. */
    [[nodiscard]] const BloomFilter<Key, HashPolicy>& primary() const { return _primary; }
    /** @brief Accesses the stash instance. */
    [[nodiscard]] const Stash& stash() const { return _stash; }

   private:
    BloomFilter<Key, HashPolicy> _primary;
    Stash _stash;
    size_t _collision_threshold;
    StashMode _mode;
    size_t _stash_count = 0;
};
