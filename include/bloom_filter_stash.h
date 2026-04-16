#pragma once

#include <cstddef>
#include <cstdint>

#include "bloom_filter.h"
#include "stash_set.h"

/**
 * @file bloom_filter_stash.h
 * @brief Stash implementation backed by a secondary Bloom filter.
 */

/**
 * @brief Stash backend that stores diverted keys in a secondary Bloom filter.
 *
 * This stash is probabilistic: stash hits can be false positives.
 *
 * @tparam Key Key type accepted by stash operations.
 * @tparam HashPolicy Hash policy used by the internal Bloom filter.
 */
template <typename Key = uint64_t, typename HashPolicy = DefaultHashPolicy>
class BloomFilterStash : public StashSet<BloomFilterStash<Key, HashPolicy>, Key> {
   public:
    /**
     * @brief Constructs a Bloom-filter stash.
     * @param num_bits Number of bits in the stash Bloom filter.
     * @param num_hashes Number of hash probes for stash operations.
     */
    BloomFilterStash(size_t num_bits, size_t num_hashes) : _bf(num_bits, num_hashes) {}

    /**
     * @brief Inserts a key into the stash.
     * @return Always `true` because Bloom filter insertion cannot fail.
     */
    bool do_insert(const Key& key) {
        _bf.insert(key);
        return true;  // Bloom filter insert never fails
    }

    /**
     * @brief Queries stash membership.
     * @return Bloom-filter query result (may be false positive).
     */
    bool do_query(const Key& key) const { return _bf.query(key); }

    /** @brief Returns stash size in bits. */
    [[nodiscard]] size_t do_size_bits() const { return _bf.num_bits(); }

    /** @brief Indicates probabilistic stash behavior. */
    [[nodiscard]] bool do_is_probabilistic() const { return true; }

    /** @brief Accesses the underlying Bloom filter instance. */
    [[nodiscard]] const BloomFilter<Key, HashPolicy>& bloom_filter() const { return _bf; }

   private:
    BloomFilter<Key, HashPolicy> _bf;
};
