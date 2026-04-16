#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bloom_filter.h"

/**
 * @file partitioned_bloom_filter.h
 * @brief Partitioned Bloom filter baseline.
 */

/**
 * @brief Bloom filter variant that partitions the bit array into hash-local
 * segments.
 *
 * The `m`-bit table is divided into `k` equal partitions of size `m / k`.
 * Hash probe `i` maps only into partition `i`.
 *
 * @tparam Key Type of keys inserted and queried.
 * @tparam HashPolicy Hash policy providing
 * `static std::pair<uint64_t, uint64_t> hash_pair(const Key&)`.
 */
template <typename Key = uint64_t, typename HashPolicy = DefaultHashPolicy>
class PartitionedBloomFilter {
   public:
    /**
     * @brief Constructs a partitioned Bloom filter.
     * @param num_bits Total bits across all partitions.
     * @param num_hashes Number of partitions and probes per operation.
     */
    PartitionedBloomFilter(size_t num_bits, size_t num_hashes)
        : _bits(num_bits, false), _num_hashes(num_hashes), _partition_size(num_bits / num_hashes) {}

    /**
     * @brief Inserts a key into the filter.
     * @param key Key to insert.
     */
    void insert(const Key& key) {
        auto [h1, h2] = HashPolicy::hash_pair(key);
        for (size_t i = 0; i < _num_hashes; ++i) {
            _bits[nth_hash(h1, h2, i)] = true;
        }
    }

    /**
     * @brief Queries key membership.
     * @param key Key to query.
     * @return `true` if all probed bits are set (possible false positive),
     * `false` otherwise.
     */
    [[nodiscard]] bool query(const Key& key) const {
        auto [h1, h2] = HashPolicy::hash_pair(key);
        for (size_t i = 0; i < _num_hashes; ++i) {
            if (!_bits[nth_hash(h1, h2, i)]) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Counts how many probe positions are already set for a key.
     * @param key Key to evaluate.
     * @return Number of set bits among partition-constrained probe positions.
     */
    [[nodiscard]] size_t count_collisions(const Key& key) const {
        auto [h1, h2] = HashPolicy::hash_pair(key);
        size_t count = 0;
        for (size_t i = 0; i < _num_hashes; ++i) {
            if (_bits[nth_hash(h1, h2, i)]) {
                ++count;
            }
        }
        return count;
    }

    /** @brief Returns the total number of bits in the table. */
    [[nodiscard]] size_t num_bits() const { return _bits.size(); }
    /** @brief Returns the number of hash probes per operation. */
    [[nodiscard]] size_t num_hashes() const { return _num_hashes; }
    /** @brief Returns bits per partition (`num_bits() / num_hashes()`). */
    [[nodiscard]] size_t partition_size() const { return _partition_size; }

    /** @brief Counts currently set bits in the table. */
    [[nodiscard]] size_t bits_set() const {
        size_t count = 0;
        for (bool b : _bits) {
            if (b) {
                ++count;
            }
        }
        return count;
    }

   private:
    /**
     * @brief Computes the i-th index constrained to partition i.
     */
    [[nodiscard]] size_t nth_hash(uint64_t h1, uint64_t h2, size_t i) const {
        return i * _partition_size + ((h1 + i * h2) % _partition_size);
    }

    std::vector<bool> _bits;
    size_t _num_hashes;
    size_t _partition_size;
};
