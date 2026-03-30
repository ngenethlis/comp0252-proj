#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bloom_filter.h"

// Partitioned Bloom filter: the m-bit array is divided into k equal partitions
// of (m/k) bits each. Hash function i maps strictly into partition i.
// This can yield different FPR characteristics compared to a standard BF.
template <typename Key = uint64_t, typename HashPolicy = DefaultHashPolicy>
class PartitionedBloomFilter {
   public:
    PartitionedBloomFilter(size_t num_bits, size_t num_hashes)
        : _bits(num_bits, false), _num_hashes(num_hashes), _partition_size(num_bits / num_hashes) {}

    void insert(const Key& key) {
        auto [h1, h2] = HashPolicy::hash_pair(key);
        for (size_t i = 0; i < _num_hashes; ++i) {
            _bits[nth_hash(h1, h2, i)] = true;
        }
    }

    [[nodiscard]] bool query(const Key& key) const {
        auto [h1, h2] = HashPolicy::hash_pair(key);
        for (size_t i = 0; i < _num_hashes; ++i) {
            if (!_bits[nth_hash(h1, h2, i)]) {
                return false;
            }
        }
        return true;
    }

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

    [[nodiscard]] size_t num_bits() const { return _bits.size(); }
    [[nodiscard]] size_t num_hashes() const { return _num_hashes; }
    [[nodiscard]] size_t partition_size() const { return _partition_size; }

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
    // Hash i maps into partition i: [i * partition_size, (i+1) * partition_size)
    [[nodiscard]] size_t nth_hash(uint64_t h1, uint64_t h2, size_t i) const {
        return i * _partition_size + ((h1 + i * h2) % _partition_size);
    }

    std::vector<bool> _bits;
    size_t _num_hashes;
    size_t _partition_size;
};
