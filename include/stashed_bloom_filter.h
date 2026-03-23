#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "bloom_filter.h"

template <typename Key = uint64_t, typename HashPolicy = DefaultHashPolicy>
class StashedBloomFilter {
   public:
    // total_bits: combined bit budget for primary + stash
    // stash_fraction: fraction of total_bits allocated to the stash (0.0–1.0)
    // num_hashes: number of hash functions for the primary filter
    // stash_hashes: number of hash functions for the stash filter
    // collision_threshold: how many of k bits already set triggers stashing
    StashedBloomFilter(size_t total_bits, double stash_fraction, size_t num_hashes,
                       size_t stash_hashes, size_t collision_threshold)
        : primary_(total_bits - static_cast<size_t>(total_bits * stash_fraction), num_hashes),
          stash_(std::max<size_t>(1, static_cast<size_t>(total_bits * stash_fraction)),
                 stash_hashes),
          collision_threshold_(collision_threshold) {}

    void insert(const Key& key) {
        size_t collisions = primary_.count_collisions(key);
        if (collisions >= collision_threshold_) {
            stash_.insert(key);
            ++stash_count_;
        } else {
            primary_.insert(key);
        }
    }

    bool query(const Key& key) const {
        return primary_.query(key) || stash_.query(key);
    }

    size_t primary_bits() const { return primary_.num_bits(); }
    size_t stash_bits() const { return stash_.num_bits(); }
    size_t total_bits() const { return primary_.num_bits() + stash_.num_bits(); }
    size_t collision_threshold() const { return collision_threshold_; }
    size_t stash_count() const { return stash_count_; }

   private:
    BloomFilter<Key, HashPolicy> primary_;
    BloomFilter<Key, HashPolicy> stash_;
    size_t collision_threshold_;
    size_t stash_count_ = 0;
};
