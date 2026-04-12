#pragma once

#include <cstddef>
#include <cstdint>

#include "bloom_filter.h"
#include "stash_set.h"

// BloomFilterStash: uses a secondary Bloom filter as the stash.
template <typename Key = uint64_t, typename HashPolicy = DefaultHashPolicy>
class BloomFilterStash : public StashSet<BloomFilterStash<Key, HashPolicy>, Key> {
   public:
    BloomFilterStash(size_t num_bits, size_t num_hashes) : _bf(num_bits, num_hashes) {}

    bool do_insert(const Key& key) {
        _bf.insert(key);
        return true;  // Bloom filter insert never fails
    }

    bool do_query(const Key& key) const { return _bf.query(key); }

    [[nodiscard]] size_t do_size_bits() const { return _bf.num_bits(); }

    [[nodiscard]] bool do_is_probabilistic() const { return true; }

    [[nodiscard]] const BloomFilter<Key, HashPolicy>& bloom_filter() const { return _bf; }

   private:
    BloomFilter<Key, HashPolicy> _bf;
};
