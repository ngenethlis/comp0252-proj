#include "bloom_filter.h"

BloomFilter::BloomFilter(size_t num_bits, size_t num_hashes)
    : bits_(num_bits, false), num_hashes_(num_hashes) {}

void BloomFilter::insert(uint64_t key) {
    // TODO: implement
}

bool BloomFilter::query(uint64_t key) const {
    // TODO: implement
    return false;
}
