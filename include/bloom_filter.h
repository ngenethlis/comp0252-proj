#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

class BloomFilter {
   public:
    BloomFilter(size_t num_bits, size_t num_hashes);

    void insert(uint64_t key);
    bool query(uint64_t key) const;

   private:
    std::vector<bool> bits_;
    size_t num_hashes_;
};
