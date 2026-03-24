#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Default hash policy using MurmurHash3-style mixing.
// Produces a pair (h1, h2) for double hashing.
struct DefaultHashPolicy {
    static uint64_t mix64(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    static std::pair<uint64_t, uint64_t> hash_pair(uint64_t key) {
        uint64_t h1 = mix64(key);
        uint64_t h2 = mix64(h1);
        h2 |= 1;  // must be odd for coprimality with power-of-2 sizes
        return {h1, h2};
    }

    static std::pair<uint64_t, uint64_t> hash_pair(const std::string& key) {
        // FNV-1a for strings
        uint64_t h = 14695981039346656037ULL;
        for (char c : key) {
            h ^= static_cast<uint64_t>(c);
            h *= 1099511628211ULL;
        }
        uint64_t h1 = mix64(h);
        uint64_t h2 = mix64(h1);
        h2 |= 1;
        return {h1, h2};
    }
};

template <typename Key = uint64_t, typename HashPolicy = DefaultHashPolicy>
class BloomFilter {
   public:
    BloomFilter(size_t num_bits, size_t num_hashes)
        : _bits(num_bits, false), _num_hashes(num_hashes) {}

    void insert(const Key& key) {
        auto [h1, h2] = HashPolicy::hash_pair(key);
        for (size_t i = 0; i < _num_hashes; ++i) {
            _bits[nth_hash(h1, h2, i)] = true;
        }
    }

    bool query(const Key& key) const {
        auto [h1, h2] = HashPolicy::hash_pair(key);
        for (size_t i = 0; i < _num_hashes; ++i) {
            if (!_bits[nth_hash(h1, h2, i)]) {
                return false;
            }
        }
        return true;
    }

    // Count how many of the k bit positions for `key` are already set.
    size_t count_collisions(const Key& key) const {
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
    [[nodiscard]] size_t nth_hash(uint64_t h1, uint64_t h2, size_t i) const {
        return (h1 + i * h2) % _bits.size();
    }

    std::vector<bool> _bits;
    size_t _num_hashes;
};
