#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

/**
 * @file bloom_filter.h
 * @brief Standard Bloom filter implementation and default hash policy.
 */

/**
 * @brief Default hashing policy used by Bloom filters in this project.
 *
 * Produces hash pairs `(h1, h2)` for double hashing where
 * `h_i(key) = (h1 + i * h2) mod m`.
 */
struct DefaultHashPolicy {
    /**
     * @brief 64-bit mix function based on MurmurHash3 finalizer steps.
     */
    static uint64_t mix64(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    /**
     * @brief Hashes an integer key into a pair for double hashing.
     *
     * The second hash is forced odd for good stride behavior with power-of-2
     * bit-table sizes.
     */
    static std::pair<uint64_t, uint64_t> hash_pair(uint64_t key) {
        uint64_t h1 = mix64(key);
        uint64_t h2 = mix64(h1);
        h2 |= 1;  // must be odd for coprimality with power-of-2 sizes
        return {h1, h2};
    }

    /**
     * @brief Hashes a string key into a pair for double hashing.
     *
     * Uses FNV-1a to create an initial 64-bit hash, then applies `mix64`.
     */
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

/**
 * @brief Standard Bloom filter with configurable key and hash policy types.
 *
 * @tparam Key Type of keys inserted and queried.
 * @tparam HashPolicy Hash policy providing
 * `static std::pair<uint64_t, uint64_t> hash_pair(const Key&)`.
 */
template <typename Key = uint64_t, typename HashPolicy = DefaultHashPolicy>
class BloomFilter {
   public:
    /**
     * @brief Constructs a Bloom filter.
     * @param num_bits Number of bits in the bit table.
     * @param num_hashes Number of hash probes per key.
     */
    BloomFilter(size_t num_bits, size_t num_hashes)
        : _bits(num_bits, false), _num_hashes(num_hashes) {}

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
    bool query(const Key& key) const {
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
     * @return Number of set bits among the `num_hashes()` probe positions.
     */
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

    /**
     * @brief Returns the total number of bits in the table.
     */
    [[nodiscard]] size_t num_bits() const { return _bits.size(); }
    /**
     * @brief Returns the number of hash probes used per operation.
     */
    [[nodiscard]] size_t num_hashes() const { return _num_hashes; }

    /**
     * @brief Counts currently set bits in the table.
     */
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
     * @brief Computes the i-th double-hashed index.
     */
    [[nodiscard]] size_t nth_hash(uint64_t h1, uint64_t h2, size_t i) const {
        return (h1 + i * h2) % _bits.size();
    }

    std::vector<bool> _bits;
    size_t _num_hashes;
};
