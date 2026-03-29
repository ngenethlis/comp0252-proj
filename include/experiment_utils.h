#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "stashed_bloom_filter.h"

// Generate n unique uniform random uint64_t values using the given seed.
// Keys are drawn from [offset, offset + range) to allow non-overlapping sets.
inline std::vector<uint64_t> generate_uniform_keys(size_t n, uint64_t seed, uint64_t offset = 0) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> keys;
    keys.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        keys.push_back(offset + rng());
    }
    return keys;
}

// Generate n keys from a Zipf distribution with exponent s over ranks [1, max_rank].
// Uses inverse CDF table lookup.
inline std::vector<uint64_t> generate_zipf_keys(size_t n, double s, uint64_t max_rank,
                                                 uint64_t seed) {
    // Build CDF
    std::vector<double> cdf(max_rank + 1, 0.0);
    double sum = 0.0;
    for (uint64_t r = 1; r <= max_rank; ++r) {
        sum += 1.0 / std::pow(static_cast<double>(r), s);
        cdf[r] = sum;
    }
    // Normalize
    for (uint64_t r = 1; r <= max_rank; ++r) {
        cdf[r] /= sum;
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<uint64_t> keys;
    keys.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double u = dist(rng);
        // Binary search for the rank
        auto it = std::lower_bound(cdf.begin() + 1, cdf.end(), u);
        uint64_t rank = static_cast<uint64_t>(it - cdf.begin());
        if (rank < 1) {
            rank = 1;
        }
        if (rank > max_rank) {
            rank = max_rank;
        }
        keys.push_back(rank);
    }
    return keys;
}

// Three disjoint key sets for experiments.
struct DataSplit {
    std::vector<uint64_t> positives;        // keys to insert into the filter
    std::vector<uint64_t> stash_negatives;  // known negatives for populating negative stash
    std::vector<uint64_t> test_negatives;   // known negatives for measuring FPR (disjoint from above)
};

// Generate three non-overlapping key sets using different seed offsets.
// Uses high bits to separate ranges, so overlap is astronomically unlikely.
inline DataSplit generate_data(size_t n_pos, size_t n_stash_neg, size_t n_test_neg,
                               uint64_t seed) {
    DataSplit data;
    data.positives = generate_uniform_keys(n_pos, seed, 0);
    data.stash_negatives = generate_uniform_keys(n_stash_neg, seed + 1, 1ULL << 62);
    data.test_negatives = generate_uniform_keys(n_test_neg, seed + 2, 2ULL << 62);
    return data;
}

// Measure false positive rate for a filter with a .query(key) -> bool interface.
template <typename Filter, typename Key>
double measure_fpr(const Filter& filter, const std::vector<Key>& negatives) {
    if (negatives.empty()) {
        return 0.0;
    }
    size_t fp = 0;
    for (const auto& key : negatives) {
        if (filter.query(key)) {
            ++fp;
        }
    }
    return static_cast<double>(fp) / static_cast<double>(negatives.size());
}

// Measure FPR for StashedBloomFilter using query_bool().
template <typename Key, typename HP, typename Stash>
double measure_fpr_stashed(const StashedBloomFilter<Key, HP, Stash>& filter,
                           const std::vector<Key>& negatives) {
    if (negatives.empty()) {
        return 0.0;
    }
    size_t fp = 0;
    for (const auto& key : negatives) {
        if (filter.query_bool(key)) {
            ++fp;
        }
    }
    return static_cast<double>(fp) / static_cast<double>(negatives.size());
}

// ---------------------------------------------------------------------------
// File I/O helpers
// ---------------------------------------------------------------------------

// Read non-empty lines from a text file (one entry per line).
inline std::vector<std::string> read_lines(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

// Generate n random alphanumeric strings of the given length.
// Useful as synthetic negatives for FPR testing with string keys.
inline std::vector<std::string> generate_random_strings(size_t n, size_t len, uint64_t seed) {
    static constexpr char kChars[] =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*";
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, sizeof(kChars) - 2);

    std::vector<std::string> strings;
    strings.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        std::string s;
        s.reserve(len);
        for (size_t j = 0; j < len; ++j) {
            s += kChars[dist(rng)];
        }
        strings.push_back(std::move(s));
    }
    return strings;
}
