#pragma once

#include <charconv>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
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
    std::vector<uint64_t>
        test_negatives;  // known negatives for measuring FPR (disjoint from above)
};

// Generate three non-overlapping key sets using different seed offsets.
// Uses high bits to separate ranges, so overlap is astronomically unlikely.
inline DataSplit generate_data(size_t n_pos, size_t n_stash_neg, size_t n_test_neg, uint64_t seed) {
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
// ProbBool query statistics
// ---------------------------------------------------------------------------

struct QueryStats {
    size_t true_count = 0;
    size_t maybe_count = 0;
    size_t false_count = 0;

    [[nodiscard]] size_t total() const { return true_count + maybe_count + false_count; }

    // Fraction of queries returning True (certain membership).
    [[nodiscard]] double true_rate() const {
        return total() > 0 ? static_cast<double>(true_count) / static_cast<double>(total()) : 0.0;
    }

    // Fraction of queries returning True or Maybe (any positive signal).
    [[nodiscard]] double positive_rate() const {
        size_t pos = true_count + maybe_count;
        return total() > 0 ? static_cast<double>(pos) / static_cast<double>(total()) : 0.0;
    }
};

// Count ProbBool outcomes from querying a stashed bloom filter.
template <typename Key, typename HP, typename Stash>
QueryStats count_query_results(const StashedBloomFilter<Key, HP, Stash>& filter,
                               const std::vector<Key>& keys) {
    QueryStats stats;
    for (const auto& key : keys) {
        switch (filter.query(key)) {
            case ProbBool::True:
                ++stats.true_count;
                break;
            case ProbBool::Maybe:
                ++stats.maybe_count;
                break;
            case ProbBool::False:
                ++stats.false_count;
                break;
        }
    }
    return stats;
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
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

struct WeightedStringEntry {
    std::string key;
    uint64_t count = 1;
};

// Read non-empty lines from a text file where entries may be either:
//   key
// or
//   key:count
// If count is missing or invalid, defaults to 1.
inline std::vector<WeightedStringEntry> read_weighted_lines(const std::string& path) {
    std::vector<WeightedStringEntry> entries;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        WeightedStringEntry entry{line, 1};
        const size_t sep = line.rfind(':');
        if (sep != std::string::npos && sep > 0 && sep + 1 < line.size()) {
            uint64_t parsed_count = 1;
            const char* begin = line.data() + sep + 1;
            const char* end = line.data() + line.size();
            while (begin < end && std::isspace(static_cast<unsigned char>(*begin))) {
                ++begin;
            }
            while (end > begin && std::isspace(static_cast<unsigned char>(*(end - 1)))) {
                --end;
            }
            const auto result = std::from_chars(begin, end, parsed_count);
            if (begin < end && result.ec == std::errc() && result.ptr == end) {
                entry.key = line.substr(0, sep);
                entry.count = parsed_count > 0 ? parsed_count : 1;
            }
        }
        entries.push_back(std::move(entry));
    }
    return entries;
}

// Draw n samples from indices [0, weights.size()) with probability proportional
// to the provided non-negative weights.
inline std::vector<size_t> sample_weighted_indices(size_t n, const std::vector<uint64_t>& weights,
                                                   uint64_t seed) {
    std::vector<size_t> indices;
    if (weights.empty() || n == 0) {
        return indices;
    }
    indices.reserve(n);

    std::vector<double> probs;
    probs.reserve(weights.size());
    for (uint64_t w : weights) {
        probs.push_back(static_cast<double>(w));
    }

    std::mt19937_64 rng(seed);
    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    for (size_t i = 0; i < n; ++i) {
        indices.push_back(dist(rng));
    }
    return indices;
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
