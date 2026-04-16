#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
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

/**
 * @file experiment_utils.h
 * @brief Utilities for synthetic/data-driven experiment generation and metrics.
 */

/**
 * @brief Generates `n` pseudo-random keys using a uniform RNG.
 * @param n Number of keys.
 * @param seed RNG seed.
 * @param offset Constant added to each generated key.
 */
inline std::vector<uint64_t> generate_uniform_keys(size_t n, uint64_t seed, uint64_t offset = 0) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> keys;
    keys.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        keys.push_back(offset + rng());
    }
    return keys;
}

/**
 * @brief Generates keys sampled from a Zipf distribution.
 * @param n Number of generated keys.
 * @param s Zipf exponent.
 * @param max_rank Maximum rank in `[1, max_rank]`.
 * @param seed RNG seed.
 */
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

/**
 * @brief Non-overlapping key sets used by experiment pipelines.
 */
struct DataSplit {
    std::vector<uint64_t> positives;       /**< Keys inserted into filters. */
    std::vector<uint64_t> stash_negatives; /**< Known negatives for negative-stash population. */
    std::vector<uint64_t> test_negatives;  /**< Known negatives used for FPR measurement. */
};

/**
 * @brief Creates a `DataSplit` with three practically disjoint key populations.
 */
inline DataSplit generate_data(size_t n_pos, size_t n_stash_neg, size_t n_test_neg, uint64_t seed) {
    DataSplit data;
    data.positives = generate_uniform_keys(n_pos, seed, 0);
    data.stash_negatives = generate_uniform_keys(n_stash_neg, seed + 1, 1ULL << 62);
    data.test_negatives = generate_uniform_keys(n_test_neg, seed + 2, 2ULL << 62);
    return data;
}

/**
 * @brief Measures false-positive rate for a filter exposing `query(key) -> bool`.
 * @tparam Filter Filter type with `bool query(const Key&) const`.
 * @tparam Key Key element type.
 * @param filter Filter under test.
 * @param negatives Known-negative query set.
 * @return False-positive fraction in `[0, 1]`.
 */
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

/**
 * @brief Measures FPR for stashed filters using positive-like query semantics.
 * @return Fraction of known negatives where `query_bool` returns true.
 */
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

/**
 * @brief Counts of tri-valued query outcomes.
 */
struct QueryStats {
    size_t true_count = 0;
    size_t maybe_count = 0;
    size_t false_count = 0;

    /** @brief Total number of recorded queries. */
    [[nodiscard]] size_t total() const { return true_count + maybe_count + false_count; }

    /** @brief Fraction of results equal to `ProbBool::True`. */
    [[nodiscard]] double true_rate() const {
        return total() > 0 ? static_cast<double>(true_count) / static_cast<double>(total()) : 0.0;
    }

    /** @brief Fraction of results considered positive (`True` or `Maybe`). */
    [[nodiscard]] double positive_rate() const {
        size_t pos = true_count + maybe_count;
        return total() > 0 ? static_cast<double>(pos) / static_cast<double>(total()) : 0.0;
    }
};

/**
 * @brief Tallies `ProbBool` outcomes across a key set.
 */
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

/**
 * @brief Reads non-empty lines from a UTF-8 text file.
 * @param path File path.
 * @return Non-empty trimmed lines (trailing CR/LF removed).
 */
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

/**
 * @brief String key with optional frequency/count weight.
 */
struct WeightedStringEntry {
    std::string key;    /**< Entry key. */
    uint64_t count = 1; /**< Parsed count (defaults to 1). */
};

/**
 * @brief Reads weighted string entries from text lines.
 *
 * Accepted formats per line are `key` or `key:count`. Invalid or missing count
 * values default to `1`.
 */
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

/**
 * @brief Samples indices according to non-negative integer weights.
 * @param n Number of draws.
 * @param weights Weight vector (one weight per index).
 * @param seed RNG seed.
 * @return Vector of sampled indices in `[0, weights.size())`.
 */
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

/**
 * @brief Generates random strings for synthetic negative-query datasets.
 * @param n Number of strings.
 * @param len Length per string.
 * @param seed RNG seed.
 */
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
