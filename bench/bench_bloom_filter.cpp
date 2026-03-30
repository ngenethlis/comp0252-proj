#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "bloom_filter.h"
#include "bloom_filter_stash.h"
#include "experiment_utils.h"
#include "linear_probing_stash.h"
#include "partitioned_bloom_filter.h"
#include "stashed_bloom_filter.h"

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

using Clock = std::chrono::high_resolution_clock;

// Sink to prevent the compiler from optimizing out query results.
static volatile bool g_sink = false;

template <typename Fn>
double time_ns(Fn&& fn) {
    auto start = Clock::now();
    fn();
    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

static void csv_header(const std::string& cols) { std::cout << cols << "\n"; }

template <typename... Args>
static void csv_row(Args&&... args) {
    bool first = true;
    auto print = [&first](const auto& val) {
        if (!first) {
            std::cout << ",";
        }
        std::cout << val;
        first = false;
    };
    (print(std::forward<Args>(args)), ...);
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Default parameters
// ---------------------------------------------------------------------------
static constexpr size_t kTotalBits = 100000;
static constexpr size_t kNumHashes = 7;
static constexpr uint64_t kSeed = 42;

// ---------------------------------------------------------------------------
// Benchmark: Insert throughput
// ---------------------------------------------------------------------------
static void bench_insert() {
    std::cerr << "=== Benchmark: Insert throughput ===\n";
    csv_header("bench,filter_type,n,total_ns,ns_per_op");

    size_t n_values[] = {1000, 5000, 10000, 50000};
    size_t stash_bits = kTotalBits / 5;
    size_t primary_bits = kTotalBits - stash_bits;
    size_t lp_capacity = stash_bits / 64;
    size_t collision_threshold = 3;

    for (size_t n : n_values) {
        auto keys = generate_uniform_keys(n, kSeed);

        // BloomFilter
        {
            double ns = time_ns([&] {
                BloomFilter bf(kTotalBits, kNumHashes);
                for (uint64_t k : keys) {
                    bf.insert(k);
                }
            });
            csv_row("insert", "bloom_filter", n, ns, ns / n);
        }
        // PartitionedBloomFilter
        {
            double ns = time_ns([&] {
                PartitionedBloomFilter pbf(kTotalBits, kNumHashes);
                for (uint64_t k : keys) {
                    pbf.insert(k);
                }
            });
            csv_row("insert", "partitioned_bf", n, ns, ns / n);
        }
        // StashedBloomFilter + BF stash
        {
            double ns = time_ns([&] {
                BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
                StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash),
                                       collision_threshold, StashMode::Positive);
                for (uint64_t k : keys) {
                    sbf.insert(k);
                }
            });
            csv_row("insert", "stashed_bf_pos", n, ns, ns / n);
        }
        // StashedBloomFilter + LP stash
        {
            double ns = time_ns([&] {
                LinearProbingStash<uint64_t> stash(lp_capacity);
                StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                    primary_bits, kNumHashes, std::move(stash), collision_threshold,
                    StashMode::Positive);
                for (uint64_t k : keys) {
                    sbf.insert(k);
                }
            });
            csv_row("insert", "stashed_lp_pos", n, ns, ns / n);
        }
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Benchmark: Query throughput (positive — keys in the filter)
// ---------------------------------------------------------------------------
static void bench_query_positive() {
    std::cerr << "=== Benchmark: Positive query throughput ===\n";
    csv_header("bench,filter_type,n,total_ns,ns_per_op");

    size_t n_values[] = {1000, 5000, 10000, 50000};
    size_t stash_bits = kTotalBits / 5;
    size_t primary_bits = kTotalBits - stash_bits;
    size_t lp_capacity = stash_bits / 64;
    size_t collision_threshold = 3;

    for (size_t n : n_values) {
        auto keys = generate_uniform_keys(n, kSeed);

        // BloomFilter
        {
            BloomFilter bf(kTotalBits, kNumHashes);
            for (uint64_t k : keys) {
                bf.insert(k);
            }
            double ns = time_ns([&] {
                for (uint64_t k : keys) {
                    g_sink = bf.query(k);
                }
            });
            csv_row("query_pos", "bloom_filter", n, ns, ns / n);
        }
        // PartitionedBloomFilter
        {
            PartitionedBloomFilter pbf(kTotalBits, kNumHashes);
            for (uint64_t k : keys) {
                pbf.insert(k);
            }
            double ns = time_ns([&] {
                for (uint64_t k : keys) {
                    g_sink = pbf.query(k);
                }
            });
            csv_row("query_pos", "partitioned_bf", n, ns, ns / n);
        }
        // StashedBloomFilter + BF stash
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash), collision_threshold,
                                   StashMode::Positive);
            for (uint64_t k : keys) {
                sbf.insert(k);
            }
            double ns = time_ns([&] {
                for (uint64_t k : keys) {
                    g_sink = is_positive(sbf.query(k));
                }
            });
            csv_row("query_pos", "stashed_bf_pos", n, ns, ns / n);
        }
        // StashedBloomFilter + LP stash
        {
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), collision_threshold,
                StashMode::Positive);
            for (uint64_t k : keys) {
                sbf.insert(k);
            }
            double ns = time_ns([&] {
                for (uint64_t k : keys) {
                    g_sink = is_positive(sbf.query(k));
                }
            });
            csv_row("query_pos", "stashed_lp_pos", n, ns, ns / n);
        }
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Benchmark: Query throughput (negative — keys NOT in the filter)
// ---------------------------------------------------------------------------
static void bench_query_negative() {
    std::cerr << "=== Benchmark: Negative query throughput ===\n";
    csv_header("bench,filter_type,n,total_ns,ns_per_op");

    size_t n_values[] = {1000, 5000, 10000, 50000};
    size_t stash_bits = kTotalBits / 5;
    size_t primary_bits = kTotalBits - stash_bits;
    size_t lp_capacity = stash_bits / 64;
    size_t collision_threshold = 3;

    for (size_t n : n_values) {
        auto keys = generate_uniform_keys(n, kSeed);
        auto neg_keys = generate_uniform_keys(n, kSeed + 100, 1ULL << 62);

        // BloomFilter
        {
            BloomFilter bf(kTotalBits, kNumHashes);
            for (uint64_t k : keys) {
                bf.insert(k);
            }
            double ns = time_ns([&] {
                for (uint64_t k : neg_keys) {
                    g_sink = bf.query(k);
                }
            });
            csv_row("query_neg", "bloom_filter", n, ns, ns / n);
        }
        // PartitionedBloomFilter
        {
            PartitionedBloomFilter pbf(kTotalBits, kNumHashes);
            for (uint64_t k : keys) {
                pbf.insert(k);
            }
            double ns = time_ns([&] {
                for (uint64_t k : neg_keys) {
                    g_sink = pbf.query(k);
                }
            });
            csv_row("query_neg", "partitioned_bf", n, ns, ns / n);
        }
        // StashedBloomFilter + BF stash
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash), collision_threshold,
                                   StashMode::Positive);
            for (uint64_t k : keys) {
                sbf.insert(k);
            }
            double ns = time_ns([&] {
                for (uint64_t k : neg_keys) {
                    g_sink = is_positive(sbf.query(k));
                }
            });
            csv_row("query_neg", "stashed_bf_pos", n, ns, ns / n);
        }
        // StashedBloomFilter + LP stash
        {
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), collision_threshold,
                StashMode::Positive);
            for (uint64_t k : keys) {
                sbf.insert(k);
            }
            double ns = time_ns([&] {
                for (uint64_t k : neg_keys) {
                    g_sink = is_positive(sbf.query(k));
                }
            });
            csv_row("query_neg", "stashed_lp_pos", n, ns, ns / n);
        }
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string mode = (argc > 1) ? argv[1] : "all";

    if (mode == "insert" || mode == "all") {
        bench_insert();
    }
    if (mode == "query_pos" || mode == "all") {
        bench_query_positive();
    }
    if (mode == "query_neg" || mode == "all") {
        bench_query_negative();
    }

    return 0;
}
