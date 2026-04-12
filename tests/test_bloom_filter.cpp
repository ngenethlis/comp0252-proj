#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

#include "bloom_filter.h"
#include "bloom_filter_stash.h"
#include "experiment_utils.h"
#include "linear_probing_stash.h"
#include "partitioned_bloom_filter.h"
#include "prob_bool.h"
#include "stash_set.h"
#include "stashed_bloom_filter.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                           \
    static void test_##name();               \
    struct Register_##name {                 \
        Register_##name() { test_##name(); } \
    } register_##name;                       \
    static void test_##name()

#define ASSERT_TRUE(expr)                                             \
    do {                                                              \
        if (!(expr)) {                                                \
            printf("  FAIL: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            ++tests_failed;                                           \
            return;                                                   \
        }                                                             \
    } while (0)

#define PASS(name)                      \
    do {                                \
        printf("  PASS: %s\n", (name)); \
        ++tests_passed;                 \
    } while (0)

// ===========================================================================
// BloomFilter tests
// ===========================================================================

TEST(bloom_no_false_negatives) {
    BloomFilter bf(10000, 7);
    for (uint64_t i = 0; i < 500; ++i) bf.insert(i);
    for (uint64_t i = 0; i < 500; ++i) {
        ASSERT_TRUE(bf.query(i));
    }
    PASS("bloom_no_false_negatives");
}

TEST(bloom_false_positive_rate) {
    BloomFilter bf(10000, 7);
    for (uint64_t i = 0; i < 500; ++i) bf.insert(i);

    size_t fp = 0;
    const size_t test_count = 10000;
    for (uint64_t i = 1000000; i < 1000000 + test_count; ++i) {
        if (bf.query(i)) ++fp;
    }
    double fpr = static_cast<double>(fp) / test_count;
    printf("    BloomFilter FPR: %.4f\n", fpr);
    ASSERT_TRUE(fpr < 0.05);
    PASS("bloom_false_positive_rate");
}

TEST(bloom_empty_query) {
    BloomFilter bf(1000, 5);
    // No insertions — everything should return false
    for (uint64_t i = 0; i < 100; ++i) {
        ASSERT_TRUE(!bf.query(i));
    }
    PASS("bloom_empty_query");
}

TEST(bloom_string_keys) {
    BloomFilter<std::string> bf(10000, 7);
    bf.insert("hello");
    bf.insert("world");
    bf.insert("bloom");

    ASSERT_TRUE(bf.query("hello"));
    ASSERT_TRUE(bf.query("world"));
    ASSERT_TRUE(bf.query("bloom"));
    PASS("bloom_string_keys");
}

TEST(bloom_count_collisions) {
    BloomFilter bf(10000, 7);
    // Before any insert, collisions for any key should be 0
    ASSERT_TRUE(bf.count_collisions(42) == 0);

    bf.insert(42);
    // After inserting 42, querying 42 should have all k bits set
    ASSERT_TRUE(bf.count_collisions(42) == 7);
    PASS("bloom_count_collisions");
}

TEST(bloom_bits_set) {
    BloomFilter bf(10000, 7);
    ASSERT_TRUE(bf.bits_set() == 0);
    bf.insert(1);
    // At most 7 bits set after one insert (could be fewer if hash collides with itself)
    ASSERT_TRUE(bf.bits_set() > 0);
    ASSERT_TRUE(bf.bits_set() <= 7);
    PASS("bloom_bits_set");
}

TEST(bloom_single_bit) {
    // Minimal filter: 1 hash, small bit array
    BloomFilter bf(64, 1);
    bf.insert(42);
    ASSERT_TRUE(bf.query(42));
    ASSERT_TRUE(bf.bits_set() == 1);
    PASS("bloom_single_bit");
}

// ===========================================================================
// Blocked Bloom filter baseline tests (implemented via PartitionedBloomFilter)
// ===========================================================================

TEST(partitioned_bf_no_false_negatives) {
    PartitionedBloomFilter pbf(10000, 7);
    for (uint64_t i = 0; i < 500; ++i) pbf.insert(i);
    for (uint64_t i = 0; i < 500; ++i) {
        ASSERT_TRUE(pbf.query(i));
    }
    PASS("partitioned_bf_no_false_negatives");
}

TEST(partitioned_bf_false_positive_rate) {
    PartitionedBloomFilter pbf(10000, 7);
    for (uint64_t i = 0; i < 500; ++i) pbf.insert(i);

    size_t fp = 0;
    const size_t test_count = 10000;
    for (uint64_t i = 1000000; i < 1000000 + test_count; ++i) {
        if (pbf.query(i)) ++fp;
    }
    double fpr = static_cast<double>(fp) / test_count;
    printf("    BlockedBF (partitioned layout) FPR: %.4f\n", fpr);
    ASSERT_TRUE(fpr < 0.05);
    PASS("partitioned_bf_false_positive_rate");
}

TEST(partitioned_bf_empty_query) {
    PartitionedBloomFilter pbf(1000, 5);
    for (uint64_t i = 0; i < 100; ++i) {
        ASSERT_TRUE(!pbf.query(i));
    }
    PASS("partitioned_bf_empty_query");
}

TEST(partitioned_bf_count_collisions) {
    PartitionedBloomFilter pbf(10000, 7);
    ASSERT_TRUE(pbf.count_collisions(42) == 0);
    pbf.insert(42);
    ASSERT_TRUE(pbf.count_collisions(42) == 7);
    PASS("partitioned_bf_count_collisions");
}

TEST(partitioned_bf_partition_size) {
    PartitionedBloomFilter pbf(7000, 7);
    ASSERT_TRUE(pbf.partition_size() == 1000);
    ASSERT_TRUE(pbf.num_bits() == 7000);
    ASSERT_TRUE(pbf.num_hashes() == 7);
    PASS("partitioned_bf_partition_size");
}

// ===========================================================================
// ProbBool tests
// ===========================================================================

TEST(prob_bool_values) {
    ASSERT_TRUE(is_positive(ProbBool::True));
    ASSERT_TRUE(is_positive(ProbBool::Maybe));
    ASSERT_TRUE(!is_positive(ProbBool::False));
    PASS("prob_bool_values");
}

// ===========================================================================
// BloomFilterStash tests
// ===========================================================================

TEST(bf_stash_insert_query) {
    BloomFilterStash<uint64_t> stash(1000, 5);
    ASSERT_TRUE(stash.insert(42));
    ASSERT_TRUE(stash.insert(99));
    ASSERT_TRUE(stash.query(42));
    ASSERT_TRUE(stash.query(99));
    PASS("bf_stash_insert_query");
}

TEST(bf_stash_no_false_negatives) {
    BloomFilterStash<uint64_t> stash(5000, 5);
    for (uint64_t i = 0; i < 200; ++i) stash.insert(i);
    for (uint64_t i = 0; i < 200; ++i) {
        ASSERT_TRUE(stash.query(i));
    }
    PASS("bf_stash_no_false_negatives");
}

TEST(bf_stash_size_bits) {
    BloomFilterStash<uint64_t> stash(2048, 3);
    ASSERT_TRUE(stash.size_bits() == 2048);
    PASS("bf_stash_size_bits");
}

TEST(bf_stash_is_probabilistic) {
    BloomFilterStash<uint64_t> stash(2048, 3);
    ASSERT_TRUE(stash.is_probabilistic());
    PASS("bf_stash_is_probabilistic");
}

// ===========================================================================
// LinearProbingStash tests
// ===========================================================================

TEST(lp_stash_insert_query) {
    LinearProbingStash<uint64_t> stash(100);
    ASSERT_TRUE(stash.insert(42));
    ASSERT_TRUE(stash.insert(99));
    ASSERT_TRUE(stash.query(42));
    ASSERT_TRUE(stash.query(99));
    ASSERT_TRUE(!stash.query(123));
    PASS("lp_stash_insert_query");
}

TEST(lp_stash_capacity_limit) {
    LinearProbingStash<uint64_t> stash(5);
    for (uint64_t i = 1; i <= 5; ++i) {
        ASSERT_TRUE(stash.insert(i));
    }
    // Should fail when full
    ASSERT_TRUE(!stash.insert(100));
    PASS("lp_stash_capacity_limit");
}

TEST(lp_stash_no_false_negatives) {
    LinearProbingStash<uint64_t> stash(500);
    for (uint64_t i = 1; i <= 200; ++i) stash.insert(i);
    for (uint64_t i = 1; i <= 200; ++i) {
        ASSERT_TRUE(stash.query(i));
    }
    PASS("lp_stash_no_false_negatives");
}

TEST(lp_stash_empty_query) {
    LinearProbingStash<uint64_t> stash(50);
    for (uint64_t i = 0; i < 50; ++i) {
        ASSERT_TRUE(!stash.query(i));
    }
    PASS("lp_stash_empty_query");
}

TEST(lp_stash_size_bits) {
    LinearProbingStash<uint64_t> stash(10);
    ASSERT_TRUE(stash.size_bits() == 10 * 64);
    PASS("lp_stash_size_bits");
}

TEST(lp_stash_is_deterministic) {
    LinearProbingStash<uint64_t> stash(10);
    ASSERT_TRUE(!stash.is_probabilistic());
    PASS("lp_stash_is_deterministic");
}

TEST(lp_stash_duplicate_insert_no_capacity_loss) {
    LinearProbingStash<uint64_t> stash(2);
    ASSERT_TRUE(stash.insert(42));
    ASSERT_TRUE(stash.insert(42));
    ASSERT_TRUE(stash.insert(42));
    ASSERT_TRUE(stash.insert(99));
    ASSERT_TRUE(!stash.insert(123));
    PASS("lp_stash_duplicate_insert_no_capacity_loss");
}

TEST(read_weighted_lines_parses_count_suffix) {
    const std::string path = "tests/tmp_weighted_lines.txt";
    {
        std::ofstream out(path);
        out << "alpha:7\n";
        out << "beta\n";
        out << "gamma:not-a-number\n";
        out << "delta:0\n";
    }

    auto entries = read_weighted_lines(path);
    ASSERT_TRUE(entries.size() == 4);
    ASSERT_TRUE(entries[0].key == "alpha");
    ASSERT_TRUE(entries[0].count == 7);
    ASSERT_TRUE(entries[1].key == "beta");
    ASSERT_TRUE(entries[1].count == 1);
    ASSERT_TRUE(entries[2].key == "gamma:not-a-number");
    ASSERT_TRUE(entries[2].count == 1);
    ASSERT_TRUE(entries[3].key == "delta");
    ASSERT_TRUE(entries[3].count == 1);

    std::remove(path.c_str());
    PASS("read_weighted_lines_parses_count_suffix");
}

// ===========================================================================
// StashedBloomFilter with BloomFilterStash (Positive mode)
// ===========================================================================

TEST(stashed_bf_pos_no_false_negatives) {
    BloomFilterStash<uint64_t> stash(2000, 5);
    StashedBloomFilter sbf(8000, 7, std::move(stash), 5, StashMode::Positive);

    for (uint64_t i = 0; i < 500; ++i) sbf.insert(i);
    for (uint64_t i = 0; i < 500; ++i) {
        ASSERT_TRUE(sbf.query_bool(i));
    }
    PASS("stashed_bf_pos_no_false_negatives");
}

TEST(stashed_bf_pos_query_returns_prob_bool) {
    BloomFilterStash<uint64_t> stash(2000, 5);
    StashedBloomFilter sbf(8000, 7, std::move(stash), 0, StashMode::Positive);

    sbf.insert(42);
    ProbBool result = sbf.query(42);
    ASSERT_TRUE(result == ProbBool::Maybe);
    PASS("stashed_bf_pos_query_returns_prob_bool");
}

TEST(stashed_bf_pos_false_positive_rate) {
    BloomFilterStash<uint64_t> stash(2000, 5);
    StashedBloomFilter sbf(8000, 7, std::move(stash), 5, StashMode::Positive);

    for (uint64_t i = 0; i < 500; ++i) sbf.insert(i);

    size_t fp = 0;
    const size_t test_count = 10000;
    for (uint64_t i = 1000000; i < 1000000 + test_count; ++i) {
        if (sbf.query_bool(i)) ++fp;
    }
    double fpr = static_cast<double>(fp) / test_count;
    printf("    StashedBF (BF stash, pos) FPR: %.4f (stash_count: %zu)\n", fpr, sbf.stash_count());
    PASS("stashed_bf_pos_false_positive_rate");
}

TEST(stashed_bf_pos_non_inserted_false) {
    BloomFilterStash<uint64_t> stash(2000, 5);
    StashedBloomFilter sbf(8000, 7, std::move(stash), 5, StashMode::Positive);

    // Query without any insertions
    ASSERT_TRUE(sbf.query(42) == ProbBool::False);
    PASS("stashed_bf_pos_non_inserted_false");
}

// ===========================================================================
// StashedBloomFilter with LinearProbingStash (Positive mode)
// ===========================================================================

TEST(stashed_lp_pos_no_false_negatives) {
    LinearProbingStash<uint64_t> stash(200);
    StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
        8000, 7, std::move(stash), 5, StashMode::Positive);

    for (uint64_t i = 0; i < 500; ++i) sbf.insert(i);
    for (uint64_t i = 0; i < 500; ++i) {
        ASSERT_TRUE(sbf.query_bool(i));
    }
    PASS("stashed_lp_pos_no_false_negatives");
}

TEST(stashed_lp_pos_stash_true) {
    // Use a very aggressive threshold (0) so everything goes to stash
    LinearProbingStash<uint64_t> stash(100);
    StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
        8000, 7, std::move(stash), 0, StashMode::Positive);

    sbf.insert(42);
    // With threshold 0, key should be in stash → ProbBool::True
    ASSERT_TRUE(sbf.query(42) == ProbBool::True);
    PASS("stashed_lp_pos_stash_true");
}

TEST(stashed_lp_pos_false_positive_rate) {
    LinearProbingStash<uint64_t> stash(200);
    StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
        8000, 7, std::move(stash), 5, StashMode::Positive);

    for (uint64_t i = 0; i < 500; ++i) sbf.insert(i);

    size_t fp = 0;
    const size_t test_count = 10000;
    for (uint64_t i = 1000000; i < 1000000 + test_count; ++i) {
        if (sbf.query_bool(i)) ++fp;
    }
    double fpr = static_cast<double>(fp) / test_count;
    printf("    StashedBF (LP stash, pos) FPR: %.4f (stash_count: %zu)\n", fpr, sbf.stash_count());
    PASS("stashed_lp_pos_false_positive_rate");
}

// ===========================================================================
// StashedBloomFilter with Negative stash mode
// ===========================================================================

TEST(stashed_bf_neg_mode) {
    BloomFilterStash<uint64_t> stash(2000, 5);
    StashedBloomFilter sbf(8000, 7, std::move(stash), 5, StashMode::Negative);

    ASSERT_TRUE(sbf.mode() == StashMode::Negative);

    // In negative mode, items in stash return False (definitely not in set)
    // Items in primary return Maybe, and items in neither return False
    for (uint64_t i = 0; i < 100; ++i) sbf.insert(i);

    // Non-inserted elements should return False
    ASSERT_TRUE(sbf.query(999999) == ProbBool::False);
    PASS("stashed_bf_neg_mode");
}

// ===========================================================================
// StashedBloomFilter accessors
// ===========================================================================

TEST(stashed_bf_accessors) {
    BloomFilterStash<uint64_t> stash(2000, 5);
    StashedBloomFilter sbf(8000, 7, std::move(stash), 5, StashMode::Positive);

    ASSERT_TRUE(sbf.primary_bits() == 8000);
    ASSERT_TRUE(sbf.stash_bits() == 2000);
    ASSERT_TRUE(sbf.total_bits() == 10000);
    ASSERT_TRUE(sbf.collision_threshold() == 5);
    ASSERT_TRUE(sbf.stash_count() == 0);
    ASSERT_TRUE(sbf.mode() == StashMode::Positive);
    PASS("stashed_bf_accessors");
}

TEST(stashed_bf_same_total_bits_as_plain_bf) {
    const size_t total_bits = 10000;

    BloomFilter bf(total_bits, 7);

    BloomFilterStash<uint64_t> stash_bf(2000, 5);
    StashedBloomFilter sbf_bf(8000, 7, std::move(stash_bf), 5);
    ASSERT_TRUE(sbf_bf.total_bits() == bf.num_bits());

    LinearProbingStash<uint64_t> stash_lp(125);  // 125 * 64 = 8000 bits
    StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf_lp(
        2000, 7, std::move(stash_lp), 5);
    ASSERT_TRUE(sbf_lp.total_bits() == bf.num_bits());

    PASS("stashed_bf_same_total_bits_as_plain_bf");
}

TEST(stashed_bf_stash_fallback) {
    // Test that when LP stash is full, elements fall back to primary
    LinearProbingStash<uint64_t> stash(5);
    StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
        8000, 7, std::move(stash), 0, StashMode::Positive);

    // threshold=0 means everything tries stash first; stash can hold 5
    for (uint64_t i = 1; i <= 20; ++i) sbf.insert(i);

    // All inserted elements should still be queryable (via primary fallback)
    for (uint64_t i = 1; i <= 20; ++i) {
        ASSERT_TRUE(sbf.query_bool(i));
    }
    ASSERT_TRUE(sbf.stash_count() == 5);
    PASS("stashed_bf_stash_fallback");
}

// ===========================================================================
// String key tests for stashed filter
// ===========================================================================

TEST(stashed_bf_string_keys) {
    BloomFilterStash<std::string> stash(2000, 5);
    StashedBloomFilter<std::string> sbf(8000, 7, std::move(stash), 5, StashMode::Positive);

    sbf.insert("alpha");
    sbf.insert("beta");
    sbf.insert("gamma");

    ASSERT_TRUE(sbf.query_bool("alpha"));
    ASSERT_TRUE(sbf.query_bool("beta"));
    ASSERT_TRUE(sbf.query_bool("gamma"));
    PASS("stashed_bf_string_keys");
}

// ===========================================================================
// main
// ===========================================================================

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    if (tests_failed > 0) {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
    printf("All tests passed.\n");
    return 0;
}
