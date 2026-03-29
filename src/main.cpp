#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
// Default experiment parameters
// ---------------------------------------------------------------------------
static constexpr size_t kTotalBits = 100000;
static constexpr size_t kNumHashes = 7;
static constexpr size_t kNumInserts = 5000;
static constexpr size_t kNumNeg = 50000;
static constexpr uint64_t kSeed = 42;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
// Experiment 1: FPR vs collision threshold
// ---------------------------------------------------------------------------
static void run_exp1_threshold() {
    std::cerr << "=== Experiment 1: FPR vs collision threshold ===\n";
    csv_header("experiment,stash_type,threshold,fpr,stash_count");

    auto data = generate_data(kNumInserts, 0, kNumNeg, kSeed);

    // Baseline: plain BF
    {
        BloomFilter bf(kTotalBits, kNumHashes);
        for (uint64_t key : data.positives) {
            bf.insert(key);
        }
        double fpr = measure_fpr(bf, data.test_negatives);
        csv_row("exp1", "baseline_bf", "-", fpr, 0);
    }

    size_t stash_bits = kTotalBits / 5;  // 20%
    size_t primary_bits = kTotalBits - stash_bits;

    for (size_t threshold = 0; threshold <= kNumHashes; ++threshold) {
        // BF stash
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash), threshold,
                                   StashMode::Positive);
            for (uint64_t key : data.positives) {
                sbf.insert(key);
            }
            double fpr = measure_fpr_stashed(sbf, data.test_negatives);
            csv_row("exp1", "bf_stash_pos", threshold, fpr, sbf.stash_count());
        }
        // LP stash
        {
            size_t lp_capacity = stash_bits / 64;
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), threshold, StashMode::Positive);
            for (uint64_t key : data.positives) {
                sbf.insert(key);
            }
            double fpr = measure_fpr_stashed(sbf, data.test_negatives);
            csv_row("exp1", "lp_stash_pos", threshold, fpr, sbf.stash_count());
        }
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Experiment 2: FPR vs stash fraction
// ---------------------------------------------------------------------------
static void run_exp2_stash_fraction() {
    std::cerr << "=== Experiment 2: FPR vs stash fraction ===\n";
    csv_header("experiment,stash_type,stash_mode,stash_fraction,fpr,stash_count");

    auto data = generate_data(kNumInserts, kNumNeg, kNumNeg, kSeed);

    // Baseline
    {
        BloomFilter bf(kTotalBits, kNumHashes);
        for (uint64_t key : data.positives) {
            bf.insert(key);
        }
        double fpr = measure_fpr(bf, data.test_negatives);
        csv_row("exp2", "baseline_bf", "-", 0.0, fpr, 0);
    }

    size_t collision_threshold = 3;

    for (int frac_pct = 0; frac_pct <= 50; frac_pct += 5) {
        double frac = frac_pct / 100.0;
        size_t stash_bits = static_cast<size_t>(kTotalBits * frac);
        size_t primary_bits = kTotalBits - stash_bits;
        if (stash_bits == 0) {
            stash_bits = 1;  // avoid zero-size
        }

        // BF stash positive
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash),
                                   collision_threshold, StashMode::Positive);
            for (uint64_t key : data.positives) {
                sbf.insert(key);
            }
            double fpr = measure_fpr_stashed(sbf, data.test_negatives);
            csv_row("exp2", "bf_stash", "positive", frac, fpr, sbf.stash_count());
        }
        // BF stash negative
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash),
                                   collision_threshold, StashMode::Negative);
            for (uint64_t key : data.positives) {
                sbf.insert(key);
            }
            sbf.populate_negative_stash(data.stash_negatives.begin(),
                                        data.stash_negatives.end());
            double fpr = measure_fpr_stashed(sbf, data.test_negatives);
            csv_row("exp2", "bf_stash", "negative", frac, fpr, sbf.stash_count());
        }
        // LP stash positive
        {
            size_t lp_capacity = stash_bits / 64;
            if (lp_capacity == 0) {
                lp_capacity = 1;
            }
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), collision_threshold,
                StashMode::Positive);
            for (uint64_t key : data.positives) {
                sbf.insert(key);
            }
            double fpr = measure_fpr_stashed(sbf, data.test_negatives);
            csv_row("exp2", "lp_stash", "positive", frac, fpr, sbf.stash_count());
        }
        // LP stash negative
        {
            size_t lp_capacity = stash_bits / 64;
            if (lp_capacity == 0) {
                lp_capacity = 1;
            }
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), collision_threshold,
                StashMode::Negative);
            for (uint64_t key : data.positives) {
                sbf.insert(key);
            }
            sbf.populate_negative_stash(data.stash_negatives.begin(),
                                        data.stash_negatives.end());
            double fpr = measure_fpr_stashed(sbf, data.test_negatives);
            csv_row("exp2", "lp_stash", "negative", frac, fpr, sbf.stash_count());
        }
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Experiment 3: Compare all filter types (multiple seeds)
// ---------------------------------------------------------------------------
static void run_exp3_comparison() {
    std::cerr << "=== Experiment 3: Filter type comparison ===\n";
    csv_header("experiment,filter_type,seed,fpr,stash_count");

    size_t stash_bits = kTotalBits / 5;
    size_t primary_bits = kTotalBits - stash_bits;
    size_t collision_threshold = 3;
    size_t lp_capacity = stash_bits / 64;
    constexpr int kTrials = 10;

    for (int trial = 0; trial < kTrials; ++trial) {
        uint64_t seed = kSeed + trial;
        auto data = generate_data(kNumInserts, kNumNeg, kNumNeg, seed);

        // 1. Plain BF
        {
            BloomFilter bf(kTotalBits, kNumHashes);
            for (uint64_t key : data.positives) {
                bf.insert(key);
            }
            csv_row("exp3", "bloom_filter", seed, measure_fpr(bf, data.test_negatives), 0);
        }
        // 2. Partitioned BF
        {
            PartitionedBloomFilter pbf(kTotalBits, kNumHashes);
            for (uint64_t key : data.positives) {
                pbf.insert(key);
            }
            csv_row("exp3", "partitioned_bf", seed, measure_fpr(pbf, data.test_negatives), 0);
        }
        // 3. Stashed BF + BF stash + positive
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash),
                                   collision_threshold, StashMode::Positive);
            for (uint64_t key : data.positives) {
                sbf.insert(key);
            }
            csv_row("exp3", "stashed_bf_pos", seed,
                    measure_fpr_stashed(sbf, data.test_negatives), sbf.stash_count());
        }
        // 4. Stashed BF + BF stash + negative
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash),
                                   collision_threshold, StashMode::Negative);
            for (uint64_t key : data.positives) {
                sbf.insert(key);
            }
            sbf.populate_negative_stash(data.stash_negatives.begin(),
                                        data.stash_negatives.end());
            csv_row("exp3", "stashed_bf_neg", seed,
                    measure_fpr_stashed(sbf, data.test_negatives), sbf.stash_count());
        }
        // 5. Stashed BF + LP stash + positive
        {
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), collision_threshold,
                StashMode::Positive);
            for (uint64_t key : data.positives) {
                sbf.insert(key);
            }
            csv_row("exp3", "stashed_lp_pos", seed,
                    measure_fpr_stashed(sbf, data.test_negatives), sbf.stash_count());
        }
        // 6. Stashed BF + LP stash + negative
        {
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), collision_threshold,
                StashMode::Negative);
            for (uint64_t key : data.positives) {
                sbf.insert(key);
            }
            sbf.populate_negative_stash(data.stash_negatives.begin(),
                                        data.stash_negatives.end());
            csv_row("exp3", "stashed_lp_neg", seed,
                    measure_fpr_stashed(sbf, data.test_negatives), sbf.stash_count());
        }
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Experiment 4: Zipf distribution
// ---------------------------------------------------------------------------
static void run_exp4_zipf() {
    std::cerr << "=== Experiment 4: Zipf distribution ===\n";
    csv_header("experiment,filter_type,zipf_s,fpr,stash_count");

    size_t stash_bits = kTotalBits / 5;
    size_t primary_bits = kTotalBits - stash_bits;
    size_t collision_threshold = 3;
    size_t lp_capacity = stash_bits / 64;
    constexpr uint64_t kMaxRank = 100000;

    double skews[] = {0.5, 0.8, 1.0, 1.2, 1.5};

    for (double s : skews) {
        std::cerr << "  zipf s=" << s << "...\n";
        auto positives = generate_zipf_keys(kNumInserts, s, kMaxRank, kSeed);
        auto test_negatives = generate_uniform_keys(kNumNeg, kSeed + 100, kMaxRank + 1);
        auto stash_negatives = generate_uniform_keys(kNumNeg, kSeed + 200, kMaxRank * 2 + 1);

        // Plain BF
        {
            BloomFilter bf(kTotalBits, kNumHashes);
            for (uint64_t key : positives) {
                bf.insert(key);
            }
            csv_row("exp4", "bloom_filter", s, measure_fpr(bf, test_negatives), 0);
        }
        // Partitioned BF
        {
            PartitionedBloomFilter pbf(kTotalBits, kNumHashes);
            for (uint64_t key : positives) {
                pbf.insert(key);
            }
            csv_row("exp4", "partitioned_bf", s, measure_fpr(pbf, test_negatives), 0);
        }
        // Stashed BF + BF stash + positive
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash),
                                   collision_threshold, StashMode::Positive);
            for (uint64_t key : positives) {
                sbf.insert(key);
            }
            csv_row("exp4", "stashed_bf_pos", s,
                    measure_fpr_stashed(sbf, test_negatives), sbf.stash_count());
        }
        // Stashed BF + BF stash + negative
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash),
                                   collision_threshold, StashMode::Negative);
            for (uint64_t key : positives) {
                sbf.insert(key);
            }
            sbf.populate_negative_stash(stash_negatives.begin(), stash_negatives.end());
            csv_row("exp4", "stashed_bf_neg", s,
                    measure_fpr_stashed(sbf, test_negatives), sbf.stash_count());
        }
        // Stashed BF + LP stash + positive
        {
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), collision_threshold,
                StashMode::Positive);
            for (uint64_t key : positives) {
                sbf.insert(key);
            }
            csv_row("exp4", "stashed_lp_pos", s,
                    measure_fpr_stashed(sbf, test_negatives), sbf.stash_count());
        }
        // Stashed BF + LP stash + negative
        {
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), collision_threshold,
                StashMode::Negative);
            for (uint64_t key : positives) {
                sbf.insert(key);
            }
            sbf.populate_negative_stash(stash_negatives.begin(), stash_negatives.end());
            csv_row("exp4", "stashed_lp_neg", s,
                    measure_fpr_stashed(sbf, test_negatives), sbf.stash_count());
        }
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Experiment 5: FPR vs number of elements (load factor)
// ---------------------------------------------------------------------------
static void run_exp5_varying_n() {
    std::cerr << "=== Experiment 5: FPR vs number of elements ===\n";
    csv_header("experiment,filter_type,n,fpr,stash_count");

    size_t stash_bits = kTotalBits / 5;
    size_t primary_bits = kTotalBits - stash_bits;
    size_t collision_threshold = 3;
    size_t lp_capacity = stash_bits / 64;

    size_t n_values[] = {500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000};

    for (size_t n : n_values) {
        std::cerr << "  n=" << n << "...\n";
        auto data = generate_data(n, kNumNeg, kNumNeg, kSeed);

        // Plain BF
        {
            BloomFilter bf(kTotalBits, kNumHashes);
            for (uint64_t key : data.positives) {
                bf.insert(key);
            }
            csv_row("exp5", "bloom_filter", n, measure_fpr(bf, data.test_negatives), 0);
        }
        // Partitioned BF
        {
            PartitionedBloomFilter pbf(kTotalBits, kNumHashes);
            for (uint64_t key : data.positives) {
                pbf.insert(key);
            }
            csv_row("exp5", "partitioned_bf", n, measure_fpr(pbf, data.test_negatives), 0);
        }
        // Stashed BF + BF stash + positive
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash),
                                   collision_threshold, StashMode::Positive);
            for (uint64_t key : data.positives) {
                sbf.insert(key);
            }
            csv_row("exp5", "stashed_bf_pos", n,
                    measure_fpr_stashed(sbf, data.test_negatives), sbf.stash_count());
        }
        // Stashed BF + BF stash + negative
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash),
                                   collision_threshold, StashMode::Negative);
            for (uint64_t key : data.positives) {
                sbf.insert(key);
            }
            sbf.populate_negative_stash(data.stash_negatives.begin(),
                                        data.stash_negatives.end());
            csv_row("exp5", "stashed_bf_neg", n,
                    measure_fpr_stashed(sbf, data.test_negatives), sbf.stash_count());
        }
        // Stashed BF + LP stash + positive
        {
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), collision_threshold,
                StashMode::Positive);
            for (uint64_t key : data.positives) {
                sbf.insert(key);
            }
            csv_row("exp5", "stashed_lp_pos", n,
                    measure_fpr_stashed(sbf, data.test_negatives), sbf.stash_count());
        }
        // Stashed BF + LP stash + negative
        {
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), collision_threshold,
                StashMode::Negative);
            for (uint64_t key : data.positives) {
                sbf.insert(key);
            }
            sbf.populate_negative_stash(data.stash_negatives.begin(),
                                        data.stash_negatives.end());
            csv_row("exp5", "stashed_lp_neg", n,
                    measure_fpr_stashed(sbf, data.test_negatives), sbf.stash_count());
        }
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Experiment 6: Password data (string keys from file)
// ---------------------------------------------------------------------------
static void run_exp6_passwords(const std::string& password_file) {
    std::cerr << "=== Experiment 6: Password data ===\n";

    auto passwords = read_lines(password_file);
    if (passwords.empty()) {
        std::cerr << "  ERROR: no passwords loaded from " << password_file << "\n";
        return;
    }
    std::cerr << "  loaded " << passwords.size() << " passwords from " << password_file << "\n";

    csv_header("experiment,filter_type,n_passwords,total_bits,fpr,stash_count");

    // Scale bits to password count: ~20 bits per element for low FPR
    size_t n = passwords.size();
    size_t total_bits = n * 20;
    size_t stash_bits = total_bits / 5;
    size_t primary_bits = total_bits - stash_bits;
    size_t collision_threshold = 3;
    size_t lp_capacity = stash_bits / 64;
    if (lp_capacity == 0) {
        lp_capacity = 1;
    }

    // Generate random strings as negatives (unlikely to collide with real passwords)
    size_t n_neg = std::max(n * 10, static_cast<size_t>(10000));
    auto negatives = generate_random_strings(n_neg, 12, kSeed);

    // Also generate a set of negatives for populating the negative stash
    auto stash_negatives = generate_random_strings(n_neg, 14, kSeed + 1);

    // 1. Plain Bloom filter
    {
        BloomFilter<std::string> bf(total_bits, kNumHashes);
        for (const auto& pw : passwords) {
            bf.insert(pw);
        }
        csv_row("exp6", "bloom_filter", n, total_bits, measure_fpr(bf, negatives), 0);
    }
    // 2. Partitioned Bloom filter
    {
        PartitionedBloomFilter<std::string> pbf(total_bits, kNumHashes);
        for (const auto& pw : passwords) {
            pbf.insert(pw);
        }
        csv_row("exp6", "partitioned_bf", n, total_bits, measure_fpr(pbf, negatives), 0);
    }
    // 3. Stashed BF + BF stash + positive
    {
        BloomFilterStash<std::string> stash(stash_bits, kNumHashes);
        StashedBloomFilter<std::string> sbf(primary_bits, kNumHashes, std::move(stash),
                                            collision_threshold, StashMode::Positive);
        for (const auto& pw : passwords) {
            sbf.insert(pw);
        }
        csv_row("exp6", "stashed_bf_pos", n, total_bits,
                measure_fpr_stashed(sbf, negatives), sbf.stash_count());
    }
    // 4. Stashed BF + BF stash + negative
    {
        BloomFilterStash<std::string> stash(stash_bits, kNumHashes);
        StashedBloomFilter<std::string> sbf(primary_bits, kNumHashes, std::move(stash),
                                            collision_threshold, StashMode::Negative);
        for (const auto& pw : passwords) {
            sbf.insert(pw);
        }
        sbf.populate_negative_stash(stash_negatives.begin(), stash_negatives.end());
        csv_row("exp6", "stashed_bf_neg", n, total_bits,
                measure_fpr_stashed(sbf, negatives), sbf.stash_count());
    }
    // 5. Stashed BF + LP stash + positive
    {
        LinearProbingStash<std::string> stash(lp_capacity);
        StashedBloomFilter<std::string, DefaultHashPolicy, LinearProbingStash<std::string>> sbf(
            primary_bits, kNumHashes, std::move(stash), collision_threshold, StashMode::Positive);
        for (const auto& pw : passwords) {
            sbf.insert(pw);
        }
        csv_row("exp6", "stashed_lp_pos", n, total_bits,
                measure_fpr_stashed(sbf, negatives), sbf.stash_count());
    }
    // 6. Stashed BF + LP stash + negative
    {
        LinearProbingStash<std::string> stash(lp_capacity);
        StashedBloomFilter<std::string, DefaultHashPolicy, LinearProbingStash<std::string>> sbf(
            primary_bits, kNumHashes, std::move(stash), collision_threshold, StashMode::Negative);
        for (const auto& pw : passwords) {
            sbf.insert(pw);
        }
        sbf.populate_negative_stash(stash_negatives.begin(), stash_negatives.end());
        csv_row("exp6", "stashed_lp_neg", n, total_bits,
                measure_fpr_stashed(sbf, negatives), sbf.stash_count());
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Demo: Interactive breached-password querier
// ---------------------------------------------------------------------------
static void run_demo(const std::string& password_file) {
    auto passwords = read_lines(password_file);
    if (passwords.empty()) {
        std::cerr << "ERROR: no passwords loaded from " << password_file << "\n";
        return;
    }

    size_t n = passwords.size();
    size_t total_bits = n * 20;
    size_t stash_bits = total_bits / 5;
    size_t primary_bits = total_bits - stash_bits;
    size_t collision_threshold = 3;
    size_t lp_capacity = stash_bits / 64;
    if (lp_capacity == 0) {
        lp_capacity = 1;
    }

    // Build LP stash (positive mode) — deterministic True for stashed keys
    LinearProbingStash<std::string> stash(lp_capacity);
    StashedBloomFilter<std::string, DefaultHashPolicy, LinearProbingStash<std::string>> sbf(
        primary_bits, kNumHashes, std::move(stash), collision_threshold, StashMode::Positive);

    for (const auto& pw : passwords) {
        sbf.insert(pw);
    }

    std::cerr << "Loaded " << n << " breached passwords (" << sbf.total_bits() << " bits, "
              << sbf.stash_count() << " stashed)\n";
    std::cerr << "Enter passwords to check (one per line, Ctrl-D to quit):\n";

    std::string query;
    while (std::getline(std::cin, query)) {
        if (query.empty()) {
            continue;
        }
        ProbBool result = sbf.query(query);
        switch (result) {
            case ProbBool::True:
                std::cout << "BREACHED (certain) — password is in the stash\n";
                break;
            case ProbBool::Maybe:
                std::cout << "BREACHED (probable) — Bloom filter positive\n";
                break;
            case ProbBool::False:
                std::cout << "SAFE — not found in filter\n";
                break;
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string mode = (argc > 1) ? argv[1] : "all";
    std::string password_file = (argc > 2) ? argv[2] : "data/breached_passwords.txt";

    if (mode == "exp1" || mode == "all") {
        run_exp1_threshold();
    }
    if (mode == "exp2" || mode == "all") {
        run_exp2_stash_fraction();
    }
    if (mode == "exp3" || mode == "all") {
        run_exp3_comparison();
    }
    if (mode == "exp4" || mode == "all") {
        run_exp4_zipf();
    }
    if (mode == "exp5" || mode == "all") {
        run_exp5_varying_n();
    }
    if (mode == "exp6" || mode == "all") {
        run_exp6_passwords(password_file);
    }
    if (mode == "demo") {
        run_demo(password_file);
    }

    return 0;
}
