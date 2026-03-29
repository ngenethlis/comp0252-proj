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
// Default experiment parameters
// ---------------------------------------------------------------------------
static constexpr size_t kTotalBits = 100000;
static constexpr size_t kNumHashes = 7;
static constexpr size_t kNumInserts = 5000;
static constexpr size_t kNumNeg = 50000;
static constexpr uint64_t kSeed = 42;

// ---------------------------------------------------------------------------
// CSV helpers
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
// Experiment 1 — Certainty analysis vs collision threshold
//
// Question: how does the threshold control certainty (True vs Maybe) for
// positive queries, and does the BF stash produce false certainty (True)
// for negative queries?
//
// LP stash stores fingerprints → True means genuinely certain.
// BF stash is itself probabilistic → True on a negative is a lie.
// ---------------------------------------------------------------------------
static void run_exp1_certainty() {
    std::cerr << "=== Exp 1: Certainty vs collision threshold ===\n";
    csv_header("experiment,stash_type,threshold,"
               "pos_true,pos_maybe,pos_false,certainty_rate,"
               "neg_true,neg_maybe,neg_false,fpr,false_certainty_rate,"
               "stash_count");

    auto data = generate_data(kNumInserts, 0, kNumNeg, kSeed);
    size_t stash_bits = kTotalBits / 5;
    size_t primary_bits = kTotalBits - stash_bits;
    size_t lp_capacity = stash_bits / 64;

    // Baseline plain BF (no stash → certainty_rate = 0, false_certainty = 0)
    {
        BloomFilter bf(kTotalBits, kNumHashes);
        for (uint64_t k : data.positives) {
            bf.insert(k);
        }
        double fpr = measure_fpr(bf, data.test_negatives);
        csv_row("exp1", "baseline_bf", "-", 0, kNumInserts, 0, 0.0, 0, 0, 0, fpr, 0.0, 0);
    }

    for (size_t t = 0; t <= kNumHashes; ++t) {
        // BF stash — positive mode
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash), t,
                                   StashMode::Positive);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }

            auto pos = count_query_results(sbf, data.positives);
            auto neg = count_query_results(sbf, data.test_negatives);

            csv_row("exp1", "bf_stash", t, pos.true_count, pos.maybe_count, pos.false_count,
                    pos.true_rate(), neg.true_count, neg.maybe_count, neg.false_count,
                    neg.positive_rate(), neg.true_rate(), sbf.stash_count());
        }
        // LP stash — positive mode
        {
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), t, StashMode::Positive);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }

            auto pos = count_query_results(sbf, data.positives);
            auto neg = count_query_results(sbf, data.test_negatives);

            csv_row("exp1", "lp_stash", t, pos.true_count, pos.maybe_count, pos.false_count,
                    pos.true_rate(), neg.true_count, neg.maybe_count, neg.false_count,
                    neg.positive_rate(), neg.true_rate(), sbf.stash_count());
        }
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Experiment 2 — Negative stash: can it reduce FPR?
//
// The positive stash cannot reduce FPR (it only upgrades Maybe → True for
// real positives). The negative stash CAN: it stores known false positives,
// turning their query result from Maybe → False.
//
// We test two scenarios:
//   practical — scan a disjoint set of negatives, stash FPs found,
//               then measure FPR on the held-out test negatives.
//   oracle    — scan the test negatives themselves (upper bound).
//
// Compared against plain BF with all kTotalBits.
// ---------------------------------------------------------------------------
static void run_exp2_negative_stash() {
    std::cerr << "=== Exp 2: Negative stash FPR reduction ===\n";
    csv_header("experiment,stash_type,scenario,stash_fraction,"
               "baseline_fpr,stashed_fpr,fpr_reduction_pct,stash_count");

    auto data = generate_data(kNumInserts, kNumNeg, kNumNeg, kSeed);

    // Baseline: plain BF with ALL bits
    BloomFilter<uint64_t> baseline_bf(kTotalBits, kNumHashes);
    for (uint64_t k : data.positives) {
        baseline_bf.insert(k);
    }
    double baseline_fpr = measure_fpr(baseline_bf, data.test_negatives);

    csv_row("exp2", "baseline_bf", "-", 0.0, baseline_fpr, baseline_fpr, 0.0, 0);

    for (int frac_pct = 5; frac_pct <= 50; frac_pct += 5) {
        double frac = frac_pct / 100.0;
        size_t stash_bits = static_cast<size_t>(kTotalBits * frac);
        size_t primary_bits = kTotalBits - stash_bits;
        if (stash_bits == 0) {
            stash_bits = 1;
        }
        size_t lp_capacity = stash_bits / 64;
        if (lp_capacity == 0) {
            lp_capacity = 1;
        }

        // --- BF stash negative ---
        auto run_bf_neg = [&](const std::string& scenario,
                              const std::vector<uint64_t>& scan_set) {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash), 0,
                                   StashMode::Negative);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            sbf.populate_negative_stash(scan_set.begin(), scan_set.end());
            double fpr = measure_fpr_stashed(sbf, data.test_negatives);
            double reduction =
                baseline_fpr > 0 ? (1.0 - fpr / baseline_fpr) * 100.0 : 0.0;
            csv_row("exp2", "bf_stash", scenario, frac, baseline_fpr, fpr, reduction,
                    sbf.stash_count());
        };

        // --- LP stash negative ---
        auto run_lp_neg = [&](const std::string& scenario,
                              const std::vector<uint64_t>& scan_set) {
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), 0, StashMode::Negative);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            sbf.populate_negative_stash(scan_set.begin(), scan_set.end());
            double fpr = measure_fpr_stashed(sbf, data.test_negatives);
            double reduction =
                baseline_fpr > 0 ? (1.0 - fpr / baseline_fpr) * 100.0 : 0.0;
            csv_row("exp2", "lp_stash", scenario, frac, baseline_fpr, fpr, reduction,
                    sbf.stash_count());
        };

        // Practical: scan disjoint negatives
        run_bf_neg("practical", data.stash_negatives);
        run_lp_neg("practical", data.stash_negatives);

        // Oracle: scan the test set itself (best case)
        run_bf_neg("oracle", data.test_negatives);
        run_lp_neg("oracle", data.test_negatives);
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Experiment 3 — FPR vs load factor for all filter types
//
// The stash steals bits from the primary BF.  At what load does this start
// to matter?  Includes ProbBool breakdown for stashed types.
// ---------------------------------------------------------------------------
static void run_exp3_load() {
    std::cerr << "=== Exp 3: FPR vs load factor ===\n";
    csv_header("experiment,filter_type,n,fpr,certainty_rate,false_certainty_rate,stash_count");

    size_t stash_bits = kTotalBits / 5;
    size_t primary_bits = kTotalBits - stash_bits;
    size_t lp_capacity = stash_bits / 64;
    size_t collision_threshold = 3;

    size_t n_values[] = {500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000};

    for (size_t n : n_values) {
        std::cerr << "  n=" << n << "...\n";
        auto data = generate_data(n, kNumNeg, kNumNeg, kSeed);

        // Plain BF
        {
            BloomFilter bf(kTotalBits, kNumHashes);
            for (uint64_t k : data.positives) {
                bf.insert(k);
            }
            csv_row("exp3", "bloom_filter", n, measure_fpr(bf, data.test_negatives), 0.0, 0.0,
                    0);
        }
        // Partitioned BF
        {
            PartitionedBloomFilter pbf(kTotalBits, kNumHashes);
            for (uint64_t k : data.positives) {
                pbf.insert(k);
            }
            csv_row("exp3", "partitioned_bf", n, measure_fpr(pbf, data.test_negatives), 0.0,
                    0.0, 0);
        }
        // Stashed BF + BF stash + positive
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash),
                                   collision_threshold, StashMode::Positive);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            auto pos = count_query_results(sbf, data.positives);
            auto neg = count_query_results(sbf, data.test_negatives);
            csv_row("exp3", "stashed_bf_pos", n, neg.positive_rate(), pos.true_rate(),
                    neg.true_rate(), sbf.stash_count());
        }
        // Stashed BF + LP stash + positive
        {
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), collision_threshold,
                StashMode::Positive);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            auto pos = count_query_results(sbf, data.positives);
            auto neg = count_query_results(sbf, data.test_negatives);
            csv_row("exp3", "stashed_lp_pos", n, neg.positive_rate(), pos.true_rate(),
                    neg.true_rate(), sbf.stash_count());
        }
        // Stashed BF + BF stash + negative (practical: scan disjoint set)
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash),
                                   collision_threshold, StashMode::Negative);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            sbf.populate_negative_stash(data.stash_negatives.begin(),
                                        data.stash_negatives.end());
            csv_row("exp3", "stashed_bf_neg", n,
                    measure_fpr_stashed(sbf, data.test_negatives), 0.0, 0.0,
                    sbf.stash_count());
        }
        // Stashed BF + LP stash + negative (practical: scan disjoint set)
        {
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), collision_threshold,
                StashMode::Negative);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            sbf.populate_negative_stash(data.stash_negatives.begin(),
                                        data.stash_negatives.end());
            csv_row("exp3", "stashed_lp_neg", n,
                    measure_fpr_stashed(sbf, data.test_negatives), 0.0, 0.0,
                    sbf.stash_count());
        }
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Experiment 4 — Stash fraction sweep
//
// How much of the bit budget should go to the stash?
// Compares positive & negative modes for both stash types.
// ---------------------------------------------------------------------------
static void run_exp4_stash_fraction() {
    std::cerr << "=== Exp 4: Stash fraction sweep ===\n";
    csv_header("experiment,stash_type,stash_mode,stash_fraction,"
               "fpr,certainty_rate,false_certainty_rate,stash_count");

    auto data = generate_data(kNumInserts, kNumNeg, kNumNeg, kSeed);
    size_t collision_threshold = 3;

    // Baseline
    {
        BloomFilter bf(kTotalBits, kNumHashes);
        for (uint64_t k : data.positives) {
            bf.insert(k);
        }
        csv_row("exp4", "baseline_bf", "-", 0.0,
                measure_fpr(bf, data.test_negatives), 0.0, 0.0, 0);
    }

    for (int frac_pct = 5; frac_pct <= 50; frac_pct += 5) {
        double frac = frac_pct / 100.0;
        size_t stash_bits = static_cast<size_t>(kTotalBits * frac);
        size_t primary_bits = kTotalBits - stash_bits;
        if (stash_bits == 0) {
            stash_bits = 1;
        }
        size_t lp_capacity = stash_bits / 64;
        if (lp_capacity == 0) {
            lp_capacity = 1;
        }

        // BF stash — positive
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash),
                                   collision_threshold, StashMode::Positive);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            auto pos = count_query_results(sbf, data.positives);
            auto neg = count_query_results(sbf, data.test_negatives);
            csv_row("exp4", "bf_stash", "positive", frac, neg.positive_rate(),
                    pos.true_rate(), neg.true_rate(), sbf.stash_count());
        }
        // BF stash — negative
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash),
                                   collision_threshold, StashMode::Negative);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            sbf.populate_negative_stash(data.stash_negatives.begin(),
                                        data.stash_negatives.end());
            csv_row("exp4", "bf_stash", "negative", frac,
                    measure_fpr_stashed(sbf, data.test_negatives), 0.0, 0.0,
                    sbf.stash_count());
        }
        // LP stash — positive
        {
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), collision_threshold,
                StashMode::Positive);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            auto pos = count_query_results(sbf, data.positives);
            auto neg = count_query_results(sbf, data.test_negatives);
            csv_row("exp4", "lp_stash", "positive", frac, neg.positive_rate(),
                    pos.true_rate(), neg.true_rate(), sbf.stash_count());
        }
        // LP stash — negative
        {
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), collision_threshold,
                StashMode::Negative);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            sbf.populate_negative_stash(data.stash_negatives.begin(),
                                        data.stash_negatives.end());
            csv_row("exp4", "lp_stash", "negative", frac,
                    measure_fpr_stashed(sbf, data.test_negatives), 0.0, 0.0,
                    sbf.stash_count());
        }
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Experiment 5 — Password workload with ProbBool breakdown
//
// Real-world string-key evaluation.  Reports per-query-class breakdown
// so the user can see how many queries get True vs Maybe vs False for
// both positives (should be True/Maybe) and negatives (should be False).
// ---------------------------------------------------------------------------
static void run_exp5_passwords(const std::string& password_file) {
    std::cerr << "=== Exp 5: Password workload ===\n";

    auto passwords = read_lines(password_file);
    if (passwords.empty()) {
        std::cerr << "  ERROR: no passwords loaded from " << password_file << "\n";
        return;
    }
    std::cerr << "  loaded " << passwords.size() << " passwords from " << password_file << "\n";

    csv_header("experiment,filter_type,n_passwords,total_bits,"
               "pos_true,pos_maybe,pos_false,certainty_rate,"
               "neg_true,neg_maybe,neg_false,fpr,false_certainty_rate,"
               "stash_count");

    size_t n = passwords.size();
    size_t total_bits = n * 20;
    size_t stash_bits = total_bits / 5;
    size_t primary_bits = total_bits - stash_bits;
    size_t collision_threshold = 3;
    size_t lp_capacity = stash_bits / 64;
    if (lp_capacity == 0) {
        lp_capacity = 1;
    }

    size_t n_neg = std::max(n * 10, static_cast<size_t>(10000));
    auto negatives = generate_random_strings(n_neg, 12, kSeed);
    auto stash_negatives = generate_random_strings(n_neg, 14, kSeed + 1);

    // Plain BF
    {
        BloomFilter<std::string> bf(total_bits, kNumHashes);
        for (const auto& pw : passwords) {
            bf.insert(pw);
        }
        // For plain BF: all positives are "Maybe", no True/False certainty
        size_t pos_maybe = 0;
        for (const auto& pw : passwords) {
            if (bf.query(pw)) {
                ++pos_maybe;
            }
        }
        double fpr = measure_fpr(bf, negatives);
        csv_row("exp5", "bloom_filter", n, total_bits, 0, pos_maybe, 0, 0.0, 0, 0, 0, fpr,
                0.0, 0);
    }
    // Partitioned BF
    {
        PartitionedBloomFilter<std::string> pbf(total_bits, kNumHashes);
        for (const auto& pw : passwords) {
            pbf.insert(pw);
        }
        size_t pos_maybe = 0;
        for (const auto& pw : passwords) {
            if (pbf.query(pw)) {
                ++pos_maybe;
            }
        }
        double fpr = measure_fpr(pbf, negatives);
        csv_row("exp5", "partitioned_bf", n, total_bits, 0, pos_maybe, 0, 0.0, 0, 0, 0, fpr,
                0.0, 0);
    }
    // Stashed BF + BF stash + positive
    {
        BloomFilterStash<std::string> stash(stash_bits, kNumHashes);
        StashedBloomFilter<std::string> sbf(primary_bits, kNumHashes, std::move(stash),
                                            collision_threshold, StashMode::Positive);
        for (const auto& pw : passwords) {
            sbf.insert(pw);
        }
        auto pos = count_query_results(sbf, passwords);
        auto neg = count_query_results(sbf, negatives);
        csv_row("exp5", "stashed_bf_pos", n, total_bits, pos.true_count, pos.maybe_count,
                pos.false_count, pos.true_rate(), neg.true_count, neg.maybe_count,
                neg.false_count, neg.positive_rate(), neg.true_rate(), sbf.stash_count());
    }
    // Stashed BF + LP stash + positive
    {
        LinearProbingStash<std::string> stash(lp_capacity);
        StashedBloomFilter<std::string, DefaultHashPolicy, LinearProbingStash<std::string>> sbf(
            primary_bits, kNumHashes, std::move(stash), collision_threshold,
            StashMode::Positive);
        for (const auto& pw : passwords) {
            sbf.insert(pw);
        }
        auto pos = count_query_results(sbf, passwords);
        auto neg = count_query_results(sbf, negatives);
        csv_row("exp5", "stashed_lp_pos", n, total_bits, pos.true_count, pos.maybe_count,
                pos.false_count, pos.true_rate(), neg.true_count, neg.maybe_count,
                neg.false_count, neg.positive_rate(), neg.true_rate(), sbf.stash_count());
    }
    // Stashed BF + BF stash + negative
    {
        BloomFilterStash<std::string> stash(stash_bits, kNumHashes);
        StashedBloomFilter<std::string> sbf(primary_bits, kNumHashes, std::move(stash),
                                            collision_threshold, StashMode::Negative);
        for (const auto& pw : passwords) {
            sbf.insert(pw);
        }
        sbf.populate_negative_stash(stash_negatives.begin(), stash_negatives.end());
        auto pos = count_query_results(sbf, passwords);
        auto neg = count_query_results(sbf, negatives);
        csv_row("exp5", "stashed_bf_neg", n, total_bits, pos.true_count, pos.maybe_count,
                pos.false_count, pos.true_rate(), neg.true_count, neg.maybe_count,
                neg.false_count, neg.positive_rate(), neg.true_rate(), sbf.stash_count());
    }
    // Stashed BF + LP stash + negative
    {
        LinearProbingStash<std::string> stash(lp_capacity);
        StashedBloomFilter<std::string, DefaultHashPolicy, LinearProbingStash<std::string>> sbf(
            primary_bits, kNumHashes, std::move(stash), collision_threshold,
            StashMode::Negative);
        for (const auto& pw : passwords) {
            sbf.insert(pw);
        }
        sbf.populate_negative_stash(stash_negatives.begin(), stash_negatives.end());
        auto pos = count_query_results(sbf, passwords);
        auto neg = count_query_results(sbf, negatives);
        csv_row("exp5", "stashed_lp_neg", n, total_bits, pos.true_count, pos.maybe_count,
                pos.false_count, pos.true_rate(), neg.true_count, neg.maybe_count,
                neg.false_count, neg.positive_rate(), neg.true_rate(), sbf.stash_count());
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
                std::cout << "BREACHED (certain) -- password is in the stash\n";
                break;
            case ProbBool::Maybe:
                std::cout << "BREACHED (probable) -- Bloom filter positive\n";
                break;
            case ProbBool::False:
                std::cout << "SAFE -- not found in filter\n";
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
        run_exp1_certainty();
    }
    if (mode == "exp2" || mode == "all") {
        run_exp2_negative_stash();
    }
    if (mode == "exp3" || mode == "all") {
        run_exp3_load();
    }
    if (mode == "exp4" || mode == "all") {
        run_exp4_stash_fraction();
    }
    if (mode == "exp5" || mode == "all") {
        run_exp5_passwords(password_file);
    }
    if (mode == "demo") {
        run_demo(password_file);
    }

    return 0;
}
