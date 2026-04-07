#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_set>
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

static std::vector<std::string> make_data_derived_negatives(
    const std::vector<std::string>& positives, size_t count) {
    std::vector<std::string> negatives;
    if (positives.empty() || count == 0) {
        return negatives;
    }
    negatives.reserve(count);

    std::unordered_set<std::string> positive_set(positives.begin(), positives.end());
    std::unordered_set<std::string> seen;
    seen.reserve(count * 2);

    for (size_t i = 0; negatives.size() < count; ++i) {
        const std::string& base = positives[i % positives.size()];
        std::string candidate = base + "|neg|" + std::to_string(i);
        if (positive_set.find(candidate) != positive_set.end()) {
            continue;
        }
        if (!seen.insert(candidate).second) {
            continue;
        }
        negatives.push_back(std::move(candidate));
    }
    return negatives;
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
    csv_header(
        "experiment,stash_type,threshold,"
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
//   practical — warm up on the first half of a query stream and evaluate on
//               the second half (same distribution, held-out keys).
//   oracle    — scan the evaluation half itself (upper bound).
//
// Compared against plain BF with all kTotalBits.
// ---------------------------------------------------------------------------
static void run_exp2_negative_stash() {
    std::cerr << "=== Exp 2: Negative stash FPR reduction ===\n";
    csv_header(
        "experiment,stash_type,scenario,stash_fraction,"
        "baseline_fpr,stashed_fpr,fpr_reduction_pct,false_negative_rate,stash_count");

    auto data = generate_data(kNumInserts, kNumNeg, kNumNeg, kSeed);
    size_t practical_warmup_count = data.test_negatives.size() / 2;
    std::vector<uint64_t> practical_scan(data.test_negatives.begin(),
                                         data.test_negatives.begin() + practical_warmup_count);
    std::vector<uint64_t> practical_eval(data.test_negatives.begin() + practical_warmup_count,
                                         data.test_negatives.end());
    if (practical_eval.empty()) {
        practical_eval = data.test_negatives;
    }

    // Baseline: plain BF with ALL bits
    BloomFilter<uint64_t> baseline_bf(kTotalBits, kNumHashes);
    for (uint64_t k : data.positives) {
        baseline_bf.insert(k);
    }
    double baseline_fpr = measure_fpr(baseline_bf, practical_eval);

    csv_row("exp2", "baseline_bf", "-", 0.0, baseline_fpr, baseline_fpr, 0.0, 0.0, 0);

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
        auto run_bf_neg = [&](const std::string& scenario, const std::vector<uint64_t>& scan_set,
                              const std::vector<uint64_t>& eval_set) {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash), 0,
                                   StashMode::Negative);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            sbf.populate_negative_stash(scan_set.begin(), scan_set.end());
            double fpr = measure_fpr_stashed(sbf, eval_set);
            auto pos = count_query_results(sbf, data.positives);
            double fnr = data.positives.empty()
                             ? 0.0
                             : static_cast<double>(pos.false_count) /
                                   static_cast<double>(data.positives.size());
            double reduction = baseline_fpr > 0 ? (1.0 - fpr / baseline_fpr) * 100.0 : 0.0;
            csv_row("exp2", "bf_stash", scenario, frac, baseline_fpr, fpr, reduction, fnr,
                    sbf.stash_count());
        };

        // --- LP stash negative ---
        auto run_lp_neg = [&](const std::string& scenario, const std::vector<uint64_t>& scan_set,
                              const std::vector<uint64_t>& eval_set) {
            LinearProbingStash<uint64_t> stash(lp_capacity);
            StashedBloomFilter<uint64_t, DefaultHashPolicy, LinearProbingStash<uint64_t>> sbf(
                primary_bits, kNumHashes, std::move(stash), 0, StashMode::Negative);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            sbf.populate_negative_stash(scan_set.begin(), scan_set.end());
            double fpr = measure_fpr_stashed(sbf, eval_set);
            auto pos = count_query_results(sbf, data.positives);
            double fnr = data.positives.empty()
                             ? 0.0
                             : static_cast<double>(pos.false_count) /
                                   static_cast<double>(data.positives.size());
            double reduction = baseline_fpr > 0 ? (1.0 - fpr / baseline_fpr) * 100.0 : 0.0;
            csv_row("exp2", "lp_stash", scenario, frac, baseline_fpr, fpr, reduction, fnr,
                    sbf.stash_count());
        };

        // Practical: warm-up on first half, evaluate on second half.
        run_bf_neg("practical", practical_scan, practical_eval);
        run_lp_neg("practical", practical_scan, practical_eval);

        // Oracle: scan the test set itself (best case)
        run_bf_neg("oracle", practical_eval, practical_eval);
        run_lp_neg("oracle", practical_eval, practical_eval);
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
    csv_header(
        "experiment,filter_type,n,fpr,certainty_rate,false_certainty_rate,false_negative_rate,"
        "stash_count");

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
            csv_row("exp3", "bloom_filter", n, measure_fpr(bf, data.test_negatives), 0.0, 0.0, 0.0,
                    0);
        }
        // Partitioned BF
        {
            PartitionedBloomFilter pbf(kTotalBits, kNumHashes);
            for (uint64_t k : data.positives) {
                pbf.insert(k);
            }
            csv_row("exp3", "partitioned_bf", n, measure_fpr(pbf, data.test_negatives), 0.0, 0.0,
                    0.0, 0);
        }
        // Stashed BF + BF stash + positive
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash), collision_threshold,
                                   StashMode::Positive);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            auto pos = count_query_results(sbf, data.positives);
            auto neg = count_query_results(sbf, data.test_negatives);
            csv_row("exp3", "stashed_bf_pos", n, neg.positive_rate(), pos.true_rate(),
                    neg.true_rate(), 0.0, sbf.stash_count());
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
                    neg.true_rate(), 0.0, sbf.stash_count());
        }
        // Stashed BF + BF stash + negative (practical: scan disjoint set)
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash), collision_threshold,
                                   StashMode::Negative);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            sbf.populate_negative_stash(data.stash_negatives.begin(), data.stash_negatives.end());
            auto pos = count_query_results(sbf, data.positives);
            double fnr = data.positives.empty()
                             ? 0.0
                             : static_cast<double>(pos.false_count) /
                                   static_cast<double>(data.positives.size());
            csv_row("exp3", "stashed_bf_neg", n, measure_fpr_stashed(sbf, data.test_negatives), 0.0,
                    0.0, fnr, sbf.stash_count());
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
            sbf.populate_negative_stash(data.stash_negatives.begin(), data.stash_negatives.end());
            auto pos = count_query_results(sbf, data.positives);
            double fnr = data.positives.empty()
                             ? 0.0
                             : static_cast<double>(pos.false_count) /
                                   static_cast<double>(data.positives.size());
            csv_row("exp3", "stashed_lp_neg", n, measure_fpr_stashed(sbf, data.test_negatives), 0.0,
                    0.0, fnr, sbf.stash_count());
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
    csv_header(
        "experiment,stash_type,stash_mode,stash_fraction,"
        "fpr,certainty_rate,false_certainty_rate,false_negative_rate,stash_count");

    auto data = generate_data(kNumInserts, kNumNeg, kNumNeg, kSeed);
    size_t collision_threshold = 3;

    // Baseline
    {
        BloomFilter bf(kTotalBits, kNumHashes);
        for (uint64_t k : data.positives) {
            bf.insert(k);
        }
        csv_row("exp4", "baseline_bf", "-", 0.0, measure_fpr(bf, data.test_negatives), 0.0, 0.0,
                0.0, 0);
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
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash), collision_threshold,
                                   StashMode::Positive);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            auto pos = count_query_results(sbf, data.positives);
            auto neg = count_query_results(sbf, data.test_negatives);
            csv_row("exp4", "bf_stash", "positive", frac, neg.positive_rate(), pos.true_rate(),
                    neg.true_rate(), 0.0, sbf.stash_count());
        }
        // BF stash — negative
        {
            BloomFilterStash<uint64_t> stash(stash_bits, kNumHashes);
            StashedBloomFilter sbf(primary_bits, kNumHashes, std::move(stash), collision_threshold,
                                   StashMode::Negative);
            for (uint64_t k : data.positives) {
                sbf.insert(k);
            }
            sbf.populate_negative_stash(data.stash_negatives.begin(), data.stash_negatives.end());
            auto pos = count_query_results(sbf, data.positives);
            double fnr = data.positives.empty()
                             ? 0.0
                             : static_cast<double>(pos.false_count) /
                                   static_cast<double>(data.positives.size());
            csv_row("exp4", "bf_stash", "negative", frac,
                    measure_fpr_stashed(sbf, data.test_negatives), 0.0, 0.0, fnr,
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
            csv_row("exp4", "lp_stash", "positive", frac, neg.positive_rate(), pos.true_rate(),
                    neg.true_rate(), 0.0, sbf.stash_count());
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
            sbf.populate_negative_stash(data.stash_negatives.begin(), data.stash_negatives.end());
            auto pos = count_query_results(sbf, data.positives);
            double fnr = data.positives.empty()
                             ? 0.0
                             : static_cast<double>(pos.false_count) /
                                   static_cast<double>(data.positives.size());
            csv_row("exp4", "lp_stash", "negative", frac,
                    measure_fpr_stashed(sbf, data.test_negatives), 0.0, 0.0, fnr,
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

    csv_header(
        "experiment,filter_type,n_passwords,total_bits,"
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
        csv_row("exp5", "bloom_filter", n, total_bits, 0, pos_maybe, 0, 0.0, 0, 0, 0, fpr, 0.0, 0);
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
        csv_row("exp5", "partitioned_bf", n, total_bits, 0, pos_maybe, 0, 0.0, 0, 0, 0, fpr, 0.0,
                0);
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
                pos.false_count, pos.true_rate(), neg.true_count, neg.maybe_count, neg.false_count,
                neg.positive_rate(), neg.true_rate(), sbf.stash_count());
    }
    // Stashed BF + LP stash + positive
    {
        LinearProbingStash<std::string> stash(lp_capacity);
        StashedBloomFilter<std::string, DefaultHashPolicy, LinearProbingStash<std::string>> sbf(
            primary_bits, kNumHashes, std::move(stash), collision_threshold, StashMode::Positive);
        for (const auto& pw : passwords) {
            sbf.insert(pw);
        }
        auto pos = count_query_results(sbf, passwords);
        auto neg = count_query_results(sbf, negatives);
        csv_row("exp5", "stashed_lp_pos", n, total_bits, pos.true_count, pos.maybe_count,
                pos.false_count, pos.true_rate(), neg.true_count, neg.maybe_count, neg.false_count,
                neg.positive_rate(), neg.true_rate(), sbf.stash_count());
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
                pos.false_count, pos.true_rate(), neg.true_count, neg.maybe_count, neg.false_count,
                neg.positive_rate(), neg.true_rate(), sbf.stash_count());
    }
    // Stashed BF + LP stash + negative
    {
        LinearProbingStash<std::string> stash(lp_capacity);
        StashedBloomFilter<std::string, DefaultHashPolicy, LinearProbingStash<std::string>> sbf(
            primary_bits, kNumHashes, std::move(stash), collision_threshold, StashMode::Negative);
        for (const auto& pw : passwords) {
            sbf.insert(pw);
        }
        sbf.populate_negative_stash(stash_negatives.begin(), stash_negatives.end());
        auto pos = count_query_results(sbf, passwords);
        auto neg = count_query_results(sbf, negatives);
        csv_row("exp5", "stashed_lp_neg", n, total_bits, pos.true_count, pos.maybe_count,
                pos.false_count, pos.true_rate(), neg.true_count, neg.maybe_count, neg.false_count,
                neg.positive_rate(), neg.true_rate(), sbf.stash_count());
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Experiment 6 — Data-driven hot-positive certainty workload
//
// Uses password data for both inserted keys and Zipf-weighted query streams.
// Goal: show when positive LP stash helps by reducing "Maybe" responses
// (fewer downstream exact checks) while preserving correctness.
// ---------------------------------------------------------------------------
static void run_exp6_hot_positive(const std::string& password_file) {
    std::cerr << "=== Exp 6: Hot-positive certainty workload ===\n";

    auto passwords = read_lines(password_file);
    if (passwords.empty()) {
        std::cerr << "  ERROR: no passwords loaded from " << password_file << "\n";
        return;
    }
    constexpr size_t kMaxPasswords = 100000;
    if (passwords.size() > kMaxPasswords) {
        passwords.resize(kMaxPasswords);
    }
    std::cerr << "  loaded " << passwords.size() << " passwords from " << password_file << "\n";

    size_t n = passwords.size();
    size_t total_bits = n * 20;
    size_t stash_bits = total_bits / 5;
    size_t primary_bits = total_bits - stash_bits;
    size_t collision_threshold = 3;
    size_t lp_capacity = std::max<size_t>(1, stash_bits / 64);

    size_t neg_pool_size = std::max<size_t>(5000, std::min<size_t>(30000, n));
    auto negatives = make_data_derived_negatives(passwords, neg_pool_size);

    size_t pos_queries = std::max<size_t>(100000, n * 3);
    size_t neg_queries = std::max<size_t>(50000, neg_pool_size * 10);
    auto pos_ranks =
        generate_zipf_keys(pos_queries, 1.2, static_cast<uint64_t>(n), kSeed + 600);
    auto neg_ranks = generate_zipf_keys(neg_queries, 1.1, static_cast<uint64_t>(negatives.size()),
                                        kSeed + 601);

    csv_header(
        "experiment,filter_type,n_passwords,total_bits,"
        "pos_queries,neg_queries,"
        "pos_true,pos_maybe,pos_false,"
        "neg_true,neg_maybe,neg_false,"
        "certainty_rate,fpr,false_certainty_rate,"
        "downstream_check_rate,downstream_reduction_pct,"
        "stash_count");

    double baseline_downstream_rate = 0.0;

    // Plain BF baseline
    {
        BloomFilter<std::string> bf(total_bits, kNumHashes);
        for (const auto& pw : passwords) {
            bf.insert(pw);
        }

        size_t pos_maybe = 0;
        size_t pos_false = 0;
        for (uint64_t rank : pos_ranks) {
            if (bf.query(passwords[rank - 1])) {
                ++pos_maybe;
            } else {
                ++pos_false;
            }
        }

        size_t neg_maybe = 0;
        size_t neg_false = 0;
        for (uint64_t rank : neg_ranks) {
            if (bf.query(negatives[rank - 1])) {
                ++neg_maybe;
            } else {
                ++neg_false;
            }
        }

        size_t total_queries = pos_ranks.size() + neg_ranks.size();
        baseline_downstream_rate = total_queries > 0
                                       ? static_cast<double>(pos_maybe + neg_maybe) /
                                             static_cast<double>(total_queries)
                                       : 0.0;
        double fpr = neg_ranks.empty()
                         ? 0.0
                         : static_cast<double>(neg_maybe) / static_cast<double>(neg_ranks.size());

        csv_row("exp6", "bloom_filter", n, total_bits, pos_ranks.size(), neg_ranks.size(), 0,
                pos_maybe, pos_false, 0, neg_maybe, neg_false, 0.0, fpr, 0.0,
                baseline_downstream_rate, 0.0, 0);
    }

    // Stashed BF with LP stash in positive mode
    {
        LinearProbingStash<std::string> stash(lp_capacity);
        StashedBloomFilter<std::string, DefaultHashPolicy, LinearProbingStash<std::string>> sbf(
            primary_bits, kNumHashes, std::move(stash), collision_threshold, StashMode::Positive);
        for (const auto& pw : passwords) {
            sbf.insert(pw);
        }

        size_t pos_true = 0;
        size_t pos_maybe = 0;
        size_t pos_false = 0;
        for (uint64_t rank : pos_ranks) {
            switch (sbf.query(passwords[rank - 1])) {
                case ProbBool::True:
                    ++pos_true;
                    break;
                case ProbBool::Maybe:
                    ++pos_maybe;
                    break;
                case ProbBool::False:
                    ++pos_false;
                    break;
            }
        }

        size_t neg_true = 0;
        size_t neg_maybe = 0;
        size_t neg_false = 0;
        for (uint64_t rank : neg_ranks) {
            switch (sbf.query(negatives[rank - 1])) {
                case ProbBool::True:
                    ++neg_true;
                    break;
                case ProbBool::Maybe:
                    ++neg_maybe;
                    break;
                case ProbBool::False:
                    ++neg_false;
                    break;
            }
        }

        size_t total_queries = pos_ranks.size() + neg_ranks.size();
        double certainty_rate =
            pos_ranks.empty() ? 0.0 : static_cast<double>(pos_true) / pos_ranks.size();
        double fpr = neg_ranks.empty()
                         ? 0.0
                         : static_cast<double>(neg_true + neg_maybe) /
                               static_cast<double>(neg_ranks.size());
        double false_certainty = neg_ranks.empty()
                                     ? 0.0
                                     : static_cast<double>(neg_true) /
                                           static_cast<double>(neg_ranks.size());
        double downstream_rate = total_queries > 0
                                     ? static_cast<double>(pos_maybe + neg_maybe) /
                                           static_cast<double>(total_queries)
                                     : 0.0;
        double downstream_reduction = baseline_downstream_rate > 0
                                          ? (1.0 - downstream_rate / baseline_downstream_rate) *
                                                100.0
                                          : 0.0;

        csv_row("exp6", "stashed_lp_pos", n, total_bits, pos_ranks.size(), neg_ranks.size(),
                pos_true, pos_maybe, pos_false, neg_true, neg_maybe, neg_false, certainty_rate, fpr,
                false_certainty, downstream_rate, downstream_reduction, sbf.stash_count());
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Experiment 7 — Repeated negatives with warm-up (data-driven)
//
// Uses data-derived negatives sampled with Zipf locality.
// Warm-up queries populate an LP negative stash; evaluation uses a fresh draw
// from the same distribution to test practical carryover.
// ---------------------------------------------------------------------------
static void run_exp7_repeated_negative(const std::string& password_file) {
    std::cerr << "=== Exp 7: Repeated-negative warm-up ===\n";

    auto passwords = read_lines(password_file);
    if (passwords.empty()) {
        std::cerr << "  ERROR: no passwords loaded from " << password_file << "\n";
        return;
    }
    constexpr size_t kMaxPasswords = 100000;
    if (passwords.size() > kMaxPasswords) {
        passwords.resize(kMaxPasswords);
    }
    std::cerr << "  loaded " << passwords.size() << " passwords from " << password_file << "\n";

    size_t n = passwords.size();
    size_t total_bits = n * 12;
    constexpr double kStashFraction = 0.20;
    size_t stash_bits = static_cast<size_t>(static_cast<double>(total_bits) * kStashFraction);
    if (stash_bits == 0) {
        stash_bits = 1;
    }
    size_t primary_bits = total_bits - stash_bits;
    size_t lp_capacity = std::max<size_t>(1, stash_bits / 64);

    size_t neg_pool_size = std::max<size_t>(5000, std::min<size_t>(30000, n));
    auto negatives = make_data_derived_negatives(passwords, neg_pool_size);
    size_t warmup_queries = std::max<size_t>(200000, neg_pool_size * 20);
    size_t eval_queries = warmup_queries;
    auto warmup_ranks = generate_zipf_keys(warmup_queries, 1.15,
                                           static_cast<uint64_t>(negatives.size()), kSeed + 700);
    auto eval_ranks = generate_zipf_keys(eval_queries, 1.15, static_cast<uint64_t>(negatives.size()),
                                         kSeed + 701);

    csv_header(
        "experiment,filter_type,n_passwords,total_bits,"
        "stash_fraction,neg_pool_size,warmup_queries,eval_queries,"
        "fpr,false_negative_rate,fpr_reduction_pct,"
        "stash_count");

    double baseline_fpr = 0.0;

    // Plain BF baseline on the evaluation query stream.
    {
        BloomFilter<std::string> bf(total_bits, kNumHashes);
        for (const auto& pw : passwords) {
            bf.insert(pw);
        }

        size_t fp = 0;
        for (uint64_t rank : eval_ranks) {
            if (bf.query(negatives[rank - 1])) {
                ++fp;
            }
        }
        baseline_fpr = eval_ranks.empty()
                           ? 0.0
                           : static_cast<double>(fp) / static_cast<double>(eval_ranks.size());

        csv_row("exp7", "bloom_filter", n, total_bits, 0.0, negatives.size(), warmup_ranks.size(),
                eval_ranks.size(), baseline_fpr, 0.0, 0.0, 0);
    }

    // LP negative stash with warm-up then holdout evaluation.
    {
        LinearProbingStash<std::string> stash(lp_capacity);
        StashedBloomFilter<std::string, DefaultHashPolicy, LinearProbingStash<std::string>> sbf(
            primary_bits, kNumHashes, std::move(stash), 0, StashMode::Negative);
        for (const auto& pw : passwords) {
            sbf.insert(pw);
        }

        for (uint64_t rank : warmup_ranks) {
            sbf.insert_negative(negatives[rank - 1]);
        }

        size_t fp_eval = 0;
        for (uint64_t rank : eval_ranks) {
            if (sbf.query_bool(negatives[rank - 1])) {
                ++fp_eval;
            }
        }
        double fpr = eval_ranks.empty()
                         ? 0.0
                         : static_cast<double>(fp_eval) / static_cast<double>(eval_ranks.size());
        auto pos = count_query_results(sbf, passwords);
        double fnr = passwords.empty()
                         ? 0.0
                         : static_cast<double>(pos.false_count) /
                               static_cast<double>(passwords.size());
        double reduction = baseline_fpr > 0 ? (1.0 - fpr / baseline_fpr) * 100.0 : 0.0;

        csv_row("exp7", "stashed_lp_neg", n, total_bits, kStashFraction, negatives.size(),
                warmup_ranks.size(), eval_ranks.size(), fpr, fnr, reduction, sbf.stash_count());
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
    if (mode == "exp6" || mode == "all") {
        run_exp6_hot_positive(password_file);
    }
    if (mode == "exp7" || mode == "all") {
        run_exp7_repeated_negative(password_file);
    }
    if (mode == "demo") {
        run_demo(password_file);
    }

    return 0;
}
