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
static constexpr uint64_t kDefaultSeed = 42;
static uint64_t gSeed = kDefaultSeed;

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
    const std::vector<std::string>& positives, size_t count, const std::string& tag = "neg") {
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
        std::string candidate = base + "|" + tag + "|" + std::to_string(i);
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

static void split_weighted_entries(const std::vector<WeightedStringEntry>& entries,
                                   std::vector<std::string>* keys, std::vector<uint64_t>* counts) {
    keys->clear();
    counts->clear();
    keys->reserve(entries.size());
    counts->reserve(entries.size());
    for (const auto& entry : entries) {
        keys->push_back(entry.key);
        counts->push_back(entry.count);
    }
}

static uint64_t sum_weights(const std::vector<uint64_t>& weights) {
    uint64_t total = 0;
    for (uint64_t w : weights) {
        total += w;
    }
    return total;
}

static std::vector<uint64_t> derive_negative_weights(const std::vector<uint64_t>& source_weights,
                                                     size_t target_size) {
    std::vector<uint64_t> weights;
    if (source_weights.empty() || target_size == 0) {
        return weights;
    }
    weights.reserve(target_size);
    for (size_t i = 0; i < target_size; ++i) {
        weights.push_back(source_weights[i % source_weights.size()]);
    }
    return weights;
}

static std::vector<size_t> ranks_to_zero_based_indices(const std::vector<uint64_t>& ranks,
                                                       size_t max_size) {
    std::vector<size_t> indices;
    if (max_size == 0) {
        return indices;
    }
    indices.reserve(ranks.size());
    for (uint64_t rank : ranks) {
        size_t idx = rank > 0 ? static_cast<size_t>(rank - 1) : 0;
        if (idx >= max_size) {
            idx = max_size - 1;
        }
        indices.push_back(idx);
    }
    return indices;
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

    auto data = generate_data(kNumInserts, 0, kNumNeg, gSeed);
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
    // Baseline blocked BF (same interface, different layout)
    {
        BlockedBloomFilter blocked_bf(kTotalBits, kNumHashes);
        for (uint64_t k : data.positives) {
            blocked_bf.insert(k);
        }
        double fpr = measure_fpr(blocked_bf, data.test_negatives);
        csv_row("exp1", "blocked_bf", "-", 0, kNumInserts, 0, 0.0, 0, 0, 0, fpr, 0.0, 0);
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

    auto data = generate_data(kNumInserts, kNumNeg, kNumNeg, gSeed);
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
    {
        BlockedBloomFilter<uint64_t> blocked_bf(kTotalBits, kNumHashes);
        for (uint64_t k : data.positives) {
            blocked_bf.insert(k);
        }
        double blocked_fpr = measure_fpr(blocked_bf, practical_eval);
        double blocked_reduction =
            baseline_fpr > 0 ? (1.0 - blocked_fpr / baseline_fpr) * 100.0 : 0.0;
        csv_row("exp2", "blocked_bf", "-", 0.0, baseline_fpr, blocked_fpr, blocked_reduction, 0.0,
                0);
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
            double fnr = data.positives.empty() ? 0.0
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
            double fnr = data.positives.empty() ? 0.0
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

    size_t n_values[] = {500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 30000};

    for (size_t n : n_values) {
        std::cerr << "  n=" << n << "...\n";
        auto data = generate_data(n, kNumNeg, kNumNeg, gSeed);

        // Plain BF
        {
            BloomFilter bf(kTotalBits, kNumHashes);
            for (uint64_t k : data.positives) {
                bf.insert(k);
            }
            csv_row("exp3", "bloom_filter", n, measure_fpr(bf, data.test_negatives), 0.0, 0.0, 0.0,
                    0);
        }
        // Blocked BF baseline (implemented via BlockedBloomFilter)
        {
            BlockedBloomFilter pbf(kTotalBits, kNumHashes);
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
            double fnr = data.positives.empty() ? 0.0
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
            double fnr = data.positives.empty() ? 0.0
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

    auto data = generate_data(kNumInserts, kNumNeg, kNumNeg, gSeed);
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
    {
        BlockedBloomFilter blocked_bf(kTotalBits, kNumHashes);
        for (uint64_t k : data.positives) {
            blocked_bf.insert(k);
        }
        csv_row("exp4", "blocked_bf", "-", 0.0, measure_fpr(blocked_bf, data.test_negatives), 0.0,
                0.0, 0.0, 0);
    }

    for (int frac_pct = 5; frac_pct <= 80; frac_pct += 5) {
        double requested_frac = frac_pct / 100.0;
        size_t stash_bits = static_cast<size_t>(kTotalBits * requested_frac);
        if (stash_bits == 0) {
            stash_bits = 1;
        }
        if (stash_bits >= kTotalBits) {
            stash_bits = kTotalBits - 1;
        }
        size_t primary_bits = kTotalBits - stash_bits;
        double frac = static_cast<double>(stash_bits) / static_cast<double>(kTotalBits);
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
            double fnr = data.positives.empty() ? 0.0
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
            double fnr = data.positives.empty() ? 0.0
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

    auto weighted_passwords = read_weighted_lines(password_file);
    if (weighted_passwords.empty()) {
        std::cerr << "  ERROR: no passwords loaded from " << password_file << "\n";
        return;
    }
    std::vector<std::string> passwords;
    std::vector<uint64_t> password_counts;
    split_weighted_entries(weighted_passwords, &passwords, &password_counts);
    uint64_t total_positive_queries = sum_weights(password_counts);

    std::cerr << "  loaded " << passwords.size() << " passwords from " << password_file
              << " (weighted queries=" << total_positive_queries << ")\n";

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
    auto negatives = generate_random_strings(n_neg, 12, gSeed);
    auto stash_negatives = generate_random_strings(n_neg, 14, gSeed + 1);

    // Plain BF
    {
        BloomFilter<std::string> bf(total_bits, kNumHashes);
        for (const auto& pw : passwords) {
            bf.insert(pw);
        }
        // Positive queries are weighted by the dataset's count field.
        uint64_t pos_maybe = 0;
        uint64_t pos_false = 0;
        for (size_t i = 0; i < passwords.size(); ++i) {
            if (bf.query(passwords[i])) {
                pos_maybe += password_counts[i];
            } else {
                pos_false += password_counts[i];
            }
        }
        double fpr = measure_fpr(bf, negatives);
        csv_row("exp5", "bloom_filter", n, total_bits, 0, pos_maybe, pos_false, 0.0, 0, 0, 0, fpr,
                0.0, 0);
    }
    // Blocked BF baseline (implemented via BlockedBloomFilter)
    {
        BlockedBloomFilter<std::string> pbf(total_bits, kNumHashes);
        for (const auto& pw : passwords) {
            pbf.insert(pw);
        }
        uint64_t pos_maybe = 0;
        uint64_t pos_false = 0;
        for (size_t i = 0; i < passwords.size(); ++i) {
            if (pbf.query(passwords[i])) {
                pos_maybe += password_counts[i];
            } else {
                pos_false += password_counts[i];
            }
        }
        double fpr = measure_fpr(pbf, negatives);
        csv_row("exp5", "partitioned_bf", n, total_bits, 0, pos_maybe, pos_false, 0.0, 0, 0, 0, fpr,
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
        uint64_t pos_true = 0;
        uint64_t pos_maybe = 0;
        uint64_t pos_false = 0;
        for (size_t i = 0; i < passwords.size(); ++i) {
            switch (sbf.query(passwords[i])) {
                case ProbBool::True:
                    pos_true += password_counts[i];
                    break;
                case ProbBool::Maybe:
                    pos_maybe += password_counts[i];
                    break;
                case ProbBool::False:
                    pos_false += password_counts[i];
                    break;
            }
        }
        auto neg = count_query_results(sbf, negatives);
        double certainty_rate =
            total_positive_queries > 0
                ? static_cast<double>(pos_true) / static_cast<double>(total_positive_queries)
                : 0.0;
        csv_row("exp5", "stashed_bf_pos", n, total_bits, pos_true, pos_maybe, pos_false,
                certainty_rate, neg.true_count, neg.maybe_count, neg.false_count,
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
        uint64_t pos_true = 0;
        uint64_t pos_maybe = 0;
        uint64_t pos_false = 0;
        for (size_t i = 0; i < passwords.size(); ++i) {
            switch (sbf.query(passwords[i])) {
                case ProbBool::True:
                    pos_true += password_counts[i];
                    break;
                case ProbBool::Maybe:
                    pos_maybe += password_counts[i];
                    break;
                case ProbBool::False:
                    pos_false += password_counts[i];
                    break;
            }
        }
        auto neg = count_query_results(sbf, negatives);
        double certainty_rate =
            total_positive_queries > 0
                ? static_cast<double>(pos_true) / static_cast<double>(total_positive_queries)
                : 0.0;
        csv_row("exp5", "stashed_lp_pos", n, total_bits, pos_true, pos_maybe, pos_false,
                certainty_rate, neg.true_count, neg.maybe_count, neg.false_count,
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
        uint64_t pos_true = 0;
        uint64_t pos_maybe = 0;
        uint64_t pos_false = 0;
        for (size_t i = 0; i < passwords.size(); ++i) {
            switch (sbf.query(passwords[i])) {
                case ProbBool::True:
                    pos_true += password_counts[i];
                    break;
                case ProbBool::Maybe:
                    pos_maybe += password_counts[i];
                    break;
                case ProbBool::False:
                    pos_false += password_counts[i];
                    break;
            }
        }
        auto neg = count_query_results(sbf, negatives);
        double certainty_rate =
            total_positive_queries > 0
                ? static_cast<double>(pos_true) / static_cast<double>(total_positive_queries)
                : 0.0;
        csv_row("exp5", "stashed_bf_neg", n, total_bits, pos_true, pos_maybe, pos_false,
                certainty_rate, neg.true_count, neg.maybe_count, neg.false_count,
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
        uint64_t pos_true = 0;
        uint64_t pos_maybe = 0;
        uint64_t pos_false = 0;
        for (size_t i = 0; i < passwords.size(); ++i) {
            switch (sbf.query(passwords[i])) {
                case ProbBool::True:
                    pos_true += password_counts[i];
                    break;
                case ProbBool::Maybe:
                    pos_maybe += password_counts[i];
                    break;
                case ProbBool::False:
                    pos_false += password_counts[i];
                    break;
            }
        }
        auto neg = count_query_results(sbf, negatives);
        double certainty_rate =
            total_positive_queries > 0
                ? static_cast<double>(pos_true) / static_cast<double>(total_positive_queries)
                : 0.0;
        csv_row("exp5", "stashed_lp_neg", n, total_bits, pos_true, pos_maybe, pos_false,
                certainty_rate, neg.true_count, neg.maybe_count, neg.false_count,
                neg.positive_rate(), neg.true_rate(), sbf.stash_count());
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Experiment 6 — Data-driven hot-positive certainty workload
//
// Uses password data for both inserted keys and two query models:
//   - count_weighted: sample queries by dataset count field
//   - zipf: synthetic heavy-tail over key rank
// Goal: compare when positive LP stash reduces "Maybe" responses.
// ---------------------------------------------------------------------------
static void run_exp6_hot_positive(const std::string& password_file) {
    std::cerr << "=== Exp 6: Hot-positive certainty workload ===\n";

    auto weighted_passwords = read_weighted_lines(password_file);
    if (weighted_passwords.empty()) {
        std::cerr << "  ERROR: no passwords loaded from " << password_file << "\n";
        return;
    }
    std::vector<std::string> passwords;
    std::vector<uint64_t> password_counts;
    split_weighted_entries(weighted_passwords, &passwords, &password_counts);
    constexpr size_t kMaxPasswords = 100000;
    if (passwords.size() > kMaxPasswords) {
        passwords.resize(kMaxPasswords);
        password_counts.resize(kMaxPasswords);
    }
    std::cerr << "  loaded " << passwords.size() << " passwords from " << password_file
              << " (weighted queries=" << sum_weights(password_counts) << ")\n";

    size_t n = passwords.size();
    size_t total_bits = n * 20;
    size_t stash_bits = total_bits / 5;
    size_t primary_bits = total_bits - stash_bits;
    size_t collision_threshold = 3;
    size_t lp_capacity = std::max<size_t>(1, stash_bits / 64);

    size_t neg_pool_size = std::max<size_t>(5000, std::min<size_t>(30000, n));
    auto negatives = make_data_derived_negatives(passwords, neg_pool_size);
    auto negative_weights = derive_negative_weights(password_counts, negatives.size());

    size_t pos_queries = std::max<size_t>(100000, n * 3);
    size_t neg_queries = std::max<size_t>(50000, neg_pool_size * 10);
    struct QueryModel {
        std::string name;
        std::vector<size_t> pos_indices;
        std::vector<size_t> neg_indices;
    };
    std::vector<QueryModel> query_models;
    query_models.push_back({"count_weighted",
                            sample_weighted_indices(pos_queries, password_counts, gSeed + 600),
                            sample_weighted_indices(neg_queries, negative_weights, gSeed + 601)});
    query_models.push_back(
        {"zipf",
         ranks_to_zero_based_indices(
             generate_zipf_keys(pos_queries, 1.2, static_cast<uint64_t>(passwords.size()),
                                gSeed + 602),
             passwords.size()),
         ranks_to_zero_based_indices(
             generate_zipf_keys(neg_queries, 1.1, static_cast<uint64_t>(negatives.size()),
                                gSeed + 603),
             negatives.size())});

    csv_header(
        "experiment,query_model,filter_type,n_passwords,total_bits,"
        "pos_queries,neg_queries,"
        "pos_true,pos_maybe,pos_false,"
        "neg_true,neg_maybe,neg_false,"
        "certainty_rate,fpr,false_certainty_rate,"
        "downstream_check_rate,downstream_reduction_pct,"
        "stash_count");

    for (const auto& model : query_models) {
        double baseline_downstream_rate = 0.0;

        // Plain BF baseline
        {
            BloomFilter<std::string> bf(total_bits, kNumHashes);
            for (const auto& pw : passwords) {
                bf.insert(pw);
            }

            size_t pos_maybe = 0;
            size_t pos_false = 0;
            for (size_t idx : model.pos_indices) {
                if (bf.query(passwords[idx])) {
                    ++pos_maybe;
                } else {
                    ++pos_false;
                }
            }

            size_t neg_maybe = 0;
            size_t neg_false = 0;
            for (size_t idx : model.neg_indices) {
                if (bf.query(negatives[idx])) {
                    ++neg_maybe;
                } else {
                    ++neg_false;
                }
            }

            size_t total_queries = model.pos_indices.size() + model.neg_indices.size();
            baseline_downstream_rate = total_queries > 0
                                           ? static_cast<double>(pos_maybe + neg_maybe) /
                                                 static_cast<double>(total_queries)
                                           : 0.0;
            double baseline_fpr = model.neg_indices.empty()
                                      ? 0.0
                                      : static_cast<double>(neg_maybe) /
                                            static_cast<double>(model.neg_indices.size());

            csv_row("exp6", model.name, "bloom_filter", n, total_bits, model.pos_indices.size(),
                    model.neg_indices.size(), 0, pos_maybe, pos_false, 0, neg_maybe, neg_false, 0.0,
                    baseline_fpr, 0.0, baseline_downstream_rate, 0.0, 0);
        }

        // Blocked BF baseline
        {
            BlockedBloomFilter<std::string> blocked_bf(total_bits, kNumHashes);
            for (const auto& pw : passwords) {
                blocked_bf.insert(pw);
            }

            size_t pos_maybe = 0;
            size_t pos_false = 0;
            for (size_t idx : model.pos_indices) {
                if (blocked_bf.query(passwords[idx])) {
                    ++pos_maybe;
                } else {
                    ++pos_false;
                }
            }

            size_t neg_maybe = 0;
            size_t neg_false = 0;
            for (size_t idx : model.neg_indices) {
                if (blocked_bf.query(negatives[idx])) {
                    ++neg_maybe;
                } else {
                    ++neg_false;
                }
            }

            size_t total_queries = model.pos_indices.size() + model.neg_indices.size();
            double downstream_rate = total_queries > 0
                                         ? static_cast<double>(pos_maybe + neg_maybe) /
                                               static_cast<double>(total_queries)
                                         : 0.0;
            double fpr = model.neg_indices.empty()
                             ? 0.0
                             : static_cast<double>(neg_maybe) /
                                   static_cast<double>(model.neg_indices.size());
            double downstream_reduction =
                baseline_downstream_rate > 0
                    ? (1.0 - downstream_rate / baseline_downstream_rate) * 100.0
                    : 0.0;

            csv_row("exp6", model.name, "partitioned_bf", n, total_bits, model.pos_indices.size(),
                    model.neg_indices.size(), 0, pos_maybe, pos_false, 0, neg_maybe, neg_false, 0.0,
                    fpr, 0.0, downstream_rate, downstream_reduction, 0);
        }

        // Stashed BF with LP stash in positive mode
        {
            LinearProbingStash<std::string> stash(lp_capacity);
            StashedBloomFilter<std::string, DefaultHashPolicy, LinearProbingStash<std::string>> sbf(
                primary_bits, kNumHashes, std::move(stash), collision_threshold,
                StashMode::Positive);
            for (const auto& pw : passwords) {
                sbf.insert(pw);
            }

            size_t pos_true = 0;
            size_t pos_maybe = 0;
            size_t pos_false = 0;
            for (size_t idx : model.pos_indices) {
                switch (sbf.query(passwords[idx])) {
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
            for (size_t idx : model.neg_indices) {
                switch (sbf.query(negatives[idx])) {
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

            size_t total_queries = model.pos_indices.size() + model.neg_indices.size();
            double certainty_rate =
                model.pos_indices.empty()
                    ? 0.0
                    : static_cast<double>(pos_true) / static_cast<double>(model.pos_indices.size());
            double fpr = model.neg_indices.empty()
                             ? 0.0
                             : static_cast<double>(neg_true + neg_maybe) /
                                   static_cast<double>(model.neg_indices.size());
            double false_certainty =
                model.neg_indices.empty()
                    ? 0.0
                    : static_cast<double>(neg_true) / static_cast<double>(model.neg_indices.size());
            double downstream_rate = total_queries > 0
                                         ? static_cast<double>(pos_maybe + neg_maybe) /
                                               static_cast<double>(total_queries)
                                         : 0.0;
            double downstream_reduction =
                baseline_downstream_rate > 0
                    ? (1.0 - downstream_rate / baseline_downstream_rate) * 100.0
                    : 0.0;

            csv_row("exp6", model.name, "stashed_lp_pos", n, total_bits, model.pos_indices.size(),
                    model.neg_indices.size(), pos_true, pos_maybe, pos_false, neg_true, neg_maybe,
                    neg_false, certainty_rate, fpr, false_certainty, downstream_rate,
                    downstream_reduction, sbf.stash_count());
        }
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Experiment 7 — Repeated negatives with warm-up (data-driven)
//
// Uses data-derived negatives under two query models:
//   - count_weighted: sample queries by dataset count field
//   - zipf: synthetic heavy-tail over key rank
// Warm-up queries populate an LP negative stash; evaluation uses a fresh draw
// from the same distribution to test practical carryover.
// ---------------------------------------------------------------------------
static void run_exp7_repeated_negative(const std::string& password_file) {
    std::cerr << "=== Exp 7: Repeated-negative warm-up ===\n";

    auto weighted_passwords = read_weighted_lines(password_file);
    if (weighted_passwords.empty()) {
        std::cerr << "  ERROR: no passwords loaded from " << password_file << "\n";
        return;
    }
    std::vector<std::string> passwords;
    std::vector<uint64_t> password_counts;
    split_weighted_entries(weighted_passwords, &passwords, &password_counts);
    constexpr size_t kMaxPasswords = 100000;
    if (passwords.size() > kMaxPasswords) {
        passwords.resize(kMaxPasswords);
        password_counts.resize(kMaxPasswords);
    }
    std::cerr << "  loaded " << passwords.size() << " passwords from " << password_file
              << " (weighted queries=" << sum_weights(password_counts) << ")\n";

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
    auto warmup_negatives = make_data_derived_negatives(passwords, neg_pool_size, "negA");
    auto cross_pool_negatives = make_data_derived_negatives(passwords, neg_pool_size, "negB");
    auto negative_weights = derive_negative_weights(password_counts, warmup_negatives.size());
    auto shifted_negative_weights = negative_weights;
    std::reverse(shifted_negative_weights.begin(), shifted_negative_weights.end());
    size_t warmup_queries = std::max<size_t>(200000, neg_pool_size * 20);
    size_t eval_queries = warmup_queries;

    struct EvalScenario {
        std::string name;
        const std::vector<std::string>* eval_pool = nullptr;
        std::vector<size_t> eval_indices;
    };

    struct QueryModel {
        std::string name;
        std::vector<size_t> warmup_indices;
        std::vector<EvalScenario> scenarios;
    };

    std::vector<QueryModel> query_models;
    {
        QueryModel count_model;
        count_model.name = "count_weighted";
        count_model.warmup_indices =
            sample_weighted_indices(warmup_queries, negative_weights, gSeed + 700);
        count_model.scenarios.push_back(
            {"in_dist", &warmup_negatives,
             sample_weighted_indices(eval_queries, negative_weights, gSeed + 701)});
        count_model.scenarios.push_back(
            {"cross_pool", &cross_pool_negatives,
             sample_weighted_indices(eval_queries, negative_weights, gSeed + 702)});
        count_model.scenarios.push_back(
            {"shifted_distribution", &warmup_negatives,
             sample_weighted_indices(eval_queries, shifted_negative_weights, gSeed + 703)});
        query_models.push_back(std::move(count_model));
    }
    {
        QueryModel zipf_model;
        zipf_model.name = "zipf";
        zipf_model.warmup_indices = ranks_to_zero_based_indices(
            generate_zipf_keys(warmup_queries, 1.15, static_cast<uint64_t>(warmup_negatives.size()),
                               gSeed + 710),
            warmup_negatives.size());
        zipf_model.scenarios.push_back(
            {"in_dist", &warmup_negatives,
             ranks_to_zero_based_indices(
                 generate_zipf_keys(eval_queries, 1.15,
                                    static_cast<uint64_t>(warmup_negatives.size()), gSeed + 711),
                 warmup_negatives.size())});
        zipf_model.scenarios.push_back(
            {"cross_pool", &cross_pool_negatives,
             ranks_to_zero_based_indices(
                 generate_zipf_keys(eval_queries, 1.15,
                                    static_cast<uint64_t>(cross_pool_negatives.size()),
                                    gSeed + 712),
                 cross_pool_negatives.size())});
        zipf_model.scenarios.push_back(
            {"shifted_distribution", &warmup_negatives,
             ranks_to_zero_based_indices(
                 generate_zipf_keys(eval_queries, 0.85,
                                    static_cast<uint64_t>(warmup_negatives.size()), gSeed + 713),
                 warmup_negatives.size())});
        query_models.push_back(std::move(zipf_model));
    }

    csv_header(
        "experiment,query_model,scenario,filter_type,n_passwords,total_bits,"
        "stash_fraction,neg_pool_size,warmup_queries,eval_queries,"
        "fpr,false_negative_rate,fpr_reduction_pct,"
        "stash_count");

    for (const auto& model : query_models) {
        for (const auto& scenario : model.scenarios) {
            const auto& eval_pool = *scenario.eval_pool;
            double baseline_fpr = 0.0;

            // Plain BF baseline on this evaluation stream.
            {
                BloomFilter<std::string> bf(total_bits, kNumHashes);
                for (const auto& pw : passwords) {
                    bf.insert(pw);
                }

                size_t fp = 0;
                for (size_t idx : scenario.eval_indices) {
                    if (bf.query(eval_pool[idx])) {
                        ++fp;
                    }
                }
                baseline_fpr = scenario.eval_indices.empty()
                                   ? 0.0
                                   : static_cast<double>(fp) /
                                         static_cast<double>(scenario.eval_indices.size());

                csv_row("exp7", model.name, scenario.name, "bloom_filter", n, total_bits, 0.0,
                        eval_pool.size(), model.warmup_indices.size(), scenario.eval_indices.size(),
                        baseline_fpr, 0.0, 0.0, 0);
            }

            // Blocked BF baseline on this evaluation stream.
            {
                BlockedBloomFilter<std::string> blocked_bf(total_bits, kNumHashes);
                for (const auto& pw : passwords) {
                    blocked_bf.insert(pw);
                }

                size_t fp = 0;
                for (size_t idx : scenario.eval_indices) {
                    if (blocked_bf.query(eval_pool[idx])) {
                        ++fp;
                    }
                }
                double blocked_fpr = scenario.eval_indices.empty()
                                         ? 0.0
                                         : static_cast<double>(fp) /
                                               static_cast<double>(scenario.eval_indices.size());
                double blocked_reduction =
                    baseline_fpr > 0 ? (1.0 - blocked_fpr / baseline_fpr) * 100.0 : 0.0;

                csv_row("exp7", model.name, scenario.name, "partitioned_bf", n, total_bits, 0.0,
                        eval_pool.size(), model.warmup_indices.size(), scenario.eval_indices.size(),
                        blocked_fpr, 0.0, blocked_reduction, 0);
            }

            // LP negative stash with warm-up then this scenario's evaluation stream.
            {
                LinearProbingStash<std::string> stash(lp_capacity);
                StashedBloomFilter<std::string, DefaultHashPolicy, LinearProbingStash<std::string>>
                    sbf(primary_bits, kNumHashes, std::move(stash), 0, StashMode::Negative);
                for (const auto& pw : passwords) {
                    sbf.insert(pw);
                }

                for (size_t idx : model.warmup_indices) {
                    sbf.insert_negative(warmup_negatives[idx]);
                }

                size_t fp_eval = 0;
                for (size_t idx : scenario.eval_indices) {
                    if (sbf.query_bool(eval_pool[idx])) {
                        ++fp_eval;
                    }
                }
                double fpr = scenario.eval_indices.empty()
                                 ? 0.0
                                 : static_cast<double>(fp_eval) /
                                       static_cast<double>(scenario.eval_indices.size());
                uint64_t pos_total = 0;
                uint64_t pos_false = 0;
                for (size_t i = 0; i < passwords.size(); ++i) {
                    pos_total += password_counts[i];
                    if (sbf.query(passwords[i]) == ProbBool::False) {
                        pos_false += password_counts[i];
                    }
                }
                double fnr = pos_total == 0
                                 ? 0.0
                                 : static_cast<double>(pos_false) / static_cast<double>(pos_total);
                double reduction = baseline_fpr > 0 ? (1.0 - fpr / baseline_fpr) * 100.0 : 0.0;

                csv_row("exp7", model.name, scenario.name, "stashed_lp_neg", n, total_bits,
                        kStashFraction, eval_pool.size(), model.warmup_indices.size(),
                        scenario.eval_indices.size(), fpr, fnr, reduction, sbf.stash_count());
            }
        }
    }
    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Experiment 8 — Warm-up budget sweep (negative LP stash)
//
// Measures how much warm-up traffic is needed before negative-stash benefits
// saturate, and how this behaves under distribution shift.
// ---------------------------------------------------------------------------
static void run_exp8_warmup_budget(const std::string& password_file) {
    std::cerr << "=== Exp 8: Warm-up budget sweep ===\n";

    auto weighted_passwords = read_weighted_lines(password_file);
    if (weighted_passwords.empty()) {
        std::cerr << "  ERROR: no passwords loaded from " << password_file << "\n";
        return;
    }
    std::vector<std::string> passwords;
    std::vector<uint64_t> password_counts;
    split_weighted_entries(weighted_passwords, &passwords, &password_counts);
    constexpr size_t kMaxPasswords = 100000;
    if (passwords.size() > kMaxPasswords) {
        passwords.resize(kMaxPasswords);
        password_counts.resize(kMaxPasswords);
    }
    std::cerr << "  loaded " << passwords.size() << " passwords from " << password_file
              << " (weighted queries=" << sum_weights(password_counts) << ")\n";

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
    auto warmup_negatives = make_data_derived_negatives(passwords, neg_pool_size, "negA");
    auto cross_pool_negatives = make_data_derived_negatives(passwords, neg_pool_size, "negB");
    auto negative_weights = derive_negative_weights(password_counts, warmup_negatives.size());
    auto shifted_negative_weights = negative_weights;
    std::reverse(shifted_negative_weights.begin(), shifted_negative_weights.end());

    size_t eval_queries = std::max<size_t>(200000, neg_pool_size * 20);
    std::vector<size_t> warmup_budgets = {
        0,
        std::max<size_t>(1000, eval_queries / 40),
        std::max<size_t>(2000, eval_queries / 20),
        std::max<size_t>(5000, eval_queries / 10),
        std::max<size_t>(10000, eval_queries / 4),
        std::max<size_t>(20000, eval_queries / 2),
        eval_queries,
        eval_queries * 2,
    };

    struct EvalScenario {
        std::string name;
        const std::vector<std::string>* eval_pool = nullptr;
        std::vector<size_t> eval_indices;
    };

    struct QueryModel {
        std::string name;
        std::vector<EvalScenario> scenarios;
    };

    std::vector<QueryModel> query_models;
    {
        QueryModel count_model;
        count_model.name = "count_weighted";
        count_model.scenarios.push_back(
            {"in_dist", &warmup_negatives,
             sample_weighted_indices(eval_queries, negative_weights, gSeed + 900)});
        count_model.scenarios.push_back(
            {"cross_pool", &cross_pool_negatives,
             sample_weighted_indices(eval_queries, negative_weights, gSeed + 901)});
        count_model.scenarios.push_back(
            {"shifted_distribution", &warmup_negatives,
             sample_weighted_indices(eval_queries, shifted_negative_weights, gSeed + 902)});
        query_models.push_back(std::move(count_model));
    }
    {
        QueryModel zipf_model;
        zipf_model.name = "zipf";
        zipf_model.scenarios.push_back(
            {"in_dist", &warmup_negatives,
             ranks_to_zero_based_indices(
                 generate_zipf_keys(eval_queries, 1.15,
                                    static_cast<uint64_t>(warmup_negatives.size()), gSeed + 910),
                 warmup_negatives.size())});
        zipf_model.scenarios.push_back(
            {"cross_pool", &cross_pool_negatives,
             ranks_to_zero_based_indices(
                 generate_zipf_keys(eval_queries, 1.15,
                                    static_cast<uint64_t>(cross_pool_negatives.size()),
                                    gSeed + 911),
                 cross_pool_negatives.size())});
        zipf_model.scenarios.push_back(
            {"shifted_distribution", &warmup_negatives,
             ranks_to_zero_based_indices(
                 generate_zipf_keys(eval_queries, 0.85,
                                    static_cast<uint64_t>(warmup_negatives.size()), gSeed + 912),
                 warmup_negatives.size())});
        query_models.push_back(std::move(zipf_model));
    }

    csv_header(
        "experiment,query_model,scenario,filter_type,n_passwords,total_bits,"
        "stash_fraction,neg_pool_size,warmup_queries,eval_queries,"
        "fpr,false_negative_rate,fpr_reduction_pct,stash_count");

    for (const auto& model : query_models) {
        for (const auto& scenario : model.scenarios) {
            const auto& eval_pool = *scenario.eval_pool;

            // Baseline for this model/scenario.
            double baseline_fpr = 0.0;
            {
                BloomFilter<std::string> bf(total_bits, kNumHashes);
                for (const auto& pw : passwords) {
                    bf.insert(pw);
                }

                size_t fp = 0;
                for (size_t idx : scenario.eval_indices) {
                    if (bf.query(eval_pool[idx])) {
                        ++fp;
                    }
                }
                baseline_fpr = scenario.eval_indices.empty()
                                   ? 0.0
                                   : static_cast<double>(fp) /
                                         static_cast<double>(scenario.eval_indices.size());

                csv_row("exp8", model.name, scenario.name, "bloom_filter", n, total_bits, 0.0,
                        eval_pool.size(), 0, scenario.eval_indices.size(), baseline_fpr, 0.0, 0.0,
                        0);
            }
            {
                BlockedBloomFilter<std::string> blocked_bf(total_bits, kNumHashes);
                for (const auto& pw : passwords) {
                    blocked_bf.insert(pw);
                }

                size_t fp = 0;
                for (size_t idx : scenario.eval_indices) {
                    if (blocked_bf.query(eval_pool[idx])) {
                        ++fp;
                    }
                }
                double blocked_fpr = scenario.eval_indices.empty()
                                         ? 0.0
                                         : static_cast<double>(fp) /
                                               static_cast<double>(scenario.eval_indices.size());
                double blocked_reduction =
                    baseline_fpr > 0 ? (1.0 - blocked_fpr / baseline_fpr) * 100.0 : 0.0;

                csv_row("exp8", model.name, scenario.name, "partitioned_bf", n, total_bits, 0.0,
                        eval_pool.size(), 0, scenario.eval_indices.size(), blocked_fpr, 0.0,
                        blocked_reduction, 0);
            }

            // Sweep warm-up budgets.
            for (size_t budget_idx = 0; budget_idx < warmup_budgets.size(); ++budget_idx) {
                size_t warmup_queries = warmup_budgets[budget_idx];

                std::vector<size_t> warmup_indices;
                if (model.name == "count_weighted") {
                    warmup_indices =
                        sample_weighted_indices(warmup_queries, negative_weights,
                                                gSeed + 920 + static_cast<uint64_t>(budget_idx));
                } else {
                    warmup_indices = ranks_to_zero_based_indices(
                        generate_zipf_keys(warmup_queries, 1.15,
                                           static_cast<uint64_t>(warmup_negatives.size()),
                                           gSeed + 940 + static_cast<uint64_t>(budget_idx)),
                        warmup_negatives.size());
                }

                LinearProbingStash<std::string> stash(lp_capacity);
                StashedBloomFilter<std::string, DefaultHashPolicy, LinearProbingStash<std::string>>
                    sbf(primary_bits, kNumHashes, std::move(stash), 0, StashMode::Negative);
                for (const auto& pw : passwords) {
                    sbf.insert(pw);
                }

                for (size_t idx : warmup_indices) {
                    sbf.insert_negative(warmup_negatives[idx]);
                }

                size_t fp_eval = 0;
                for (size_t idx : scenario.eval_indices) {
                    if (sbf.query_bool(eval_pool[idx])) {
                        ++fp_eval;
                    }
                }
                double fpr = scenario.eval_indices.empty()
                                 ? 0.0
                                 : static_cast<double>(fp_eval) /
                                       static_cast<double>(scenario.eval_indices.size());

                uint64_t pos_total = 0;
                uint64_t pos_false = 0;
                for (size_t i = 0; i < passwords.size(); ++i) {
                    pos_total += password_counts[i];
                    if (sbf.query(passwords[i]) == ProbBool::False) {
                        pos_false += password_counts[i];
                    }
                }
                double fnr = pos_total == 0
                                 ? 0.0
                                 : static_cast<double>(pos_false) / static_cast<double>(pos_total);
                double reduction = baseline_fpr > 0 ? (1.0 - fpr / baseline_fpr) * 100.0 : 0.0;

                csv_row("exp8", model.name, scenario.name, "stashed_lp_neg", n, total_bits,
                        kStashFraction, eval_pool.size(), warmup_queries,
                        scenario.eval_indices.size(), fpr, fnr, reduction, sbf.stash_count());
            }
        }
    }

    std::cerr << "  done.\n";
}

// ---------------------------------------------------------------------------
// Demo: Interactive breached-password querier
// ---------------------------------------------------------------------------
static void run_demo(const std::string& password_file) {
    auto weighted_passwords = read_weighted_lines(password_file);
    if (weighted_passwords.empty()) {
        std::cerr << "ERROR: no passwords loaded from " << password_file << "\n";
        return;
    }
    std::vector<std::string> passwords;
    std::vector<uint64_t> password_counts;
    split_weighted_entries(weighted_passwords, &passwords, &password_counts);

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
    std::string password_file = "data/breached_passwords.txt";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--seed=", 0) == 0) {
            try {
                gSeed = static_cast<uint64_t>(std::stoull(arg.substr(7)));
            } catch (...) {
                std::cerr << "WARNING: invalid seed argument '" << arg
                          << "', using default seed=" << gSeed << "\n";
            }
        } else {
            password_file = arg;
        }
    }

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
    if (mode == "exp8" || mode == "all") {
        run_exp8_warmup_budget(password_file);
    }
    if (mode == "demo") {
        run_demo(password_file);
    }

    return 0;
}
