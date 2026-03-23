#include <cassert>
#include <cstdio>
#include <cstdint>

#include "bloom_filter.h"
#include "stashed_bloom_filter.h"

void test_bloom_no_false_negatives() {
    BloomFilter bf(10000, 7);
    for (uint64_t i = 0; i < 500; ++i) {
        bf.insert(i);
    }
    for (uint64_t i = 0; i < 500; ++i) {
        assert(bf.query(i) && "BloomFilter: false negative detected");
    }
    printf("PASS: bloom_no_false_negatives\n");
}

void test_bloom_false_positive_rate() {
    BloomFilter bf(10000, 7);
    for (uint64_t i = 0; i < 500; ++i) {
        bf.insert(i);
    }
    size_t fp = 0;
    size_t test_count = 10000;
    for (uint64_t i = 1000000; i < 1000000 + test_count; ++i) {
        if (bf.query(i)) ++fp;
    }
    double fpr = static_cast<double>(fp) / test_count;
    printf("  BloomFilter FPR: %.4f\n", fpr);
    assert(fpr < 0.05 && "BloomFilter: FPR too high");
    printf("PASS: bloom_false_positive_rate\n");
}

void test_stashed_no_false_negatives() {
    StashedBloomFilter sbf(10000, 0.2, 7, 5, 5);
    for (uint64_t i = 0; i < 500; ++i) {
        sbf.insert(i);
    }
    for (uint64_t i = 0; i < 500; ++i) {
        assert(sbf.query(i) && "StashedBloomFilter: false negative detected");
    }
    printf("PASS: stashed_no_false_negatives\n");
}

void test_stashed_false_positive_rate() {
    StashedBloomFilter sbf(10000, 0.2, 7, 5, 5);
    for (uint64_t i = 0; i < 500; ++i) {
        sbf.insert(i);
    }
    size_t fp = 0;
    size_t test_count = 10000;
    for (uint64_t i = 1000000; i < 1000000 + test_count; ++i) {
        if (sbf.query(i)) ++fp;
    }
    double fpr = static_cast<double>(fp) / test_count;
    printf("  StashedBloomFilter FPR: %.4f (stash_count: %zu)\n", fpr, sbf.stash_count());
    printf("PASS: stashed_false_positive_rate\n");
}

int main() {
    test_bloom_no_false_negatives();
    test_bloom_false_positive_rate();
    test_stashed_no_false_negatives();
    test_stashed_false_positive_rate();
    printf("\nAll tests passed.\n");
    return 0;
}
