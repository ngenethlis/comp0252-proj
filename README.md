# Proposal

The Cuckoo filter as presented in lectures features a stash where elements can
be placed when insertion fails, to reduce critical load. We propose doing
something similar with a Bloom filter to target a lower false positive rate
given the same space. Specifically, we aim to sacrifice some number of bits to
use as a deterministic stash, where elements can be stored if inserting them
into the Bloom filter would cause some number of collisions. This also has the
added benefit that anything in the stash can be queried as deterministically
yes rather than possibly yes. We hypothesise that this would be most effective
in a heavy hitter scenario. We’d run experiments to determine the optimal value
for the number of collisions (e.g. is it best to stash it if only one bit
collides or if all bits collide) and the optimal trade off between Bloom filter
size and stash size, to determine if this could improve false positive rate
while maintaining the same storage and time complexities. We'd compare it to
the performance of a standard Bloom filter given the extra space. We would
trial the performance of these approaches in a real-world scenario via
attempting to make a breached password querier, as this allows both for heavy
hitter scenarios (e.g. with popular passwords) and due to the classification as
deterministically yes, deterministically no, or maybe being useful.

## Current experimental findings vs a standard Bloom filter

### Where stashed Bloom filter can be better

- **Positive stash mode can increase certainty for known positives**: some positive
  queries return `True` (certain) instead of only `Maybe`.
- This can be useful when downstream logic benefits from a "certain hit" signal
  for a subset of inserted keys.

### Where stashed Bloom filter is worse

- **As a general FPR optimizer, it is usually worse than plain BF** at equal total
  bits: moving bits from the primary Bloom filter into a stash increases primary
  false positives in practical settings.
- **Negative stash mode can introduce false negatives** (especially at high load,
  when a probabilistic stash is used and becomes saturated).
- The strong "oracle" negative-stash results are an **upper bound only** and rely
  on pre-scanning the evaluation queries; they are not practical generalization
  results.

### Robustness (overfitting) check under distribution shift

We added robustness scenarios for Exp 7 using two query models:
- `count_weighted` (from dataset `:count` values)
- `zipf` (synthetic heavy-tail baseline)

Each model is evaluated in:
- `in_dist` (same warm-up/eval pool and distribution)
- `cross_pool` (disjoint eval pool)
- `shifted_distribution` (same pool, shifted eval distribution)

In current runs (hibp-100k), negative stash shows strong in-distribution gains
but can fail under shift:
- **count_weighted**: +95.07% (`in_dist`), -410.33% (`cross_pool`),
  -40.63% (`shifted_distribution`)
- **zipf**: +84.90% (`in_dist`), -93.23% (`cross_pool`),
  +74.77% (`shifted_distribution`)

So negative stash performance is **locality-sensitive**: it can be excellent when
future queries resemble warm-up traffic, but may degrade sharply when traffic
moves to different keys.

### Practical takeaway

Use plain Bloom filter if your objective is the classic one: **no false negatives
and best overall FPR for a fixed bit budget**. Use stashed variants only when you
explicitly want the extra certainty semantics (`True` vs `Maybe`) and can accept
the associated trade-offs.
