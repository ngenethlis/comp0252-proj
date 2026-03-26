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
