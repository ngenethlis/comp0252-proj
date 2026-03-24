# Proposal

The Cuckoo filter as presented in lectures features a stash where elements can
be placed when insertion fails, to reduce critical load. We propose doing
something similar with a Bloom filter to target a lower false positive rate
given the same space. Specifically, we aim to sacrifice some number of bits to
use as a deterministic stash, where elements can be stored if inserting them
into the Bloom filter would cause some number of collisions. We predict that
there's two possible ways a stash could be useful. Either, it can act as
overflow for the Bloom filter by storing positive elements that otherwise would
have collided, or it can store negative elements not in the set that would
otherwise have been flagged as a false positive by the Bloom filter. A benefit
to the first approach is that we can classify things not just as
deterministically no or maybe yes, but as deterministically yes as well. We
hypothesise that either of these approaches would be most effective in a heavy
hitter scenario. We’d run experiments to determine the optimal value for
the number of collisions (e.g. is it best to stash it if only one bit collides
or if all bits collide) and the optimal trade off between Bloom filter size and
stash size, to determine if this could improve false positive rate while
maintaining the same storage and time complexities. We'd compare it to
the performance of a standard Bloom filter given the extra space, as well as a
partitioned Bloom filter (again by using the second set either for positive or
negative elements). We would trial the performance of these approaches in a
real-world scenario via attempting to make a malware detector, as this allows
both for heavy hitter scenarios (e.g. with popular benignware) and for
classification as deterministically yes, deterministically no, or maybe.
