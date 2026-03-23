# Proposal

The Cuckoo filter as presented in lectures features a stash where elements can be placed when insertion fails, to
reduce critical load. We propose doing something similar with a Bloom filter to target a lower false positive rate
given the same space. Specifically, we aim to sacrifice some number of bits to use as a deterministic stash, where
elements can be stored if inserting them into the Bloom filter would cause some number of collisions. We’d then
run experiments to determine the optimal value for the number of collisions (e.g. is it best to stash it if only one bit
collides or if all bits collide) and the optimal trade off between Bloom filter size and stash size, to determine if
this could improve false positive rate while maintaining roughly the same storage and time complexities. We can
also compare it to the performance of a learned Bloom filter through the literature. Our hypothesis is that this new
approach will be particularly useful in heavy hitter scenarios.
