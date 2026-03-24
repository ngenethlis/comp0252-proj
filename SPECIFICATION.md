# Project Bloomfilter specifications

## Normal bloomfilter

attirbutes : vector<bits>


methods:
ctr : size, number of hashes

insert : key -> void

query : key -> bool

## Stashed Bloomfilter

attributes : normal bloomfilter / vector<bits>, stash set, collision_limit (number of bits set by hash_f that are already set), is stash +ve or -ve

methods:
ctr : size, number of hashes, size of stash (number of elements in stash), collisions to insert key in stash

insert : key -> void :
  if insertion into bloomfilter would cause more than collision_limit : insert inside stash (if that fails insert to bf)
  else : insert to bloomfilter:

query : key -> prob_bool:
  if (stash_set.querry) return prob_bool(True);
  else if (bf.querry(key)) return prob_bool(Maybe);
  else return False;

Properties: sizeof(stashed_bloomfiter) = sizeof(bloomfilter)

### stash set:

define as an interface so we can try multiple types of stash set

Methods:

insert : key -> bool (if false we add to bloomfilter)

query : true or false

Properties: small size wise, efficient

EACH stash set implementation can be used in 2 ways (defined at ctor time):
- either stores only +ve keys, i.e definately yes
- either stores only -ve keys, i.e definately no


#### Stash sets to try:

- another bloomfilter
- deterministic set
  - linear probing


### Prob bool:
struct: True,Maybe,False
