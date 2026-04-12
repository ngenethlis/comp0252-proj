#pragma once

#include <cstddef>

// StashSet interface (CRTP for static polymorphism).
//
// Implementations must provide:
//   bool do_insert(const Key& key)  — returns true if stored, false if full
//   bool do_query(const Key& key) const
//   size_t do_size_bits() const     — number of bits consumed
//   bool do_is_probabilistic() const — whether stash membership can be false-positive
template <typename Derived, typename Key>
class StashSet {
   public:
    // Insert a key into the stash. Returns true if successful, false if full.
    bool insert(const Key& key) { return static_cast<Derived*>(this)->do_insert(key); }

    // Query whether a key is in the stash.
    bool query(const Key& key) const { return static_cast<const Derived*>(this)->do_query(key); }

    // Number of bits this stash occupies.
    [[nodiscard]] size_t size_bits() const {
        return static_cast<const Derived*>(this)->do_size_bits();
    }

    // True if stash lookup can return false positives (e.g., Bloom-filter stash).
    [[nodiscard]] bool is_probabilistic() const {
        return static_cast<const Derived*>(this)->do_is_probabilistic();
    }
};
