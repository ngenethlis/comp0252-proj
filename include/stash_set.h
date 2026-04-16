#pragma once

#include <cstddef>

/**
 * @file stash_set.h
 * @brief CRTP interface for stash implementations used by StashedBloomFilter.
 */

/**
 * @brief CRTP interface for stash storage backends.
 *
 * Implementations must define:
 * - `bool do_insert(const Key& key)` returning false when capacity is exhausted.
 * - `bool do_query(const Key& key) const`.
 * - `size_t do_size_bits() const` returning total bit budget consumed.
 * - `bool do_is_probabilistic() const` indicating false-positive possibility.
 *
 * @tparam Derived Concrete stash type implementing the `do_*` methods.
 * @tparam Key Key type accepted by stash operations.
 */
template <typename Derived, typename Key>
class StashSet {
   public:
    /**
     * @brief Inserts a key into the stash.
     * @return `true` if stored, `false` if the stash rejected insertion.
     */
    bool insert(const Key& key) { return static_cast<Derived*>(this)->do_insert(key); }

    /**
     * @brief Queries whether a key is present in the stash.
     */
    bool query(const Key& key) const { return static_cast<const Derived*>(this)->do_query(key); }

    /**
     * @brief Returns the stash bit footprint.
     */
    [[nodiscard]] size_t size_bits() const {
        return static_cast<const Derived*>(this)->do_size_bits();
    }

    /**
     * @brief Returns whether stash queries may produce false positives.
     */
    [[nodiscard]] bool is_probabilistic() const {
        return static_cast<const Derived*>(this)->do_is_probabilistic();
    }
};
