#pragma once

#include <ostream>

/**
 * @file prob_bool.h
 * @brief Three-valued boolean-like result for probabilistic membership queries.
 */

/**
 * @brief Probabilistic query outcome.
 */
enum class ProbBool {
    True,   /**< definitely in the set */
    Maybe,  /**< probably in the set (Bloom filter positive) */
    False,  /**< definitely not in the set */
};

/**
 * @brief Writes a `ProbBool` value as `True`, `Maybe`, or `False`.
 */
inline std::ostream& operator<<(std::ostream& os, ProbBool pb) {
    switch (pb) {
        case ProbBool::True:
            os << "True";
            break;
        case ProbBool::Maybe:
            os << "Maybe";
            break;
        case ProbBool::False:
            os << "False";
            break;
    }
    return os;
}

/**
 * @brief Returns whether a `ProbBool` result is positive-like.
 * @return `true` for `ProbBool::True` and `ProbBool::Maybe`, `false` otherwise.
 */
inline bool is_positive(ProbBool pb) { return pb != ProbBool::False; }
