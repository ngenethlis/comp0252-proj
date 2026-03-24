#pragma once

#include <ostream>

enum class ProbBool {
    True,   // definitely in the set
    Maybe,  // probably in the set (Bloom filter positive)
    False,  // definitely not in the set
};

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

inline bool is_positive(ProbBool pb) { return pb != ProbBool::False; }
