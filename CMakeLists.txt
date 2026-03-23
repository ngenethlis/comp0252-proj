cmake_minimum_required(VERSION 3.16)
project(stashed_bloom_filter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Header-only library
add_library(bloom_filter INTERFACE)
target_include_directories(bloom_filter INTERFACE include)

# Main executable
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE bloom_filter)

# Tests
enable_testing()
add_executable(tests tests/test_bloom_filter.cpp)
target_link_libraries(tests PRIVATE bloom_filter)
add_test(NAME bloom_filter_tests COMMAND tests)

# Benchmarks
add_executable(bench bench/bench_bloom_filter.cpp)
target_link_libraries(bench PRIVATE bloom_filter)
