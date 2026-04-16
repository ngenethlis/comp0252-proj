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

# Documentation (optional): requires sphinx-build and doxygen installed.
find_program(SPHINX_BUILD_EXECUTABLE sphinx-build)
find_package(Doxygen QUIET)
if(SPHINX_BUILD_EXECUTABLE AND DOXYGEN_FOUND)
  add_custom_target(
    docs
    COMMAND ${SPHINX_BUILD_EXECUTABLE} -b html -W docs docs/_build/html
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building API documentation with Sphinx and Doxygen"
  )
endif()
