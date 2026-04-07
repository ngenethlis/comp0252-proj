#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
RESULTS_DIR="$PROJECT_DIR/results"

mkdir -p "$RESULTS_DIR"

# Build first
echo "Building..."
cmake -B "$BUILD_DIR" "$PROJECT_DIR" > /dev/null
cmake --build "$BUILD_DIR" > /dev/null
echo "Build complete."

EXP5_DATASET="$PROJECT_DIR/data/hibp-100k.txt"
if [[ ! -f "$EXP5_DATASET" ]]; then
    EXP5_DATASET="$PROJECT_DIR/data/hibp-1k.txt"
fi

# Run each experiment, saving CSV to results/
for exp in exp1 exp2 exp3 exp4 exp5; do
    echo "Running $exp..."
    if [[ "$exp" == "exp5" ]]; then
        "$BUILD_DIR/main" "$exp" "$EXP5_DATASET" > "$RESULTS_DIR/$exp.csv" 2>/dev/null
    else
        "$BUILD_DIR/main" "$exp" > "$RESULTS_DIR/$exp.csv" 2>/dev/null
    fi
    echo "  -> $RESULTS_DIR/$exp.csv"
done

# Benchmarks
echo "Running benchmarks..."
"$BUILD_DIR/bench" > "$RESULTS_DIR/bench.csv" 2>/dev/null
echo "  -> $RESULTS_DIR/bench.csv"

echo "All experiments complete. Results in $RESULTS_DIR/"
