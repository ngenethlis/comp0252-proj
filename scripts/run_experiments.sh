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

EXP_DATASET="$PROJECT_DIR/data/hibp-100k.txt"
if [[ ! -f "$EXP_DATASET" ]]; then
    EXP_DATASET="$PROJECT_DIR/data/hibp-1k.txt"
fi

SEEDS=(42 43 44 45 46)
TMP_DIR="$RESULTS_DIR/.seed_runs"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

aggregate_seed_csvs() {
    local out_csv="$1"
    shift
    /usr/bin/env python3 - "$out_csv" "$@" <<'PY'
import math
import sys
import pandas as pd

out_csv = sys.argv[1]
in_files = sys.argv[2:]
n_runs = len(in_files)

dfs = []
for i, path in enumerate(in_files):
    df = pd.read_csv(path)
    df["__seed_run"] = i
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

numeric_cols = [
    c for c in all_df.columns
    if c != "__seed_run" and pd.api.types.is_numeric_dtype(all_df[c])
]
# Numeric columns that define experiment configuration and must stay as grouping keys.
param_numeric_cols = {
    "threshold",
    "stash_fraction",
    "n",
    "n_passwords",
    "total_bits",
    "pos_queries",
    "neg_queries",
    "neg_pool_size",
    "warmup_queries",
    "eval_queries",
}
metric_cols = [c for c in numeric_cols if c not in param_numeric_cols]
group_cols = [c for c in all_df.columns if c not in metric_cols + ["__seed_run"]]

mean_df = all_df.groupby(group_cols, dropna=False, as_index=False)[metric_cols].mean()
std_df = all_df.groupby(group_cols, dropna=False, as_index=False)[metric_cols].std(ddof=1)
median_df = all_df.groupby(group_cols, dropna=False, as_index=False)[metric_cols].median()
min_df = all_df.groupby(group_cols, dropna=False, as_index=False)[metric_cols].min()
max_df = all_df.groupby(group_cols, dropna=False, as_index=False)[metric_cols].max()
std_df = std_df.fillna(0.0)

merged = mean_df.copy()
for col in metric_cols:
    merged[f"{col}_std"] = std_df[col]
    merged[f"{col}_ci95"] = 1.96 * std_df[col] / math.sqrt(n_runs)
    merged[f"{col}_median"] = median_df[col]
    merged[f"{col}_min"] = min_df[col]
    merged[f"{col}_max"] = max_df[col]

merged.to_csv(out_csv, index=False)
PY
}

# Run each experiment over 5 seeds, then aggregate to mean/std/ci95.
for exp in exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8; do
    echo "Running $exp across ${#SEEDS[@]} seeds..."
    run_files=()
    for seed in "${SEEDS[@]}"; do
        out="$TMP_DIR/${exp}_seed${seed}.csv"
        run_files+=("$out")
        if [[ "$exp" == "exp5" || "$exp" == "exp6" || "$exp" == "exp7" || "$exp" == "exp8" ]]; then
            "$BUILD_DIR/main" "$exp" "$EXP_DATASET" "--seed=${seed}" > "$out" 2>/dev/null
        else
            "$BUILD_DIR/main" "$exp" "--seed=${seed}" > "$out" 2>/dev/null
        fi
    done
    aggregate_seed_csvs "$RESULTS_DIR/$exp.csv" "${run_files[@]}"
    echo "  -> $RESULTS_DIR/$exp.csv (mean + std + ci95 from ${#SEEDS[@]} seeds)"
done

# Benchmarks
echo "Running benchmarks..."
"$BUILD_DIR/bench" > "$RESULTS_DIR/bench.csv" 2>/dev/null
echo "  -> $RESULTS_DIR/bench.csv"

echo "All experiments complete. Results in $RESULTS_DIR/"
