#!/usr/bin/env python3
"""Plot experiment results from CSV files in results/."""

import os
import sys
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Consistent style
FILTER_COLORS = {
    "baseline_bf": "#333333",
    "bloom_filter": "#333333",
    "partitioned_bf": "#666666",
    "bf_stash": "#e74c3c",
    "lp_stash": "#2980b9",
    "stashed_bf_pos": "#e74c3c",
    "stashed_lp_pos": "#2980b9",
    "stashed_bf_neg": "#e67e22",
    "stashed_lp_neg": "#27ae60",
}

FILTER_LABELS = {
    "baseline_bf": "Plain BF (baseline)",
    "bloom_filter": "Plain BF",
    "partitioned_bf": "Partitioned BF",
    "bf_stash": "BF stash",
    "lp_stash": "LP stash",
    "stashed_bf_pos": "Stashed BF (BF, +ve)",
    "stashed_lp_pos": "Stashed BF (LP, +ve)",
    "stashed_bf_neg": "Stashed BF (BF, -ve)",
    "stashed_lp_neg": "Stashed BF (LP, -ve)",
}


def color_of(name):
    return FILTER_COLORS.get(name, "#999999")


def label_of(name):
    return FILTER_LABELS.get(name, name)


def savefig(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  saved {path}")
    plt.close(fig)


# -------------------------------------------------------------------------
# Exp 1: Certainty vs collision threshold
# -------------------------------------------------------------------------
def plot_exp1():
    path = os.path.join(RESULTS_DIR, "exp1.csv")
    if not os.path.exists(path):
        print("Skipping exp1 (no CSV)")
        return
    df = pd.read_csv(path)
    stash_df = df[df["threshold"] != "-"].copy()
    stash_df["threshold"] = stash_df["threshold"].astype(int)

    # 1a: Certainty rate for positives
    fig, ax = plt.subplots(figsize=(7, 4))
    for st in ["bf_stash", "lp_stash"]:
        sub = stash_df[stash_df["stash_type"] == st]
        ax.plot(sub["threshold"], sub["certainty_rate"], "o-",
                color=color_of(st), label=label_of(st))
    ax.set_xlabel("Collision threshold")
    ax.set_ylabel("Certainty rate (fraction of positives returning True)")
    ax.set_title("Exp 1a: Positive query certainty vs threshold")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.03,
            "LP plateau at low thresholds is a capacity effect "
            "(312-slot stash for this setup).",
            transform=ax.transAxes, fontsize=8, color="#555555")
    savefig(fig, "exp1a_certainty_rate.png")

    # 1b: False certainty rate for negatives
    fig, ax = plt.subplots(figsize=(7, 4))
    for st in ["bf_stash", "lp_stash"]:
        sub = stash_df[stash_df["stash_type"] == st]
        ax.plot(sub["threshold"], sub["false_certainty_rate"], "o-",
                color=color_of(st), label=label_of(st))
    ax.set_xlabel("Collision threshold")
    ax.set_ylabel("False certainty rate (negatives returning True)")
    ax.set_title("Exp 1b: False certainty (BF stash flaw)")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    savefig(fig, "exp1b_false_certainty.png")

    # 1c: Overall FPR vs threshold
    fig, ax = plt.subplots(figsize=(7, 4))
    baseline = df[df["stash_type"] == "baseline_bf"]["fpr"].iloc[0]
    ax.axhline(baseline, color=color_of("baseline_bf"), linestyle="--",
               label=label_of("baseline_bf"))
    for st in ["bf_stash", "lp_stash"]:
        sub = stash_df[stash_df["stash_type"] == st]
        ax.plot(sub["threshold"], sub["fpr"], "o-",
                color=color_of(st), label=label_of(st))
    ax.set_xlabel("Collision threshold")
    ax.set_ylabel("False positive rate")
    ax.set_title("Exp 1c: FPR vs threshold (positive stash mode)")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    savefig(fig, "exp1c_fpr_vs_threshold.png")


# -------------------------------------------------------------------------
# Exp 2: Negative stash FPR reduction
# -------------------------------------------------------------------------
def plot_exp2():
    path = os.path.join(RESULTS_DIR, "exp2.csv")
    if not os.path.exists(path):
        print("Skipping exp2 (no CSV)")
        return
    df = pd.read_csv(path)
    stash_df = df[df["scenario"] != "-"]
    baseline_fpr = df[df["stash_type"] == "baseline_bf"]["baseline_fpr"].iloc[0]

    # 2a: FPR vs stash fraction (practical vs oracle)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, scenario in zip(axes, ["practical", "oracle"]):
        sub = stash_df[stash_df["scenario"] == scenario]
        ax.axhline(baseline_fpr, color=color_of("baseline_bf"), linestyle="--",
                   label="Plain BF baseline")
        for st in ["bf_stash", "lp_stash"]:
            s = sub[sub["stash_type"] == st]
            ax.plot(s["stash_fraction"], s["stashed_fpr"], "o-",
                    color=color_of(st), label=label_of(st))
        ax.set_xlabel("Stash fraction of total bits")
        ax.set_ylabel("FPR")
        if scenario == "oracle":
            ax.set_title("Negative stash (oracle upper bound)")
        else:
            ax.set_title("Negative stash (practical warm-up)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Exp 2: Negative stash FPR reduction", fontsize=13)
    fig.text(0.5, 0.01,
             "Practical uses first-half warm-up and second-half holdout. "
             "Oracle scans the same holdout set (upper bound only).",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    savefig(fig, "exp2_negative_stash_fpr.png")

    # 2b: FPR reduction % (practical)
    fig, ax = plt.subplots(figsize=(7, 4))
    prac = stash_df[stash_df["scenario"] == "practical"]
    for st in ["bf_stash", "lp_stash"]:
        s = prac[prac["stash_type"] == st]
        ax.plot(s["stash_fraction"], s["fpr_reduction_pct"], "o-",
                color=color_of(st), label=label_of(st))
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_xlabel("Stash fraction of total bits")
    ax.set_ylabel("FPR reduction vs baseline (%)")
    ax.set_title("Exp 2b: Negative stash FPR reduction (practical)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, "exp2b_fpr_reduction_practical.png")

    # 2c: False-negative rate vs stash fraction (negative mode side effect)
    if "false_negative_rate" in stash_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, scenario in zip(axes, ["practical", "oracle"]):
            sub = stash_df[stash_df["scenario"] == scenario]
            for st in ["bf_stash", "lp_stash"]:
                s = sub[sub["stash_type"] == st]
                ax.plot(s["stash_fraction"], s["false_negative_rate"], "o-",
                        color=color_of(st), label=label_of(st))
            ax.set_xlabel("Stash fraction of total bits")
            ax.set_ylabel("False negative rate")
            ax.set_title(f"False negatives ({scenario})")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.suptitle("Exp 2c: Negative stash false negatives", fontsize=13)
        fig.tight_layout()
        savefig(fig, "exp2c_false_negative_rate.png")


# -------------------------------------------------------------------------
# Exp 3: FPR vs load factor
# -------------------------------------------------------------------------
def plot_exp3():
    path = os.path.join(RESULTS_DIR, "exp3.csv")
    if not os.path.exists(path):
        print("Skipping exp3 (no CSV)")
        return
    df = pd.read_csv(path)

    # 3a: FPR vs n
    fig, ax = plt.subplots(figsize=(8, 5))
    for ft in ["bloom_filter", "partitioned_bf", "stashed_bf_pos", "stashed_lp_pos",
               "stashed_bf_neg", "stashed_lp_neg"]:
        sub = df[df["filter_type"] == ft]
        # Skip zeros for log scale
        sub_nonzero = sub[sub["fpr"] > 0]
        if not sub_nonzero.empty:
            ax.plot(sub_nonzero["n"], sub_nonzero["fpr"], "o-",
                    color=color_of(ft), label=label_of(ft))
    ax.set_xlabel("Number of inserted elements (n)")
    ax.set_ylabel("False positive rate")
    ax.set_title("Exp 3a: FPR vs load factor")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, "exp3a_fpr_vs_n.png")

    # 3b: Certainty rate vs n (positive stash types only)
    fig, ax = plt.subplots(figsize=(7, 4))
    for ft in ["stashed_bf_pos", "stashed_lp_pos"]:
        sub = df[df["filter_type"] == ft]
        ax.plot(sub["n"], sub["certainty_rate"], "o-",
                color=color_of(ft), label=label_of(ft))
    ax.set_xlabel("Number of inserted elements (n)")
    ax.set_ylabel("Certainty rate (True / total positive queries)")
    ax.set_title("Exp 3b: Certainty rate vs load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, "exp3b_certainty_vs_n.png")

    # 3c: False certainty vs n
    fig, ax = plt.subplots(figsize=(7, 4))
    for ft in ["stashed_bf_pos", "stashed_lp_pos"]:
        sub = df[df["filter_type"] == ft]
        ax.plot(sub["n"], sub["false_certainty_rate"], "o-",
                color=color_of(ft), label=label_of(ft))
    ax.set_xlabel("Number of inserted elements (n)")
    ax.set_ylabel("False certainty rate (negatives returning True)")
    ax.set_title("Exp 3c: False certainty vs load (BF stash flaw)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, "exp3c_false_certainty_vs_n.png")

    # 3d: False-negative rate vs n (negative stash modes)
    if "false_negative_rate" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for ft in ["stashed_bf_neg", "stashed_lp_neg"]:
            sub = df[df["filter_type"] == ft]
            ax.plot(sub["n"], sub["false_negative_rate"], "o-",
                    color=color_of(ft), label=label_of(ft))
        ax.set_xlabel("Number of inserted elements (n)")
        ax.set_ylabel("False negative rate")
        ax.set_title("Exp 3d: False negatives vs load (negative stash)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        savefig(fig, "exp3d_false_negative_vs_n.png")


# -------------------------------------------------------------------------
# Exp 4: Stash fraction sweep
# -------------------------------------------------------------------------
def plot_exp4():
    path = os.path.join(RESULTS_DIR, "exp4.csv")
    if not os.path.exists(path):
        print("Skipping exp4 (no CSV)")
        return
    df = pd.read_csv(path)
    baseline = df[df["stash_type"] == "baseline_bf"]
    stash_df = df[df["stash_type"] != "baseline_bf"]
    baseline_fpr = baseline["fpr"].iloc[0]

    # 4a: FPR vs stash fraction (all types/modes)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(baseline_fpr, color=color_of("baseline_bf"), linestyle="--",
               label="Plain BF baseline")
    for (st, mode), sub in stash_df.groupby(["stash_type", "stash_mode"]):
        key = f"stashed_{st.replace('_stash', '')}_{'pos' if mode == 'positive' else 'neg'}"
        lbl = f"{label_of(st)} ({mode})"
        c = color_of(key) if key in FILTER_COLORS else color_of(st)
        style = "-" if mode == "positive" else "--"
        ax.plot(sub["stash_fraction"], sub["fpr"], f"o{style}", color=c, label=lbl)
    ax.set_xlabel("Stash fraction of total bits")
    ax.set_ylabel("False positive rate")
    ax.set_title("Exp 4a: FPR vs stash fraction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.03,
            "At this load the BF and LP negative-mode lines can overlap "
            "because LP capacity is not yet the bottleneck.",
            transform=ax.transAxes, fontsize=8, color="#555555")
    savefig(fig, "exp4a_fpr_vs_fraction.png")

    # 4b: Certainty rate vs stash fraction (positive mode only)
    fig, ax = plt.subplots(figsize=(7, 4))
    pos = stash_df[stash_df["stash_mode"] == "positive"]
    for st in ["bf_stash", "lp_stash"]:
        sub = pos[pos["stash_type"] == st]
        ax.plot(sub["stash_fraction"], sub["certainty_rate"], "o-",
                color=color_of(st), label=label_of(st))
    ax.set_xlabel("Stash fraction of total bits")
    ax.set_ylabel("Certainty rate")
    ax.set_title("Exp 4b: Certainty rate vs stash fraction (positive mode)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, "exp4b_certainty_vs_fraction.png")

    # 4c: False-negative rate vs stash fraction (negative mode)
    if "false_negative_rate" in stash_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        neg_mode = stash_df[stash_df["stash_mode"] == "negative"]
        for st in ["bf_stash", "lp_stash"]:
            sub = neg_mode[neg_mode["stash_type"] == st]
            ax.plot(sub["stash_fraction"], sub["false_negative_rate"], "o-",
                    color=color_of(st), label=label_of(st))
        ax.set_xlabel("Stash fraction of total bits")
        ax.set_ylabel("False negative rate")
        ax.set_title("Exp 4c: False negatives vs stash fraction (negative mode)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        savefig(fig, "exp4c_false_negative_vs_fraction.png")


# -------------------------------------------------------------------------
# Exp 5: Password workload
# -------------------------------------------------------------------------
def plot_exp5():
    path = os.path.join(RESULTS_DIR, "exp5.csv")
    if not os.path.exists(path):
        print("Skipping exp5 (no CSV)")
        return
    df = pd.read_csv(path)

    # 5a: ProbBool breakdown for positive queries (stacked bar)
    fig, ax = plt.subplots(figsize=(9, 5))
    filters = df["filter_type"].tolist()
    true_vals = df["pos_true"].tolist()
    maybe_vals = df["pos_maybe"].tolist()
    false_vals = df["pos_false"].tolist()
    x = range(len(filters))
    labels = [label_of(f) for f in filters]

    ax.bar(x, true_vals, color="#27ae60", label="True (certain)")
    ax.bar(x, maybe_vals, bottom=true_vals, color="#f39c12", label="Maybe (probable)")
    ax.bar(x, false_vals,
           bottom=[t + m for t, m in zip(true_vals, maybe_vals)],
           color="#e74c3c", label="False (missed)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Number of positive queries")
    ax.set_title("Exp 5a: ProbBool breakdown for positive queries (passwords)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "exp5a_password_pos_breakdown.png")

    # 5b: FPR + false certainty for negative queries
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, df["fpr"].tolist(), color="#3498db", label="FPR")
    ax2 = ax.twinx()
    ax2.plot(list(x), df["false_certainty_rate"].tolist(), "ro-",
             label="False certainty rate", markersize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("False positive rate", color="#3498db")
    ax2.set_ylabel("False certainty rate", color="red")
    ax.set_title("Exp 5b: Negative query errors (passwords)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)
    fig.tight_layout()
    savefig(fig, "exp5b_password_neg_errors.png")


# -------------------------------------------------------------------------
# Exp 6: Hot-positive certainty workload
# -------------------------------------------------------------------------
def plot_exp6():
    path = os.path.join(RESULTS_DIR, "exp6.csv")
    if not os.path.exists(path):
        print("Skipping exp6 (no CSV)")
        return
    df = pd.read_csv(path)
    order = ["bloom_filter", "stashed_lp_pos"]
    present = [f for f in order if f in set(df["filter_type"])]
    df = df.set_index("filter_type").loc[present].reset_index()

    x = range(len(df))
    labels = [label_of(f) for f in df["filter_type"]]
    colors = [color_of(f) for f in df["filter_type"]]

    # 6a: Downstream check rate
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(x, df["downstream_check_rate"].tolist(), color=colors)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Downstream check rate (Maybe / total queries)")
    ax.set_title("Exp 6a: Hot-positive workload downstream checks")
    ax.set_ylim(0, 1)
    if "downstream_reduction_pct" in df.columns:
        for i, row in df.iterrows():
            if row["filter_type"] != "bloom_filter":
                ax.text(bars[i].get_x() + bars[i].get_width() / 2,
                        bars[i].get_height() + 0.02,
                        f"{row['downstream_reduction_pct']:.1f}% fewer checks",
                        ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "exp6a_downstream_checks.png")

    # 6b: Positive query outcome breakdown
    fig, ax = plt.subplots(figsize=(7, 4))
    true_vals = df["pos_true"].tolist()
    maybe_vals = df["pos_maybe"].tolist()
    false_vals = df["pos_false"].tolist()
    ax.bar(x, true_vals, color="#27ae60", label="True")
    ax.bar(x, maybe_vals, bottom=true_vals, color="#f39c12", label="Maybe")
    ax.bar(x, false_vals,
           bottom=[t + m for t, m in zip(true_vals, maybe_vals)],
           color="#e74c3c", label="False")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Positive query count")
    ax.set_title("Exp 6b: Positive query outcomes (count-weighted hot keys)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "exp6b_positive_query_breakdown.png")


# -------------------------------------------------------------------------
# Exp 7: Repeated negatives with warm-up
# -------------------------------------------------------------------------
def plot_exp7():
    path = os.path.join(RESULTS_DIR, "exp7.csv")
    if not os.path.exists(path):
        print("Skipping exp7 (no CSV)")
        return
    df = pd.read_csv(path)
    order = ["bloom_filter", "stashed_lp_neg"]
    present = [f for f in order if f in set(df["filter_type"])]
    df = df.set_index("filter_type").loc[present].reset_index()

    x = range(len(df))
    labels = [label_of(f) for f in df["filter_type"]]
    colors = [color_of(f) for f in df["filter_type"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 7a: Eval FPR comparison
    bars = axes[0].bar(x, df["fpr"].tolist(), color=colors)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("False positive rate")
    axes[0].set_title("Exp 7a: Eval FPR after warm-up")
    axes[0].grid(True, axis="y", alpha=0.3)
    if "fpr_reduction_pct" in df.columns:
        for i, row in df.iterrows():
            if row["filter_type"] == "stashed_lp_neg":
                axes[0].text(bars[i].get_x() + bars[i].get_width() / 2,
                             bars[i].get_height() + 0.001,
                             f"{row['fpr_reduction_pct']:.1f}% vs plain",
                             ha="center", va="bottom", fontsize=8)

    # 7b: False-negative rate on positives
    axes[1].bar(x, df["false_negative_rate"].tolist(), color=colors)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("False negative rate")
    axes[1].set_title("Exp 7b: Positive-set false negatives")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Exp 7: Repeated-negative warm-up (data-derived queries)", fontsize=12)
    fig.tight_layout()
    savefig(fig, "exp7_repeated_negative_warmup.png")


# -------------------------------------------------------------------------
# Benchmarks
# -------------------------------------------------------------------------
def plot_bench():
    path = os.path.join(RESULTS_DIR, "bench.csv")
    if not os.path.exists(path):
        print("Skipping bench (no CSV)")
        return

    # bench.csv has multiple headers (one per benchmark type). Read in chunks.
    chunks = []
    current_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("bench,"):
                if current_lines:
                    chunks.append(current_lines)
                current_lines = [line]
            else:
                current_lines.append(line)
    if current_lines:
        chunks.append(current_lines)

    dfs = {}
    for chunk in chunks:
        df = pd.read_csv(StringIO("\n".join(chunk)))
        bench_type = df["bench"].iloc[0]
        dfs[bench_type] = df

    for bench_type, df in dfs.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for ft in df["filter_type"].unique():
            sub = df[df["filter_type"] == ft]
            ax.plot(sub["n"], sub["ns_per_op"], "o-",
                    color=color_of(ft), label=label_of(ft))
        ax.set_xlabel("Number of elements")
        ax.set_ylabel("ns / operation")
        ax.set_title(f"Benchmark: {bench_type}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        savefig(fig, f"bench_{bench_type}.png")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    plt.style.use("seaborn-v0_8-whitegrid")

    which = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    plotters = {
        "exp1": plot_exp1,
        "exp2": plot_exp2,
        "exp3": plot_exp3,
        "exp4": plot_exp4,
        "exp5": plot_exp5,
        "exp6": plot_exp6,
        "exp7": plot_exp7,
        "bench": plot_bench,
    }

    for name, fn in plotters.items():
        if "all" in which or name in which:
            print(f"Plotting {name}...")
            fn()

    print("Done.")


if __name__ == "__main__":
    main()
