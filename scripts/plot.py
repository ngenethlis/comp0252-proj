#!/ usr / bin / env python3
"""Plot experiment results from CSV files in results/."""

import os
import sys
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

#Consistent style
FILTER_COLORS = {
    "baseline_bf": "#333333",
    "partitioned_bf": "#666666",
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
    "baseline_bf": "Plain BF",
    "partitioned_bf": "Partitioned BF",
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


def plot_baseline_dots(ax, x_values, y_value, baseline_key):
    x = np.asarray(list(x_values), dtype=float)
    if x.size == 0:
        return
    y = np.full(x.shape, float(y_value), dtype=float)
    ax.plot(x, y, linestyle="None", marker="o", markersize=4,
            color=color_of(baseline_key), label=label_of(baseline_key))


def query_model_label(name):
    mapping = {
        "zipf": "Synthetic Zipf",
        "count_weighted": "Count-weighted from data",
    }
    return mapping.get(name, str(name))


def scenario_label(name):
    mapping = {
        "in_dist": "In-distribution",
        "cross_pool": "Cross-pool",
        "shifted_distribution": "Shifted distribution",
    }
    return mapping.get(name, str(name))


def savefig(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  saved {path}")
    plt.close(fig)

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Exp 1 : Certainty vs collision threshold
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
def plot_exp1():
    path = os.path.join(RESULTS_DIR, "exp1.csv")
    if not os.path.exists(path):
        print("Skipping exp1 (no CSV)")
        return
    df = pd.read_csv(path)
    stash_df = df[df["threshold"] != "-"].copy()
    stash_df["threshold"] = stash_df["threshold"].astype(int)

# 1a : Certainty rate for positives
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

# 1b : False certainty rate for negatives
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

# 1c : Overall FPR vs threshold
    fig, ax = plt.subplots(figsize=(7, 4))
    x_baseline = sorted(stash_df["threshold"].unique().tolist())
    plain_row = df[df["stash_type"] == "baseline_bf"]
    if not plain_row.empty:
        plot_baseline_dots(ax, x_baseline, float(plain_row["fpr"].iloc[0]), "baseline_bf")
    partitioned_row = df[df["stash_type"] == "partitioned_bf"]
    if not partitioned_row.empty:
        plot_baseline_dots(ax, x_baseline, float(partitioned_row["fpr"].iloc[0]), "partitioned_bf")
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

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Exp 2 : Negative stash FPR reduction
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
def plot_exp2():
    path = os.path.join(RESULTS_DIR, "exp2.csv")
    if not os.path.exists(path):
        print("Skipping exp2 (no CSV)")
        return
    df = pd.read_csv(path)
    stash_df = df[df["scenario"] != "-"]
    baseline_fpr = df[df["stash_type"] == "baseline_bf"]["baseline_fpr"].iloc[0]
    partitioned_row = df[df["stash_type"] == "partitioned_bf"]
    partitioned_fpr = (float(partitioned_row["stashed_fpr"].iloc[0])
                       if not partitioned_row.empty else None)

# 2a : FPR vs stash fraction(practical vs oracle)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, scenario in zip(axes, ["practical", "oracle"]):
        sub = stash_df[stash_df["scenario"] == scenario]
        x_baseline = sorted(sub["stash_fraction"].unique().tolist())
        plot_baseline_dots(ax, x_baseline, float(baseline_fpr), "baseline_bf")
        if partitioned_fpr is not None:
            plot_baseline_dots(ax, x_baseline, float(partitioned_fpr), "partitioned_bf")
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
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    savefig(fig, "exp2_negative_stash_fpr.png")

# 2b : FPR reduction %(practical)
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

# 2c : False - negative rate vs stash fraction(negative mode side effect)
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

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Exp 3 : FPR vs load factor
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
def plot_exp3():
    path = os.path.join(RESULTS_DIR, "exp3.csv")
    if not os.path.exists(path):
        print("Skipping exp3 (no CSV)")
        return
    df = pd.read_csv(path)

# 3a : FPR vs n
    fig, ax = plt.subplots(figsize=(8, 5))
    for ft in ["bloom_filter", "partitioned_bf", "stashed_bf_pos", "stashed_lp_pos",
               "stashed_bf_neg", "stashed_lp_neg"]:
        sub = df[df["filter_type"] == ft]
#Skip zeros for log scale
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

# 3b : Certainty rate vs n(positive stash types only)
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

# 3c : False certainty vs n
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

# 3d : False - negative rate vs n(negative stash modes)
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

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Exp 4 : Stash fraction sweep
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
def plot_exp4():
    path = os.path.join(RESULTS_DIR, "exp4.csv")
    if not os.path.exists(path):
        print("Skipping exp4 (no CSV)")
        return
    df = pd.read_csv(path)
    plain_baseline = df[df["stash_type"] == "baseline_bf"]
    partitioned_baseline = df[df["stash_type"] == "partitioned_bf"]
    stash_df = df[~df["stash_type"].isin(["baseline_bf", "partitioned_bf"])]
    baseline_fpr = plain_baseline["fpr"].iloc[0]

# 4a : FPR vs stash fraction(all types / modes)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_baseline = sorted(stash_df["stash_fraction"].unique().tolist())
    plot_baseline_dots(ax, x_baseline, float(baseline_fpr), "baseline_bf")
    if not partitioned_baseline.empty:
        plot_baseline_dots(ax, x_baseline, float(partitioned_baseline["fpr"].iloc[0]),
                           "partitioned_bf")
    for (st, mode), sub in stash_df.groupby(["stash_type", "stash_mode"]):
        st_name = str(st)
        key = f"stashed_{st_name.replace('_stash', '')}_{'pos' if mode == 'positive' else 'neg'}"
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

# 4b : Certainty rate vs stash fraction(positive mode only)
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

# 4c : False - negative rate vs stash fraction(negative mode)
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

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Exp 5 : Password workload
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
def plot_exp5():
    path = os.path.join(RESULTS_DIR, "exp5.csv")
    if not os.path.exists(path):
        print("Skipping exp5 (no CSV)")
        return
    df = pd.read_csv(path)

# 5a : ProbBool breakdown for positive queries(stacked bar)
    fig, ax = plt.subplots(figsize=(9, 5))
    filters = df["filter_type"].tolist()
    true_vals = df["pos_true"].tolist()
    maybe_vals = df["pos_maybe"].tolist()
    false_vals = df["pos_false"].tolist()
    x = range(len(filters))
    labels = [str(label_of(f)) for f in filters]

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

# 5b : FPR + false certainty for negative queries
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

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Exp 6 : Hot - positive certainty workload
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
def plot_exp6():
    path = os.path.join(RESULTS_DIR, "exp6.csv")
    if not os.path.exists(path):
        print("Skipping exp6 (no CSV)")
        return
    df = pd.read_csv(path)
    if "query_model" not in df.columns:
        df["query_model"] = "count_weighted"

    order = ["bloom_filter", "partitioned_bf", "stashed_lp_pos"]
    order_6a = ["bloom_filter", "stashed_lp_pos"]
    preferred_models = ["zipf", "count_weighted"]
    models = [m for m in preferred_models if m in set(df["query_model"])]
    for m in sorted(set(df["query_model"])):
        if m not in models:
            models.append(m)

# 6a : Downstream check rate(per query model)
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = df[(df["query_model"] == model) & (df["filter_type"].isin(order_6a))].copy()
        sub["filter_type"] = pd.Categorical(sub["filter_type"], categories=order_6a, ordered=True)
        sub = sub.sort_values("filter_type")

        x = range(len(sub))
        labels = [label_of(f) for f in sub["filter_type"]]
        colors = [color_of(f) for f in sub["filter_type"]]
        bars = ax.bar(x, sub["downstream_check_rate"].tolist(), color=colors)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Downstream check rate (Maybe / total queries)")
        ax.set_title(query_model_label(model))
        ax.set_ylim(0, 1)
        if "downstream_reduction_pct" in sub.columns:
            for i, row in sub.reset_index(drop=True).iterrows():
                if row["filter_type"] != "bloom_filter":
                    ax.text(bars[i].get_x() + bars[i].get_width() / 2,
                            bars[i].get_height() + 0.02,
                            f"{row['downstream_reduction_pct']:.1f}% fewer checks",
                            ha="center", va="bottom", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Exp 6a: Hot-positive downstream checks", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    savefig(fig, "exp6a_downstream_checks.png")

# 6b : Positive query outcome breakdown(per query model)
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = df[(df["query_model"] == model) & (df["filter_type"].isin(order))].copy()
        sub["filter_type"] = pd.Categorical(sub["filter_type"], categories=order, ordered=True)
        sub = sub.sort_values("filter_type")

        x = range(len(sub))
        labels = [label_of(f) for f in sub["filter_type"]]
        true_vals = sub["pos_true"].tolist()
        maybe_vals = sub["pos_maybe"].tolist()
        false_vals = sub["pos_false"].tolist()
        ax.bar(x, true_vals, color="#27ae60", label="True")
        ax.bar(x, maybe_vals, bottom=true_vals, color="#f39c12", label="Maybe")
        ax.bar(x, false_vals,
               bottom=[t + m for t, m in zip(true_vals, maybe_vals)],
               color="#e74c3c", label="False")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Positive query count")
        ax.set_title(query_model_label(model))
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    fig.suptitle("Exp 6b: Positive query outcomes", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    savefig(fig, "exp6b_positive_query_breakdown.png")

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Exp 7 : Repeated negatives with warm - up
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
def plot_exp7():
    path = os.path.join(RESULTS_DIR, "exp7.csv")
    if not os.path.exists(path):
        print("Skipping exp7 (no CSV)")
        return
    df = pd.read_csv(path)
    if "query_model" not in df.columns:
        df["query_model"] = "count_weighted"
    if "scenario" not in df.columns:
        df["scenario"] = "in_dist"

    order = ["bloom_filter", "partitioned_bf", "stashed_lp_neg"]
    preferred_models = ["zipf", "count_weighted"]
    preferred_scenarios = ["in_dist", "cross_pool", "shifted_distribution"]

    models = [m for m in preferred_models if m in set(df["query_model"])]
    for m in sorted(set(df["query_model"])):
        if m not in models:
            models.append(m)

    scenarios = [s for s in preferred_scenarios if s in set(df["scenario"])]
    for s in sorted(set(df["scenario"])):
        if s not in scenarios:
            scenarios.append(s)

    fig, axes = plt.subplots(len(models), 2, figsize=(13, 4 * len(models)), squeeze=False)

    for row_idx, model in enumerate(models):
        sub = df[(df["query_model"] == model) & (df["filter_type"].isin(order))].copy()

        pivot_fpr = sub.pivot_table(index="scenario", columns="filter_type", values="fpr", aggfunc="first")
        pivot_fnr = sub.pivot_table(index="scenario", columns="filter_type", values="false_negative_rate", aggfunc="first")
        pivot_red = sub.pivot_table(index="scenario", columns="filter_type",
                                    values="fpr_reduction_pct", aggfunc="first")

        x = list(range(len(scenarios)))
        present_filters = [f for f in order if f in pivot_fpr.columns]
        width = 0.8 / max(1, len(present_filters))

        filter_to_bars_fpr = {}
        for i, ft in enumerate(present_filters):
            offset = (i - (len(present_filters) - 1) / 2) * width
            xs = [v + offset for v in x]
            vals = [pivot_fpr.at[s, ft] if s in pivot_fpr.index else 0.0 for s in scenarios]
            bars = axes[row_idx, 0].bar(xs, vals, width=width, color=color_of(ft), label=label_of(ft))
            filter_to_bars_fpr[ft] = bars

        for i, ft in enumerate(present_filters):
            offset = (i - (len(present_filters) - 1) / 2) * width
            xs = [v + offset for v in x]
            vals = [pivot_fnr.at[s, ft] if (s in pivot_fnr.index and ft in pivot_fnr.columns)
                    else 0.0 for s in scenarios]
            axes[row_idx, 1].bar(xs, vals, width=width, color=color_of(ft), label=label_of(ft))

        axes[row_idx, 0].set_xticks(x)
        axes[row_idx, 0].set_xticklabels([scenario_label(s) for s in scenarios], rotation=15, ha="right")
        axes[row_idx, 0].set_ylabel("False positive rate")
        axes[row_idx, 0].set_title(f"{query_model_label(model)}: Eval FPR")
        axes[row_idx, 0].grid(True, axis="y", alpha=0.3)

        if ("fpr_reduction_pct" in sub.columns and "stashed_lp_neg" in pivot_red.columns and
                "stashed_lp_neg" in filter_to_bars_fpr):
            stash_vals = [pivot_fpr.at[s, "stashed_lp_neg"] if s in pivot_fpr.index else 0.0
                          for s in scenarios]
            max_stash = (float(np.nanmax(np.asarray(stash_vals, dtype=np.float64)))
                         if stash_vals else 0.0)
            for i, s in enumerate(scenarios):
                if s in pivot_red.index and pd.notna(pivot_red.at[s, "stashed_lp_neg"]):
                    bar = filter_to_bars_fpr["stashed_lp_neg"][i]
                    axes[row_idx, 0].text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (max_stash * 0.03 if max_stash > 0 else 0.001),
                        f"{pivot_red.at[s, 'stashed_lp_neg']:.1f}%",
                        ha="center", va="bottom", fontsize=8
                    )
        axes[row_idx, 1].set_xticks(x)
        axes[row_idx, 1].set_xticklabels([scenario_label(s) for s in scenarios], rotation=15, ha="right")
        axes[row_idx, 1].set_ylabel("False negative rate")
        axes[row_idx, 1].set_title(f"{query_model_label(model)}: Positive-set FNR")
        axes[row_idx, 1].grid(True, axis="y", alpha=0.3)

    axes[0, 0].legend()
    fig.suptitle("Exp 7: Repeated-negative robustness under distribution shift", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    savefig(fig, "exp7_repeated_negative_warmup.png")

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Exp 8 : Warm - up budget sweep
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
def plot_exp8():
    path = os.path.join(RESULTS_DIR, "exp8.csv")
    if not os.path.exists(path):
        print("Skipping exp8 (no CSV)")
        return
    df = pd.read_csv(path)
    if "query_model" not in df.columns or "scenario" not in df.columns:
        print("Skipping exp8 (unexpected schema)")
        return

    st = df[df["filter_type"] == "stashed_lp_neg"].copy()
    if st.empty:
        print("Skipping exp8 (no stashed rows)")
        return

    st["warmup_queries"] = st["warmup_queries"].astype(int)

    preferred_models = ["zipf", "count_weighted"]
    preferred_scenarios = ["in_dist", "cross_pool", "shifted_distribution"]
    models = [m for m in preferred_models if m in set(st["query_model"])]
    for m in sorted(set(st["query_model"])):
        if m not in models:
            models.append(m)
    scenarios = [s for s in preferred_scenarios if s in set(st["scenario"])]
    for s in sorted(set(st["scenario"])):
        if s not in scenarios:
            scenarios.append(s)

# 8a : FPR reduction vs warm - up budget.
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    scenario_styles = {
        "in_dist": {"color": color_of("stashed_lp_neg"), "linestyle": "-", "marker": "o"},
        "cross_pool": {
            "color": color_of("stashed_bf_neg"),
            "linestyle": "--",
            "marker": "s",
        },
        "shifted_distribution": {
            "color": color_of("partitioned_bf"),
            "linestyle": "-.",
            "marker": "^",
        },
    }
    for ax, model in zip(axes, models):
        sub_model = st[st["query_model"] == model]
        for sc in scenarios:
            sub = sub_model[sub_model["scenario"] == sc].sort_values("warmup_queries")
            if sub.empty:
                continue
            style = scenario_styles.get(sc, {"color": "#999999", "linestyle": "-", "marker": "o"})
            ax.plot(
                sub["warmup_queries"],
                sub["fpr_reduction_pct"],
                label=scenario_label(sc),
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
            )
        ax.axhline(0.0, color=color_of("baseline_bf"), linestyle=":", label="0% gain to plain BF baseline")
        ax.set_xscale("symlog", linthresh=1000)
        ax.set_xlabel("Warm-up queries")
        ax.set_ylabel("FPR reduction vs baseline (%)")
        ax.set_title(query_model_label(model))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Exp 8a: Benefit vs warm-up budget", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    savefig(fig, "exp8a_warmup_budget_fpr_reduction.png")

# 8b : FNR and stash occupancy vs warm - up budget(in - distribution only).
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 4), sharey=False)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = st[(st["query_model"] == model) & (st["scenario"] == "in_dist")].sort_values(
            "warmup_queries")
        if sub.empty:
            continue
        line_fnr = ax.plot(sub["warmup_queries"], sub["false_negative_rate"], "o-",
                color=color_of("stashed_bf_neg"),
                label="False negative rate")
        ax.set_xscale("symlog", linthresh=1000)
        ax.set_xlabel("Warm-up queries")
        ax.set_ylabel("False negative rate", color=color_of("stashed_bf_neg"))
        ax.tick_params(axis="y", labelcolor=color_of("stashed_bf_neg"))
        ax.set_title(query_model_label(model) + " (in-dist)")
        ax.grid(True, which="both", alpha=0.3)

        ax2 = ax.twinx()
        line_stash = ax2.plot(sub["warmup_queries"], sub["stash_count"], "s--",
                 color=color_of("stashed_lp_pos"),
                 label="Stash count")
        ax2.set_ylabel("Stash count", color=color_of("stashed_lp_pos"))
        ax2.tick_params(axis="y", labelcolor=color_of("stashed_lp_pos"))

        handles = [line_fnr[0], line_stash[0]]
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, fontsize=8, loc="best")

    fig.suptitle("Exp 8b: Safety and capacity vs warm-up budget", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    savefig(fig, "exp8b_warmup_budget_fnr_stash.png")

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Transfer decision boundary : where LP negative stash helps vs hurts
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
def plot_decision_boundary():
    path8 = os.path.join(RESULTS_DIR, "exp8.csv")
    if not os.path.exists(path8):
        print("Skipping decision boundary plot (need exp8.csv)")
        return

    df8 = pd.read_csv(path8)

    st8 = df8[df8["filter_type"] == "stashed_lp_neg"].copy()
    if st8.empty:
        print("Skipping decision boundary plot (no points)")
        return

    st8["warmup_queries"] = st8["warmup_queries"].astype(int)
    scenario_order = ["cross_pool", "shifted_distribution", "in_dist"]
    preferred_models = ["zipf", "count_weighted"]

    models = [m for m in preferred_models if m in set(st8["query_model"])]
    for m in sorted(set(st8["query_model"])):
        if m not in models:
            models.append(m)
    scenarios = [s for s in scenario_order if s in set(st8["scenario"])]
    for s in sorted(set(st8["scenario"])):
        if s not in scenarios:
            scenarios.append(s)

    warmups = sorted(st8["warmup_queries"].unique().tolist())
    if not warmups:
        print("Skipping decision boundary plot (no warm-up values)")
        return

    vmin = float(st8["fpr_reduction_pct"].min())
    vmax = float(st8["fpr_reduction_pct"].max())
    norm = None
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.8), sharey=True)
    if len(models) == 1:
        axes = [axes]

    im = None
    for ax, model in zip(axes, models):
        mat = np.full((len(scenarios), len(warmups)), np.nan)
        sub = st8[st8["query_model"] == model]
        for i, sc in enumerate(scenarios):
            row = sub[sub["scenario"] == sc]
            for j, w in enumerate(warmups):
                cell = row[row["warmup_queries"] == w]
                if not cell.empty:
                    mat[i, j] = float(cell["fpr_reduction_pct"].iloc[0])

        if norm is not None:
            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", norm=norm)
        else:
            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)

        for i in range(len(scenarios)):
            for j in range(len(warmups)):
                if np.isnan(mat[i, j]):
                    continue
                ax.text(j, i, f"{mat[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                        color="#111111")

        ax.set_title(query_model_label(model))
        ax.set_xticks(range(len(warmups)))
        ax.set_xticklabels([f"{int(w/1000)}k" if w >= 1000 else str(int(w)) for w in warmups],
                           rotation=30, ha="right")
        ax.set_xlabel("Warm-up queries")
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels([scenario_label(s) for s in scenarios])

    axes[0].set_ylabel("Evaluation scenario")
    fig.suptitle("Decision Boundary: Transfer x Warm-up (LP negative stash)", fontsize=12)
    if im is None:
        print("Skipping decision boundary plot (no renderable matrix)")
        plt.close(fig)
        return

    cbar = fig.colorbar(im, ax=axes, shrink=0.9)
    cbar.set_label("FPR reduction vs plain BF (%)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    savefig(fig, "decision_boundary_transfer.png")

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Benchmarks
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
def plot_bench():
    path = os.path.join(RESULTS_DIR, "bench.csv")
    if not os.path.exists(path):
        print("Skipping bench (no CSV)")
        return

#bench.csv has multiple headers(one per benchmark type).Read in chunks.
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

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Main
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
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
        "exp8": plot_exp8,
        "decision": plot_decision_boundary,
        "bench": plot_bench,
    }

    for name, fn in plotters.items():
        if "all" in which or name in which:
            print(f"Plotting {name}...")
            fn()

    print("Done.")


if __name__ == "__main__":
    main()
