#!/usr/bin/env python3
"""Plot frequency distributions for datasets in data/.

Supported input formats:
1) HIBP-style lines: <35-hex-hash>:<count>
2) Plain token lists: one token per line (frequency inferred from repeats)

Outputs are written to results/plots/frequency/ as:
- <dataset>_hist.png      (histogram of frequency counts)
- <dataset>_rank.png      (rank-frequency curve, log-log)
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

HIBP_LINE_RE = re.compile(r"^[A-F0-9]{35}:(\d+)$")


def parse_counts(path: Path) -> np.ndarray:
    """Return frequency counts for entries in a dataset file."""
    hibp_counts: list[int] = []
    tokens: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            m = HIBP_LINE_RE.match(line)
            if m:
                hibp_counts.append(int(m.group(1)))
            else:
                tokens.append(line)

    if hibp_counts and not tokens:
        return np.array(hibp_counts, dtype=np.int64)

    # Fallback for plain text lists: frequency from repeated tokens.
    freq = Counter(tokens)
    return np.array(list(freq.values()), dtype=np.int64)


def plot_histogram(counts: np.ndarray, title: str, out_path: Path) -> None:
    """Plot histogram of frequency counts using log-spaced bins."""
    if counts.size == 0:
        return

    cmin = int(max(1, counts.min()))
    cmax = int(counts.max())

    if cmin == cmax:
        bins = np.array([cmin, cmax + 1])
    else:
        bins = np.logspace(np.log10(cmin), np.log10(cmax), num=50)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(counts, bins=bins, color="#2E5EAA", alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency count")
    ax.set_ylabel("Number of entries")
    ax.set_title(f"Frequency histogram: {title}")
    ax.grid(True, which="both", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rank_frequency(counts: np.ndarray, title: str, out_path: Path, top_n: int) -> None:
    """Plot rank-frequency distribution (Zipf-style) on log-log axes."""
    if counts.size == 0:
        return

    sorted_counts = np.sort(counts)[::-1]
    if top_n > 0:
        sorted_counts = sorted_counts[:top_n]

    ranks = np.arange(1, len(sorted_counts) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ranks, sorted_counts, color="#C24E00", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency count")
    ax.set_title(f"Rank-frequency: {title}")
    ax.grid(True, which="both", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def default_inputs(repo_root: Path) -> list[Path]:
    return sorted((repo_root / "data").glob("*.txt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot frequency distributions for dataset files")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional input dataset paths. Defaults to data/*.txt",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=2000000,
        help="Maximum number of ranked points to show in rank-frequency plots",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.paths:
        inputs = [Path(p).resolve() for p in args.paths]
    else:
        inputs = default_inputs(repo_root)

    if not inputs:
        print("No input files found.")
        return 1

    out_dir = repo_root / "results" / "plots" / "frequency"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in inputs:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        counts = parse_counts(path)
        if counts.size == 0:
            print(f"Skipping empty/unsupported file: {path}")
            continue

        stem = path.stem
        title = path.name
        hist_out = out_dir / f"{stem}_hist.png"
        rank_out = out_dir / f"{stem}_rank.png"

        plot_histogram(counts, title, hist_out)
        plot_rank_frequency(counts, title, rank_out, args.top_n)

        print(
            f"{path}: n={counts.size}, min={int(counts.min())}, "
            f"max={int(counts.max())}, mean={float(counts.mean()):.2f}"
        )
        print(f"  saved {hist_out}")
        print(f"  saved {rank_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
