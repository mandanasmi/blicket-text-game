"""
Alternative visualizations for the four rule/object outcomes:
- Grouped bars: 4 outcomes on x-axis, Conjunctive vs Disjunctive side-by-side per outcome. Easy to compare.
- Heatmap: outcomes x rule types, % in cells. Compact, no legend.
Uses same data as plot_accuracy_four_outcomes_stacked (no prior experience).
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]


def _bool(s):
    v = s if hasattr(s, "item") else s
    return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")


def _get_data(df):
    df = df.copy()
    df["obj_ok"] = df["object_identification_correct"].map(_bool)
    df["rule_ok"] = df["rule_choice_correct"].map(_bool)
    df["rule_ok_obj_wrong"] = df["rule_ok"] & ~df["obj_ok"]
    df["rule_wrong_obj_ok"] = ~df["rule_ok"] & df["obj_ok"]
    df["both_ok"] = df["rule_ok"] & df["obj_ok"]
    df["both_wrong"] = ~df["rule_ok"] & ~df["obj_ok"]

    order = ["both_wrong", "rule_ok_obj_wrong", "rule_wrong_obj_ok", "both_ok"]
    rule_types = ["Conjunctive", "Disjunctive"]
    pct = {k: [] for k in order}
    counts = {k: [] for k in order}
    n_per_rule = []

    for rt in ["conjunctive", "disjunctive"]:
        sub = df[df["ground_truth_rule"] == rt]
        n = len(sub)
        n_per_rule.append(n)
        for key in order:
            k = int(sub[key].sum())
            counts[key].append(k)
            pct[key].append(k / n * 100 if n else 0)

    colors = {
        "both_wrong": "#94A3B8",
        "rule_ok_obj_wrong": "#FBBF24",
        "rule_wrong_obj_ok": "#F87171",
        "both_ok": "#34D399",
    }
    return order, rule_types, pct, counts, n_per_rule, colors


def plot_grouped(args, order, rule_types, pct, counts, n_per_rule, colors):
    out = args.output_grouped or "accuracy_four_outcomes_grouped.png"
    short_labels = [
        "Both wrong",
        "Rule c., Obj w.",
        "Rule w., Obj c.",
        "Both correct",
    ]
    x = np.arange(len(order))
    width = 0.35
    off = [-width / 2, width / 2]
    rule_colors = ["#2a9d8f", "#e76f51"]  # Conj, Disj

    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    for j, (rt, rc) in enumerate(zip(rule_types, rule_colors)):
        vals = [pct[k][j] for k in order]
        bars = ax.bar(x + off[j], vals, width, label=rt, color=rc, alpha=0.85, edgecolor="#333333", linewidth=0.8)
        for i, (v, k) in enumerate(zip(vals, order)):
            n = n_per_rule[j]
            c = counts[k][j]
            if c == 0 and v < 0.5:
                continue
            txt = f"{v:.1f}%" if v < 0.5 else f"{v:.1f}% ({int(c)}/{n})"
            ax.text(x[i] + off[j], v + 1.5, txt, ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("Percentage (%)", fontsize=8)
    ax.set_ylim(0, 105)
    ax.axhline(100, color="#6a3d9a", ls="--", lw=1.0, alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=True, fancybox=False, shadow=False)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=8)
    plt.subplots_adjust(left=0.12, right=0.96, top=0.94, bottom=0.12)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Saved: {out}")


def plot_heatmap(args, order, rule_types, pct, counts, n_per_rule, colors):
    out = args.output_heatmap or "accuracy_four_outcomes_heatmap.png"
    short_labels = ["Both wrong", "Rule c., Obj w.", "Rule w., Obj c.", "Both correct"]
    M = np.array([[pct[k][j] for j in range(len(rule_types))] for k in order])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(M, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    for i in range(len(order)):
        for j in range(len(rule_types)):
            v = M[i, j]
            c = counts[order[i]][j]
            n = n_per_rule[j]
            txt = f"{v:.1f}%\n({int(c)}/{n})" if c else f"{v:.1f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    ax.set_xticks(range(len(rule_types)))
    ax.set_xticklabels(rule_types, fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel("Rule type", fontsize=9)
    plt.colorbar(im, ax=ax, label="Percentage (%)")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser(description="Alternative four-outcomes plots: grouped bars or heatmap.")
    ap.add_argument("--input", default="main_game_data_with_prior_experience.csv", help="Input CSV")
    ap.add_argument("--variant", choices=["grouped", "heatmap", "both"], default="both",
                    help="Which plot(s) to generate")
    ap.add_argument("--output-grouped", default=None, help="Output path for grouped bars")
    ap.add_argument("--output-heatmap", default=None, help="Output path for heatmap")
    ap.add_argument("--max-per-rule", type=int, default=50, help="Max participants per rule type (match stacked)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = df[df["has_prior_experience"] == False].copy()
    rng = np.random.default_rng(args.seed)
    keep = np.zeros(len(df), dtype=bool)
    for rt in ["conjunctive", "disjunctive"]:
        sub = df[df["ground_truth_rule"] == rt]
        uids = sub["user_id"].unique()
        n = min(len(uids), args.max_per_rule)
        chosen = rng.choice(uids, size=n, replace=False) if len(uids) > n else uids
        keep |= (df["ground_truth_rule"] == rt) & (df["user_id"].isin(chosen))
    df = df[keep].copy()
    order, rule_types, pct, counts, n_per_rule, colors = _get_data(df)

    if args.variant in ("grouped", "both"):
        plot_grouped(args, order, rule_types, pct, counts, n_per_rule, colors)
    if args.variant in ("heatmap", "both"):
        plot_heatmap(args, order, rule_types, pct, counts, n_per_rule, colors)


if __name__ == "__main__":
    main()
