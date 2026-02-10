"""
Stacked bar chart: p(rule=c & obj=w), p(rule=w & obj=c), p(rule=c & obj=c), p(rule=w & obj=w)
by rule type (Conjunctive, Disjunctive). One stacked bar per rule type.
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]


def _bool(s):
    v = s if hasattr(s, "item") else s
    return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")


def main():
    parser = argparse.ArgumentParser(description="Stacked bar: four rule/object outcomes by rule type.")
    parser.add_argument("--input", default="../main_game_data_with_prior_experience.csv", help="Input CSV (default: from round7)")
    parser.add_argument("--output", default="accuracy_four_outcomes_stacked.png", help="Output PNG")
    parser.add_argument("--max-per-rule", type=int, default=50, help="Max participants per rule type (sample if exceeded)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[df["has_prior_experience"] == False].copy()

    # Sample up to --max-per-rule participants per rule type
    rng = np.random.default_rng(args.seed)
    keep = np.zeros(len(df), dtype=bool)
    for rt in ["conjunctive", "disjunctive"]:
        sub = df[df["ground_truth_rule"] == rt]
        uids = sub["user_id"].unique()
        n = min(len(uids), args.max_per_rule)
        chosen = rng.choice(uids, size=n, replace=False) if len(uids) > n else uids
        keep |= (df["ground_truth_rule"] == rt) & (df["user_id"].isin(chosen))
    df = df[keep].copy()

    df["obj_ok"] = df["object_identification_correct"].map(_bool)
    df["rule_ok"] = df["rule_choice_correct"].map(_bool)

    # Four mutually exclusive outcomes
    df["rule_ok_obj_wrong"] = df["rule_ok"] & ~df["obj_ok"]
    df["rule_wrong_obj_ok"] = ~df["rule_ok"] & df["obj_ok"]
    df["both_ok"] = df["rule_ok"] & df["obj_ok"]
    df["both_wrong"] = ~df["rule_ok"] & ~df["obj_ok"]

    rule_types = ["Conjunctive", "Disjunctive"]
    pct = {  # each list has 2 elements (conjunctive, disjunctive)
        "rule_ok_obj_wrong": [],
        "rule_wrong_obj_ok": [],
        "both_ok": [],
        "both_wrong": [],
    }
    counts = {k: [] for k in pct}
    n_per_rule = []

    for rt in ["conjunctive", "disjunctive"]:
        sub = df[df["ground_truth_rule"] == rt]
        n = len(sub)
        n_per_rule.append(n)
        for key in pct:
            k = int(sub[key].sum())
            counts[key].append(k)
            pct[key].append(k / n * 100 if n else 0)

    # Softer, cohesive palette (colorblind-friendly)
    colors = {
        "both_wrong": "#94A3B8",
        "rule_ok_obj_wrong": "#FBBF24",
        "rule_wrong_obj_ok": "#F87171",
        "both_ok": "#34D399",
    }
    labels = {
        "rule_ok_obj_wrong": "Rule correct & Objects wrong",
        "rule_wrong_obj_ok": "Rule wrong & Objects correct",
        "both_ok": "Rule correct & Objects correct",
        "both_wrong": "Rule wrong & Objects wrong",
    }

    # Stack order bottom -> top
    order = ["both_wrong", "rule_ok_obj_wrong", "rule_wrong_obj_ok", "both_ok"]
    min_seg = 14.0  # minimum drawn height (%) so labels fit in compartment

    def redistribute(p):
        """Ensure each positive segment has at least min_seg drawn height; borrow from larger. Sum preserved."""
        draw = list(p)
        small = [i for i in range(4) if 0 < draw[i] < min_seg]
        big = [i for i in range(4) if draw[i] >= min_seg]
        need = sum(min_seg - draw[i] for i in small)
        available = sum(draw[i] - min_seg for i in big)
        if need <= 0 or available <= 0 or need > available:
            return draw
        factor = available
        for i in big:
            take = need * (draw[i] - min_seg) / factor
            draw[i] = draw[i] - take
        for i in small:
            draw[i] = min_seg
        return draw

    # Per rule type: compute drawn heights
    draw = {k: [] for k in order}
    for j in range(len(rule_types)):
        p_vec = [pct[k][j] for k in order]
        d_vec = redistribute(p_vec)
        for i, k in enumerate(order):
            draw[k].append(d_vec[i])

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.2))
    # Bar width; centers 0.44, 0.56
    x = np.array([0.44, 0.56])
    width = 0.07

    bottoms = np.zeros(len(rule_types))
    for key in order:
        vals = np.array(draw[key])
        ax.bar(x, vals, width, bottom=bottoms, color=colors[key], alpha=0.85,
               edgecolor="#333333", linewidth=0.8)
        for i in range(len(rule_types)):
            n = n_per_rule[i]
            v_actual = pct[key][i]
            v_draw = vals[i]
            k = counts[key][i]
            if v_actual < 0.5 and k == 0:
                continue
            bot = bottoms[i]
            mid = bot + v_draw / 2
            if v_actual < 0.5:
                txt = f"{v_actual:.1f}%"
            else:
                txt = f"{v_actual:.1f}% ({k}/{n})"
            if v_draw >= min_seg - 1:
                ax.text(x[i], mid, txt, ha="center", va="center", fontsize=7)
            elif v_actual >= 0.5:
                ax.text(x[i], bot + v_draw + 0.4, txt, ha="center", va="bottom", fontsize=6)
        bottoms = bottoms + vals

    ax.set_xticks(x)
    ax.set_xticklabels(rule_types, fontsize=8)
    ax.set_xlim(0.38, 0.62)
    ax.set_ylabel("Percentage (%)", fontsize=8)
    ax.set_ylim(0, 105)
    ax.axhline(100, color="#6a3d9a", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=8)

    legend_handles = [
        mpatches.Patch(facecolor=colors[k], edgecolor="#333333", linewidth=0.8, label=labels[k])
        for k in order
    ]
    ax.legend(handles=legend_handles, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              frameon=True, fancybox=False, shadow=False, ncol=1)

    plt.subplots_adjust(left=0.12, right=0.96, top=0.94, bottom=0.12)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
