"""
Same as accuracy_three_metrics (stacked bars) but: no brackets, no equation text;
two extra bars per rule type: Rule correct regardless (purple), Objects correct
regardless (blue), placed next to the stacked bars.
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
    parser = argparse.ArgumentParser(description="Stacked + Rule/Objects regardless bars.")
    parser.add_argument(
        "--input",
        default="main_game_data_with_prior_experience_3_no_prior.csv",
        help="Input CSV (no-prior)",
    )
    parser.add_argument("--output", default="accuracy_rule_obj_regardless.png", help="Output PNG")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["obj_ok"] = df["object_identification_correct"].map(_bool)
    df["rule_ok"] = df["rule_choice_correct"].map(_bool)
    df["rule_ok_obj_wrong"] = df["rule_ok"] & ~df["obj_ok"]
    df["obj_ok_rule_wrong"] = df["obj_ok"] & ~df["rule_ok"]
    df["both_ok"] = df["obj_ok"] & df["rule_ok"]

    rule_types = ["Conjunctive", "Disjunctive"]
    rule_obj_wrong_pct = []
    both_pct = []
    obj_rule_wrong_pct = []
    rule_obj_wrong_k = []
    both_k = []
    obj_rule_wrong_k = []
    n_per_rule = []

    for rt in ["conjunctive", "disjunctive"]:
        sub = df[df["ground_truth_rule"] == rt]
        n = len(sub)
        n_per_rule.append(n)
        k1 = int(sub["rule_ok_obj_wrong"].sum())
        k2 = int(sub["both_ok"].sum())
        k3 = int(sub["obj_ok_rule_wrong"].sum())
        rule_obj_wrong_pct.append(k1 / n * 100 if n else 0)
        both_pct.append(k2 / n * 100 if n else 0)
        obj_rule_wrong_pct.append(k3 / n * 100 if n else 0)
        rule_obj_wrong_k.append(k1)
        both_k.append(k2)
        obj_rule_wrong_k.append(k3)

    # Colorblind-friendly (Okabe–Ito / Paul Tol–style)
    colors = {
        "rule_obj_wrong": "#E69F00",
        "obj_rule_wrong": "#C7656B",
        "both": "#00E676",
    }
    purple = "#5B8FC4"
    blue = "#56B4E9"

    width = 0.12
    width_solid = 0.09
    x_offset = 0.72
    x = np.arange(len(rule_types)) * 0.70 + x_offset
    # Four bars per group: rule stacked, rule regardless (purple), obj stacked, obj regardless (blue)
    off = 0.42
    x_rule = x - off
    x_obj = x - off + width * 2.3
    gap_ref = (1.06 * width - width_solid / 2) - (0 + width / 2)
    small_gap = gap_ref * 0.35
    rule_regardless_offset = width / 2 + small_gap + width_solid / 2
    x_rule_regardless = x - off + rule_regardless_offset
    blue_center_offset = width * 2.3 + width / 2 + small_gap + width_solid / 2
    x_obj_regardless = x - off + blue_center_offset
    x_center = x - off + blue_center_offset / 2

    min_seg = 10.0

    def draw_heights(bot, top):
        if bot < min_seg and top >= min_seg - bot:
            return min_seg, bot + top - min_seg
        if top < min_seg and bot >= min_seg - top:
            return bot + top - min_seg, min_seg
        return bot, top

    rule1_draw, rule2_draw = [], []
    obj1_draw, obj2_draw = [], []
    rule_regardless_pct = []
    obj_regardless_pct = []
    for j in range(len(rule_types)):
        r1, r2 = draw_heights(rule_obj_wrong_pct[j], both_pct[j])
        o1, o2 = draw_heights(obj_rule_wrong_pct[j], both_pct[j])
        rule1_draw.append(r1)
        rule2_draw.append(r2)
        obj1_draw.append(o1)
        obj2_draw.append(o2)
        rule_regardless_pct.append(r1 + r2)
        obj_regardless_pct.append(o1 + o2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    # Stacked bars (same as accuracy_three_metrics)
    ax.bar(x_rule, rule1_draw, width, color=colors["rule_obj_wrong"], alpha=0.85, edgecolor="#333333", linewidth=1.0)
    ax.bar(x_rule, rule2_draw, width, bottom=rule1_draw, color=colors["both"], alpha=0.85, edgecolor="#333333", linewidth=1.0)
    ax.bar(x_obj, obj1_draw, width, color=colors["obj_rule_wrong"], alpha=0.85, edgecolor="#333333", linewidth=1.0)
    ax.bar(x_obj, obj2_draw, width, bottom=obj1_draw, color=colors["both"], alpha=0.85, edgecolor="#333333", linewidth=1.0)

    # Two new bars: Rule correct (purple), Objects correct (blue) — narrower
    ax.bar(x_rule_regardless, rule_regardless_pct, width_solid, color=purple, alpha=0.85, edgecolor="#333333", linewidth=1.0)
    ax.bar(x_obj_regardless, obj_regardless_pct, width_solid, color=blue, alpha=0.85, edgecolor="#333333", linewidth=1.0)

    def seg_label(pct, k, n):
        return f"{pct:.1f}%\n({k}/{n})"

    fs = 7
    pad = 0.8
    for j in range(len(rule_types)):
        n = n_per_rule[j]
        h1, h2 = rule_obj_wrong_pct[j], both_pct[j]
        d1, d2 = rule1_draw[j], rule2_draw[j]
        xx = x_rule[j]
        if d1 >= min_seg:
            ax.text(xx, d1 / 2, seg_label(h1, rule_obj_wrong_k[j], n), ha="center", va="center", fontsize=fs, fontweight="bold")
        else:
            ax.text(xx, d1 + pad, seg_label(h1, rule_obj_wrong_k[j], n), ha="center", va="bottom", fontsize=fs, fontweight="bold")
        if d2 >= min_seg:
            ax.text(xx, d1 + d2 / 2, seg_label(h2, both_k[j], n), ha="center", va="center", fontsize=fs, fontweight="bold")
        else:
            ax.text(xx, d1 + d2 + pad, seg_label(h2, both_k[j], n), ha="center", va="bottom", fontsize=fs, fontweight="bold")
        h3 = obj_rule_wrong_pct[j]
        o1, o2 = obj1_draw[j], obj2_draw[j]
        xx = x_obj[j]
        if o1 >= min_seg:
            ax.text(xx, o1 / 2, seg_label(h3, obj_rule_wrong_k[j], n), ha="center", va="center", fontsize=fs, fontweight="bold")
        else:
            ax.text(xx, o1 + pad, seg_label(h3, obj_rule_wrong_k[j], n), ha="center", va="bottom", fontsize=fs, fontweight="bold")
        if o2 >= min_seg:
            ax.text(xx, o1 + o2 / 2, seg_label(h2, both_k[j], n), ha="center", va="center", fontsize=fs, fontweight="bold")
        else:
            ax.text(xx, o1 + o2 + pad, seg_label(h2, both_k[j], n), ha="center", va="bottom", fontsize=fs, fontweight="bold")

    # Labels inside new bars (purple, blue): % and k/n centered in bar
    for j in range(len(rule_types)):
        n = n_per_rule[j]
        kr = int(rule_obj_wrong_k[j]) + int(both_k[j])
        ko = int(obj_rule_wrong_k[j]) + int(both_k[j])
        ax.text(x_rule_regardless[j], rule_regardless_pct[j] / 2, seg_label(rule_regardless_pct[j], kr, n),
                ha="center", va="center", fontsize=fs, fontweight="bold")
        ax.text(x_obj_regardless[j], obj_regardless_pct[j] / 2, seg_label(obj_regardless_pct[j], ko, n),
                ha="center", va="center", fontsize=fs, fontweight="bold")

    ax.set_xlabel("Rule Type", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_xticks(x_center)
    ax.set_xticklabels(rule_types, fontsize=13)
    ax.set_ylim(0, 110)
    ax.axhline(100, color="#6a3d9a", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.grid(False)
    ax.set_xlim(0.18, 1.48)

    legend_handles = [
        mpatches.Patch(facecolor=colors["rule_obj_wrong"], edgecolor="#333333", linewidth=1.0, label="Rule correct & Objects wrong"),
        mpatches.Patch(facecolor=colors["obj_rule_wrong"], edgecolor="#333333", linewidth=1.0, label="Objects correct & Rule wrong"),
        mpatches.Patch(facecolor=colors["both"], edgecolor="#333333", linewidth=1.0, label="Objects correct & Rule correct"),
        mpatches.Patch(facecolor=purple, edgecolor="#333333", linewidth=1.0, label="Rule correct"),
        mpatches.Patch(facecolor=blue, edgecolor="#333333", linewidth=1.0, label="Objects correct"),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(pad=0.25)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
