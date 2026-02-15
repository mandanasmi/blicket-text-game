"""
Stacked bar plot: Rule correct (regardless) = Rule correct obj wrong + Both correct;
Objects correct (regardless) = Objects correct rule wrong + Both correct.
All five pieces of information via stacked segments. Percent and (k/n) on each segment.
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.transforms as mtrans
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]


def _bool(s):
    v = s if hasattr(s, "item") else s
    return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")


def main():
    parser = argparse.ArgumentParser(description="Stacked accuracy bars by rule type.")
    parser.add_argument(
        "--input",
        default="main_game_data_with_prior_experience_3_no_prior.csv",
        help="Input CSV (no-prior)",
    )
    parser.add_argument("--output", default="accuracy_three_metrics.png", help="Output PNG")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["obj_ok"] = df["object_identification_correct"].map(_bool)
    df["rule_ok"] = df["rule_choice_correct"].map(_bool)
    df["rule_ok_obj_wrong"] = df["rule_ok"] & ~df["obj_ok"]
    df["obj_ok_rule_wrong"] = df["obj_ok"] & ~df["rule_ok"]
    df["both_ok"] = df["obj_ok"] & df["rule_ok"]

    rule_types = ["Conjunctive", "Disjunctive"]
    # Rule regardless = rule_ok_obj_wrong + both; Objects regardless = obj_ok_rule_wrong + both
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
        "obj_rule_wrong": "#0072B2",
        "both": "#009E73",
    }
    width = 0.16
    x_offset = 0.38
    x = np.arange(len(rule_types)) * 1.15 + x_offset
    x_rule = x - width * 1.4
    x_obj = x + width * 1.4

    min_seg = 10.0  # min drawn height (%) so small segments can fit label text

    def draw_heights(bot, top):
        """Ensure bottom or top segment has at least min_seg height; adjust the other. Total preserved."""
        if bot < min_seg and top >= min_seg - bot:
            return min_seg, bot + top - min_seg
        if top < min_seg and bot >= min_seg - top:
            return bot + top - min_seg, min_seg
        return bot, top

    rule1_draw, rule2_draw = [], []
    obj1_draw, obj2_draw = [], []
    for j in range(len(rule_types)):
        r1, r2 = draw_heights(rule_obj_wrong_pct[j], both_pct[j])
        o1, o2 = draw_heights(obj_rule_wrong_pct[j], both_pct[j])
        rule1_draw.append(r1)
        rule2_draw.append(r2)
        obj1_draw.append(o1)
        obj2_draw.append(o2)

    fig, ax = plt.subplots(1, 1, figsize=(9, 4))

    # Stacked bar: Rule correct (regardless) = [rule ok obj wrong | both]
    b1 = ax.bar(
        x_rule,
        rule1_draw,
        width,
        color=colors["rule_obj_wrong"],
        alpha=0.85,
        edgecolor="#333333",
        linewidth=1.0,
        label="Rule correct, objects wrong",
    )
    b2 = ax.bar(
        x_rule,
        rule2_draw,
        width,
        bottom=rule1_draw,
        color=colors["both"],
        alpha=0.85,
        edgecolor="#333333",
        linewidth=1.0,
        label="Both correct",
    )
    # Stacked bar: Objects correct (regardless) = [obj ok rule wrong | both]
    b3 = ax.bar(
        x_obj,
        obj1_draw,
        width,
        color=colors["obj_rule_wrong"],
        alpha=0.85,
        edgecolor="#333333",
        linewidth=1.0,
        label="Objects correct, rule wrong",
    )
    b4 = ax.bar(
        x_obj,
        obj2_draw,
        width,
        bottom=obj1_draw,
        color=colors["both"],
        alpha=0.85,
        edgecolor="#333333",
        linewidth=1.0,
    )

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

    ax.set_xlabel("Rule Type", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(rule_types, fontsize=13)
    ax.set_ylim(0, 110)
    ax.axhline(100, color="#6a3d9a", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.grid(False)

    # Curly brace "}" on the right of each bar: exact bar height, thin gray line
    gap = 0.008
    txt_offset = 0.02
    brace_width = 0.025
    brace_color = "#888888"
    brace_lw = 0.7

    def draw_brace(ax, x_left, y_bot, y_top):
        h = y_top - y_bot
        # Unit "}" path: (x 0..1, y 0..1), opens right
        verts = [(0.5, 0), (0, 0.25), (0.5, 0.5), (0, 0.75), (0.5, 1)]
        codes = [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3, mpath.Path.CURVE3, mpath.Path.CURVE3]
        path = mpath.Path(verts, codes)
        tr = mtrans.Affine2D().scale(brace_width, h).translate(x_left, y_bot) + ax.transData
        patch = mpatches.PathPatch(path, transform=tr, facecolor="none", edgecolor=brace_color, linewidth=brace_lw)
        ax.add_patch(patch)

    for i in range(len(rule_types)):
        # Rule regardless bar (first bar)
        hr = rule1_draw[i] + rule2_draw[i]
        xr = x_rule[i] + width + gap
        draw_brace(ax, xr, 0, hr)
        ax.text(xr + brace_width + txt_offset, hr / 2, "p(rule=c | obj=w) + p(rule=c | obj=c) = p(rule=c)", ha="left", va="center", fontsize=9, color="#333333", rotation=90)
        # Objects regardless bar (second bar)
        ho = obj1_draw[i] + obj2_draw[i]
        xo = x_obj[i] + width + gap
        draw_brace(ax, xo, 0, ho)
        ax.text(xo + brace_width + txt_offset, ho / 2 + 6, "p(obj=c | rule=w) + p(obj=c | rule=c) = p(obj=c)", ha="left", va="center", fontsize=9, color="#333333", rotation=90)

    ax.set_xlim(-0.05, 2.2)

    legend_handles = [
        mpatches.Patch(facecolor=colors["rule_obj_wrong"], edgecolor="#333333", linewidth=1.0, label="Rule correct & Objects wrong"),
        mpatches.Patch(facecolor=colors["obj_rule_wrong"], edgecolor="#333333", linewidth=1.0, label="Objects correct & Rule wrong"),
        mpatches.Patch(facecolor=colors["both"], edgecolor="#333333", linewidth=1.0, label="Objects correct & Rule correct"),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout(pad=0.25)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
