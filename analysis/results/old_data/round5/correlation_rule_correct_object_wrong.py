"""
Compute correlation between rule correct and object incorrect
(rule correct AND object wrong) for conjunctive vs disjunctive cases.
Uses main_game_data_with_prior_experience_3.csv.
Optionally saves a bar-chart visualization.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "sans-serif"]


def _plot(stats, output_path):
    """Bar chart: r(object_correct, rule_correct) only, by rule type."""
    stats = [s for s in stats if s["n"] > 0]
    if not stats:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    rule_types = [s["rule_type"].capitalize() for s in stats]
    r_vals = [s.get("r_obj_rule", np.nan) for s in stats]
    r_vals = [0.0 if np.isnan(r) else float(r) for r in r_vals]
    colors = ["#2a9d8f", "#e76f51"]
    x = np.arange(len(rule_types))
    width = 0.5
    bars = ax.bar(x, r_vals, width, color=colors[: len(rule_types)], edgecolor="#333333", linewidth=1.2)
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(rule_types, fontsize=12)
    ax.set_ylabel("r(object correct, rule correct)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ground-truth rule type", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.6, 0.6)
    ax.set_title("Correlation: object identification correct vs rule correct", fontsize=14, fontweight="bold", pad=12)
    for bar, s in zip(bars, stats):
        ror = s.get("r_obj_rule", np.nan)
        if np.isnan(ror):
            continue
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        dy = 0.04 if h >= 0 else -0.04
        ax.text(bar.get_x() + bar.get_width() / 2, h + dy, f"r = {ror:.2f}", ha="center", va=va, fontsize=11)
    plt.tight_layout(pad=0.25)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Correlation: rule correct vs object wrong by rule type.")
    parser.add_argument("--input", default="main_game_data_with_prior_experience_3.csv", help="Input CSV")
    parser.add_argument("--output", default="correlation_rule_correct_object_wrong.png", help="Output PNG")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving plot")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["object_wrong"] = ~(df["object_identification_correct"].fillna(False).astype(bool))
    df["rule_correct"] = df["rule_choice_correct"].fillna(False).astype(bool)
    df["object_correct"] = df["object_identification_correct"].fillna(False).astype(bool)

    print("=" * 70)
    print("Correlation: Rule Correct vs Object (by ground-truth rule)")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Total N: {len(df)}")
    print()

    stats = []
    for rule_type in ["conjunctive", "disjunctive"]:
        sub = df[df["ground_truth_rule"] == rule_type].copy()
        if len(sub) == 0:
            print(f"[{rule_type.upper()}] No data.")
            print()
            stats.append({"rule_type": rule_type, "n": 0, "both": 0, "pct": 0, "r": np.nan, "r_obj_rule": np.nan})
            continue

        n = len(sub)
        rule_ok = sub["rule_correct"].astype(int)
        obj_wrong = sub["object_wrong"].astype(int)
        obj_ok = sub["object_correct"].astype(int)

        # Contingency: rule_correct x object_wrong
        both = (sub["rule_correct"] & sub["object_wrong"]).sum()
        rule_ok_only = (sub["rule_correct"] & ~sub["object_wrong"]).sum()
        obj_wrong_only = (~sub["rule_correct"] & sub["object_wrong"]).sum()
        neither = (~sub["rule_correct"] & ~sub["object_wrong"]).sum()

        # r(rule_correct, object_wrong)
        r = np.corrcoef(rule_ok, obj_wrong)[0, 1] if n > 1 else np.nan
        r = np.nan if np.isnan(r) else float(r)

        # r(object_correct, rule_correct)
        r_obj_rule = np.corrcoef(obj_ok, rule_ok)[0, 1] if n > 1 else np.nan
        r_obj_rule = np.nan if np.isnan(r_obj_rule) else float(r_obj_rule)

        p_rule_ok_obj_wrong = both / n if n else 0
        stats.append({"rule_type": rule_type, "n": n, "both": int(both), "pct": 100 * p_rule_ok_obj_wrong, "r": r, "r_obj_rule": r_obj_rule})

        print(f"[{rule_type.upper()}] N = {n}")
        print(f"  Rule correct, object wrong: {int(both)} / {n} ({100 * p_rule_ok_obj_wrong:.1f}%)")
        print(f"  Contingency (rule_correct x object_wrong):")
        print(f"                    object_wrong=0   object_wrong=1")
        print(f"    rule_correct=0      {int(neither):4}              {int(obj_wrong_only):4}")
        print(f"    rule_correct=1      {int(rule_ok_only):4}              {int(both):4}")
        print(f"  r(rule_correct, object_wrong): {r:.3f}" if not np.isnan(r) else "  r(rule_correct, object_wrong): N/A")
        print(f"  r(object_correct, rule_correct): {r_obj_rule:.3f}" if not np.isnan(r_obj_rule) else "  r(object_correct, rule_correct): N/A")
        print()

    # Plot
    if not getattr(args, "no_plot", False):
        _plot(stats, args.output)

    # Summary
    cj = df[df["ground_truth_rule"] == "conjunctive"]
    dj = df[df["ground_truth_rule"] == "disjunctive"]
    cj_both = ((cj["rule_choice_correct"] == True) & (cj["object_identification_correct"] == False)).sum()
    dj_both = ((dj["rule_choice_correct"] == True) & (dj["object_identification_correct"] == False)).sum()
    print("SUMMARY: Rule correct, object wrong")
    print(f"  Conjunctive:  {int(cj_both)} / {len(cj)} ({100 * cj_both / len(cj):.1f}%)")
    print(f"  Disjunctive:  {int(dj_both)} / {len(dj)} ({100 * dj_both / len(dj):.1f}%)")
    print()
    print("SUMMARY: r(object_correct, rule_correct)")
    for s in stats:
        if s["n"] > 0:
            ror = s.get("r_obj_rule", np.nan)
            ror_str = f"{ror:.3f}" if not np.isnan(ror) else "N/A"
            print(f"  {s['rule_type'].capitalize()}: r = {ror_str}")
    print("=" * 70)

if __name__ == "__main__":
    main()
