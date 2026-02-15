"""
Plot Full Causal Recovery accuracy: proportion of participants who got BOTH
object identification and rule inference correct. One bar per rule type.
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]


def _bool(s):
    if hasattr(s, "item"):
        v = s
    else:
        v = s
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def main():
    parser = argparse.ArgumentParser(description="Plot Full Causal Recovery accuracy.")
    parser.add_argument(
        "--input",
        default="main_game_data_with_prior_experience_3_no_prior.csv",
        help="Input CSV",
    )
    parser.add_argument(
        "--output",
        default="full_causal_recovery.png",
        help="Output PNG",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["obj_ok"] = df["object_identification_correct"].map(_bool)
    df["rule_ok"] = df["rule_choice_correct"].map(_bool)
    df["full_recovery"] = df["obj_ok"] & df["rule_ok"]

    rule_types = ["Conjunctive", "Disjunctive"]
    accs = []
    counts = []
    totals = []
    ses = []

    for rt in ["conjunctive", "disjunctive"]:
        sub = df[df["ground_truth_rule"] == rt]
        n = len(sub)
        k = sub["full_recovery"].sum()
        p = (k / n * 100) if n else 0
        se = 100 * np.sqrt((k / n) * (1 - k / n) / n) if n else 0
        accs.append(p)
        counts.append(int(k))
        totals.append(int(n))
        ses.append(se)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    x = np.arange(len(rule_types))
    w = 0.4
    bars = ax.bar(
        x,
        accs,
        w,
        yerr=ses,
        capsize=6,
        color=["#2a9d8f", "#e76f51"],
        alpha=0.85,
        edgecolor="#333333",
        linewidth=1.2,
        error_kw={"color": "#333333", "linewidth": 1.2},
    )
    ax.axhline(100, color="#6a3d9a", linestyle="--", linewidth=1.5, alpha=0.5, label="Perfect (100%)")
    ax.set_xticks(x)
    ax.set_xticklabels(rule_types, fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Rule Type", fontsize=14, fontweight="bold")
    ax.set_title("Full Causal Recovery\n(both objects and rule correct)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(False)

    for i, (bar, acc, k, n) in enumerate(zip(bars, accs, counts, totals)):
        h = bar.get_height()
        se = ses[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + se + 2,
            f"{acc:.1f}%\n({k}/{n})",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout(pad=0.25)
    plt.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    print("Full Causal Recovery (objects + rule correct):")
    for rt, acc, k, n in zip(rule_types, accs, counts, totals):
        print(f"  {rt}: {acc:.1f}% ({k}/{n})")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
