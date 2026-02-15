#!/usr/bin/env python3
"""
Plot accuracy for o3-mini and o4-mini models by reasoning effort.
Separate panels for conjunctive vs disjunctive rules.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot o3-mini and o4-mini accuracy by rule")
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to llm_o3_o4_accuracy_by_rule.csv (default: auto-detect). Use llm_o3_o4_accuracy_by_rule_4obj.csv for 4 objects.",
    )
    parser.add_argument(
        "--output",
        default="llm_o3_o4_accuracy.png",
        help="Output PNG filename (default: llm_o3_o4_accuracy.png)",
    )
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv) if args.csv else script_dir / "llm_o3_o4_accuracy_by_rule.csv"
    output_path = script_dir / args.output

    if not csv_path.exists():
        print(f"ERROR: CSV file not found at {csv_path}")
        print("Run compute_llm_o3_o4_accuracy.py first to generate the data.")
        return 1

    df = pd.read_csv(csv_path)

    # Define order for reasoning effort
    reasoning_order = ["low", "medium", "high", "default"]
    df["reasoning_effort"] = pd.Categorical(
        df["reasoning_effort"], categories=reasoning_order, ordered=True
    )
    df = df.sort_values(["model", "reasoning_effort", "rule"])

    rules = ["conjunctive", "disjunctive"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    colors = {
        "o3-mini": "#3498db",  # blue
        "o4-mini": "#e67e22",  # orange
    }

    for ax_idx, rule in enumerate(rules):
        ax = axes[ax_idx]
        sub = df[df["rule"] == rule]

        if sub.empty:
            ax.set_title(f"{rule.capitalize()}")
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("All-Correct Accuracy")
            continue

        models = sub["model"].unique()
        reasoning_efforts = sub["reasoning_effort"].dropna().unique()
        reasoning_efforts = sorted(reasoning_efforts, key=lambda x: reasoning_order.index(x) if x in reasoning_order else 99)

        bar_width = 0.35
        x_positions = np.arange(len(reasoning_efforts))

        for i, model in enumerate(models):
            model_data = sub[sub["model"] == model]
            means = []
            ses = []
            counts = []
            for re in reasoning_efforts:
                row = model_data[model_data["reasoning_effort"] == re]
                if len(row) > 0:
                    means.append(row["mean"].values[0])
                    ses.append(row["se"].values[0])
                    counts.append(row["count"].values[0])
                else:
                    means.append(0)
                    ses.append(0)
                    counts.append(0)

            offset = (i - len(models) / 2 + 0.5) * bar_width
            ax.bar(
                x_positions + offset,
                means,
                bar_width,
                yerr=ses,
                label=model,
                color=colors.get(model, "#95a5a6"),
                capsize=4,
                edgecolor="#333",
                linewidth=0.8,
            )

            for j, (x, y, m, c) in enumerate(zip(x_positions + offset, means, means, counts)):
                if y > 0 and c > 0:
                    num_correct = int(round(m * c))
                    ax.text(
                        x,
                        y + ses[j] + 0.02,
                        f"{num_correct}/{int(c)}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="#555",
                    )

        ax.set_title(rule.capitalize(), fontsize=13)
        ax.set_xlabel("Reasoning Effort", fontsize=11)
        ax.set_ylabel("All-Correct Accuracy", fontsize=11)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(re) for re in reasoning_efforts])
        ax.set_ylim(0, 1.1)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle=":")

    fig.suptitle("LLM Accuracy: o3-mini and o4-mini by Reasoning Effort and Rule", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close()

    return 0


if __name__ == "__main__":
    exit(main())
