"""
Grouped bar plot: Average number of tests per LLM trial by rule type (conjunctive vs disjunctive).
Data from llm_num_tests_by_rule_4obj.csv (extracted from action logs).
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"]

MODEL_ORDER = ["deepseek-reasoner", "gpt-4o", "deepseek-chat", "gpt-4o-mini", "gemma3:27b", "qwq"]
model_colors = {
    "deepseek-reasoner": "#4477AA",
    "gpt-4o": "#CCBB44",
    "deepseek-chat": "#228833",
    "gpt-4o-mini": "#BB88CC",
    "gemma3:27b": "#88CCEE",
    "qwq": "#44AA99",
}


def main():
    parser = argparse.ArgumentParser(description="Plot average num_tests per LLM by rule")
    parser.add_argument("--csv", default="llm_num_tests_by_rule_4obj.csv", help="Input CSV")
    parser.add_argument("--output", default="llm_num_tests_by_rule.png", help="Output PNG")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / args.csv
    output_path = script_dir / args.output

    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No data in CSV.")
        return 0

    # Prepare data: one row per (model, rule) with mean and se
    plot_data = []
    for model in MODEL_ORDER:
        conj = df[(df["model"] == model) & (df["rule"] == "conjunctive")]
        disj = df[(df["model"] == model) & (df["rule"] == "disjunctive")]
        if len(conj) > 0 and len(disj) > 0:
            plot_data.append({
                "model": model,
                "conj_mean": float(conj.iloc[0]["mean"]),
                "conj_se": float(conj.iloc[0]["se"]),
                "disj_mean": float(disj.iloc[0]["mean"]),
                "disj_se": float(disj.iloc[0]["se"]),
            })

    if not plot_data:
        print("No data to plot.")
        return 0

    plot_df = pd.DataFrame(plot_data)
    n = len(plot_df)
    width = 0.3
    x = np.arange(n)
    gap = 1.2  # Increased gap to prevent overlap

    fig, ax = plt.subplots(figsize=(12, 6))

    # Two groups: disjunctive (left) and conjunctive (right)
    # Position bars so they don't overlap: disjunctive centered at x - gap/2, conjunctive at x + gap/2
    x_disj = x - gap / 2
    x_conj = x + gap / 2

    colors = [model_colors.get(m, "#999999") for m in plot_df["model"]]

    bars_conj = ax.bar(
        x_conj, plot_df["conj_mean"], width,
        yerr=plot_df["conj_se"],
        color=colors, edgecolor="#333", linewidth=0.8,
        capsize=3, error_kw={"elinewidth": 1, "capthick": 1},
        label="Conjunctive"
    )
    bars_disj = ax.bar(
        x_disj, plot_df["disj_mean"], width,
        yerr=plot_df["disj_se"],
        color=colors, edgecolor="#333", linewidth=0.8,
        capsize=3, error_kw={"elinewidth": 1, "capthick": 1},
        label="Disjunctive", alpha=0.7
    )

    # Add mean values above bars
    for i in range(n):
        row = plot_df.iloc[i]
        ax.text(x_conj[i], row["conj_mean"] + row["conj_se"] + 0.3,
                f"{row['conj_mean']:.1f}", ha="center", va="bottom", fontsize=8, color="#333")
        ax.text(x_disj[i], row["disj_mean"] + row["disj_se"] + 0.3,
                f"{row['disj_mean']:.1f}", ha="center", va="bottom", fontsize=8, color="#333")

    # X-axis: model names
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], fontsize=10)
    ax.set_ylabel("Average number of tests per trial", fontsize=11)
    ax.set_ylim(0, max(plot_df["conj_mean"].max(), plot_df["disj_mean"].max()) + max(plot_df["conj_se"].max(), plot_df["disj_se"].max()) + 1.5)

    # Legend: rule types
    ax.legend(loc="upper right", fontsize=10)

    # Add group labels
    ax.text(np.mean(x_disj), ax.get_ylim()[1] - 0.5, "Disjunctive", ha="center", fontsize=11, fontweight="bold")
    ax.text(np.mean(x_conj), ax.get_ylim()[1] - 0.5, "Conjunctive", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {output_path}")
    plt.close()
    return 0


if __name__ == "__main__":
    exit(main())
