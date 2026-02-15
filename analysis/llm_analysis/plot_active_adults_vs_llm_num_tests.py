"""
Grouped bar plot: Number of tests per trial — Active adults vs LLMs.

- Human: Adults (Active) average number of tests from comprehensive_correlation_data
  (num_unique_tests or num_tests_before_qa), by rule.
- LLM: Average number of tests per trial (from action logs), with one bar per model
  (aggregated across all prompt types for that model).

Layout: two groups — Disjunctive (Human + LLMs in one group, ranked by disjunctive score) and
Conjunctive (Human + LLMs in one group, ranked by conjunctive score). Rank shown by bar order (best first).
Lower num_tests = more efficient exploration.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"]

MODEL_ORDER = ["deepseek-reasoner", "gpt-4o", "deepseek-chat", "gpt-4o-mini", "gemma3:27b", "qwq"]


def load_human_num_tests(csv_path: Path):
    """Human (active adults) num_tests: use num_unique_tests if available, else num_tests_before_qa."""
    df = pd.read_csv(csv_path)
    
    # Prefer num_unique_tests, fallback to num_tests_before_qa
    if "num_unique_tests" in df.columns:
        num_tests_col = "num_unique_tests"
    elif "num_tests_before_qa" in df.columns:
        num_tests_col = "num_tests_before_qa"
    else:
        print("WARNING: No num_tests column found in CSV")
        return None
    
    df = df[df[num_tests_col].notna()]
    if df.empty:
        return None
    
    acc = df.groupby("true_rule")[num_tests_col].agg(["mean", "std", "count"]).reset_index()
    acc["se"] = acc["std"] / np.sqrt(acc["count"])
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-csv", default="llm_num_tests_by_rule_4obj.csv", help="Path to LLM num_tests CSV")
    parser.add_argument("--human-csv", default=None, help="Path to comprehensive_correlation_data.csv")
    parser.add_argument("--output", default="active_adults_vs_llm_num_tests.png", help="Output PNG")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    llm_csv = script_dir / args.llm_csv
    human_csv = (
        Path(args.human_csv)
        if args.human_csv
        else project_root / "analysis" / "results" / "round7" / "comprehensive_correlation_data_no_prior_102.csv"
    )
    output_path = script_dir / args.output

    if not llm_csv.exists():
        print(f"ERROR: LLM CSV not found at {llm_csv}")
        return 1

    # --- Human: Adults (Active) num_tests by rule ---
    human_data = load_human_num_tests(human_csv)
    human_plot_rows = []
    if human_data is not None:
        conj_row = human_data[human_data["true_rule"] == "conjunctive"]
        disj_row = human_data[human_data["true_rule"] == "disjunctive"]
        if len(conj_row) > 0 and len(disj_row) > 0:
            human_plot_rows.append({
                "label": "Adults (Active)",
                "conj": float(conj_row["mean"].iloc[0]),
                "conj_se": float(conj_row["se"].iloc[0]),
                "disj": float(disj_row["mean"].iloc[0]),
                "disj_se": float(disj_row["se"].iloc[0]),
                "source": "human",
            })

    # --- LLM: one bar per model (CSV already aggregated by model and rule) ---
    print("Loading LLM data...")
    llm_df = pd.read_csv(llm_csv)
    llm_plot_rows = []
    if not llm_df.empty:
        for model in MODEL_ORDER:
            conj_row = llm_df[(llm_df["model"] == model) & (llm_df["rule"] == "conjunctive")]
            disj_row = llm_df[(llm_df["model"] == model) & (llm_df["rule"] == "disjunctive")]
            if len(conj_row) > 0 and len(disj_row) > 0:
                llm_plot_rows.append({
                    "label": model,
                    "conj": float(conj_row["mean"].iloc[0]),
                    "conj_se": float(conj_row["se"].iloc[0]),
                    "disj": float(disj_row["mean"].iloc[0]),
                    "disj_se": float(disj_row["se"].iloc[0]),
                    "source": "llm",
                    "model": model,
                })

    # Combine human + LLM into one dataframe
    all_rows = human_plot_rows + llm_plot_rows
    if not all_rows:
        print("No data to plot.")
        return 0

    plot_df = pd.DataFrame(all_rows)
    human_color = "#F06B85"
    human_edge = "#888"
    model_colors = {
        "deepseek-reasoner": "#4477AA",
        "gpt-4o": "#CCBB44",
        "deepseek-chat": "#228833",
        "gpt-4o-mini": "#BB88CC",
        "gemma3:27b": "#88CCEE",
        "qwq": "#44AA99",
    }

    def bar_color(row):
        if row["source"] == "llm":
            return model_colors.get(row["model"], "#999999")
        return human_color

    # Rank by num_tests (ascending = better, so lower is better)
    disj_df = plot_df.sort_values("disj", ascending=True).reset_index(drop=True)
    conj_df = plot_df.sort_values("conj", ascending=True).reset_index(drop=True)
    n = len(plot_df)
    width = 0.75
    gap = 1.5
    x_disj = np.arange(n)
    x_conj = np.arange(n) + n + gap

    fig, ax = plt.subplots(figsize=(14, 7))

    colors_disj = [bar_color(disj_df.iloc[i]) for i in range(n)]
    colors_conj = [bar_color(conj_df.iloc[i]) for i in range(n)]

    bars_disj = ax.bar(
        x_disj, disj_df["disj"], width, yerr=disj_df["disj_se"],
        color=colors_disj, edgecolor=[human_edge if disj_df.iloc[i]["source"] == "human" else "#333" for i in range(n)],
        linewidth=0.8, capsize=2, error_kw={"elinewidth": 1, "capthick": 1}
    )
    bars_conj = ax.bar(
        x_conj, conj_df["conj"], width, yerr=conj_df["conj_se"],
        color=colors_conj, edgecolor=[human_edge if conj_df.iloc[i]["source"] == "human" else "#333" for i in range(n)],
        linewidth=0.8, capsize=2, error_kw={"elinewidth": 1, "capthick": 1}
    )

    # Mean above each bar
    for i in range(n):
        r_disj = disj_df.iloc[i]
        y_disj = r_disj["disj"] + r_disj["disj_se"] + 0.3
        ax.text(x_disj[i], y_disj, f"{r_disj['disj']:.1f}", ha="center", va="bottom", fontsize=8, color="#333")
    for i in range(n):
        r_conj = conj_df.iloc[i]
        y_conj = r_conj["conj"] + r_conj["conj_se"] + 0.3
        ax.text(x_conj[i], y_conj, f"{r_conj['conj']:.1f}", ha="center", va="bottom", fontsize=8, color="#333")

    # X-axis: two groups with bar labels under each bar
    ax.set_xticks(np.concatenate([x_disj, x_conj]))
    disj_labels = [disj_df.iloc[i]["label"] for i in range(n)]
    conj_labels = [conj_df.iloc[i]["label"] for i in range(n)]
    ax.set_xticklabels(disj_labels + conj_labels, rotation=30, ha="right", fontsize=8)
    # Add group labels as text above the two clusters
    max_y = max(plot_df["conj"].max() + plot_df["conj_se"].max(), plot_df["disj"].max() + plot_df["disj_se"].max())
    ax.text(np.mean(x_disj), max_y * 1.08, "Disjunctive", ha="center", fontsize=11, fontweight="bold")
    ax.text(np.mean(x_conj), max_y * 1.08, "Conjunctive", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Average number of tests per trial", fontsize=11)
    ax.set_ylim(0, max_y * 1.15)

    # Legend: Adults (Active) + LLM models that have data (so legend matches bars)
    leg_handles = [mpatches.Patch(facecolor=human_color, edgecolor=human_edge, label="Adults (Active)")]
    models_in_plot = plot_df[plot_df["source"] == "llm"]["model"].unique().tolist()
    for m in MODEL_ORDER:
        if m in model_colors and m in models_in_plot:
            leg_handles.append(mpatches.Patch(facecolor=model_colors[m], edgecolor="#333", label=m))
    ax.legend(handles=leg_handles, loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")
    plt.close()
    return 0


if __name__ == "__main__":
    exit(main())
