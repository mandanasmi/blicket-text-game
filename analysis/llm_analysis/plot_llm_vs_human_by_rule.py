"""
Two bar plots: LLM all-correct accuracy (4 objects) vs human object identification,
one for conjunctive and one for disjunctive rule.

- LLM: all_correct (num_correct == num_questions), 4 objects only, averaged over all CoT variants
- Human: object identification accuracy from comprehensive_correlation_data
"""

MODEL_DISPLAY = {
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
    "gpt-4o": "gpt-4o",
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "ollama/gemma3:27b": "gemma3:27b",
    "ollama/qwq": "qwq",
}
MODEL_ORDER = ["deepseek-reasoner", "gpt-4o", "deepseek-chat", "gpt-4o-mini", "gemma3:27b", "qwq"]

import argparse
import ast
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

sns.set_style("white")
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"]

PATH_WCS = [
    "2025.03.26/144924/*/results.jsonl",
    "2025.03.17/230803/*/results.jsonl",
    "2025.03.14/010645/*/results.jsonl",
    "2025.03.14/010942/*/results.jsonl",
    "2025.03.14/195735/*/results.jsonl",
    "2025.03.14/200242/*/results.jsonl",
    "2025.03.14/201822/*/results.jsonl",
    "2025.03.14/202836/*/results.jsonl",
    "2025.03.14/203513/*/results.jsonl",
    "2025.03.14/203725/*/results.jsonl",
    "2025.03.14/203810/*/results.jsonl",
    "2025.03.16/170513/*/results.jsonl",
    "2025.03.16/170649/*/results.jsonl",
    "2025.03.16/170800/*/results.jsonl",
    "2025.03.16/171457/*/results.jsonl",
    "2025.03.16/171607/*/results.jsonl",
    "2025.03.16/171730/*/results.jsonl",
    "2025.03.24/213007/*/results.jsonl",
    "2025.03.24/214357/*/results.jsonl",
    "2025.03.25/002055/*/results.jsonl",
    "2025.03.25/002205/*/results.jsonl",
    "2025.03.25/002257/*/results.jsonl",
    "2025.03.23/165130/*/results.jsonl",
    "2025.03.24/165351/*/results.jsonl",
    "2025.03.24/165605/*/results.jsonl",
    "2025.03.25/002758/*/results.jsonl",
    "2025.03.25/165819/*/results.jsonl",
    "2025.03.25/165942/*/results.jsonl",
    "2025.03.25/171404/*/results.jsonl",
    "2025.03.23/165243/*/results.jsonl",
    "2025.03.24/165513/*/results.jsonl",
    "2025.03.25/003419/*/results.jsonl",
    "2025.03.25/164038/*/results.jsonl",
    "2025.03.25/164214/*/results.jsonl",
    "2025.03.25/001402/*/results.jsonl",
    "2025.03.23/204923/*/results.jsonl",
    "2025.03.25/170432/*/results.jsonl",
]


def load_llm_data(llm_data_dir: Path, rule_type: str):
    """Load LLM results with 4 objects and given rule, averaged over all CoT variants."""
    rows = []
    for wc in PATH_WCS:
        for results_path in llm_data_dir.glob(wc):
            config_path = results_path.parent / ".hydra" / "config.yaml"
            if not config_path.exists():
                continue

            with open(config_path) as f:
                config = yaml.safe_load(f)
            if not config:
                continue

            env = config.get("env_kwargs", {})
            if env.get("num_objects") != 4:
                continue
            if env.get("rule") != rule_type:
                continue

            raw_model = (config.get("agent") or {}).get("model", "")
            model = MODEL_DISPLAY.get(raw_model)
            if model is None:
                continue

            with open(results_path) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    num_correct = row.get("num_correct", 0)
                    num_questions = row.get("num_questions", 1)
                    all_correct = 1.0 if num_correct == num_questions else 0.0
                    rows.append({"model": model, "all_correct": all_correct})

    return pd.DataFrame(rows)


def load_human_object_accuracy(csv_path: Path):
    """Load human object identification accuracy by rule from comprehensive_correlation_data."""
    df = pd.read_csv(csv_path)

    def parse_objects(s):
        try:
            return set(ast.literal_eval(str(s)))
        except (ValueError, SyntaxError):
            return None

    df["true_set"] = df["true_blicket_objects"].apply(parse_objects)
    df["chosen_set"] = df["chosen_objects"].apply(parse_objects)
    df["obj_correct"] = df.apply(
        lambda r: 1.0 if r["true_set"] == r["chosen_set"] else 0.0, axis=1
    )

    acc = df.groupby("true_rule")["obj_correct"].agg(["mean", "std", "count"]).reset_index()
    acc["se"] = acc["std"] / np.sqrt(acc["count"])
    return acc


def make_plot(plot_rows, title, output_path):
    """Create and save a single bar plot (aligned with codebase style)."""
    if not plot_rows:
        print(f"No data for {output_path}, skipping.")
        return

    plot_df = pd.DataFrame(plot_rows)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(plot_df))
    colors = ["#2a9d8f" if s == "llm" else "#e76f51" for s in plot_df["source"]]
    ax.bar(x, plot_df["mean"], yerr=plot_df["se"], capsize=4,
           color=colors, alpha=0.9, edgecolor="none",
           error_kw={"color": "#333333", "linewidth": 1, "capthick": 1})

    for i, (m, se) in enumerate(zip(plot_df["mean"], plot_df["se"])):
        y_pos = m + se + 0.035
        ax.text(i, y_pos, f"{m:.2f} (±{se:.2f})", ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=25, ha="right", fontsize=10, color="#333333")
    ax.set_ylabel("Accuracy", fontsize=10, color="#333333")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(title, fontsize=11, color="#333333")
    ax.axhline(1.0, color="#6a3d9a", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.grid(False)

    legend_handles = [
        mpatches.Patch(facecolor="#2a9d8f", edgecolor="none", label="LLM (all correct)"),
        mpatches.Patch(facecolor="#e76f51", edgecolor="none", label="Human (object identification)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right", frameon=True, fancybox=False, shadow=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved: {output_path}")
    plt.close()


def make_merged_plot(data_conj, data_disj, output_path, labels=None):
    """Create merged plot: all conjunctive bars grouped, then all disjunctive bars grouped."""
    if labels is None:
        labels = MODEL_ORDER + ["human"]
    n = len(labels)
    width = 0.6
    gap = 2
    x_disj = np.arange(n)
    x_conj = np.arange(n) + n + gap

    means_conj = []
    se_conj = []
    means_disj = []
    se_disj = []

    for lbl in labels:
        c = next((r for r in data_conj if r["label"] == lbl), None)
        d = next((r for r in data_disj if r["label"] == lbl), None)
        means_conj.append(c["mean"] if c else 0)
        se_conj.append(c["se"] if c else 0)
        means_disj.append(d["mean"] if d else 0)
        se_disj.append(d["se"] if d else 0)

    colors_llm = "#3B82F6"   # blue
    color_human = "#EA580C"  # orange
    bar_colors_conj = [colors_llm if lbl != "human" else color_human for lbl in labels]
    bar_colors_disj = [colors_llm if lbl != "human" else color_human for lbl in labels]

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    for i in range(n):
        ax.bar(x_disj[i], means_disj[i], width, yerr=se_disj[i] if se_disj[i] > 0 else 0, capsize=4,
               color=bar_colors_disj[i], alpha=0.9, edgecolor="none",
               error_kw={"color": "#333333", "linewidth": 1, "capthick": 1})
        ax.bar(x_conj[i], means_conj[i], width, yerr=se_conj[i] if se_conj[i] > 0 else 0, capsize=4,
               color=bar_colors_conj[i], alpha=0.9, edgecolor="none",
               error_kw={"color": "#333333", "linewidth": 1, "capthick": 1})

    for i, (m_c, se_c, m_d, se_d) in enumerate(zip(means_conj, se_conj, means_disj, se_disj)):
        if m_d > 0 or se_d > 0:
            y_pos = m_d + se_d + 0.035
            ax.text(x_disj[i], y_pos, f"{m_d:.2f} (±{se_d:.2f})", ha="center", va="bottom", fontsize=7, color="#333333")
        if m_c > 0 or se_c > 0:
            y_pos = m_c + se_c + 0.035
            ax.text(x_conj[i], y_pos, f"{m_c:.2f} (±{se_c:.2f})", ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_xticks(np.concatenate([x_disj, x_conj]))
    ax.set_xticklabels(labels + labels, rotation=25, ha="right", fontsize=9, color="#333333")
    ax.set_ylabel("Accuracy", fontsize=10, color="#333333")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("All-Correct: LLM vs Human by Rule Type (4 objects)", fontsize=11, color="#333333")
    ax.axhline(1.0, color="#6a3d9a", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.grid(False)
    legend_handles = [
        mpatches.Patch(facecolor=colors_llm, edgecolor="none", label="LLM"),
        mpatches.Patch(facecolor=color_human, edgecolor="none", label="Human"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right", frameon=True, fancybox=False, shadow=False)

    # Section labels above bars (Disjunctive first, then Conjunctive)
    ax.text((n - 1) / 2, 1.15, "Disjunctive", ha="center", va="bottom", fontsize=10, color="#333333")
    ax.text(n + gap + (n - 1) / 2, 1.15, "Conjunctive", ha="center", va="bottom", fontsize=10, color="#333333")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-data", default=None, help="Path to llm_data directory")
    parser.add_argument("--human-csv", default=None, help="Path to comprehensive_correlation_data.csv")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    llm_data_dir = Path(args.llm_data) if args.llm_data else Path("/tmp/llm_data")
    human_csv = (
        Path(args.human_csv)
        if args.human_csv
        else project_root / "analysis" / "results" / "round7" / "comprehensive_correlation_data.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else script_dir

    human_acc = load_human_object_accuracy(human_csv)

    data_conj = []
    data_disj = []

    for rule_type in ["conjunctive", "disjunctive"]:
        print(f"Loading LLM data for {rule_type}...")
        llm_df = load_llm_data(llm_data_dir, rule_type)
        llm_rows = []
        if not llm_df.empty:
            llm_agg = llm_df.groupby("model")["all_correct"].agg(["mean", "std", "count"]).reset_index()
            llm_agg["se"] = llm_agg["std"] / np.sqrt(llm_agg["count"])
            for m in MODEL_ORDER:
                sub = llm_agg[llm_agg["model"] == m]
                if len(sub) > 0:
                    r = sub.iloc[0]
                    llm_rows.append({"label": m, "mean": r["mean"], "se": r["se"], "source": "llm"})

        sub = human_acc[human_acc["true_rule"] == rule_type]
        human_row = None
        if len(sub) > 0:
            r = sub.iloc[0]
            human_row = {"label": "human", "mean": r["mean"], "se": r["se"], "source": "human"}

        plot_rows = llm_rows + ([human_row] if human_row else [])
        if rule_type == "conjunctive":
            data_conj = plot_rows
        else:
            data_disj = plot_rows

        title = f"All-Correct: LLM vs Human ({rule_type.capitalize()}, 4 objects)"
        out_path = output_dir / f"llm_vs_human_{rule_type}.png"
        make_plot(plot_rows, title, out_path)

    # Merged plot: conjunctive vs disjunctive in one figure
    merged_labels = MODEL_ORDER + ["human"]
    make_merged_plot(data_conj, data_disj, output_dir / "llm_vs_human_merged.png", labels=merged_labels)


if __name__ == "__main__":
    main()
