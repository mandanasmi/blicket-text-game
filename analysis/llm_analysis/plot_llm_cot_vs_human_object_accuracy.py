"""
Bar plot: LLM all-correct accuracy (4 objects) by model vs human object identification accuracy.

- LLM: all_correct (num_correct == num_questions) for agents with 4 objects, grouped by model
- Human: object identification accuracy from comprehensive_correlation_data
"""

# Map raw model names from config to display labels (only include target models)
MODEL_DISPLAY = {
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
    "gpt-4o": "gpt-4o",
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "ollama/gemma3:27b": "gemma3:27b",
}
MODEL_ORDER = ["deepseek-chat", "deepseek-reasoner", "gpt-4o", "gpt-4o-mini", "gemma3:27b"]

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


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


def load_llm_data(llm_data_dir: Path):
    """Load LLM results with 4 objects from PATH_WCS only, grouped by COT variant."""
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
    """Load human object identification accuracy from comprehensive_correlation_data."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-data", default=None, help="Path to llm_data directory")
    parser.add_argument("--human-csv", default=None, help="Path to comprehensive_correlation_data.csv")
    parser.add_argument("--output", default="llm_cot_vs_human_object_accuracy.png", help="Output PNG")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    llm_data_dir = Path(args.llm_data) if args.llm_data else Path("/tmp/llm_data")
    human_csv = (
        Path(args.human_csv)
        if args.human_csv
        else project_root / "analysis" / "results" / "round7" / "comprehensive_correlation_data.csv"
    )
    output_path = script_dir / args.output

    print("Loading LLM data...")
    llm_df = load_llm_data(llm_data_dir)
    llm_rows = []
    if not llm_df.empty:
        llm_agg = llm_df.groupby("model")["all_correct"].agg(["mean", "std", "count"]).reset_index()
        llm_agg["se"] = llm_agg["std"] / np.sqrt(llm_agg["count"])
        for m in MODEL_ORDER:
            sub = llm_agg[llm_agg["model"] == m]
            if len(sub) > 0:
                r = sub.iloc[0]
                llm_rows.append({
                    "label": m,
                    "mean": r["mean"],
                    "se": r["se"],
                    "source": "llm",
                })

    print("Loading human data...")
    human_acc = load_human_object_accuracy(human_csv)
    human_rows = []
    for rt in ["conjunctive", "disjunctive"]:
        sub = human_acc[human_acc["true_rule"] == rt]
        if len(sub) > 0:
            human_rows.append({
                "label": f"Human {rt.capitalize()}",
                "mean": sub["mean"].values[0],
                "se": sub["se"].values[0],
                "source": "human",
            })

    plot_rows = llm_rows + human_rows
    if not plot_rows:
        print("No data to plot.")
        return

    plot_df = pd.DataFrame(plot_rows)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    x = np.arange(len(plot_df))
    colors = ["#2a9d8f" if s == "llm" else "#e76f51" for s in plot_df["source"]]
    ax.bar(x, plot_df["mean"], yerr=plot_df["se"], capsize=4, color=colors, edgecolor="#333", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("All-Correct: LLM (4 objects, by model) vs Human Object Identification")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2a9d8f", label="LLM (all correct)"),
        Patch(facecolor="#e76f51", label="Human (object identification)"),
    ], loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()