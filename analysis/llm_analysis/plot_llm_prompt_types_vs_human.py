"""
Scatter plot: LLM all-correct accuracy (4 objects) by prompt type vs human object identification.
- X = conjunctive accuracy, Y = disjunctive accuracy
- Each point = prompt type or Human
"""

import argparse
import ast
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"]

# Map agent class to display label for prompt type
AGENT_TO_PROMPT = {
    "PromptsAgent": "Basic prompts",
    "CoTPromptAgent": "Chain-of-thought",
    "ReflexionPromptsAgent": "Reflexion",
    "NaiveLLM": "Naive",
}
PROMPT_ORDER = ["Basic prompts", "Chain-of-thought", "Reflexion", "Naive"]


def get_prompt_type(agent_target: str) -> str:
    """Extract prompt type label from agent._target_."""
    if not agent_target:
        return None
    for key, label in AGENT_TO_PROMPT.items():
        if key in agent_target:
            return label
    return None


def load_llm_by_prompt_type(llm_data_dir: Path):
    """Load LLM results with 4 objects, grouped by prompt type and rule."""
    rows = []
    for results_path in llm_data_dir.rglob("results.jsonl"):
        config_path = results_path.parent / ".hydra" / "config.yaml"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            config = yaml.safe_load(f)
        if not config:
            continue
        env = config.get("env_kwargs") or {}
        if env.get("num_objects") != 4:
            continue
        rule = env.get("rule", "unknown")
        if rule not in ("conjunctive", "disjunctive"):
            continue
        agent = config.get("agent") or {}
        agent_target = agent.get("_target_", "")
        prompt_type = get_prompt_type(agent_target)
        if prompt_type is None:
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
                rows.append({"prompt_type": prompt_type, "rule": rule, "all_correct": all_correct})
    return pd.DataFrame(rows)


def load_human_object_accuracy(csv_path: Path):
    """Load human object identification accuracy by rule."""
    df = pd.read_csv(csv_path)
    def parse_objects(s):
        try:
            return set(ast.literal_eval(str(s)))
        except (ValueError, SyntaxError):
            return None
    df["true_set"] = df["true_blicket_objects"].apply(parse_objects)
    df["chosen_set"] = df["chosen_objects"].apply(parse_objects)
    df["obj_correct"] = df.apply(lambda r: 1.0 if r["true_set"] == r["chosen_set"] else 0.0, axis=1)
    acc = df.groupby("true_rule")["obj_correct"].agg(["mean", "std", "count"]).reset_index()
    acc["se"] = acc["std"] / np.sqrt(acc["count"])
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-data", default=None, help="Path to llm_data directory")
    parser.add_argument("--human-csv", default=None, help="Path to comprehensive_correlation_data.csv")
    parser.add_argument("--output", default="llm_prompt_types_vs_human.png", help="Output PNG")
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

    if not llm_data_dir.exists():
        print(f"ERROR: llm_data not found at {llm_data_dir}")
        return 1

    print("Loading LLM data by prompt type...")
    llm_df = load_llm_by_prompt_type(llm_data_dir)
    if llm_df.empty:
        print("No 4-object LLM data found.")
        return 0

    llm_agg = llm_df.groupby(["prompt_type", "rule"])["all_correct"].agg(["mean", "std", "count"]).reset_index()
    llm_agg["se"] = llm_agg["std"] / np.sqrt(llm_agg["count"])

    print("Loading human data...")
    human_acc = load_human_object_accuracy(human_csv)

    # Pivot to (label, conj_mean, conj_se, disj_mean, disj_se, source)
    plot_rows = []
    for pt in PROMPT_ORDER:
        conj = llm_agg[(llm_agg["prompt_type"] == pt) & (llm_agg["rule"] == "conjunctive")]
        disj = llm_agg[(llm_agg["prompt_type"] == pt) & (llm_agg["rule"] == "disjunctive")]
        if len(conj) > 0 and len(disj) > 0:
            plot_rows.append({
                "label": pt,
                "conj": conj.iloc[0]["mean"],
                "conj_se": conj.iloc[0]["se"],
                "disj": disj.iloc[0]["mean"],
                "disj_se": disj.iloc[0]["se"],
                "source": "llm",
            })
    conj_h = human_acc[human_acc["true_rule"] == "conjunctive"]
    disj_h = human_acc[human_acc["true_rule"] == "disjunctive"]
    if len(conj_h) > 0 and len(disj_h) > 0:
        plot_rows.append({
            "label": "Human",
            "conj": conj_h["mean"].values[0],
            "conj_se": conj_h["se"].values[0],
            "disj": disj_h["mean"].values[0],
            "disj_se": disj_h["se"].values[0],
            "source": "human",
        })

    if not plot_rows:
        print("No data to plot.")
        return 0

    plot_df = pd.DataFrame(plot_rows)
    colors_llm = "#3B82F6"
    color_human = "#EA580C"

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    for _, row in plot_df.iterrows():
        c = color_human if row["source"] == "human" else colors_llm
        ax.scatter(row["conj"], row["disj"], s=120, c=c, alpha=0.9, edgecolors="#333", linewidth=1, zorder=3)
        ax.errorbar(row["conj"], row["disj"], xerr=row["conj_se"], yerr=row["disj_se"],
                   fmt="none", ecolor="#666", capsize=3, alpha=0.7, zorder=2)
        ax.annotate(row["label"], (row["conj"], row["disj"]), xytext=(6, 6), textcoords="offset points",
                    fontsize=9, color="#333")

    ax.set_xlabel("Conjunctive accuracy", fontsize=10)
    ax.set_ylabel("Disjunctive accuracy", fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axhline(1.0, color="#999", linestyle="--", alpha=0.4)
    ax.axvline(1.0, color="#999", linestyle="--", alpha=0.4)
    ax.legend(handles=[
        mpatches.Patch(facecolor=colors_llm, edgecolor="none", label="LLM (all correct)"),
        mpatches.Patch(facecolor=color_human, edgecolor="none", label="Human (object identification)"),
    ], loc="lower right", fontsize=9)
    ax.set_title("LLM accuracy by prompt type vs Human (4 objects)", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved: {output_path}")
    plt.close()
    return 0


if __name__ == "__main__":
    exit(main())
