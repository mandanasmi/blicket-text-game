"""
Plot LLM object identification accuracy by rule type (conjunctive vs disjunctive).

Reads llm_object_accuracy_by_prompt.csv. Optionally adds Adults (Active) human object accuracy
from comprehensive_correlation_data.csv. Two groups of bars: Conjunctive (ranked by conjunctive
accuracy) and Disjunctive (ranked by disjunctive accuracy).
"""

import argparse
import ast
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"]

MODEL_ORDER = ["deepseek-reasoner", "gpt-4o", "deepseek-chat", "gpt-4o-mini", "gemma3:27b", "qwq"]
PROMPT_ORDER = ["default", "cot", "react", "reflexion"]
model_colors = {
    "deepseek-reasoner": "#4477AA",
    "gpt-4o": "#CCBB44",
    "deepseek-chat": "#228833",
    "gpt-4o-mini": "#BB88CC",
    "gemma3:27b": "#88CCEE",
    "qwq": "#44AA99",
}
prompt_hatch = {"default": "", "cot": "///", "react": "ooo", "reflexion": "..."}
human_color = "#F06B85"
human_edge = "#888"


def load_human_object_accuracy(csv_path: Path):
    """Human (active adults) object identification: correct iff chosen_objects == true_blicket_objects."""
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
    parser.add_argument("--input", default="llm_object_accuracy_by_prompt.csv", help="Input CSV")
    parser.add_argument("--human-csv", default=None, help="Path to comprehensive_correlation_data.csv (optional)")
    parser.add_argument("--output", default="llm_accuracy_by_rule.png", help="Output PNG")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    csv_path = script_dir / args.input
    human_csv = (
        Path(args.human_csv)
        if args.human_csv
        else project_root / "analysis" / "results" / "round7" / "comprehensive_correlation_data.csv"
    )
    output_path = script_dir / args.output

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run llm_object_accuracy_by_prompt_table.py --by-rule first.")
        return 1

    df = pd.read_csv(csv_path)
    if "conjunctive" not in df.columns or "disjunctive" not in df.columns:
        print("ERROR: CSV must have conjunctive and disjunctive columns (use --by-rule).")
        return 1

    df["label"] = df["model"] + " (" + df["prompt_type"] + ")"
    df["source"] = "llm"

    # Add Adults (Active) human object accuracy if CSV exists
    if human_csv.exists():
        human_acc = load_human_object_accuracy(human_csv)
        conj_row = human_acc[human_acc["true_rule"] == "conjunctive"]
        disj_row = human_acc[human_acc["true_rule"] == "disjunctive"]
        if len(conj_row) and len(disj_row):
            human_row = pd.DataFrame([{
                "model": "human",
                "prompt_type": "",
                "conjunctive": float(conj_row["mean"].iloc[0]),
                "disjunctive": float(disj_row["mean"].iloc[0]),
                "label": "Adults (Active)",
                "source": "human",
            }])
            df = pd.concat([human_row, df], ignore_index=True)

    df["label"] = df.apply(lambda r: r["label"] if r["source"] == "llm" else "Adults (Active)", axis=1)

    conj_df = df.sort_values("conjunctive", ascending=False).reset_index(drop=True)
    disj_df = df.sort_values("disjunctive", ascending=False).reset_index(drop=True)
    n = len(df)
    width = 0.75
    gap = 1.5
    x_conj = np.arange(n)
    x_disj = np.arange(n) + n + gap

    def bar_color(row):
        if row.get("source") == "human":
            return human_color
        return model_colors.get(row["model"], "#999999")

    def bar_hatch(row):
        if row.get("source") == "human":
            return ""
        return prompt_hatch.get(row["prompt_type"], "")

    def bar_edge(row):
        return human_edge if row.get("source") == "human" else "#333"

    fig, ax = plt.subplots(figsize=(14, 7))

    colors_conj = [bar_color(conj_df.iloc[i]) for i in range(n)]
    colors_disj = [bar_color(disj_df.iloc[i]) for i in range(n)]
    edges_conj = [bar_edge(conj_df.iloc[i]) for i in range(n)]
    edges_disj = [bar_edge(disj_df.iloc[i]) for i in range(n)]

    bars_conj = ax.bar(
        x_conj, conj_df["conjunctive"], width,
        color=colors_conj, edgecolor=edges_conj, linewidth=0.8,
    )
    bars_disj = ax.bar(
        x_disj, disj_df["disjunctive"], width,
        color=colors_disj, edgecolor=edges_disj, linewidth=0.8,
    )

    for i in range(n):
        bars_conj[i].set_hatch(bar_hatch(conj_df.iloc[i]))
        bars_disj[i].set_hatch(bar_hatch(disj_df.iloc[i]))

    for i in range(n):
        r = conj_df.iloc[i]
        ax.text(x_conj[i], r["conjunctive"] + 0.02, f"{r['conjunctive']:.2f}",
                ha="center", va="bottom", fontsize=7, color="#333")
    for i in range(n):
        r = disj_df.iloc[i]
        ax.text(x_disj[i], r["disjunctive"] + 0.02, f"{r['disjunctive']:.2f}",
                ha="center", va="bottom", fontsize=7, color="#333")

    ax.set_xticks(np.concatenate([x_conj, x_disj]))
    conj_labels = [conj_df.iloc[i]["label"] for i in range(n)]
    disj_labels = [disj_df.iloc[i]["label"] for i in range(n)]
    ax.set_xticklabels(conj_labels + disj_labels, rotation=35, ha="right", fontsize=7)
    ax.text(np.mean(x_conj), 1.06, "Conjunctive", ha="center", fontsize=11, fontweight="bold")
    ax.text(np.mean(x_disj), 1.06, "Disjunctive", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Object identification accuracy", fontsize=11)
    ax.set_ylim(0, 1.15)
    CHANCE_LEVEL = 1.0 / 6.0
    ax.axhline(CHANCE_LEVEL, color="#999", linestyle="--", alpha=0.7, linewidth=1)

    leg_handles = [mpatches.Patch(facecolor=human_color, edgecolor=human_edge, label="Adults (Active)")]
    for m in MODEL_ORDER:
        if m in model_colors:
            leg_handles.append(mpatches.Patch(facecolor=model_colors[m], edgecolor="#333", label=m))
    for p in PROMPT_ORDER:
        leg_handles.append(mpatches.Patch(facecolor="#e0e0e0", edgecolor="#333", hatch=prompt_hatch[p], label=p))
    ax.legend(handles=leg_handles, loc="upper right", fontsize=8, ncol=2)

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
