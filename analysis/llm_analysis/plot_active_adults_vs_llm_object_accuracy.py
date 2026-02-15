"""
Grouped bar plot: Object identification accuracy — Active adults vs LLMs (best prompt per model).

- Human: Adults (Active) object identification accuracy from comprehensive_correlation_data
  (proportion of trials where chosen_objects == true_blicket_objects), by rule.
- LLM: Proportion of trials in which all objects are correctly identified (num_correct == num_questions),
  with one bar per model using the prompting strategy that gave the best mean for that rule.

Layout: two groups — Disjunctive (Human + LLMs in one group, ranked by disjunctive score) and
Conjunctive (Human + LLMs in one group, ranked by conjunctive score). Rank shown by bar order (best first).
"""

import argparse
import ast
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"]

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


def get_prompt_type(agent: dict) -> str | None:
    target = agent.get("_target_", "") or ""
    if "CoTPromptAgent" in target:
        return "cot"
    if "ReflexionPromptsAgent" in target:
        return "reflexion"
    if "OReasonPromptsAgent" in target:
        return "oreason"
    if "PromptsAgent" in target:
        react = agent.get("react", False)
        return "react" if react else "default"
    return None


def load_llm_by_model_prompt(llm_data_dir: Path):
    """LLM trials: all_correct = 1 if num_correct == num_questions (all objects correct)."""
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
        rule = env.get("rule", "")
        if rule not in ("conjunctive", "disjunctive"):
            continue
        agent = config.get("agent") or {}
        raw_model = agent.get("model", "")
        model = MODEL_DISPLAY.get(raw_model)
        if model is None:
            continue
        prompt_type = get_prompt_type(agent)
        if prompt_type is None:
            continue
        with open(results_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                nc = row.get("num_correct", 0)
                nq = row.get("num_questions", 1)
                all_correct = 1.0 if nc == nq else 0.0
                rows.append({
                    "model": model,
                    "prompt_type": prompt_type,
                    "rule": rule,
                    "all_correct": all_correct,
                })
    return pd.DataFrame(rows)


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
    parser.add_argument("--llm-data", default=None, help="Path to llm_data directory")
    parser.add_argument("--human-csv", default=None, help="Path to comprehensive_correlation_data.csv")
    parser.add_argument("--output", default="active_adults_vs_llm_object_accuracy.png", help="Output PNG")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    llm_data_dir = Path(args.llm_data) if args.llm_data else (script_dir / "llm_data")
    human_csv = (
        Path(args.human_csv)
        if args.human_csv
        else project_root / "analysis" / "results" / "round7" / "comprehensive_correlation_data.csv"
    )
    output_path = script_dir / args.output

    if not llm_data_dir.exists():
        print(f"ERROR: llm_data not found at {llm_data_dir}")
        return 1

    # --- Human: Adults (Active) object identification by rule ---
    human_acc = load_human_object_accuracy(human_csv)
    conj_row = human_acc[human_acc["true_rule"] == "conjunctive"]
    disj_row = human_acc[human_acc["true_rule"] == "disjunctive"]
    human_conj_mean = float(conj_row["mean"].iloc[0]) if len(conj_row) else None
    human_conj_se = float(conj_row["se"].iloc[0]) if len(conj_row) else 0.0
    human_disj_mean = float(disj_row["mean"].iloc[0]) if len(disj_row) else None
    human_disj_se = float(disj_row["se"].iloc[0]) if len(disj_row) else 0.0

    human_plot_rows = []
    if human_conj_mean is not None and human_disj_mean is not None:
        human_plot_rows.append({
            "label": "Adults (Active)",
            "conj": human_conj_mean,
            "conj_se": human_conj_se,
            "disj": human_disj_mean,
            "disj_se": human_disj_se,
            "source": "human",
        })

    # --- LLM: one bar per model, best prompt per rule ---
    print("Loading LLM data...")
    llm_df = load_llm_by_model_prompt(llm_data_dir)
    llm_plot_rows = []
    if not llm_df.empty:
        agg = (
            llm_df.groupby(["model", "prompt_type", "rule"])["all_correct"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["se"] = agg["std"] / np.sqrt(agg["count"])

        for model in MODEL_ORDER:
            conj_candidates = agg[(agg["model"] == model) & (agg["rule"] == "conjunctive")]
            disj_candidates = agg[(agg["model"] == model) & (agg["rule"] == "disjunctive")]
            if conj_candidates.empty or disj_candidates.empty:
                continue
            best_conj = conj_candidates.loc[conj_candidates["mean"].idxmax()]
            best_disj = disj_candidates.loc[disj_candidates["mean"].idxmax()]
            llm_plot_rows.append({
                "label": model,
                "conj": float(best_conj["mean"]),
                "conj_se": float(best_conj["se"]),
                "disj": float(best_disj["mean"]),
                "disj_se": float(best_disj["se"]),
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

    # Rank by score within each condition: Disjunctive group and Conjunctive group
    disj_df = plot_df.sort_values("disj", ascending=False).reset_index(drop=True)
    conj_df = plot_df.sort_values("conj", ascending=False).reset_index(drop=True)
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
        y_disj = r_disj["disj"] + r_disj["disj_se"] + 0.02
        ax.text(x_disj[i], y_disj, f"{r_disj['disj']:.2f}", ha="center", va="bottom", fontsize=13, color="#333")
    for i in range(n):
        r_conj = conj_df.iloc[i]
        y_conj = r_conj["conj"] + r_conj["conj_se"] + 0.02
        ax.text(x_conj[i], y_conj, f"{r_conj['conj']:.2f}", ha="center", va="bottom", fontsize=13, color="#333")

    # X-axis: two groups with bar labels under each bar
    ax.set_xticks(np.concatenate([x_disj, x_conj]))
    disj_labels = [disj_df.iloc[i]["label"] for i in range(n)]
    conj_labels = [conj_df.iloc[i]["label"] for i in range(n)]
    ax.set_xticklabels(disj_labels + conj_labels, rotation=30, ha="right", fontsize=12)
    # Add group labels as text above the two clusters
    ax.text(np.mean(x_disj), 1.08, "Disjunctive", ha="center", fontsize=15, fontweight="bold")
    ax.text(np.mean(x_conj) - 0.5, 1.08, "Conjunctive", ha="center", fontsize=15, fontweight="bold")

    ax.set_ylabel("Object identification accuracy", fontsize=15)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_ylim(0, 1.2)

    # Chance level: 4 objects, 2 blickets -> 1 / C(4,2) = 1/6
    CHANCE_LEVEL = 1.0 / 6.0
    ax.axhline(CHANCE_LEVEL, color="#999", linestyle="--", alpha=0.7, linewidth=1)

    # Legend: Adults (Active) + LLM models that have data (so legend matches bars)
    leg_handles = [mpatches.Patch(facecolor=human_color, edgecolor=human_edge, label="Adults (Active)")]
    models_in_plot = plot_df[plot_df["source"] == "llm"]["model"].unique().tolist()
    for m in MODEL_ORDER:
        if m in model_colors and m in models_in_plot:
            leg_handles.append(mpatches.Patch(facecolor=model_colors[m], edgecolor="#333", label=m))
    ax.legend(handles=leg_handles, loc="upper right", fontsize=14)

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
