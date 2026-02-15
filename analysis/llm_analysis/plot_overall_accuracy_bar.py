"""
Bar plot: Overall accuracy (average of conjunctive and disjunctive) for all models/prompt variants
and human baselines shown in llm_models_prompt_types_scatter.png.
"""

import argparse
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
PROMPT_TYPES = ["cot", "default", "react", "reflexion"]


def get_prompt_type(agent: dict) -> str | None:
    """Extract prompt type: cot, default, react, reflexion."""
    target = agent.get("_target_", "") or ""
    if "CoTPromptAgent" in target:
        return "cot"
    if "ReflexionPromptsAgent" in target:
        return "reflexion"
    if "PromptsAgent" in target:
        react = agent.get("react", False)
        return "react" if react else "default"
    return None


def load_llm_by_model_prompt(llm_data_dir: Path):
    """Load LLM results with 4 objects, grouped by (model, prompt_type, rule)."""
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


def load_human_active(round7_dir: Path):
    """Load Human Active rule inference from current study."""
    path = round7_dir / "human_active_data_no_prior_experience.json"
    if not path.exists():
        return None, None

    def normalize(s):
        if not s:
            return None
        s = str(s).lower()
        if "conjunctive" in s or "all" in s:
            return "conjunctive"
        if "disjunctive" in s or "any" in s:
            return "disjunctive"
        return None

    with open(path) as f:
        data = json.load(f)
    conj, disj = [], []
    for user_data in data.values():
        mg = (user_data or {}).get("main_game") or {}
        config_rule = (mg.get("config") or {}).get("rule") or mg.get("rule") or mg.get("true_rule")
        true_rule = normalize(config_rule)
        user_choice = normalize(mg.get("rule_type", ""))
        correct = 1.0 if (user_choice and user_choice == true_rule) else 0.0
        if true_rule == "conjunctive":
            conj.append(correct)
        elif true_rule == "disjunctive":
            disj.append(correct)
    if not conj or not disj:
        return None, None
    conj_a, disj_a = np.array(conj), np.array(disj)
    return (
        (float(conj_a.mean()), float(conj_a.std(ddof=1) / np.sqrt(len(conj_a))) if len(conj_a) > 1 else 0),
        (float(disj_a.mean()), float(disj_a.std(ddof=1) / np.sqrt(len(disj_a))) if len(disj_a) > 1 else 0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-data", default=None, help="Path to llm_data directory")
    parser.add_argument("--output", default="overall_accuracy_bar.png", help="Output PNG")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    llm_data_dir = Path(args.llm_data) if args.llm_data else Path("/tmp/llm_data")
    round7_dir = project_root / "analysis" / "results" / "round7"
    output_path = script_dir / args.output

    if not llm_data_dir.exists():
        print(f"ERROR: llm_data not found at {llm_data_dir}")
        return 1

    print("Loading LLM data by model and prompt type...")
    llm_df = load_llm_by_model_prompt(llm_data_dir)
    if llm_df.empty:
        print("No 4-object LLM data found.")
        return 0

    agg = llm_df.groupby(["model", "prompt_type", "rule"])["all_correct"].agg(["mean", "std", "count"]).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"])

    plot_rows = []
    for model in MODEL_ORDER:
        for pt in PROMPT_TYPES:
            conj = agg[(agg["model"] == model) & (agg["prompt_type"] == pt) & (agg["rule"] == "conjunctive")]
            disj = agg[(agg["model"] == model) & (agg["prompt_type"] == pt) & (agg["rule"] == "disjunctive")]
            if len(conj) > 0 and len(disj) > 0:
                conj_mean, conj_se = conj.iloc[0]["mean"], conj.iloc[0]["se"]
                disj_mean, disj_se = disj.iloc[0]["mean"], disj.iloc[0]["se"]
                overall = (conj_mean + disj_mean) / 2
                # Combined SE using sqrt of sum of squared SEs divided by 2
                overall_se = np.sqrt(conj_se**2 + disj_se**2) / 2
                plot_rows.append({
                    "label": f"{model} ({pt})",
                    "overall": overall,
                    "overall_se": overall_se,
                    "conj": conj_mean,
                    "disj": disj_mean,
                    "source": "llm",
                    "model": model,
                    "prompt_type": pt,
                })

    # Human Active (current study)
    active = load_human_active(round7_dir)
    if active[0] and active[1]:
        (conj_m, conj_se), (disj_m, disj_se) = active
        overall = (conj_m + disj_m) / 2
        overall_se = np.sqrt(conj_se**2 + disj_se**2) / 2
        plot_rows.append({
            "label": "Human (Active)",
            "overall": overall,
            "overall_se": overall_se,
            "conj": conj_m,
            "disj": disj_m,
            "source": "human_active",
        })

    # Human Passive (PNAS Adults)
    conj_m, conj_se, disj_m, disj_se = 0.25, 0.02, 0.90, 0.03
    overall = (conj_m + disj_m) / 2
    overall_se = np.sqrt(conj_se**2 + disj_se**2) / 2
    plot_rows.append({
        "label": "Human (Passive)",
        "overall": overall,
        "overall_se": overall_se,
        "conj": conj_m,
        "disj": disj_m,
        "source": "human_passive",
    })

    # PNAS developmental age groups
    PNAS_CHILDREN = [
        ("4-y-olds", 0.92, 0.01, 0.68, 0.05),
        ("6-7-y-olds", 0.56, 0.02, 0.82, 0.05),
        ("9-11-y-olds", 0.60, 0.02, 0.96, 0.02),
        ("12-14-y-olds", 0.28, 0.02, 0.97, 0.02),
    ]
    for label, conj, conj_se, disj, disj_se in PNAS_CHILDREN:
        overall = (conj + disj) / 2
        overall_se = np.sqrt(conj_se**2 + disj_se**2) / 2
        plot_rows.append({
            "label": label,
            "overall": overall,
            "overall_se": overall_se,
            "conj": conj,
            "disj": disj,
            "source": "pnas_child",
        })

    if not plot_rows:
        print("No data to plot.")
        return 0

    plot_df = pd.DataFrame(plot_rows)
    # Sort by overall accuracy descending
    plot_df = plot_df.sort_values("overall", ascending=False).reset_index(drop=True)

    # Same color per model (colorblind-friendly); different texture per prompt
    model_colors = {
        "deepseek-reasoner": "#4477AA",
        "gpt-4o": "#CCBB44",
        "deepseek-chat": "#228833",
        "gpt-4o-mini": "#BB88CC",
        "gemma3:27b": "#88CCEE",
        "qwq": "#44AA99",
    }
    human_color = "#E85A75"
    prompt_hatch = {"cot": "", "default": "///", "react": "ooo", "reflexion": "..."}

    def bar_color(row):
        if row["source"] == "llm":
            return model_colors.get(row["model"], "#999999")
        return human_color

    human_hatch = "OOO"  # large circles (distinct from prompt textures)
    human_edge = "#888"  # lighter edge so texture reads lighter

    def bar_hatch(row):
        if row["source"] in ("human_active", "human_passive", "pnas_child"):
            return human_hatch
        return prompt_hatch.get(row.get("prompt_type", ""), "")

    colors = [bar_color(plot_df.iloc[i]) for i in range(len(plot_df))]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(plot_df))
    bars = ax.bar(x, plot_df["overall"], yerr=plot_df["overall_se"],
                  color=colors, edgecolor="#333", linewidth=0.8,
                  capsize=3, error_kw={"elinewidth": 1, "capthick": 1})

    # Apply texture (hatch) per bar: human = lighter edge for texture, LLM = by prompt type
    human_sources = ("human_active", "human_passive", "pnas_child")
    for i in range(len(plot_df)):
        bars[i].set_hatch(bar_hatch(plot_df.iloc[i]))
        bars[i].set_edgecolor(human_edge if plot_df.iloc[i]["source"] in human_sources else "#333")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Overall Accuracy (avg of conj. & disj.)", fontsize=11)
    ax.set_ylim(0, 1.05)

    # Legend: models (color) + Human + prompts (texture)
    leg_handles = [
        mpatches.Patch(facecolor=human_color, edgecolor=human_edge, hatch=human_hatch, label="Human"),
    ]
    for m in MODEL_ORDER:
        leg_handles.append(mpatches.Patch(facecolor=model_colors[m], edgecolor="#333", label=m))
    for pt in PROMPT_TYPES:
        leg_handles.append(mpatches.Patch(facecolor="#e0e0e0", edgecolor="#333", hatch=prompt_hatch[pt], label=pt))
    ax.legend(handles=leg_handles, loc="upper right", fontsize=8, ncol=2)

    # Add value labels on top of bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.text(i, row["overall"] + row["overall_se"] + 0.02, f"{row['overall']:.2f}", 
                ha="center", va="bottom", fontsize=7, color="#333")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {output_path}")
    plt.close()
    return 0


if __name__ == "__main__":
    exit(main())
