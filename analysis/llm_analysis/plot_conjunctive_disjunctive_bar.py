"""
Grouped bar plot: Conjunctive vs Disjunctive accuracy across models/prompts and human baselines.
Same categories as overall_accuracy_bar.png, with two bars per category (conjunctive and disjunctive).
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
    "o3-mini-2025-01-31": "o3-mini",
    "o4-mini-2025-04-16": "o4-mini",
}
MODEL_ORDER = ["deepseek-reasoner", "gpt-4o", "deepseek-chat", "gpt-4o-mini", "o3-mini", "o4-mini", "gemma3:27b", "qwq"]
PROMPT_TYPES = ["cot", "default", "react", "reflexion", "oreason"]


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
        n_obj = env.get("num_objects")
        if n_obj not in (4, 8):
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
    parser.add_argument("--output", default="conjunctive_disjunctive_bar.png", help="Output PNG")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    llm_data_dir = Path(args.llm_data) if args.llm_data else Path("/tmp/llm_data")
    round7_dir = project_root / "analysis" / "results" / "round7"
    output_path = script_dir / args.output

    if not llm_data_dir.exists():
        print(f"ERROR: llm_data not found at {llm_data_dir}")
        return 1

    print("Loading LLM data...")
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
                plot_rows.append({
                    "label": f"{model} ({pt})",
                    "conj": conj.iloc[0]["mean"],
                    "conj_se": conj.iloc[0]["se"],
                    "disj": disj.iloc[0]["mean"],
                    "disj_se": disj.iloc[0]["se"],
                    "source": "llm",
                    "model": model,
                    "prompt_type": pt,
                })

    active = load_human_active(round7_dir)
    if active[0] and active[1]:
        (conj_m, conj_se), (disj_m, disj_se) = active
        plot_rows.append({
            "label": "Adults (Active)",
            "conj": conj_m, "conj_se": conj_se,
            "disj": disj_m, "disj_se": disj_se,
            "source": "human_active",
        })

    plot_rows.append({
        "label": "Adults (Passive)",
        "conj": 0.25, "conj_se": 0.02,
        "disj": 0.90, "disj_se": 0.03,
        "source": "human_passive",
    })

    PNAS_CHILDREN = [
        ("4-y-olds (passive)", 0.92, 0.01, 0.68, 0.05),
        ("6-7-y-olds (passive)", 0.56, 0.02, 0.82, 0.05),
        ("9-11-y-olds (passive)", 0.60, 0.02, 0.96, 0.02),
        ("12-14-y-olds (passive)", 0.28, 0.02, 0.97, 0.02),
    ]
    for label, conj, conj_se, disj, disj_se in PNAS_CHILDREN:
        plot_rows.append({
            "label": label,
            "conj": conj, "conj_se": conj_se,
            "disj": disj, "disj_se": disj_se,
            "source": "pnas_child",
        })

    if not plot_rows:
        print("No data to plot.")
        return 0

    plot_df = pd.DataFrame(plot_rows)
    plot_df["overall"] = (plot_df["conj"] + plot_df["disj"]) / 2
    human_sources = ("human_active", "human_passive", "pnas_child")
    human_df = plot_df[plot_df["source"].isin(human_sources)].copy()
    llm_df = plot_df[plot_df["source"] == "llm"].copy()

    # Human order: Adults Active, Adults Passive, then children by age
    human_order = ["Adults (Active)", "Adults (Passive)", "4-y-olds (passive)", "6-7-y-olds (passive)", "9-11-y-olds (passive)", "12-14-y-olds (passive)"]
    human_df["_order"] = human_df["label"].apply(lambda l: human_order.index(l) if l in human_order else len(human_order))
    human_df = human_df.sort_values("_order").reset_index(drop=True)
    llm_df = llm_df.sort_values("overall", ascending=False).reset_index(drop=True)

    n_human = len(human_df)
    n_llm = len(llm_df)
    width = 0.75
    gap = 1.0  # gap between the four groups
    # Four groups: Disjunctive/Human, Disjunctive/LLM, Conjunctive/Human, Conjunctive/LLM
    start_disj_h = 0
    start_disj_llm = n_human + gap
    start_conj_h = start_disj_llm + n_llm + gap
    start_conj_llm = start_conj_h + n_human + gap

    x_disj_h = np.arange(n_human) + start_disj_h
    x_disj_llm = np.arange(n_llm) + start_disj_llm
    x_conj_h = np.arange(n_human) + start_conj_h
    x_conj_llm = np.arange(n_llm) + start_conj_llm

    model_colors = {
        "deepseek-reasoner": "#4477AA",
        "gpt-4o": "#CCBB44",
        "deepseek-chat": "#228833",
        "gpt-4o-mini": "#BB88CC",
        "o3-mini": "#EE7733",
        "o4-mini": "#AA3377",
        "gemma3:27b": "#88CCEE",
        "qwq": "#44AA99",
    }
    human_label_colors = {
        "Adults (Active)": "#F06B85",
        "Adults (Passive)": "#6EC5F5",
        "4-y-olds (passive)": "#00C07F",
        "6-7-y-olds (passive)": "#F5EB50",
        "9-11-y-olds (passive)": "#FFB020",
        "12-14-y-olds (passive)": "#E08FBF",
    }
    prompt_hatch = {"cot": "", "default": "///", "react": "ooo", "reflexion": "...", "oreason": "***"}
    human_hatch = "OOO"
    human_edge = "#888"

    def bar_color(row):
        if row["source"] == "llm":
            return model_colors.get(row["model"], "#999999")
        return human_label_colors.get(row["label"], "#E85A75")

    def bar_hatch(row):
        if row["source"] in human_sources:
            return human_hatch
        return prompt_hatch.get(row.get("prompt_type", ""), "")

    fig, ax = plt.subplots(figsize=(16, 8))

    colors_disj_h = [bar_color(human_df.iloc[i]) for i in range(n_human)]
    colors_disj_llm = [bar_color(llm_df.iloc[i]) for i in range(n_llm)]
    colors_conj_h = colors_disj_h
    colors_conj_llm = colors_disj_llm

    bars_disj_h = ax.bar(x_disj_h, human_df["disj"], width, yerr=human_df["disj_se"],
                         color=colors_disj_h, edgecolor=human_edge, linewidth=0.8,
                         capsize=2, error_kw={"elinewidth": 1, "capthick": 1})
    bars_disj_llm = ax.bar(x_disj_llm, llm_df["disj"], width, yerr=llm_df["disj_se"],
                           color=colors_disj_llm, edgecolor="#333", linewidth=0.8,
                           capsize=2, error_kw={"elinewidth": 1, "capthick": 1})
    bars_conj_h = ax.bar(x_conj_h, human_df["conj"], width, yerr=human_df["conj_se"],
                         color=colors_conj_h, edgecolor=human_edge, linewidth=0.8,
                         capsize=2, error_kw={"elinewidth": 1, "capthick": 1})
    bars_conj_llm = ax.bar(x_conj_llm, llm_df["conj"], width, yerr=llm_df["conj_se"],
                           color=colors_conj_llm, edgecolor="#333", linewidth=0.8,
                           capsize=2, error_kw={"elinewidth": 1, "capthick": 1})

    for i in range(n_human):
        row = human_df.iloc[i]
        for bar in (bars_disj_h[i], bars_conj_h[i]):
            bar.set_hatch(bar_hatch(row))
    for i in range(n_llm):
        row = llm_df.iloc[i]
        for bar in (bars_disj_llm[i], bars_conj_llm[i]):
            bar.set_hatch(bar_hatch(row))

    # Mean above each bar
    for i in range(n_human):
        row = human_df.iloc[i]
        ax.text(x_disj_h[i], row["disj"] + row["disj_se"] + 0.02, f"{row['disj']:.2f}",
                ha="center", va="bottom", fontsize=6, color="#333")
        ax.text(x_conj_h[i], row["conj"] + row["conj_se"] + 0.02, f"{row['conj']:.2f}",
                ha="center", va="bottom", fontsize=6, color="#333")
    for i in range(n_llm):
        row = llm_df.iloc[i]
        ax.text(x_disj_llm[i], row["disj"] + row["disj_se"] + 0.02, f"{row['disj']:.2f}",
                ha="center", va="bottom", fontsize=6, color="#333")
        ax.text(x_conj_llm[i], row["conj"] + row["conj_se"] + 0.02, f"{row['conj']:.2f}",
                ha="center", va="bottom", fontsize=6, color="#333")

    # X-axis: four group labels
    tick_positions = [
        np.mean(x_disj_h),
        np.mean(x_disj_llm),
        np.mean(x_conj_h),
        np.mean(x_conj_llm),
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([
        "Disjunctive\n(Human)",
        "Disjunctive\n(LLM)",
        "Conjunctive\n(Human)",
        "Conjunctive\n(LLM)",
    ], fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="#999", linestyle="--", alpha=0.4)

    # Legend: human categories + models that have data + prompt types that have data
    leg_handles = []
    for label, c in human_label_colors.items():
        leg_handles.append(mpatches.Patch(facecolor=c, edgecolor=human_edge, hatch=human_hatch, label=label))
    models_in_plot = llm_df["model"].unique().tolist()
    for m in MODEL_ORDER:
        if m in models_in_plot:
            leg_handles.append(mpatches.Patch(facecolor=model_colors[m], edgecolor="#333", label=m))
    prompt_types_in_plot = llm_df["prompt_type"].unique().tolist()
    for pt in PROMPT_TYPES:
        if pt in prompt_types_in_plot:
            leg_handles.append(mpatches.Patch(facecolor="#e0e0e0", edgecolor="#333", hatch=prompt_hatch[pt], label=pt))
    ax.legend(handles=leg_handles, loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {output_path}")
    plt.close()
    return 0


if __name__ == "__main__":
    exit(main())
