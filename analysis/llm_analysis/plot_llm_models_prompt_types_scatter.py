"""
Scatter plot: LLM all-correct accuracy (4 objects) by model and prompt type vs human baselines.
- X = conjunctive accuracy, Y = disjunctive accuracy
- Each point = (model, prompt_type) or Human Active / Human Passive
- Prompt types: cot, default, react, reflexion
- Models: deepseek-reasoner, gpt-4o, deepseek-chat, gpt-4o-mini, gemma3:27b, qwq
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
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
    parser.add_argument("--output", default="llm_models_prompt_types_scatter.png", help="Output PNG")
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
                plot_rows.append({
                    "label": f"{model}\n({pt})",
                    "short": f"{model} {pt}",
                    "conj": conj.iloc[0]["mean"],
                    "conj_se": conj.iloc[0]["se"],
                    "disj": disj.iloc[0]["mean"],
                    "disj_se": disj.iloc[0]["se"],
                    "source": "llm",
                    "model": model,
                    "prompt_type": pt,
                })

    # Human Active (current study)
    active = load_human_active(round7_dir)
    if active[0] and active[1]:
        (conj_m, conj_se), (disj_m, disj_se) = active
        plot_rows.append({
            "label": "Human\n(Active)",
            "short": "Human Active",
            "conj": conj_m,
            "conj_se": conj_se,
            "disj": disj_m,
            "disj_se": disj_se,
            "source": "human_active",
        })

    # Human Passive (PNAS Adults)
    plot_rows.append({
        "label": "Human\n(Passive)",
        "short": "Human Passive",
        "conj": 0.25,
        "conj_se": 0.02,
        "disj": 0.90,
        "disj_se": 0.03,
        "source": "human_passive",
    })

    # PNAS developmental age groups (4-y-olds, 6-7-y-olds, 9-11-y-olds, 12-14-y-olds)
    PNAS_CHILDREN = [
        ("4-y-olds", 0.92, 0.01, 0.68, 0.05),
        ("6-7-y-olds", 0.56, 0.02, 0.82, 0.05),
        ("9-11-y-olds", 0.60, 0.02, 0.96, 0.02),
        ("12-14-y-olds", 0.28, 0.02, 0.97, 0.02),
    ]
    for label, conj, conj_se, disj, disj_se in PNAS_CHILDREN:
        plot_rows.append({
            "label": label,
            "short": label,
            "conj": conj,
            "conj_se": conj_se,
            "disj": disj,
            "disj_se": disj_se,
            "source": "pnas_child",
            "age_label": label,
        })

    if not plot_rows:
        print("No data to plot.")
        return 0

    plot_df = pd.DataFrame(plot_rows)

    # Unique color per (model, prompt_type); marker shape for prompt type
    model_colors = {
        "deepseek-reasoner": "#1f77b4",
        "gpt-4o": "#ff7f0e",
        "deepseek-chat": "#2ca02c",
        "gpt-4o-mini": "#d62728",
        "gemma3:27b": "#9467bd",
        "qwq": "#8c564b",
    }
    prompt_markers = {"cot": "o", "default": "s", "react": "^", "reflexion": "D"}
    # Shade per prompt: default=full, cot=slightly lighter, react=medium, reflexion=lightest
    prompt_alpha = {"default": 1.0, "cot": 0.9, "react": 0.75, "reflexion": 0.6}

    def shade_hex(hex_c, amount):
        """Lighten hex by amount (0=no change, 0.5=halfway to white)."""
        hex_c = hex_c.lstrip("#")
        r = min(255, int(int(hex_c[0:2], 16) + (255 - int(hex_c[0:2], 16)) * amount))
        g = min(255, int(int(hex_c[2:4], 16) + (255 - int(hex_c[2:4], 16)) * amount))
        b = min(255, int(int(hex_c[4:6], 16) + (255 - int(hex_c[4:6], 16)) * amount))
        return f"#{r:02x}{g:02x}{b:02x}"

    pnas_child_colors = {
        "4-y-olds": "#27ae60",
        "6-7-y-olds": "#f39c12",
        "9-11-y-olds": "#e67e22",
        "12-14-y-olds": "#8e44ad",
    }

    def point_style(row):
        if row["source"] == "human_active":
            return "#c0392b", "*"
        if row["source"] == "human_passive":
            return "#2980b9", "*"
        if row["source"] == "pnas_child":
            return pnas_child_colors.get(row.get("age_label", ""), "#7f7f7f"), "*"
        base = model_colors.get(row["model"], "#7f7f7f")
        amt = 1 - prompt_alpha.get(row["prompt_type"], 0.8)
        c = shade_hex(base, amt * 0.4)
        m = prompt_markers.get(row["prompt_type"], "o")
        return c, m

    from matplotlib.lines import Line2D

    # Section headers (invisible handle, label only)
    def section_label(text):
        return Line2D([0], [0], linestyle="", marker="", markersize=0, label=text)

    human_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#c0392b", markersize=11, label="Human (Active)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#2980b9", markersize=11, label="Human (Passive)"),
    ]
    for label, c in pnas_child_colors.items():
        human_handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor=c, markersize=11, label=label))

    llm_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=model_colors[m], markersize=9, label=m)
        for m in MODEL_ORDER
    ]

    prompt_handles = [
        Line2D([0], [0], marker=prompt_markers[pt], color="w", markerfacecolor="#444", markeredgecolor="#333", markersize=8, label=pt)
        for pt in PROMPT_TYPES
    ]

    # Single combined legend: Humans, LLMs, Prompts
    all_handles = (
        [section_label("Humans")] + human_handles
        + [section_label("LLMs")] + llm_handles
        + [section_label("Prompts")] + prompt_handles
    )

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    for i, row in plot_df.iterrows():
        c, m = point_style(row)
        ax.scatter(row["conj"], row["disj"], s=75, c=c, marker=m, alpha=0.9, edgecolors="#333", linewidth=0.8, zorder=3)
        ax.errorbar(row["conj"], row["disj"], xerr=row["conj_se"], yerr=row["disj_se"],
                   fmt="none", ecolor=c, capsize=2, alpha=0.6, zorder=2)
        txt = f"c: {row['conj']:.2f}\u00b1{row['conj_se']:.2f}\nd: {row['disj']:.2f}\u00b1{row['disj_se']:.2f}"
        if row.get("model") == "gpt-4o" and row.get("prompt_type") == "default":
            ax.annotate(txt, (row["conj"], row["disj"]), xytext=(8, 4), textcoords="offset points",
                        fontsize=5, color=c, va="bottom", ha="left")
        elif row.get("age_label") == "4-y-olds":
            ax.annotate(txt, (row["conj"], row["disj"]), xytext=(-12, -3), textcoords="offset points",
                        fontsize=5, color=c, va="center", ha="right")
        elif row.get("age_label") == "9-11-y-olds":
            ax.annotate(txt, (row["conj"], row["disj"]), xytext=(-6, 0), textcoords="offset points",
                        fontsize=5, color=c, va="bottom", ha="right")
        elif row.get("model") == "gpt-4o" and row.get("prompt_type") == "react":
            ax.annotate(txt, (row["conj"], row["disj"]), xytext=(-6, -6), textcoords="offset points",
                        fontsize=5, color=c, va="top", ha="right")
        elif row.get("model") == "deepseek-reasoner" and row.get("prompt_type") == "default":
            ax.annotate(txt, (row["conj"], row["disj"]), xytext=(0, -14), textcoords="offset points",
                        fontsize=5, color=c, va="top", ha="center")
        else:
            ax.annotate(txt, (row["conj"], row["disj"]), xytext=(6, 2), textcoords="offset points",
                        fontsize=5, color=c, va="bottom")
    ax.set_xlabel("Conjunctive accuracy", fontsize=11)
    ax.set_ylabel("Disjunctive accuracy", fontsize=11)
    # Place vertical reference line just right of the rightmost point; x-axis ends there with tick "1.0"
    x_max_data = plot_df["conj"].max() + (plot_df.loc[plot_df["conj"].idxmax(), "conj_se"] if len(plot_df) else 0)
    x_line = min(1.0, x_max_data + 0.03)
    ax.set_xlim(0, x_line)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axhline(1.0, color="#999", linestyle="--", alpha=0.35)
    ax.axvline(x_line, color="#999", linestyle="--", alpha=0.35)
    # Put "1.0" tick where the gray line ends
    xticks = [0, 0.2, 0.4, 0.6, 0.8, x_line]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Single legend at bottom right (Humans, LLMs, Prompts)
    leg = ax.legend(
        handles=all_handles, loc="lower right", bbox_to_anchor=(0.98, 0.0), fontsize=7,
        handlelength=1.8, handletextpad=0.8, labelspacing=0.5, framealpha=0.95,
    )
    # Slightly bold section headers (Humans, LLMs, Prompts)
    for t in leg.get_texts():
        if t.get_text() in ("Humans", "LLMs", "Prompts"):
            t.set_fontweight("semibold")
    plt.tight_layout(pad=0.3)
    fig.subplots_adjust(left=0.1, bottom=0.12, top=0.96, right=0.96)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")
    plt.close()
    return 0


if __name__ == "__main__":
    exit(main())
