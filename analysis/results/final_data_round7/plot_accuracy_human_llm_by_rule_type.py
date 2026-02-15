"""
Generate one bar chart: Conjunctive and Disjunctive side-by-side.
Shows All Correct Accuracy (objects + rule) for Human (from round7 data)
and LLM agents (from llm_accuracy_by_rule_type.csv) in a single figure
with two groups (Conjunctive | Disjunctive), one bar per agent per group.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_style("white")
import matplotlib as mpl
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"]

# Colors for agents: RGB tuples in 0-1 (matplotlib accepts these)
AGENT_COLORS = {
    "infoGain oracle": (0.5, 0.5, 0.5),
    "baseline random": (0.8, 0.8, 0.8),
    "gpt-4o": (0.4, 0.7607843137254902, 0.6470588235294118),
    "gpt-4o-mini": (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
    "deepseek-chat": (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
    "deepseek-reasoner": (0.9058823529411765, 0.5411764705882353, 0.7647058823529411),
    "gemma3:27b": (0.6509803921568628, 0.8470588235294118, 0.32941176470588235),
    "qwq": (1.0, 0.8509803921568627, 0.1843137254901961),
}
HUMAN_COLOR = (0.161, 0.502, 0.725)  # #2980b9 in 0-1


def main():
    parser = argparse.ArgumentParser(description="Plot Human + LLM All Correct Accuracy by rule type (single plot).")
    parser.add_argument("--human-csv", default="main_game_data_with_prior_experience.csv", help="Human round data")
    parser.add_argument("--llm-csv", default="llm_accuracy_by_rule_type.csv", help="LLM accuracies (agent, conjunctive_accuracy, disjunctive_accuracy, conjunctive_se, disjunctive_se)")
    parser.add_argument("--max-per-rule", type=int, default=50, help="Max participants per rule type for human data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="accuracy_human_llm_by_rule_type.png", help="Single output PNG")
    args = parser.parse_args()

    # Load human data
    df = pd.read_csv(args.human_csv)
    df = df[df["has_prior_experience"] == False].copy()
    rng = np.random.default_rng(args.seed)
    keep = np.zeros(len(df), dtype=bool)
    for rt in ["conjunctive", "disjunctive"]:
        sub = df[df["ground_truth_rule"] == rt]
        uids = sub["user_id"].unique()
        n = min(len(uids), args.max_per_rule)
        chosen = rng.choice(uids, size=n, replace=False) if len(uids) > n else uids
        keep |= (df["ground_truth_rule"] == rt) & (df["user_id"].isin(chosen))
    df = df[keep].copy()

    # Human all-correct by rule type (proportion 0-1 and SE)
    human_conj = df[df["ground_truth_rule"] == "conjunctive"]
    human_disj = df[df["ground_truth_rule"] == "disjunctive"]
    all_correct = (df["object_identification_correct"] == True) & (df["rule_choice_correct"] == True)

    def rate_and_se(rule_df, mask_all_correct):
        sub = rule_df.loc[mask_all_correct[rule_df.index]]
        n = len(rule_df)
        correct = (rule_df["object_identification_correct"] == True) & (rule_df["rule_choice_correct"] == True)
        p = correct.sum() / n if n > 0 else 0.0
        se = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        return p, se

    human_conj_acc, human_conj_se = rate_and_se(human_conj, all_correct)
    human_disj_acc, human_disj_se = rate_and_se(human_disj, all_correct)

    # Load LLM data
    llm = pd.read_csv(args.llm_csv)
    llm["conjunctive_se"] = llm.get("conjunctive_se", np.nan).fillna(0)
    llm["disjunctive_se"] = llm.get("disjunctive_se", np.nan).fillna(0)
    llm_by_agent = llm.set_index("agent")

    # Order: baseline random first, then infoGain oracle, then remaining LLM agents, then Human last
    first_two = ["baseline random", "infoGain oracle"]
    rest_llm = [a for a in llm["agent"] if a not in first_two]
    ordered_agents = first_two + rest_llm + ["Human"]

    def row_for(agent):
        if agent == "Human":
            return human_conj_acc, human_conj_se, human_disj_acc, human_disj_se
        r = llm_by_agent.loc[agent]
        return r["conjunctive_accuracy"], r["conjunctive_se"], r["disjunctive_accuracy"], r["disjunctive_se"]

    labels = ordered_agents
    n_agents = len(labels)
    fallback = (0.6, 0.6, 0.6)
    colors = [HUMAN_COLOR if a == "Human" else AGENT_COLORS.get(a, fallback) for a in labels]
    conj_acc = [row_for(a)[0] for a in labels]
    conj_se = [row_for(a)[1] for a in labels]
    disj_acc = [row_for(a)[2] for a in labels]
    disj_se = [row_for(a)[3] for a in labels]

    # X positions: thin bars with no gap (width = spacing)
    bar_width = 0.55
    gap = 3 * bar_width  # visible space between the two rule-type groups
    x1 = np.arange(n_agents) * bar_width
    x2 = (n_agents * bar_width + gap) + np.arange(n_agents) * bar_width
    x_all = np.concatenate([x1, x2])
    # Disjunctive first (left), then Conjunctive (right)
    acc_all = np.concatenate([disj_acc, conj_acc])
    se_all = np.concatenate([disj_se, conj_se])
    colors_all = colors + colors

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.bar(x_all, acc_all, bar_width, yerr=se_all, capsize=3, color=colors_all,
           edgecolor="#333333", linewidth=1, error_kw={"color": "#333333", "linewidth": 1})
    ax.set_ylabel("All Correct Accuracy", fontsize=11)
    ax.set_xlabel("Rule Type", fontsize=11)
    ax.set_title("Best Model Accuracy (4 objects) – Human and LLM agents", fontsize=12)
    # Tick at center of each group: Disjunctive first, Conjunctive second
    ax.set_xticks([(n_agents - 1) * bar_width / 2, (n_agents * bar_width + gap) + (n_agents - 1) * bar_width / 2])
    ax.set_xticklabels(["Disjunctive", "Conjunctive"], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="#6a3d9a", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    # Legend: below the plot, outside the axes
    legend_handles = [mpatches.Patch(facecolor=c, edgecolor="#333333", label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=True)
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave space below for legend
    plt.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {args.output}")
    print("Human all-correct rates (0-1): conjunctive = {:.3f} (SE={:.3f}), disjunctive = {:.3f} (SE={:.3f})".format(
        human_conj_acc, human_conj_se, human_disj_acc, human_disj_se))


if __name__ == "__main__":
    main()
