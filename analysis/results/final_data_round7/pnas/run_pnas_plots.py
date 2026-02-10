"""
Single entry point for PNAS-related comparison plots.

Usage:
  python run_pnas_plots.py <task> [task_options...]
  python run_pnas_plots.py --list

Tasks:
  comparison_simplified     PNAS developmental + current study (line + bar + raw numbers txt)
  conjunctive_disjunctive   Adults (Passive/Active) vs LLMs, conjunctive and disjunctive bars
  conjunctive_only          Conjunctive accuracy only: Adults (Passive/Active) vs LLMs

Run from round7/pnas/ so ../ paths resolve to round7/.
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("white")
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"]

# PNAS developmental data (Gopnik et al.)
_disj_raw = [0.32, 0.18, 0.04, 0.03, 0.10]
_disj_raw_se = [0.05, 0.05, 0.02, 0.02, 0.03]
PNAS_DATA = {
    "age_group": ["4-y-olds", "6-7-y-olds", "9-11-y-olds", "12-14-y-olds", "Adults"],
    "conjunctive": [0.92, 0.56, 0.60, 0.28, 0.25],
    "conjunctive_se": [0.01, 0.02, 0.02, 0.02, 0.02],
    "disjunctive": [1 - v for v in _disj_raw],
    "disjunctive_se": _disj_raw_se,
}

# Human baselines for LLM comparison plots
HUMAN_PASSIVE_CONJ = (0.25, 0.02)
HUMAN_PASSIVE_DISJ = (0.90, 0.03)
HUMAN_ACTIVE_CONJ = (0.82, 0.05)
HUMAN_ACTIVE_DISJ = (0.96, 0.03)
LLM_SKIP = {"baseline random", "infoGain oracle"}


def normalize_rule(s):
    if not s:
        return None
    s = str(s).lower()
    if "conjunctive" in s or "all" in s:
        return "conjunctive"
    if "disjunctive" in s or "any" in s:
        return "disjunctive"
    return None


def calc_rule_accuracy_and_se_from_json(data, rule_type):
    correct = []
    for user_id, user_data in data.items():
        if "main_game" not in user_data:
            continue
        mg = user_data["main_game"]
        config_rule = (mg.get("config") or {}).get("rule") or mg.get("rule") or mg.get("true_rule")
        true_rule = normalize_rule(config_rule)
        if true_rule != rule_type:
            continue
        user_choice = normalize_rule(mg.get("rule_type", ""))
        correct.append(1.0 if (user_choice and user_choice == true_rule) else 0.0)
    arr = np.array(correct)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    mean = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return mean, se


def cmd_comparison_simplified(args):
    script_dir = Path(__file__).resolve().parent
    json_path = (script_dir / args.json).resolve()
    with open(json_path, "r") as f:
        human_data = json.load(f)
    conj_acc, conj_se = calc_rule_accuracy_and_se_from_json(human_data, "conjunctive")
    disj_acc, disj_se = calc_rule_accuracy_and_se_from_json(human_data, "disjunctive")

    x_current = len(PNAS_DATA["age_group"])
    labels_all = PNAS_DATA["age_group"] + ["Adults\n(Active)"]
    conj_all = PNAS_DATA["conjunctive"] + [conj_acc]
    conj_se_all = PNAS_DATA["conjunctive_se"] + [conj_se]
    disj_all = PNAS_DATA["disjunctive"] + [disj_acc]
    disj_se_all = PNAS_DATA["disjunctive_se"] + [disj_se]
    x_all = list(np.arange(x_current)) + [x_current]

    # Line plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.errorbar(x_all, conj_all, yerr=conj_se_all, fmt="o-", color="#2980b9", linewidth=2, markersize=8,
                capsize=4, capthick=1.5, elinewidth=1.5, label="Conjunctive")
    ax.errorbar(x_all, disj_all, yerr=disj_se_all, fmt="s-", color="#e74c3c", linewidth=2, markersize=8,
                capsize=4, capthick=1.5, elinewidth=1.5, label="Disjunctive")
    ax.axvline(x=x_current - 0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("Rule inference accuracy", fontsize=14)
    ax.set_xticks(x_all)
    ax.set_xticklabels(labels_all, fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.set_xlim(-0.3, x_current + 0.3)
    ax.legend(fontsize=11, loc="upper left", frameon=True, fancybox=True)
    ax.grid(True, axis="y", alpha=0.3, linestyle="-")
    ax.set_axisbelow(True)
    ax.annotate("Current Study", xy=(x_current - 0.12, max(conj_acc, disj_acc) + 0.14), fontsize=9, ha="center", style="italic", color="gray")
    for xi, (m, se) in enumerate(zip(conj_all, conj_se_all)):
        if xi == x_current:
            ax.annotate(f"{m:.2f}\n(+- {se:.2f})", xy=(xi, m - se - 0.04), ha="center", va="top", fontsize=9, color="#2980b9")
        elif xi == 0:
            ax.annotate(f"{m:.2f}\n(+- {se:.2f})", xy=(xi, m + se + 0.02), ha="center", va="bottom", fontsize=9, color="#2980b9")
        else:
            ax.annotate(f"{m:.2f}\n(+- {se:.2f})", xy=(xi, m - se - 0.02), ha="center", va="top", fontsize=9, color="#2980b9")
    for xi, (m, se) in enumerate(zip(disj_all, disj_se_all)):
        if xi == x_current:
            ax.annotate(f"{m:.2f}\n(+- {se:.2f})", xy=(xi, disj_acc + disj_se + 0.02), ha="center", va="bottom", fontsize=9, color="#e74c3c")
        else:
            ax.annotate(f"{m:.2f}\n(+- {se:.2f})", xy=(xi, m + se + 0.02), ha="center", va="bottom", fontsize=9, color="#e74c3c")
    plt.tight_layout()
    out_line = script_dir / args.output_line
    fig.savefig(out_line, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_line}")

    # Bar chart version
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(len(labels_all))
    width = 0.35
    bars1 = ax.bar(x - width / 2, conj_all, width, yerr=conj_se_all, label="Conjunctive", color="#2980b9", alpha=0.85,
                   edgecolor="#333333", linewidth=1.2, capsize=3, error_kw={"color": "#333333", "linewidth": 1.2})
    bars2 = ax.bar(x + width / 2, disj_all, width, yerr=disj_se_all, label="Disjunctive", color="#e74c3c", alpha=0.85,
                   edgecolor="#333333", linewidth=1.2, capsize=3, error_kw={"color": "#333333", "linewidth": 1.2})
    ax.axvline(x=x_current - 0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_ylabel("Rule inference accuracy", fontsize=14)
    ax.set_title("Accuracy of identifying blicket object given conjunctive/disjunctive evidence", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_all, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc="upper left", frameon=True, fancybox=True)
    ax.grid(True, axis="y", alpha=0.3, linestyle="-")
    ax.set_axisbelow(True)
    for bars, values, ses in [(bars1, conj_all, conj_se_all), (bars2, disj_all, disj_se_all)]:
        for bar, val, se in zip(bars, values, ses):
            ax.annotate(f"{val:.2f}\n(+- {se:.2f})", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + se),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    out_bars = script_dir / args.output_bars
    fig.savefig(out_bars, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_bars}")

    # Raw numbers table
    raw_path = script_dir / args.output_raw
    with open(raw_path, "w") as f:
        f.write("Raw numbers: Mean (M) and Standard Error (SE)\n")
        f.write("=" * 60 + "\n\n")
        f.write("PNAS (Gopnik et al.):\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Age group':<18} {'Conjunctive M':>12} {'Conjunctive SE':>14} {'Disjunctive M':>12} {'Disjunctive SE':>14}\n")
        f.write("-" * 60 + "\n")
        for i, age in enumerate(PNAS_DATA["age_group"]):
            c, cs = PNAS_DATA["conjunctive"][i], PNAS_DATA["conjunctive_se"][i]
            d, ds = PNAS_DATA["disjunctive"][i], PNAS_DATA["disjunctive_se"][i]
            f.write(f"{age.replace(chr(10), ' '):<18} {c:>12.2f} {cs:>14.2f} {d:>12.2f} {ds:>14.2f}\n")
        f.write("\n")
        f.write("Current study (Adult Active, rule inference accuracy, n=102; 51 conjunctive, 51 disjunctive):\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Adults (Active)':<18} {conj_acc:>12.2f} {conj_se:>14.2f} {disj_acc:>12.2f} {disj_se:>14.2f}\n")
        f.write("\n")
        f.write("Note: PNAS disjunctive values in this table are plotted as (1 - raw)\n")
        f.write("      in the figure; raw from paper figure: " + ", ".join(f"{v:.2f}" for v in _disj_raw) + "\n")
    print(f"Saved: {raw_path}")


def cmd_conjunctive_disjunctive(args):
    script_dir = Path(__file__).resolve().parent
    llm_path = (script_dir / args.llm_csv).resolve()
    output_path = script_dir / args.output
    llm = pd.read_csv(llm_path)
    llm = llm[~llm["agent"].isin(LLM_SKIP)].copy()
    llm["conjunctive_se"] = llm.get("conjunctive_se", np.nan).fillna(0)
    llm["disjunctive_se"] = llm.get("disjunctive_se", np.nan).fillna(0)
    rows = [
        {"label": "Adults (Passive)", "conj_mean": HUMAN_PASSIVE_CONJ[0], "conj_se": HUMAN_PASSIVE_CONJ[1],
         "disj_mean": HUMAN_PASSIVE_DISJ[0], "disj_se": HUMAN_PASSIVE_DISJ[1], "category": "human_passive"},
        {"label": "Adults (Active)", "conj_mean": HUMAN_ACTIVE_CONJ[0], "conj_se": HUMAN_ACTIVE_CONJ[1],
         "disj_mean": HUMAN_ACTIVE_DISJ[0], "disj_se": HUMAN_ACTIVE_DISJ[1], "category": "human_active"},
    ]
    for _, r in llm.iterrows():
        rows.append({"label": r["agent"], "conj_mean": r["conjunctive_accuracy"], "conj_se": r["conjunctive_se"],
                     "disj_mean": r["disjunctive_accuracy"], "disj_se": r["disjunctive_se"], "category": "llm"})
    df = pd.DataFrame(rows)
    llm_part = df[df["category"] == "llm"].copy().sort_values("conj_mean", ascending=False)
    label_order = ["Adults (Passive)", "Adults (Active)"] + llm_part["label"].tolist()
    df["label"] = pd.Categorical(df["label"], categories=label_order, ordered=True)
    df = df.sort_values("label")
    n = len(df)
    width = 0.36
    x = np.arange(n)
    x_conj = x - width / 2
    x_disj = x + width / 2
    colors = ["#6EC5F5" if row["category"] == "human_passive" else "#F06B85" if row["category"] == "human_active" else "#4477AA" for _, row in df.iterrows()]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars_conj = ax.bar(x_conj, df["conj_mean"], width, yerr=df["conj_se"], capsize=3, color=colors, edgecolor="#333", linewidth=0.8, label="Conjunctive")
    bars_disj = ax.bar(x_disj, df["disj_mean"], width, yerr=df["disj_se"], capsize=3, color=colors, edgecolor="#333", linewidth=0.8, alpha=0.85, label="Disjunctive")
    for b in bars_disj:
        b.set_hatch("///")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("Conjunctive and disjunctive accuracy: Adults (Passive / Active) vs LLMs", fontsize=12)
    for i in range(n):
        row = df.iloc[i]
        ax.text(x_conj[i], row["conj_mean"] + row["conj_se"] + 0.02, f"{row['conj_mean']:.2f}", ha="center", va="bottom", fontsize=7, color="#333")
        ax.text(x_disj[i], row["disj_mean"] + row["disj_se"] + 0.02, f"{row['disj_mean']:.2f}", ha="center", va="bottom", fontsize=7, color="#333")
    ax.legend(handles=[
        mpatches.Patch(facecolor="gray", edgecolor="#333", label="Conjunctive"),
        mpatches.Patch(facecolor="gray", edgecolor="#333", hatch="///", label="Disjunctive"),
        mpatches.Patch(facecolor="#6EC5F5", edgecolor="#333", label="Adults (Passive)"),
        mpatches.Patch(facecolor="#F06B85", edgecolor="#333", label="Adults (Active)"),
        mpatches.Patch(facecolor="#4477AA", edgecolor="#333", label="LLMs"),
    ], loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def cmd_conjunctive_only(args):
    script_dir = Path(__file__).resolve().parent
    llm_path = (script_dir / args.llm_csv).resolve()
    output_path = script_dir / args.output
    llm = pd.read_csv(llm_path)
    llm = llm[~llm["agent"].isin(LLM_SKIP)].copy()
    llm["conjunctive_se"] = llm.get("conjunctive_se", np.nan).fillna(0)
    rows = [
        {"label": "Adults (Passive)", "mean": HUMAN_PASSIVE_CONJ[0], "se": HUMAN_PASSIVE_CONJ[1], "category": "human_passive"},
        {"label": "Adults (Active)", "mean": HUMAN_ACTIVE_CONJ[0], "se": HUMAN_ACTIVE_CONJ[1], "category": "human_active"},
    ]
    for _, r in llm.iterrows():
        rows.append({"label": r["agent"], "mean": r["conjunctive_accuracy"], "se": r["conjunctive_se"], "category": "llm"})
    df = pd.DataFrame(rows)
    llm_order = df[df["category"] == "llm"].sort_values("mean", ascending=False)["label"].tolist()
    label_order = ["Adults (Passive)", "Adults (Active)"] + llm_order
    df["label"] = pd.Categorical(df["label"], categories=label_order, ordered=True)
    df = df.sort_values("label")
    colors = ["#6EC5F5" if row["category"] == "human_passive" else "#F06B85" if row["category"] == "human_active" else "#4477AA" for _, row in df.iterrows()]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(df)), df["mean"], yerr=df["se"], capsize=4, color=colors, edgecolor="#333", linewidth=0.8)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Conjunctive accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("Conjunctive accuracy: Adults (Passive / Active) vs LLMs", fontsize=12)
    for i in range(len(df)):
        row = df.iloc[i]
        ax.text(i, row["mean"] + row["se"] + 0.02, f"{row['mean']:.2f}", ha="center", va="bottom", fontsize=8)
    ax.legend(handles=[
        mpatches.Patch(facecolor="#6EC5F5", edgecolor="#333", label="Adults (Passive)"),
        mpatches.Patch(facecolor="#F06B85", edgecolor="#333", label="Adults (Active)"),
        mpatches.Patch(facecolor="#4477AA", edgecolor="#333", label="LLMs"),
    ], loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PNAS comparison plots")
    parser.add_argument("--list", action="store_true", help="List tasks and exit")
    sub = parser.add_subparsers(dest="task", help="Task to run")

    p = sub.add_parser("comparison_simplified", help="PNAS developmental + current study (line, bar, raw numbers)")
    p.add_argument("--json", default="../human_active_data_no_prior_experience.json", help="Current study JSON")
    p.add_argument("--output-line", default="pnas_comparison_simplified.png", help="Line plot PNG")
    p.add_argument("--output-bars", default="pnas_comparison_simplified_bars.png", help="Bar chart PNG")
    p.add_argument("--output-raw", default="pnas_comparison_raw_numbers.txt", help="Raw numbers table")
    p.set_defaults(func=cmd_comparison_simplified)

    p = sub.add_parser("conjunctive_disjunctive", help="Adults (Passive/Active) vs LLMs, conj + disj bars")
    p.add_argument("--llm-csv", default="../llm_accuracy_by_rule_type.csv", help="LLM accuracy CSV")
    p.add_argument("--output", default="conjunctive_disjunctive_accuracy_merged.png", help="Output PNG")
    p.set_defaults(func=cmd_conjunctive_disjunctive)

    p = sub.add_parser("conjunctive_only", help="Conjunctive accuracy: Adults vs LLMs")
    p.add_argument("--llm-csv", default="../llm_accuracy_by_rule_type.csv", help="LLM accuracy CSV")
    p.add_argument("--output", default="conjunctive_accuracy_comparison.png", help="Output PNG")
    p.set_defaults(func=cmd_conjunctive_only)

    args = parser.parse_args()
    if args.list:
        print("Tasks: comparison_simplified, conjunctive_disjunctive, conjunctive_only")
        return 0
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
