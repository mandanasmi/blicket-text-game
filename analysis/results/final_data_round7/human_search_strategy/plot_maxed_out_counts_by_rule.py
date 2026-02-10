"""
Plot: Number of participants who max out (total and successful) by rule type.
Shows grouped bars: Total maxed out vs Successful maxed out for Conjunctive vs Disjunctive.
Uses human_active_data_no_prior_experience.json.
Output: maxed_out_counts_by_rule.png
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]

MAX_TESTS_ALLOWED = 16


def normalize_rule(s):
    if not s:
        return None
    s = str(s).lower()
    if "conjunctive" in s or "all" in s:
        return "conjunctive"
    if "disjunctive" in s or "any" in s:
        return "disjunctive"
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot maxed out counts (total and successful) by rule type.")
    parser.add_argument("--input", default="../human_active_data_no_prior_experience.json", help="Input JSON (default: from round7)")
    parser.add_argument("--output", default="maxed_out_counts_by_rule.png", help="Output PNG")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    with open(args.input, "r") as f:
        data = json.load(f)

    records = {"conjunctive": [], "disjunctive": []}
    for user_id, user_data in data.items():
        if "main_game" not in user_data:
            continue
        mg = user_data["main_game"]
        config_rule = (mg.get("config") or {}).get("rule") or mg.get("rule") or mg.get("true_rule")
        true_rule = normalize_rule(config_rule)
        if true_rule not in ("conjunctive", "disjunctive"):
            continue
        true_blickets = set(mg.get("true_blicket_indices", []))
        user_blickets = set(mg.get("user_chosen_blickets", []))
        obj_ok = true_blickets == user_blickets
        user_choice = normalize_rule(mg.get("rule_type", ""))
        rule_ok = user_choice == true_rule if user_choice else False
        both_ok = obj_ok and rule_ok
        tt = mg.get("test_timings") or []
        num_tests = len(tt) if isinstance(tt, list) else 0
        records[true_rule].append({"num_tests": num_tests, "both_ok": both_ok})

    n_conj = len(records["conjunctive"])
    n_disj = len(records["disjunctive"])
    n_maxed_conj = sum(1 for r in records["conjunctive"] if r["num_tests"] >= MAX_TESTS_ALLOWED)
    n_maxed_disj = sum(1 for r in records["disjunctive"] if r["num_tests"] >= MAX_TESTS_ALLOWED)
    n_maxed_ok_conj = sum(1 for r in records["conjunctive"] if r["num_tests"] >= MAX_TESTS_ALLOWED and r["both_ok"])
    n_maxed_ok_disj = sum(1 for r in records["disjunctive"] if r["num_tests"] >= MAX_TESTS_ALLOWED and r["both_ok"])
    n_success_conj = sum(1 for r in records["conjunctive"] if r["both_ok"])
    n_success_disj = sum(1 for r in records["disjunctive"] if r["both_ok"])

    rule_labels = ["Conjunctive", "Disjunctive"]
    x = np.arange(len(rule_labels))
    w = 0.35
    colors = {"conjunctive": "#3B82F6", "disjunctive": "#EA580C"}

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    b1 = ax.bar(x - w / 2, [n_maxed_conj, n_maxed_disj], w, label="Total maxed out",
                color=[colors["conjunctive"], colors["disjunctive"]], edgecolor="#333", linewidth=0.8)
    b2 = ax.bar(x + w / 2, [n_maxed_ok_conj, n_maxed_ok_disj], w, label="Successful maxed out",
                color=["#93C5FD", "#FDBA74"], edgecolor="#333", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(rule_labels)
    ax.set_ylabel("Number of participants")
    ax.set_title(f"Participants who maxed out ({MAX_TESTS_ALLOWED} tests)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, max(n_maxed_conj, n_maxed_disj) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, bar in enumerate(b1):
        h = bar.get_height()
        n_tot = n_conj if i == 0 else n_disj
        ax.annotate(f"{int(h)}/{n_tot}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for i, bar in enumerate(b2):
        h = bar.get_height()
        n_ok = n_success_conj if i == 0 else n_success_disj
        ax.annotate(f"{int(h)}/{n_ok}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
