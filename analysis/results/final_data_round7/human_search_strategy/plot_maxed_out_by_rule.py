"""
Plot: Did participants max out? (stacked bar by rule type)
Among participants who successfully solved the task (both objects and rule correct).
Shows proportion who did not max out vs maxed out (hit the test limit).
Uses human_active_data_no_prior_experience.json.
Output: maxed_out_by_rule.png
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
    parser = argparse.ArgumentParser(description="Plot maxed out vs not maxed out by rule type.")
    parser.add_argument("--input", default="../human_active_data_no_prior_experience.json", help="Input JSON (default: from round7)")
    parser.add_argument("--output", default="maxed_out_by_rule.png", help="Output PNG")
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
        if not both_ok:
            continue
        tt = mg.get("test_timings") or []
        num_tests = len(tt) if isinstance(tt, list) else 0
        records[true_rule].append({"num_tests": num_tests})

    n_conj = len(records["conjunctive"])
    n_disj = len(records["disjunctive"])
    n_maxed_conj = sum(1 for r in records["conjunctive"] if r["num_tests"] >= MAX_TESTS_ALLOWED)
    n_maxed_disj = sum(1 for r in records["disjunctive"] if r["num_tests"] >= MAX_TESTS_ALLOWED)
    prop_maxed_conj = n_maxed_conj / max(1, n_conj)
    prop_maxed_disj = n_maxed_disj / max(1, n_disj)
    prop_not_maxed_conj = 1 - prop_maxed_conj
    prop_not_maxed_disj = 1 - prop_maxed_disj

    rule_labels = ["Conjunctive", "Disjunctive"]
    x = np.arange(len(rule_labels))
    w = 0.5
    color_not_maxed = "#67e8f9"
    color_maxed = "#f97316"

    fig, ax = plt.subplots(1, 1, figsize=(4, 2.8))
    b1 = ax.bar(x, [prop_not_maxed_conj, prop_not_maxed_disj], w, color=color_not_maxed, edgecolor="none")
    b2 = ax.bar(x, [prop_maxed_conj, prop_maxed_disj], w, bottom=[prop_not_maxed_conj, prop_not_maxed_disj],
                color=color_maxed, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(rule_labels, fontsize=8)
    ax.set_ylabel("Proportion of successful participants (out of 1)", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(axis="both", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate ALL segments: label + count
    for i, (prop_m, prop_not, n_m, n_tot) in enumerate([
        (prop_maxed_conj, prop_not_maxed_conj, n_maxed_conj, n_conj),
        (prop_maxed_disj, prop_not_maxed_disj, n_maxed_disj, n_disj),
    ]):
        pct_not = 100 * prop_not
        pct_m = 100 * prop_m
        n_not = n_tot - int(n_m)
        # Did not max out segment (always show)
        fsize_not = 5 if prop_not < 0.15 else 6
        ax.annotate(f"Did not max out\n{n_not}/{n_tot} ({pct_not:.0f}%)",
                    xy=(x[i], prop_not / 2), xytext=(0, 0), textcoords="offset points",
                    ha="center", va="center", fontsize=fsize_not, color="#0f172a")
        # Maxed segment (always show)
        fsize_m = 5 if prop_m < 0.1 else 6
        label = f"Maxed ({MAX_TESTS_ALLOWED})\n{int(n_m)}/{n_tot} ({pct_m:.0f}%)"
        ax.annotate(label, xy=(x[i], prop_not + prop_m / 2), xytext=(0, 0), textcoords="offset points",
                    ha="center", va="center", fontsize=fsize_m, color="white")

    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.18)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
