"""
Visualize actions, tests, time, and maxed-out results by rule type.
Uses human_active_data_no_prior_experience.json.
Output: actions_tests_time_by_rule.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

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


def parse_duration_iso(start_str, end_str):
    try:
        start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        return (end - start).total_seconds()
    except Exception:
        return None


def mean_se(arr):
    arr = np.array([x for x in arr if np.isfinite(x)])
    n = len(arr)
    if n == 0:
        return np.nan, np.nan
    m = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return m, se


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot actions/tests/time and maxed-out by rule type.")
    parser.add_argument("--input", default="../human_active_data_no_prior_experience.json", help="Input JSON (default: from round7)")
    parser.add_argument("--output", default="actions_tests_time_by_rule.png", help="Output PNG")
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
        ah = mg.get("action_history") or []
        num_actions = len(ah) if isinstance(ah, list) else mg.get("action_history_length", 0)
        tt = mg.get("test_timings") or []
        num_tests = len(tt) if isinstance(tt, list) else 0
        total_time = mg.get("total_test_time_seconds")
        if total_time is None and isinstance(tt, list) and tt:
            total_time = sum(float(t.get("time_since_previous_seconds") or 0) for t in tt if isinstance(t, dict))
        if total_time is None:
            total_time = parse_duration_iso(mg.get("start_time"), mg.get("end_time"))
        total_time = float(total_time) if total_time is not None else np.nan
        true_blickets = set(mg.get("true_blicket_indices", []))
        user_blickets = set(mg.get("user_chosen_blickets", []))
        obj_ok = true_blickets == user_blickets
        user_choice = normalize_rule(mg.get("rule_type", ""))
        rule_ok = user_choice == true_rule if user_choice else False
        both_ok = obj_ok and rule_ok
        records[true_rule].append({
            "num_actions": num_actions,
            "num_tests": num_tests,
            "total_time": total_time,
            "both_ok": both_ok,
        })

    results = {}
    for rt in ["conjunctive", "disjunctive"]:
        recs = records[rt]
        n = len(recs)
        if n == 0:
            results[rt] = {"n": 0}
            continue
        actions = [r["num_actions"] for r in recs]
        tests = [r["num_tests"] for r in recs]
        total_times = [r["total_time"] for r in recs]
        time_per_test = [r["total_time"] / r["num_tests"] for r in recs if r["num_tests"] > 0 and np.isfinite(r["total_time"])]
        actions_per_test = [r["num_actions"] / r["num_tests"] for r in recs if r["num_tests"] > 0]
        n_maxed = sum(1 for r in recs if r["num_tests"] >= MAX_TESTS_ALLOWED)
        n_maxed_ok = sum(1 for r in recs if r["num_tests"] >= MAX_TESTS_ALLOWED and r["both_ok"])
        m_actions, se_actions = mean_se(actions)
        m_tests, se_tests = mean_se(tests)
        m_total, se_total = mean_se(total_times)
        m_tpt, se_tpt = mean_se(time_per_test)
        m_apt, se_apt = mean_se(actions_per_test)
        results[rt] = {
            "n": n,
            "mean_tests": m_tests, "se_tests": se_tests,
            "mean_total_time": m_total, "se_total_time": se_total,
            "mean_time_per_test": m_tpt, "se_time_per_test": se_tpt,
            "mean_actions_per_test": m_apt, "se_actions_per_test": se_apt,
            "mean_actions": m_actions, "se_actions": se_actions,
            "n_maxed_out": n_maxed, "n_maxed_out_both_ok": n_maxed_ok,
        }

    rule_labels = ["Conjunctive", "Disjunctive"]
    x = np.arange(len(rule_labels))
    width = 0.45
    colors = {"conjunctive": "#3B82F6", "disjunctive": "#EA580C"}

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    # Panel A: Number of tests (with max line at 16)
    ax = axes[0, 0]
    means = [results["conjunctive"]["mean_tests"], results["disjunctive"]["mean_tests"]]
    ses = [results["conjunctive"]["se_tests"], results["disjunctive"]["se_tests"]]
    bars = ax.bar(x - width / 2, means, width, yerr=ses, capsize=4, color=[colors["conjunctive"], colors["disjunctive"]], edgecolor="#333", linewidth=0.8)
    for i, (m, se) in enumerate(zip(means, ses)):
        ax.annotate(f"{m:.2f} $\\pm$ {se:.2f}", xy=(x[i] - width / 2 + width / 2, m + se), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.axhline(MAX_TESTS_ALLOWED, color="gray", linestyle="--", linewidth=1, label=f"Max allowed ({MAX_TESTS_ALLOWED})")
    ax.set_xticks(x)
    ax.set_xticklabels(rule_labels)
    ax.set_ylabel("Number of tests")
    ax.set_title("Tests used")
    ax.set_ylim(0, max(max(means) + max(ses) * 2, MAX_TESTS_ALLOWED) * 1.15)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Actions per test
    ax = axes[0, 1]
    means = [results["conjunctive"]["mean_actions_per_test"], results["disjunctive"]["mean_actions_per_test"]]
    ses = [results["conjunctive"]["se_actions_per_test"], results["disjunctive"]["se_actions_per_test"]]
    ax.bar(x - width / 2, means, width, yerr=ses, capsize=4, color=[colors["conjunctive"], colors["disjunctive"]], edgecolor="#333", linewidth=0.8)
    for i, (m, se) in enumerate(zip(means, ses)):
        ax.annotate(f"{m:.2f} $\\pm$ {se:.2f}", xy=(x[i] - width / 2 + width / 2, m + se), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(rule_labels)
    ax.set_ylabel("Actions per test")
    ax.set_title("Actions per test")
    ax.set_ylim(0, (max(means) + max(ses) * 2.5 if ses else max(means) * 0.15) * 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel C: Total test time (s)
    ax = axes[1, 0]
    means = [results["conjunctive"]["mean_total_time"], results["disjunctive"]["mean_total_time"]]
    ses = [results["conjunctive"]["se_total_time"], results["disjunctive"]["se_total_time"]]
    ax.bar(x - width / 2, means, width, yerr=ses, capsize=4, color=[colors["conjunctive"], colors["disjunctive"]], edgecolor="#333", linewidth=0.8)
    for i, (m, se) in enumerate(zip(means, ses)):
        ax.annotate(f"{m:.1f} $\\pm$ {se:.1f}", xy=(x[i] - width / 2 + width / 2, m + se), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(rule_labels)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Total test time")
    ax.set_ylim(0, (max(means) + max(ses) * 2.5) * 1.12 if ses else max(means) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel D: Time per test (s)
    ax = axes[1, 1]
    means = [results["conjunctive"]["mean_time_per_test"], results["disjunctive"]["mean_time_per_test"]]
    ses = [results["conjunctive"]["se_time_per_test"], results["disjunctive"]["se_time_per_test"]]
    ax.bar(x - width / 2, means, width, yerr=ses, capsize=4, color=[colors["conjunctive"], colors["disjunctive"]], edgecolor="#333", linewidth=0.8)
    for i, (m, se) in enumerate(zip(means, ses)):
        ax.annotate(f"{m:.2f} $\\pm$ {se:.2f}", xy=(x[i] - width / 2 + width / 2, m + se), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(rule_labels)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Time per test")
    ax.set_ylim(0, (max(means) + (max(ses) * 2.5 if ses else max(means) * 0.15)) * 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
