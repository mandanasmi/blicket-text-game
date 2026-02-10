"""
Single entry point for info-gain and hypothesis-remaining analysis and plots.

Usage:
  python run_infogain_hypothesis.py <task> [task_options...]
  python run_infogain_hypothesis.py --list

Tasks:
  resolution_vs_exit   Build hypothesis_resolution_vs_exit.csv + scatter + stacked + stats (from JSON)
  info_gain_evolution  Plot info_gain_evolution_by_rule.png from comprehensive_correlation_data.csv
  hypotheses_remaining Plot conjunctive_hypotheses_remaining.png (N hypotheses vs steps)
  combined            Plot hypotheses_and_info_gain_combined.png (3 panels)
  exit_categories      Plot exit_categories_by_rule.png (requires resolution_vs_exit CSV)
  exit_categories_scatter  Plot exit_categories_and_scatter_combined.png
  tests_vs_resolution Plot tests_vs_resolution.png (bar: mean +/- SE)
  elimination_style    Plot hypothesis_elimination_style.png from main_game CSV

Run from infogain_hypothesis/ so ../ paths resolve to round7/. Inputs from parent by default.
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]

NUM_OBJECTS = 4
MAX_TESTS = 16
CONJ_COLOR = "#2a9d8f"
DISJ_COLOR = "#e76f51"

# Exit categories (shared by exit_categories and exit_categories_scatter)
EXIT_RULE_LABELS = ["Conjunctive", "Disjunctive"]
EXIT_CATEGORIES = ["never_reached", "at_resolution", "at_max", "exited_later"]
EXIT_CATEGORY_LABELS = ["N>1 at exit", "Exited at N=1", "Hit max (16 tests)", "N=1, exited after"]
EXIT_CATEGORY_COLORS = ["#94a3b8", "#34d399", "#f97316", "#a78bfa"]


def all_subsets(n):
    out = []
    for b in range(0, 1 << n):
        out.append(frozenset(i for i in range(n) if (b >> i) & 1))
    return out


def all_full_hypotheses(n):
    subsets = all_subsets(n)
    out = []
    for s in subsets:
        out.append((s, "conjunctive"))
        out.append((s, "disjunctive"))
    return out


def consistent_with_test_full(hyp, objects_on, machine_lit):
    s, rule = hyp
    o = set(objects_on) if isinstance(objects_on, (list, tuple)) else set()
    if rule == "conjunctive":
        pred_on = s <= o
    else:
        pred_on = bool(s & o)
    return pred_on == bool(machine_lit)


def trajectory_full(state_history, hyps_full):
    out = []
    remaining = set(hyps_full)
    for ent in state_history:
        ob = ent.get("objects_on_machine")
        lit = ent.get("machine_lit")
        if ob is None or lit is None:
            continue
        o = list(ob) if isinstance(ob, (list, tuple)) else []
        remaining = {h for h in remaining if consistent_with_test_full(h, o, lit)}
        out.append(len(remaining))
    return out


def _get_true_rule(rd):
    out = rd.get("true_rule") or (rd.get("config") or {}).get("rule") or rd.get("rule")
    return (out or "").strip().lower()


def _parse_rule(s):
    if not s or not isinstance(s, str):
        return None
    s = s.lower()
    if "conjunctive" in s:
        return "conjunctive"
    if "disjunctive" in s or "any" in s:
        return "disjunctive"
    return None


def _has_prior(entry):
    comp = entry.get("comprehension") or {}
    sg = (comp if isinstance(comp, dict) else {}).get("similar_game_experience") or {}
    if not isinstance(sg, dict):
        return None
    a = (sg.get("answer") or "").strip().lower()
    if "yes" in a:
        return True
    if "no" in a:
        return False
    return None


def _get_true_blickets(rd):
    out = rd.get("true_blicket_indices")
    if out is not None:
        return out
    return (rd.get("config") or {}).get("blicket_indices")


def _get_chosen_blickets(rd):
    if "user_chosen_blickets" in rd:
        chosen = [x for x in (rd["user_chosen_blickets"] or []) if x is not None]
        return sorted(chosen) if chosen else None
    c = rd.get("blicket_classifications") or {}
    if not isinstance(c, dict):
        return None
    chosen = []
    for k, v in c.items():
        if v == "Yes":
            try:
                chosen.append(int(k.split("_")[1]))
            except (ValueError, IndexError):
                pass
    return sorted(chosen) if chosen else None


def normalize_rule(s):
    if not s:
        return None
    s = str(s).lower()
    if "conjunctive" in s or "all" in s:
        return "conjunctive"
    if "disjunctive" in s or "any" in s:
        return "disjunctive"
    return None


def load_hypotheses_remaining(json_path, no_prior_only=True, max_per_rule=60, seed=42):
    """Load N hypotheses remaining trajectories (with valid state_history)."""
    with open(json_path, "r") as f:
        data = json.load(f)
    hyps_full = all_full_hypotheses(NUM_OBJECTS)
    traj_conj = []
    traj_disj = []
    for _pid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        if no_prior_only and _has_prior(entry) is True:
            continue
        mg = entry.get("main_game") or {}
        if not isinstance(mg, dict):
            continue
        has_rounds = any(isinstance(k, str) and k.startswith("round_") for k in mg)
        rounds = (
            [(k, mg[k]) for k in mg if isinstance(k, str) and k.startswith("round_") and isinstance(mg.get(k), dict)]
            if has_rounds
            else [("main", mg)]
        )
        for _rk, rd in rounds:
            rule = _get_true_rule(rd)
            if not rule or _parse_rule(rule) is None:
                continue
            sh = rd.get("state_history") or []
            if not isinstance(sh, list) or len(sh) < 1:
                continue
            conj = rule == "conjunctive"
            traj = trajectory_full(sh, hyps_full)
            if traj:
                traj_with_init = [len(hyps_full)] + traj
                if conj:
                    traj_conj.append(traj_with_init)
                else:
                    traj_disj.append(traj_with_init)
            break
    rng = np.random.default_rng(seed)
    if len(traj_conj) > max_per_rule:
        idx = rng.choice(len(traj_conj), size=max_per_rule, replace=False)
        traj_conj = [traj_conj[i] for i in np.sort(idx)]
    if len(traj_disj) > max_per_rule:
        idx = rng.choice(len(traj_disj), size=max_per_rule, replace=False)
        traj_disj = [traj_disj[i] for i in np.sort(idx)]
    return traj_conj, traj_disj


def load_info_gain_data(csv_path):
    """Load info gain per test from comprehensive_correlation_data."""
    df = pd.read_csv(csv_path)
    if "info_gain_per_test" not in df.columns:
        raise SystemExit("Input CSV must have 'info_gain_per_test' column")
    conj_ig_by_test = {t: [] for t in range(1, MAX_TESTS + 1)}
    disj_ig_by_test = {t: [] for t in range(1, MAX_TESTS + 1)}
    for _, row in df.iterrows():
        rule = str(row.get("true_rule", "")).strip().lower()
        ig_str = row.get("info_gain_per_test", "[]")
        try:
            ig_list = json.loads(ig_str) if isinstance(ig_str, str) else ig_str
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(ig_list, list):
            continue
        target = conj_ig_by_test if "conjunctive" in rule else disj_ig_by_test if "disjunctive" in rule else None
        if target is None:
            continue
        for i, ig in enumerate(ig_list):
            t = i + 1
            if t <= MAX_TESTS and isinstance(ig, (int, float)):
                target[t].append(float(ig))
    test_positions = list(range(1, MAX_TESTS + 1))
    conj_means, conj_sems, disj_means, disj_sems = [], [], [], []
    conj_cum, conj_cum_sem, disj_cum, disj_cum_sem = [], [], [], []
    for t in test_positions:
        c_vals = conj_ig_by_test[t]
        d_vals = disj_ig_by_test[t]
        conj_means.append(np.mean(c_vals) if c_vals else np.nan)
        conj_sems.append(np.std(c_vals, ddof=1) / np.sqrt(len(c_vals)) if len(c_vals) > 1 else 0.0)
        disj_means.append(np.mean(d_vals) if d_vals else np.nan)
        disj_sems.append(np.std(d_vals, ddof=1) / np.sqrt(len(d_vals)) if len(d_vals) > 1 else 0.0)
    for t in test_positions:
        c_cum_list = []
        for _, row in df.iterrows():
            rule = str(row.get("true_rule", "")).strip().lower()
            if "conjunctive" not in rule:
                continue
            ig_str = row.get("info_gain_per_test", "[]")
            try:
                ig_list = json.loads(ig_str) if isinstance(ig_str, str) else ig_str
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(ig_list, list) or len(ig_list) < t:
                continue
            c_cum_list.append(sum(float(ig_list[i]) for i in range(t)))
        d_cum_list = []
        for _, row in df.iterrows():
            rule = str(row.get("true_rule", "")).strip().lower()
            if "disjunctive" not in rule:
                continue
            ig_str = row.get("info_gain_per_test", "[]")
            try:
                ig_list = json.loads(ig_str) if isinstance(ig_str, str) else ig_str
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(ig_list, list) or len(ig_list) < t:
                continue
            d_cum_list.append(sum(float(ig_list[i]) for i in range(t)))
        conj_cum.append(np.mean(c_cum_list) if c_cum_list else np.nan)
        conj_cum_sem.append(np.std(c_cum_list, ddof=1) / np.sqrt(len(c_cum_list)) if len(c_cum_list) > 1 else 0.0)
        disj_cum.append(np.mean(d_cum_list) if d_cum_list else np.nan)
        disj_cum_sem.append(np.std(d_cum_list, ddof=1) / np.sqrt(len(d_cum_list)) if len(d_cum_list) > 1 else 0.0)
    return {
        "test_positions": test_positions,
        "conj_means": conj_means, "conj_sems": conj_sems,
        "disj_means": disj_means, "disj_sems": disj_sems,
        "conj_cum": conj_cum, "conj_cum_sem": conj_cum_sem,
        "disj_cum": disj_cum, "disj_cum_sem": disj_cum_sem,
        "n_conj": len(conj_ig_by_test[1]) if conj_ig_by_test[1] else 0,
        "n_disj": len(disj_ig_by_test[1]) if disj_ig_by_test[1] else 0,
    }


def exit_categorize(row):
    step = row["step"]
    num_tests = row["num_tests"]
    if pd.isna(step) or step == "":
        return "never_reached"
    if num_tests == step:
        return "at_resolution"
    if num_tests >= MAX_TESTS:
        return "at_max"
    if num_tests > step:
        return "exited_later"
    return "exited_early"


def draw_exit_categories_ax(df, ax, title, show_ylabel=True, width=0.52, annot_min_p=0.005):
    n_conj = len(df[df["true_rule"] == "conjunctive"])
    n_disj = len(df[df["true_rule"] == "disjunctive"])
    n_totals = [n_conj, n_disj]
    counts = {cat: [] for cat in EXIT_CATEGORIES}
    for rt in ["conjunctive", "disjunctive"]:
        sub = df[df["true_rule"] == rt]
        for cat in EXIT_CATEGORIES:
            counts[cat].append(len(sub[sub["category"] == cat]))
    props = {cat: [counts[cat][i] / max(1, n_totals[i]) for i in range(2)] for cat in EXIT_CATEGORIES}
    bottoms = {}
    b = np.array([0.0, 0.0])
    for cat in EXIT_CATEGORIES:
        bottoms[cat] = b.copy()
        b = b + np.array(props[cat])
    x = np.arange(len(EXIT_RULE_LABELS))
    for i, (cat, label) in enumerate(zip(EXIT_CATEGORIES, EXIT_CATEGORY_LABELS)):
        ax.bar(x, props[cat], width, bottom=bottoms[cat], label=label, color=EXIT_CATEGORY_COLORS[i], edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(EXIT_RULE_LABELS)
    if show_ylabel:
        ax.set_ylabel("Proportion of participants (out of 1)")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="both", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for cat, label in zip(EXIT_CATEGORIES, EXIT_CATEGORY_LABELS):
        for i in range(2):
            cnt = counts[cat][i]
            p = props[cat][i]
            if p <= annot_min_p:
                continue
            y_center = bottoms[cat][i] + p / 2
            prop_val = cnt / max(1, n_totals[i])
            if p < 0.05:
                text = f"{cnt}/{n_totals[i]} ({prop_val:.2f})"
                fsize = 5
            elif p < 0.08:
                text = f"{cnt}/{n_totals[i]} ({prop_val:.2f})"
                fsize = 6
            else:
                text = f"{cnt}/{n_totals[i]}\n({prop_val:.2f})"
                fsize = 7 if p < 0.15 else 8
            ax.annotate(text, xy=(x[i], y_center), xytext=(0, 0), textcoords="offset points",
                        ha="center", va="center", fontsize=fsize, color="#000")


def cmd_info_gain_evolution(args):
    df = pd.read_csv(args.input)
    if "info_gain_per_test" not in df.columns:
        raise SystemExit("Input CSV must have 'info_gain_per_test' column")
    conj_ig_by_test = {t: [] for t in range(1, MAX_TESTS + 1)}
    disj_ig_by_test = {t: [] for t in range(1, MAX_TESTS + 1)}
    for _, row in df.iterrows():
        rule = str(row.get("true_rule", "")).strip().lower()
        ig_str = row.get("info_gain_per_test", "[]")
        try:
            ig_list = json.loads(ig_str) if isinstance(ig_str, str) else ig_str
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(ig_list, list):
            continue
        target = conj_ig_by_test if "conjunctive" in rule else disj_ig_by_test if "disjunctive" in rule else None
        if target is None:
            continue
        for i, ig in enumerate(ig_list):
            t = i + 1
            if t <= MAX_TESTS and isinstance(ig, (int, float)):
                target[t].append(float(ig))
    test_positions = list(range(1, MAX_TESTS + 1))
    conj_means, conj_sems, disj_means, disj_sems = [], [], [], []
    conj_n, disj_n = [], []
    for t in test_positions:
        c_vals = conj_ig_by_test[t]
        d_vals = disj_ig_by_test[t]
        conj_n.append(len(c_vals))
        disj_n.append(len(d_vals))
        conj_means.append(np.mean(c_vals) if c_vals else np.nan)
        conj_sems.append(np.std(c_vals, ddof=1) / np.sqrt(len(c_vals)) if len(c_vals) > 1 else 0.0)
        disj_means.append(np.mean(d_vals) if d_vals else np.nan)
        disj_sems.append(np.std(d_vals, ddof=1) / np.sqrt(len(d_vals)) if len(d_vals) > 1 else 0.0)
    conj_cum, disj_cum = [], []
    conj_cum_sem, disj_cum_sem = [], []
    for t in test_positions:
        c_cum_list = []
        for _, row in df.iterrows():
            rule = str(row.get("true_rule", "")).strip().lower()
            if "conjunctive" not in rule:
                continue
            ig_str = row.get("info_gain_per_test", "[]")
            try:
                ig_list = json.loads(ig_str) if isinstance(ig_str, str) else ig_str
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(ig_list, list) or len(ig_list) < t:
                continue
            c_cum_list.append(sum(float(ig_list[i]) for i in range(t)))
        d_cum_list = []
        for _, row in df.iterrows():
            rule = str(row.get("true_rule", "")).strip().lower()
            if "disjunctive" not in rule:
                continue
            ig_str = row.get("info_gain_per_test", "[]")
            try:
                ig_list = json.loads(ig_str) if isinstance(ig_str, str) else ig_str
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(ig_list, list) or len(ig_list) < t:
                continue
            d_cum_list.append(sum(float(ig_list[i]) for i in range(t)))
        conj_cum.append(np.mean(c_cum_list) if c_cum_list else np.nan)
        conj_cum_sem.append(np.std(c_cum_list, ddof=1) / np.sqrt(len(c_cum_list)) if len(c_cum_list) > 1 else 0.0)
        disj_cum.append(np.mean(d_cum_list) if d_cum_list else np.nan)
        disj_cum_sem.append(np.std(d_cum_list, ddof=1) / np.sqrt(len(d_cum_list)) if len(d_cum_list) > 1 else 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.array(test_positions)
    ax = axes[0]
    ax.plot(x, conj_means, "o-", color=CONJ_COLOR, label="Conjunctive", linewidth=2, markersize=5)
    ax.fill_between(x, np.array(conj_means) - np.array(conj_sems), np.array(conj_means) + np.array(conj_sems), color=CONJ_COLOR, alpha=0.25)
    ax.plot(x, disj_means, "s-", color=DISJ_COLOR, label="Disjunctive", linewidth=2, markersize=5)
    ax.fill_between(x, np.array(disj_means) - np.array(disj_sems), np.array(disj_means) + np.array(disj_sems), color=DISJ_COLOR, alpha=0.25)
    ax.set_xlabel("Test number")
    ax.set_ylabel("Information gain (bits)")
    ax.set_title("Mean IG per test")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0.5, MAX_TESTS + 0.5)
    ax.set_xticks(test_positions)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    ax.plot(x, conj_cum, "o-", color=CONJ_COLOR, label="Conjunctive", linewidth=2, markersize=5)
    ax.fill_between(x, np.array(conj_cum) - np.array(conj_cum_sem), np.array(conj_cum) + np.array(conj_cum_sem), color=CONJ_COLOR, alpha=0.25)
    ax.plot(x, disj_cum, "s-", color=DISJ_COLOR, label="Disjunctive", linewidth=2, markersize=5)
    ax.fill_between(x, np.array(disj_cum) - np.array(disj_cum_sem), np.array(disj_cum) + np.array(disj_cum_sem), color=DISJ_COLOR, alpha=0.25)
    ax.set_xlabel("Test number")
    ax.set_ylabel("Cumulative information gain (bits)")
    ax.set_title("Mean cumulative IG up to test k")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0.5, MAX_TESTS + 0.5)
    ax.set_xticks(test_positions)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {args.output}")


def cmd_hypotheses_remaining(args):
    with open(args.input, "r") as f:
        data = json.load(f)
    hyps_full = all_full_hypotheses(NUM_OBJECTS)
    traj_conj = []
    traj_disj = []
    for _pid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        if not args.no_prior_only and _has_prior(entry) is True:
            continue
        mg = entry.get("main_game") or {}
        if not isinstance(mg, dict):
            continue
        has_rounds = any(isinstance(k, str) and k.startswith("round_") for k in mg)
        rounds = (
            [(k, mg[k]) for k in mg if isinstance(k, str) and k.startswith("round_") and isinstance(mg.get(k), dict)]
            if has_rounds
            else [("main", mg)]
        )
        for _rk, rd in rounds:
            rule = _get_true_rule(rd)
            true_b = _get_true_blickets(rd)
            user_b = _get_chosen_blickets(rd)
            obj_ok = true_b is not None and user_b is not None and set(true_b) == set(user_b)
            rule_str = rd.get("rule_type") or ""
            user_r = _parse_rule(rule_str)
            rule_ok = user_r and rule and user_r == rule
            if not (obj_ok and rule_ok):
                continue
            sh = rd.get("state_history") or []
            if not isinstance(sh, list) or len(sh) < 1:
                continue
            conj = rule == "conjunctive"
            traj = trajectory_full(sh, hyps_full)
            if traj:
                traj_with_init = [len(hyps_full)] + traj
                if conj:
                    traj_conj.append(traj_with_init)
                else:
                    traj_disj.append(traj_with_init)
            break
    if not traj_conj and not traj_disj:
        print("No both-correct participants with state_history.")
        return
    rng = np.random.default_rng(args.seed)
    if len(traj_conj) > args.max_per_rule:
        idx = rng.choice(len(traj_conj), size=args.max_per_rule, replace=False)
        traj_conj = [traj_conj[i] for i in np.sort(idx)]
    if len(traj_disj) > args.max_per_rule:
        idx = rng.choice(len(traj_disj), size=args.max_per_rule, replace=False)
        traj_disj = [traj_disj[i] for i in np.sort(idx)]
    all_traj = traj_conj + traj_disj
    max_len = max(len(t) for t in all_traj)
    steps = np.arange(0, max_len, dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(3.8, 2.4))
    y_max = 0
    for trajectories, label, color in [(traj_conj, "Conjunctive", CONJ_COLOR), (traj_disj, "Disjunctive", DISJ_COLOR)]:
        if not trajectories:
            continue
        mean_n = []
        se_n = []
        for t in range(max_len):
            vals = [tr[t] for tr in trajectories if len(tr) > t]
            mean_n.append(np.mean(vals) if vals else np.nan)
            se_n.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        mean_n = np.array(mean_n)
        se_n = np.array(se_n)
        valid = np.isfinite(mean_n)
        if not np.any(valid):
            continue
        y_max = max(y_max, np.nanmax(mean_n + se_n))
        ax.fill_between(steps, mean_n - se_n, mean_n + se_n, alpha=0.25, color=color)
        ax.plot(steps, mean_n, color=color, linewidth=2, marker="o", markersize=3, markevery=max(1, max_len // 8), label=label)
    ax.axhline(1, color="#555", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Number of tests", fontsize=8)
    ax.set_ylabel("N hypotheses remaining", fontsize=8)
    ax.set_xlim(-0.5, max_len - 0.5)
    ax.set_ylim(0, max(1, y_max + 1.2))
    ax.legend(loc="upper right", fontsize=7, framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved: {args.output} (conjunctive: {len(traj_conj)}, disjunctive: {len(traj_disj)})")


def cmd_resolution_vs_exit(args):
    with open(args.input, "r") as f:
        data = json.load(f)
    hyps_full = all_full_hypotheses(NUM_OBJECTS)
    rows = []
    for user_id, entry in data.items():
        if not isinstance(entry, dict):
            continue
        mg = entry.get("main_game") or {}
        if not mg:
            continue
        config_rule = (mg.get("config") or {}).get("rule") or mg.get("rule") or mg.get("true_rule")
        true_rule = normalize_rule(config_rule)
        if true_rule not in ("conjunctive", "disjunctive"):
            continue
        sh = mg.get("state_history") or []
        if not isinstance(sh, list) or len(sh) < 1:
            continue
        num_tests = len(sh)
        traj = trajectory_full(sh, hyps_full)
        step_to_one = None
        for i, n in enumerate(traj):
            if n == 1:
                step_to_one = i + 1
                break
        never_reached_one = step_to_one is None
        exited_early = step_to_one is not None and num_tests < step_to_one
        exited_exact = step_to_one is not None and num_tests == step_to_one
        exited_late = step_to_one is not None and num_tests > step_to_one
        true_b = set(mg.get("true_blicket_indices", []))
        user_b = set(mg.get("user_chosen_blickets", []))
        obj_ok = true_b == user_b
        user_choice = normalize_rule(mg.get("rule_type", ""))
        rule_ok = user_choice == true_rule if user_choice else False
        both_ok = obj_ok and rule_ok
        rows.append({
            "user_id": user_id, "true_rule": true_rule, "num_tests": num_tests,
            "step_when_n_reaches_1": step_to_one if step_to_one is not None else "",
            "never_reached_one": never_reached_one, "exited_early": exited_early,
            "exited_exact": exited_exact, "exited_late": exited_late, "both_correct": both_ok,
        })
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "true_rule", "num_tests", "step_when_n_reaches_1",
                                          "never_reached_one", "exited_early", "exited_exact", "exited_late", "both_correct"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {args.output_csv} ({len(rows)} participants)")
    stats_path = os.path.join(os.path.dirname(args.output_csv) or ".", "hypothesis_resolution_vs_exit_stats.txt")
    sub_c = [r for r in rows if r["true_rule"] == "conjunctive"]
    sub_d = [r for r in rows if r["true_rule"] == "disjunctive"]
    n_c, n_d = len(sub_c), len(sub_d)
    never_c = sum(1 for r in sub_c if r["never_reached_one"])
    never_d = sum(1 for r in sub_d if r["never_reached_one"])
    early_c = sum(1 for r in sub_c if r["exited_early"])
    early_d = sum(1 for r in sub_d if r["exited_early"])
    exact_c = sum(1 for r in sub_c if r["exited_exact"])
    exact_d = sum(1 for r in sub_d if r["exited_exact"])
    late_c = sum(1 for r in sub_c if r["exited_late"])
    late_d = sum(1 for r in sub_d if r["exited_late"])
    with open(stats_path, "w") as f:
        f.write("Exit vs Resolution Stats (hypothesis_resolution_vs_exit)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Category':<30} {'Conjunctive':<20} {'Disjunctive':<20}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Never reached N=1':<30} {never_c}/{n_c} ({100*never_c/n_c:.1f}%){'':<8} {never_d}/{n_d} ({100*never_d/n_d:.1f}%){'':<8}\n")
        f.write(f"{'Exited early':<30} {early_c}/{n_c} ({100*early_c/n_c:.1f}%){'':<8} {early_d}/{n_d} ({100*early_d/n_d:.1f}%){'':<8}\n")
        f.write(f"{'Exited at resolution':<30} {exact_c}/{n_c} ({100*exact_c/n_c:.1f}%){'':<8} {exact_d}/{n_d} ({100*exact_d/n_d:.1f}%){'':<8}\n")
        f.write(f"{'Exited late':<30} {late_c}/{n_c} ({100*late_c/n_c:.1f}%){'':<8} {late_d}/{n_d} ({100*late_d/n_d:.1f}%){'':<8}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'N participants':<30} {n_c:<20} {n_d:<20}\n")
    print(f"Saved: {stats_path}")
    base = args.output_png.replace(".png", "")
    reached = [r for r in rows if r["step_when_n_reaches_1"] != ""]
    agg = defaultdict(lambda: {"conjunctive": 0, "disjunctive": 0})
    for r in reached:
        key = (r["step_when_n_reaches_1"], r["num_tests"])
        agg[key][r["true_rule"]] += 1
    fig1, ax = plt.subplots(1, 1, figsize=(4, 4))
    jitter = 0.12
    for (sx, ny), counts in agg.items():
        n_conj = counts["conjunctive"]
        n_disj = counts["disjunctive"]
        if n_conj > 0:
            size = 80 + n_conj * 55
            x_pos = sx - jitter if n_disj > 0 else sx
            y_pos = ny - jitter if n_disj > 0 else ny
            ax.scatter([x_pos], [y_pos], c="#3B82F6", alpha=0.75, edgecolor="#333", linewidth=0.6, s=size, zorder=2)
            if n_conj > 1:
                ax.annotate(str(n_conj), xy=(x_pos, y_pos), xytext=(0, 0), textcoords="offset points", ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=3)
        if n_disj > 0:
            size = 80 + n_disj * 55
            x_pos = sx + jitter if n_conj > 0 else sx
            y_pos = ny + jitter if n_conj > 0 else ny
            ax.scatter([x_pos], [y_pos], c="#EA580C", alpha=0.75, edgecolor="#333", linewidth=0.6, s=size, zorder=2)
            if n_disj > 1:
                ax.annotate(str(n_disj), xy=(x_pos, y_pos), xytext=(0, 0), textcoords="offset points", ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=3)
    step_vals = [r["step_when_n_reaches_1"] for r in reached]
    num_vals = [r["num_tests"] for r in reached]
    max_val = max(max(step_vals) if step_vals else 0, max(num_vals) if num_vals else 0)
    ax.plot([0, max_val + 1], [0, max_val + 1], "k--", linewidth=1, alpha=0.7, label="Exit = resolution")
    ax.set_xlabel("Number of tests when N hypotheses = 1", fontsize=8)
    ax.set_ylabel("Number of tests when human exited", fontsize=8)
    ax.set_xlim(0, max_val + 1.5)
    ax.set_ylim(0, max_val + 1.5)
    ax.set_xticks(list(range(0, int(max_val) + 2, 2)))
    ax.set_yticks(list(range(0, int(max_val) + 2, 2)))
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Patch(facecolor="#3B82F6", label="Conjunctive"),
        Patch(facecolor="#EA580C", label="Disjunctive"),
        Line2D([0], [0], color="k", linestyle="--", linewidth=1, label="Exit = resolution"),
    ], fontsize=6, title="Size = count (N shown if N>1)", title_fontsize=7)
    plt.tight_layout(pad=0.2)
    fig1.savefig(f"{base}_scatter.png", dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig1)
    print(f"Saved: {base}_scatter.png")
    n_conj = sum(1 for r in rows if r["true_rule"] == "conjunctive")
    n_disj = sum(1 for r in rows if r["true_rule"] == "disjunctive")
    never_counts = [sum(1 for r in rows if r["true_rule"] == rt and r["never_reached_one"]) for rt in ["conjunctive", "disjunctive"]]
    early_counts = [sum(1 for r in rows if r["true_rule"] == rt and r["exited_early"]) for rt in ["conjunctive", "disjunctive"]]
    exact_counts = [sum(1 for r in rows if r["true_rule"] == rt and r["exited_exact"]) for rt in ["conjunctive", "disjunctive"]]
    late_counts = [sum(1 for r in rows if r["true_rule"] == rt and r["exited_late"]) for rt in ["conjunctive", "disjunctive"]]
    n_totals = [n_conj, n_disj]
    never_prop = [never_counts[i] / max(1, n_totals[i]) for i in range(2)]
    early_prop = [early_counts[i] / max(1, n_totals[i]) for i in range(2)]
    exact_prop = [exact_counts[i] / max(1, n_totals[i]) for i in range(2)]
    late_prop = [late_counts[i] / max(1, n_totals[i]) for i in range(2)]
    fig2, ax = plt.subplots(1, 1, figsize=(6, 5))
    x = np.arange(2)
    width = 0.65
    bottom_never = np.array(never_prop)
    bottom_early = bottom_never + np.array(early_prop)
    bottom_exact = bottom_early + np.array(exact_prop)
    ax.bar(x, never_prop, width, label="Never reached N=1", color="#94a3b8", edgecolor="none")
    ax.bar(x, early_prop, width, bottom=bottom_never, label="Exited early", color="#fbbf24", edgecolor="none")
    ax.bar(x, exact_prop, width, bottom=bottom_early, label="Exited at resolution", color="#34d399", edgecolor="none")
    ax.bar(x, late_prop, width, bottom=bottom_exact, label="Exited late", color="#a78bfa", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(EXIT_RULE_LABELS)
    ax.set_ylabel("Proportion of participants (out of 1)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, fontsize=5, frameon=True)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for props, bottoms, counts in [
        (never_prop, [0] * 2, never_counts),
        (early_prop, bottom_never, early_counts),
        (exact_prop, bottom_early, exact_counts),
        (late_prop, bottom_exact, late_counts),
    ]:
        for i in range(2):
            if props[i] > 0.03:
                prop_val = counts[i] / max(1, n_totals[i])
                y_center = bottoms[i] + props[i] / 2
                label = f"{counts[i]}/{n_totals[i]}\n({prop_val:.2f})"
                fsize = 7 if props[i] < 0.1 else 8
                ax.annotate(label, xy=(x[i], y_center), xytext=(0, 0), textcoords="offset points", ha="center", va="center", fontsize=fsize, color="#000")
    plt.tight_layout()
    fig2.savefig(f"{base}_stacked.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig2)
    print(f"Saved: {base}_stacked.png")


def cmd_combined(args):
    traj_conj, traj_disj = load_hypotheses_remaining(
        args.json, no_prior_only=args.prior_filter, max_per_rule=args.max_per_rule, seed=args.seed
    )
    ig_data = load_info_gain_data(args.csv)
    all_traj = traj_conj + traj_disj
    max_len = max(len(t) for t in all_traj) if all_traj else MAX_TESTS + 1
    steps = np.arange(0, max_len, dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(6, 10))
    ax = axes[0]
    y_max = 0
    for trajectories, label, color in [(traj_conj, "Conjunctive", CONJ_COLOR), (traj_disj, "Disjunctive", DISJ_COLOR)]:
        if not trajectories:
            continue
        mean_n = []
        se_n = []
        for t in range(max_len):
            vals = [tr[t] for tr in trajectories if len(tr) > t]
            mean_n.append(np.mean(vals) if vals else np.nan)
            se_n.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        mean_n = np.array(mean_n)
        se_n = np.array(se_n)
        if np.any(np.isfinite(mean_n)):
            y_max = max(y_max, np.nanmax(mean_n + se_n))
            ax.fill_between(steps, mean_n - se_n, mean_n + se_n, alpha=0.25, color=color)
            ax.plot(steps, mean_n, color=color, linewidth=2, marker="o", markersize=3, markevery=max(1, max_len // 8), label=label)
    ax.axhline(1, color="#555", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Number of tests")
    ax.set_ylabel("N hypotheses remaining")
    ax.set_title("a. Hypotheses remaining", fontsize=10)
    ax.set_xlim(-0.5, max_len - 0.5)
    ax.set_ylim(0, max(1, y_max + 1.2))
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax = axes[1]
    x = np.array(ig_data["test_positions"])
    ax.plot(x, ig_data["conj_cum"], "o-", color=CONJ_COLOR, label="Conjunctive", linewidth=2, markersize=4)
    ax.fill_between(x, np.array(ig_data["conj_cum"]) - np.array(ig_data["conj_cum_sem"]), np.array(ig_data["conj_cum"]) + np.array(ig_data["conj_cum_sem"]), color=CONJ_COLOR, alpha=0.25)
    ax.plot(x, ig_data["disj_cum"], "s-", color=DISJ_COLOR, label="Disjunctive", linewidth=2, markersize=4)
    ax.fill_between(x, np.array(ig_data["disj_cum"]) - np.array(ig_data["disj_cum_sem"]), np.array(ig_data["disj_cum"]) + np.array(ig_data["disj_cum_sem"]), color=DISJ_COLOR, alpha=0.25)
    ax.set_xlabel("Test number")
    ax.set_ylabel("Cumulative information gain (bits)")
    ax.set_title("b. Mean cumulative information gain", fontsize=10)
    ax.set_xlim(0.5, MAX_TESTS + 0.5)
    ax.set_xticks(ig_data["test_positions"])
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax = axes[2]
    ax.plot(x, ig_data["conj_means"], "o-", color=CONJ_COLOR, label="Conjunctive", linewidth=2, markersize=4)
    ax.fill_between(x, np.array(ig_data["conj_means"]) - np.array(ig_data["conj_sems"]), np.array(ig_data["conj_means"]) + np.array(ig_data["conj_sems"]), color=CONJ_COLOR, alpha=0.25)
    ax.plot(x, ig_data["disj_means"], "s-", color=DISJ_COLOR, label="Disjunctive", linewidth=2, markersize=4)
    ax.fill_between(x, np.array(ig_data["disj_means"]) - np.array(ig_data["disj_sems"]), np.array(ig_data["disj_means"]) + np.array(ig_data["disj_sems"]), color=DISJ_COLOR, alpha=0.25)
    ax.set_xlabel("Test number")
    ax.set_ylabel("Information gain (bits)")
    ax.set_title("c. Mean information gain per test", fontsize=10)
    ax.set_xlim(0.5, MAX_TESTS + 0.5)
    ax.set_xticks(ig_data["test_positions"])
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {args.output}")


def cmd_exit_categories(args):
    df = pd.read_csv(args.input)
    df["step"] = pd.to_numeric(df["step_when_n_reaches_1"], errors="coerce")
    df["category"] = df.apply(exit_categorize, axis=1)
    both_ok = df["both_correct"].astype(str).str.lower() == "true"
    df_success = df[both_ok].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 7.5))
    draw_exit_categories_ax(df, axes[0], "All participants", show_ylabel=True)
    draw_exit_categories_ax(df_success, axes[1], "Successful participants (both rule and objects correct)", show_ylabel=False)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=7, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {args.output}")


def cmd_exit_categories_scatter(args):
    df = pd.read_csv(args.input)
    df["step"] = pd.to_numeric(df["step_when_n_reaches_1"], errors="coerce")
    df["category"] = df.apply(exit_categorize, axis=1)
    both_ok = df["both_correct"].astype(str).str.lower() == "true"
    df_success = df[both_ok].copy()
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    reached = df[df["step"].notna()].copy()
    ax = axes[0]
    if len(reached) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        agg = defaultdict(lambda: {"conjunctive": 0, "disjunctive": 0})
        for _, r in reached.iterrows():
            key = (int(r["step"]), int(r["num_tests"]))
            agg[key][r["true_rule"]] += 1
        jitter = 0.12
        for (sx, ny), counts in agg.items():
            n_conj = counts["conjunctive"]
            n_disj = counts["disjunctive"]
            if n_conj > 0:
                size = 80 + n_conj * 55
                x_pos = sx - jitter if n_disj > 0 else sx
                y_pos = ny - jitter if n_disj > 0 else ny
                ax.scatter([x_pos], [y_pos], c="#3B82F6", alpha=0.75, edgecolor="#333", linewidth=0.6, s=size, zorder=2)
                if n_conj > 1:
                    ax.annotate(str(n_conj), xy=(x_pos, y_pos), xytext=(0, 0), textcoords="offset points", ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=3)
            if n_disj > 0:
                size = 80 + n_disj * 55
                x_pos = sx + jitter if n_conj > 0 else sx
                y_pos = ny + jitter if n_conj > 0 else ny
                ax.scatter([x_pos], [y_pos], c="#EA580C", alpha=0.75, edgecolor="#333", linewidth=0.6, s=size, zorder=2)
                if n_disj > 1:
                    ax.annotate(str(n_disj), xy=(x_pos, y_pos), xytext=(0, 0), textcoords="offset points", ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=3)
        step_vals = reached["step"].tolist()
        num_vals = reached["num_tests"].tolist()
        max_val = max(max(step_vals), max(num_vals))
        ax.plot([0, max_val + 1], [0, max_val + 1], "k--", linewidth=1, alpha=0.7, label="Exit = N=1")
        ax.set_xlabel("Number of tests when N hypotheses = 1", fontsize=9)
        ax.set_ylabel("Number of tests when human exited", fontsize=9)
        ax.set_xlim(0, max_val + 1.5)
        ax.set_ylim(0, max_val + 1.5)
        tick_vals = list(range(0, int(max_val) + 2, 2))
        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Patch(facecolor="#3B82F6", label="Conjunctive"),
            Patch(facecolor="#EA580C", label="Disjunctive"),
            Line2D([0], [0], color="k", linestyle="--", linewidth=1, label="Exit = N=1"),
        ], fontsize=6, title="Size = count (N shown if N>1)", title_fontsize=6)
    draw_exit_categories_ax(df, axes[1], "All participants", show_ylabel=True, width=0.65)
    draw_exit_categories_ax(df_success, axes[2], "Successful participants (both rule and objects correct)", show_ylabel=False, width=0.65)
    handles, leg_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", bbox_to_anchor=(0.65, 1.02), ncol=2, fontsize=6, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {args.output}")


def cmd_tests_vs_resolution(args):
    def mean_se(arr):
        arr = np.array(arr)
        n = len(arr)
        if n == 0:
            return np.nan, np.nan
        m = np.mean(arr)
        se = np.std(arr, ddof=1) / np.sqrt(n) if n > 1 else 0.0
        return m, se
    df = pd.read_csv(args.input)
    reached = df[df["step_when_n_reaches_1"].notna() & (df["step_when_n_reaches_1"] != "")].copy()
    reached["step"] = pd.to_numeric(reached["step_when_n_reaches_1"], errors="coerce")
    reached_c = reached[reached["true_rule"] == "conjunctive"]
    reached_d = reached[reached["true_rule"] == "disjunctive"]
    m_num_c, se_num_c = mean_se(reached_c["num_tests"].tolist())
    m_step_c, se_step_c = mean_se(reached_c["step"].tolist())
    m_num_d, se_num_d = mean_se(reached_d["num_tests"].tolist())
    m_step_d, se_step_d = mean_se(reached_d["step"].tolist())
    x = np.arange(2)
    width = 0.35
    colors = {"conjunctive": "#3B82F6", "disjunctive": "#EA580C"}
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    bars1 = ax.bar(x - width / 2, [m_step_c, m_step_d], width, yerr=[se_step_c, se_step_d], capsize=4, label="Test when N=1", color=[colors["conjunctive"], colors["disjunctive"]], edgecolor="#333", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, [m_num_c, m_num_d], width, yerr=[se_num_c, se_num_d], capsize=4, label="Tests when exited", color=["#93C5FD", "#FDBA74"], edgecolor="#333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(EXIT_RULE_LABELS)
    ax.set_ylabel("Number of tests")
    ax.legend(loc="upper right", fontsize=8)
    y_max = max(m_num_c + se_num_c, m_num_d + se_num_d, m_step_c + se_step_c, m_step_d + se_step_d) * 1.15
    ax.set_ylim(0, y_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, (m, se) in zip(list(bars1) + list(bars2), [(m_step_c, se_step_c), (m_step_d, se_step_d), (m_num_c, se_num_c), (m_num_d, se_num_d)]):
        h = bar.get_height()
        ax.annotate(f"{m:.2f} +/- {se:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h + se), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {args.output}")


def cmd_elimination_style(args):
    def _bool(s):
        v = s if hasattr(s, "item") else s
        return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")
    df = pd.read_csv(args.input)
    df["obj_ok"] = df["object_identification_correct"].map(_bool)
    df["rule_ok"] = df["rule_choice_correct"].map(_bool)
    df["both_correct"] = df["obj_ok"] & df["rule_ok"]
    df["prop_incorrect"] = 1.0 - df["both_correct"].astype(float)
    bins = [0, 4, 8, 12, 16, 20]
    labels_bin = [2.5, 6.5, 10.5, 14.5, 18.5]
    df["step_bin"] = pd.cut(df["num_tests"], bins=bins, labels=labels_bin, include_lowest=True)
    df["step_bin"] = df["step_bin"].astype(float)
    rule_configs = [("conjunctive", "Rule: conjunctive, Objects: 4"), ("disjunctive", "Rule: disjunctive, Objects: 4")]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for ax, (rule, title) in zip(axes, rule_configs):
        sub = df[df["ground_truth_rule"] == rule]
        if len(sub) > args.n:
            sub = sub.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
        else:
            sub = sub.reset_index(drop=True)
        g = sub.groupby("step_bin", dropna=True).agg(
            mean_incorrect=("prop_incorrect", "mean"),
            se_incorrect=("prop_incorrect", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0),
            n=("prop_incorrect", "count"),
        ).reindex(labels_bin)
        g = g.dropna(subset=["mean_incorrect"]).sort_index()
        if g.empty:
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Steps", fontsize=9)
            ax.set_ylabel("Proportion incorrect (log scale)", fontsize=9)
            continue
        x = g.index.astype(float).values
        mu = g["mean_incorrect"].values
        se = g["se_incorrect"].values
        eps = 0.01
        y = np.maximum(mu + eps, eps)
        y_lo = np.maximum(mu - se + eps, eps * 0.5)
        y_hi = np.minimum(mu + se + eps, 1.0 + eps)
        y_lo, y_hi = np.minimum(y_lo, y_hi), np.maximum(y_lo, y_hi)
        ax.fill_between(x, y_lo, y_hi, alpha=0.25, color=CONJ_COLOR)
        ax.plot(x, y, color=CONJ_COLOR, linewidth=2, label="Participants")
        ax.axhline(eps, color="gray", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_ylim(0.005, 1.5)
        ax.set_xlim(0, 20)
        ax.set_xlabel("Steps", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yticks([0.01, 0.1, 1.0])
        ax.set_yticklabels(["0.01", "0.1", "1.0"])
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Proportion incorrect (log scale)", fontsize=9)
    fig.suptitle("Hypothesis elimination efficiency (50 participants each)", fontsize=11, y=1.01)
    plt.tight_layout(pad=0.25)
    plt.subplots_adjust(wspace=0.25)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Info-gain and hypothesis-remaining analysis and plots")
    parser.add_argument("--list", action="store_true", help="List tasks and exit")
    sub = parser.add_subparsers(dest="task", help="Task to run")
    # resolution_vs_exit
    p = sub.add_parser("resolution_vs_exit", help="Build hypothesis_resolution_vs_exit CSV + scatter + stacked + stats")
    p.add_argument("--input", default="../human_active_data_no_prior_experience.json", help="Input JSON")
    p.add_argument("--output-csv", default="hypothesis_resolution_vs_exit.csv", help="Output CSV")
    p.add_argument("--output-png", default="hypothesis_resolution_vs_exit.png", help="Output PNG base (also writes _scatter.png, _stacked.png)")
    p.set_defaults(func=cmd_resolution_vs_exit)
    # info_gain_evolution
    p = sub.add_parser("info_gain_evolution", help="Plot info_gain_evolution_by_rule.png")
    p.add_argument("--input", default="../correlations/comprehensive_correlation_data.csv", help="Input CSV")
    p.add_argument("--output", default="info_gain_evolution_by_rule.png", help="Output PNG")
    p.set_defaults(func=cmd_info_gain_evolution)
    # hypotheses_remaining
    p = sub.add_parser("hypotheses_remaining", help="Plot conjunctive_hypotheses_remaining.png")
    p.add_argument("--input", default="../human_active_data_no_prior_experience.json", help="Input JSON")
    p.add_argument("--output", default="conjunctive_hypotheses_remaining.png", help="Output PNG")
    p.add_argument("--no-prior-only", action="store_true", help="Include prior-experience participants")
    p.add_argument("--max-per-rule", type=int, default=50, help="Max participants per rule type")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_hypotheses_remaining)
    # combined
    p = sub.add_parser("combined", help="Plot hypotheses_and_info_gain_combined.png")
    p.add_argument("--json", default="../human_active_data_no_prior_experience.json", help="Input JSON")
    p.add_argument("--csv", default="../correlations/comprehensive_correlation_data_no_prior_102.csv", help="Input CSV")
    p.add_argument("--output", default="hypotheses_and_info_gain_combined.png", help="Output PNG")
    p.add_argument("--no-prior-only", action="store_false", dest="prior_filter", default=True, help="Include prior-experience participants")
    p.add_argument("--max-per-rule", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_combined)
    # exit_categories
    p = sub.add_parser("exit_categories", help="Plot exit_categories_by_rule.png")
    p.add_argument("--input", default="hypothesis_resolution_vs_exit.csv", help="Input CSV")
    p.add_argument("--output", default="exit_categories_by_rule.png", help="Output PNG")
    p.set_defaults(func=cmd_exit_categories)
    # exit_categories_scatter
    p = sub.add_parser("exit_categories_scatter", help="Plot exit_categories_and_scatter_combined.png")
    p.add_argument("--input", default="hypothesis_resolution_vs_exit.csv", help="Input CSV")
    p.add_argument("--output", default="exit_categories_and_scatter_combined.png", help="Output PNG")
    p.set_defaults(func=cmd_exit_categories_scatter)
    # tests_vs_resolution
    p = sub.add_parser("tests_vs_resolution", help="Plot tests_vs_resolution.png")
    p.add_argument("--input", default="hypothesis_resolution_vs_exit.csv", help="Input CSV")
    p.add_argument("--output", default="tests_vs_resolution.png", help="Output PNG")
    p.set_defaults(func=cmd_tests_vs_resolution)
    # elimination_style
    p = sub.add_parser("elimination_style", help="Plot hypothesis_elimination_style.png")
    p.add_argument("--input", default="../main_game_data_with_prior_experience.csv", help="Input CSV")
    p.add_argument("--output", default="hypothesis_elimination_style.png", help="Output PNG")
    p.add_argument("--n", type=int, default=50, help="Participants per rule type")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_elimination_style)

    args = parser.parse_args()
    if args.list:
        print("Tasks: resolution_vs_exit, info_gain_evolution, hypotheses_remaining, combined,")
        print("       exit_categories, exit_categories_scatter, tests_vs_resolution, elimination_style")
        return 0
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
