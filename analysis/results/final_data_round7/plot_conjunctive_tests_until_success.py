"""
Plot evolution of tests until success (correct rule + correct blickets) for conjunctive
and disjunctive participants. Uses round7 data.json. CDF of num_tests among those who got both correct.
"""

import argparse
import json

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]


def _get_true_blickets(round_data):
    out = round_data.get("true_blicket_indices")
    if out is not None:
        return out
    cfg = round_data.get("config") or {}
    return cfg.get("blicket_indices")


def _get_true_rule(round_data):
    out = round_data.get("true_rule")
    if out:
        return out
    cfg = round_data.get("config") or {}
    out = cfg.get("rule")
    if out:
        return out
    return round_data.get("rule") or ""


def _parse_rule(s):
    if not s or not isinstance(s, str):
        return None
    s = s.lower()
    if "conjunctive" in s:
        return "conjunctive"
    if "disjunctive" in s or "any" in s:
        return "disjunctive"
    return None


def _get_chosen_blickets(round_data):
    if "user_chosen_blickets" in round_data:
        chosen = round_data["user_chosen_blickets"]
        if isinstance(chosen, list):
            chosen = [x for x in chosen if x is not None]
            return sorted(chosen)
    if "blicket_classifications" in round_data:
        c = round_data["blicket_classifications"]
        if isinstance(c, dict):
            chosen = []
            for k, v in c.items():
                if v == "Yes":
                    try:
                        chosen.append(int(k.split("_")[1]))
                    except (ValueError, IndexError):
                        pass
            return sorted(chosen)
    return None


def _has_prior(entry):
    comp = entry.get("comprehension") or {}
    if not isinstance(comp, dict):
        return None
    sg = comp.get("similar_game_experience") or {}
    if not isinstance(sg, dict):
        return None
    a = (sg.get("answer") or "").strip().lower()
    if "yes" in a:
        return True
    if "no" in a:
        return False
    return None


def main():
    ap = argparse.ArgumentParser(description="Conjunctive tests-until-success CDF from round7 data.json")
    ap.add_argument("--input", default="data.json", help="Input JSON (round7 data.json)")
    ap.add_argument("--output", default="conjunctive_tests_until_success.png", help="Output PNG")
    ap.add_argument("--no-prior-only", action="store_true", help="Include participants with prior experience")
    args = ap.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    # Collect rounds by rule type: num_tests, both_correct
    rows_conj = []
    rows_disj = []
    for _pid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        if not args.no_prior_only and _has_prior(entry) is True:
            continue
        mg = entry.get("main_game") or {}
        if not isinstance(mg, dict):
            continue

        has_rounds = any(isinstance(k, str) and k.startswith("round_") for k in mg)
        if has_rounds:
            rounds = [(k, mg[k]) for k in mg if isinstance(k, str) and k.startswith("round_") and isinstance(mg[k], dict)]
        else:
            rounds = [("main", mg)]

        for _rk, rd in rounds:
            rule = (_get_true_rule(rd) or "").strip().lower()
            if rule not in ("conjunctive", "disjunctive"):
                continue
            tt = rd.get("test_timings") or []
            num_tests = len(tt) if isinstance(tt, list) else 0
            true_b = _get_true_blickets(rd)
            user_b = _get_chosen_blickets(rd)
            rule_str = rd.get("rule_type") or ""
            user_rule = _parse_rule(rule_str)
            obj_ok = False
            if true_b is not None and user_b is not None:
                obj_ok = set(true_b) == set(user_b)
            rule_ok = rule and user_rule and (rule == user_rule)
            both = obj_ok and rule_ok
            row = {"num_tests": num_tests, "both_correct": both}
            if rule == "conjunctive":
                rows_conj.append(row)
            else:
                rows_disj.append(row)

    success_conj = np.array([r["num_tests"] for r in rows_conj if r["both_correct"] and r["num_tests"] >= 1])
    success_disj = np.array([r["num_tests"] for r in rows_disj if r["both_correct"] and r["num_tests"] >= 1])
    if success_conj.size == 0 and success_disj.size == 0:
        print("No conjunctive or disjunctive participants with both correct found.")
        return

    t_max = int(max(
        np.max(success_conj) if success_conj.size else 0,
        np.max(success_disj) if success_disj.size else 0,
    ))
    steps = np.arange(1, t_max + 1, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.8))
    for success, label, color in [
        (success_conj, "Conjunctive", "#2a9d8f"),
        (success_disj, "Disjunctive", "#e76f51"),
    ]:
        if success.size == 0:
            continue
        cdf = np.array([np.sum(success <= t) / len(success) for t in steps])
        ax.plot(steps, cdf, color=color, linewidth=1.5, marker="o", markersize=3.5, label=label)

    ax.set_xlabel("Number of tests performed", fontsize=9)
    ax.set_ylabel("Full hypothesis accuracy", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlim(0.5, t_max + 0.5)
    ax.set_ylim(-0.01, 1.03)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(loc="lower right", fontsize=8)
    plt.subplots_adjust(left=0.12, right=0.96, top=0.96, bottom=0.12)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.02)
    n_conj = len(success_conj)
    n_disj = len(success_disj)
    print(f"Saved: {args.output} (conjunctive: {n_conj}, disjunctive: {n_disj} with both correct)")


if __name__ == "__main__":
    main()
