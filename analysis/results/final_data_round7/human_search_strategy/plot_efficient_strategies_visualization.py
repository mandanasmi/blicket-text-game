"""
Plot test pattern frequency and order: single vs pair vs multiple objects per test,
by conjunctive vs disjunctive rule.

Data: human_active_data_no_prior_experience.json (default; 102 participants, no prior).
Outputs: efficient_strategies_visualization.png, efficient_strategies_order.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]

CONJ_COLOR = "#2a9d9f"
DISJ_COLOR = "#e76f51"
SINGLE_COLOR = "#4d7c8c"
PAIR_COLOR = "#c9a227"
MULTIPLE_COLOR = "#8b5a7c"


def _get_true_rule(rd):
    out = rd.get("true_rule") or (rd.get("config") or {}).get("rule") or rd.get("rule")
    return (out or "").strip().lower()


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


def _test_type(n_objects):
    """Return 'single', 'pair', or 'multiple' from number of objects on machine."""
    if n_objects <= 0:
        return None
    if n_objects == 1:
        return "single"
    if n_objects == 2:
        return "pair"
    return "multiple"


def load_test_patterns(json_path, no_prior_only=True):
    """
    Load per-test pattern (single/pair/multiple) for each participant round.
    Returns:
      conj_sequence: list of lists, each inner list is [type1, type2, ...] for one participant
      disj_sequence: same for disjunctive
      conj_all: flat list of all test types (for frequency)
      disj_all: flat list for disjunctive
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    conj_sequences = []
    disj_sequences = []
    conj_all = []
    disj_all = []

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
            if rule not in ("conjunctive", "disjunctive"):
                continue
            sh = rd.get("state_history") or []
            if not isinstance(sh, list) or len(sh) < 1:
                continue

            seq = []
            for ent in sh:
                ob = ent.get("objects_on_machine")
                if ob is None:
                    continue
                n = len(ob) if isinstance(ob, (list, tuple)) else 0
                t = _test_type(n)
                if t:
                    seq.append(t)
                    if rule == "conjunctive":
                        conj_all.append(t)
                    else:
                        disj_all.append(t)
            if seq:
                if rule == "conjunctive":
                    conj_sequences.append(seq)
                else:
                    disj_sequences.append(seq)
            break  # one round per participant

    return conj_sequences, disj_sequences, conj_all, disj_all


def main():
    parser = argparse.ArgumentParser(description="Test pattern frequency and order by rule")
    parser.add_argument("--json", default="../human_active_data_no_prior_experience.json", help="Input JSON (default: from round7)")
    parser.add_argument("--output", default="efficient_strategies_visualization.png", help="Output PNG")
    parser.add_argument("--no-prior-only", action="store_false", dest="prior_filter", default=True)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = args.json if os.path.isabs(args.json) else os.path.normpath(os.path.join(script_dir, args.json))
    out_path = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)

    conj_seqs, disj_seqs, conj_all, disj_all = load_test_patterns(json_path, no_prior_only=args.prior_filter)
    print(f"Data: {os.path.abspath(json_path)}")

    types = ["single", "pair", "multiple"]
    type_labels = ["Single object", "Pair", "Multiple"]

    # Frequency counts by rule
    conj_counts = [conj_all.count(t) for t in types]
    disj_counts = [disj_all.count(t) for t in types]
    conj_total = len(conj_all)
    disj_total = len(disj_all)
    n_conj = len(conj_seqs)
    n_disj = len(disj_seqs)
    conj_pct = [100 * conj_counts[i] / conj_total if conj_total else 0 for i in range(len(types))]
    disj_pct = [100 * disj_counts[i] / disj_total if disj_total else 0 for i in range(len(types))]

    # Per-participant proportions (for SEM across participants)
    conj_props = [[100 * seq.count(t) / len(seq) if seq else 0 for t in types] for seq in conj_seqs]
    disj_props = [[100 * seq.count(t) / len(seq) if seq else 0 for t in types] for seq in disj_seqs]
    conj_mean_pct = [np.mean([p[i] for p in conj_props]) for i in range(len(types))]
    conj_sem_pct = [np.std([p[i] for p in conj_props], ddof=1) / np.sqrt(n_conj) if n_conj > 1 else 0 for i in range(len(types))]
    disj_mean_pct = [np.mean([p[i] for p in disj_props]) for i in range(len(types))]
    disj_sem_pct = [np.std([p[i] for p in disj_props], ddof=1) / np.sqrt(n_disj) if n_disj > 1 else 0 for i in range(len(types))]

    # Order: at each test position (1, 2, 3, ...), proportion single / pair / multiple
    max_len = max(len(s) for s in conj_seqs + disj_seqs) if (conj_seqs or disj_seqs) else 16
    conj_by_pos = {t: [] for t in types}
    disj_by_pos = {t: [] for t in types}
    for pos in range(max_len):
        for t in types:
            c = sum(1 for seq in conj_seqs if len(seq) > pos and seq[pos] == t)
            d = sum(1 for seq in disj_seqs if len(seq) > pos and seq[pos] == t)
            conj_by_pos[t].append(c)
            disj_by_pos[t].append(d)

    # Normalize by number of participants with at least that many tests
    conj_n_at_pos = [sum(1 for seq in conj_seqs if len(seq) > pos) for pos in range(max_len)]
    disj_n_at_pos = [sum(1 for seq in disj_seqs if len(seq) > pos) for pos in range(max_len)]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    # Panel 1: Frequency (grouped bars) - mean % with SEM across participants
    x = np.arange(len(types))
    w = 0.35
    bars_c = ax.bar(x - w / 2, conj_mean_pct, w, yerr=conj_sem_pct, label=f"Conjunctive (n={n_conj})", color=CONJ_COLOR, alpha=0.9, capsize=3)
    bars_d = ax.bar(x + w / 2, disj_mean_pct, w, yerr=disj_sem_pct, label=f"Disjunctive (n={n_disj})", color=DISJ_COLOR, alpha=0.9, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Objects per test")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, b in enumerate(bars_c):
        h = b.get_height()
        cnt = int(conj_counts[i])
        if conj_total > 0:
            x_center = b.get_x() + b.get_width() / 2
            ax.text(x_center, h + conj_sem_pct[i] + 1.5, f"{cnt}/{conj_total}\n{conj_mean_pct[i]:.1f}% \u00b1 {conj_sem_pct[i]:.1f}", ha="center", va="bottom", fontsize=7)
    for i, b in enumerate(bars_d):
        h = b.get_height()
        cnt = int(disj_counts[i])
        if disj_total > 0:
            x_center = b.get_x() + b.get_width() / 2
            ax.text(x_center, h + disj_sem_pct[i] + 1.5, f"{cnt}/{disj_total}\n{disj_mean_pct[i]:.1f}% \u00b1 {disj_sem_pct[i]:.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {out_path}")

    # Separate figure: Order bar plots (16 tests, three bars per test)
    n_tests = 16
    def _pad(v, n):
        return np.array(list(v) + [0] * (n - len(v))) if len(v) < n else np.array(v[:n])
    conj_single = _pad([conj_by_pos["single"][i] / conj_n_at_pos[i] * 100 if conj_n_at_pos[i] else 0 for i in range(max_len)], n_tests)
    conj_pair   = _pad([conj_by_pos["pair"][i]   / conj_n_at_pos[i] * 100 if conj_n_at_pos[i] else 0 for i in range(max_len)], n_tests)
    conj_mult   = _pad([conj_by_pos["multiple"][i] / conj_n_at_pos[i] * 100 if conj_n_at_pos[i] else 0 for i in range(max_len)], n_tests)
    disj_single = _pad([disj_by_pos["single"][i] / disj_n_at_pos[i] * 100 if disj_n_at_pos[i] else 0 for i in range(max_len)], n_tests)
    disj_pair   = _pad([disj_by_pos["pair"][i]   / disj_n_at_pos[i] * 100 if disj_n_at_pos[i] else 0 for i in range(max_len)], n_tests)
    disj_mult   = _pad([disj_by_pos["multiple"][i] / disj_n_at_pos[i] * 100 if disj_n_at_pos[i] else 0 for i in range(max_len)], n_tests)

    x = np.arange(n_tests)
    bar_w = 0.25

    order_path = os.path.join(os.path.dirname(out_path), "efficient_strategies_order.png")
    fig2, axes2 = plt.subplots(2, 1, figsize=(8, 7))

    ax = axes2[0]
    ax.bar(x - bar_w, conj_single, bar_w, label="Single", color=SINGLE_COLOR, alpha=0.9)
    ax.bar(x,         conj_pair,   bar_w, label="Pair", color=PAIR_COLOR, alpha=0.9)
    ax.bar(x + bar_w, conj_mult,   bar_w, label="Multiple", color=MULTIPLE_COLOR, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(n_tests)])
    ax.set_xlabel("Test (1–16)")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Order of test types: Conjunctive")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    ax = axes2[1]
    ax.bar(x - bar_w, disj_single, bar_w, label="Single", color=SINGLE_COLOR, alpha=0.9)
    ax.bar(x,         disj_pair,   bar_w, label="Pair", color=PAIR_COLOR, alpha=0.9)
    ax.bar(x + bar_w, disj_mult,   bar_w, label="Multiple", color=MULTIPLE_COLOR, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(n_tests)])
    ax.set_xlabel("Test (1–16)")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Order of test types: Disjunctive")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig2.savefig(order_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {order_path}")
    print(f"Conjunctive: {len(conj_seqs)} participants, {len(conj_all)} tests (single {conj_counts[0]}, pair {conj_counts[1]}, multiple {conj_counts[2]})")
    print(f"Disjunctive: {len(disj_seqs)} participants, {len(disj_all)} tests (single {disj_counts[0]}, pair {disj_counts[1]}, multiple {disj_counts[2]})")


if __name__ == "__main__":
    main()
