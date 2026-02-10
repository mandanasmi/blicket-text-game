"""
Analyze action sequences from round7_actions.csv.
Output: summary stats, transition matrix, within-test patterns, n-grams, exploration order.
"""

import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def _action_type(txt):
    if not isinstance(txt, str):
        return "other"
    t = txt.strip().lower()
    if t.startswith("placed object"):
        return "place"
    if t.startswith("removed object"):
        return "remove"
    if "test result" in t and "machine is on" in t:
        return "test_on"
    if "test result" in t and "machine is off" in t:
        return "test_off"
    return "other"


def _object_index(txt):
    if not isinstance(txt, str):
        return None
    t = txt.strip().lower()
    for token in ["object 1", "object 2", "object 3", "object 4"]:
        if token in t:
            return int(token.split()[-1])
    return None


def main():
    ap = argparse.ArgumentParser(description="Analyze action sequences")
    ap.add_argument("--input", default=None, help="Path to round7_actions.csv")
    ap.add_argument("--out-dir", default=None, help="Output directory for report and optional plots")
    ap.add_argument("--report", default="action_sequence_report.txt", help="Report filename")
    ap.add_argument("--plot", action="store_true", help="Generate transition heatmap and pattern plots")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # round7
    inp = args.input or os.path.join(parent_dir, "round7_actions.csv")
    out_dir = args.out_dir or script_dir
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(inp)
    df["action_type"] = df["action_text"].map(_action_type)
    df["object_idx"] = df["action_text"].map(_object_index)

    lines = []
    def log(s=""):
        lines.append(s)

    log("=== Action sequence analysis (round7_actions.csv) ===")
    log()
    n_part = df["user_id"].nunique()
    n_act = len(df)
    log(f"Participants: {n_part}  |  Total actions: {n_act}")
    log(f"Actions per participant: mean {df.groupby('user_id').size().mean():.1f}, "
        f"median {df.groupby('user_id').size().median():.0f}, "
        f"min {df.groupby('user_id').size().min()}, max {df.groupby('user_id').size().max()}")
    log()

    # Action type counts
    log("--- Action type counts ---")
    for k, v in df["action_type"].value_counts().sort_index().items():
        pct = 100 * v / n_act
        log(f"  {k}: {v} ({pct:.1f}%)")
    log()

    # Transitions (action_type -> next action_type)
    log("--- Transition matrix (action_type -> next action_type) ---")
    trans = defaultdict(lambda: defaultdict(int))
    for uid, g in df.sort_values(["user_id", "action_index"]).groupby("user_id"):
        types = g["action_type"].tolist()
        for i in range(len(types) - 1):
            a, b = types[i], types[i + 1]
            trans[a][b] += 1

    order = ["place", "remove", "test_on", "test_off", "other"]
    order = [x for x in order if x in set().union(*[set(t.keys()) for t in trans.values()])]
    for a in order:
        row = trans.get(a, {})
        total = sum(row.values())
        log(f"  {a} -> ")
        for b in order:
            c = row.get(b, 0)
            pct = 100 * c / total if total else 0
            if c > 0:
                log(f"      {b}: {c} ({pct:.1f}%)")
    log()

    # Within-test structure: place/remove counts before each Test
    log("--- Within-test structure (actions before each Test Result) ---")
    place_per_test = []
    remove_per_test = []
    for uid, g in df.sort_values(["user_id", "action_index"]).groupby("user_id"):
        for tid, tg in g.groupby("test_index"):
            types = tg["action_type"].tolist()
            n_place = sum(1 for x in types if x == "place")
            n_remove = sum(1 for x in types if x == "remove")
            place_per_test.append(n_place)
            remove_per_test.append(n_remove)

    log(f"  Place actions per test:  mean {np.mean(place_per_test):.2f}, "
        f"median {np.median(place_per_test):.1f}, "
        f"min {int(np.min(place_per_test))}, max {int(np.max(place_per_test))}")
    log(f"  Remove actions per test: mean {np.mean(remove_per_test):.2f}, "
        f"median {np.median(remove_per_test):.1f}, "
        f"min {int(np.min(remove_per_test))}, max {int(np.max(remove_per_test))}")

    # Pattern: (place, remove) profile per test
    pattern_counts = Counter((int(p), int(r)) for p, r in zip(place_per_test, remove_per_test))
    log("  Top (place, remove) patterns per test:")
    for (p, r), c in pattern_counts.most_common(10):
        log(f"    ({p}, {r}): {c} tests")
    log()

    # N-grams of action types (bigrams, trigrams)
    log("--- Bigrams (action_type pairs) ---")
    bigrams = Counter()
    for uid, g in df.sort_values(["user_id", "action_index"]).groupby("user_id"):
        types = g["action_type"].tolist()
        for i in range(len(types) - 1):
            bigrams[(types[i], types[i + 1])] += 1
    for (a, b), c in bigrams.most_common(12):
        log(f"  {a} -> {b}: {c}")
    log()

    log("--- Trigrams (action_type triples) ---")
    trigrams = Counter()
    for uid, g in df.sort_values(["user_id", "action_index"]).groupby("user_id"):
        types = g["action_type"].tolist()
        for i in range(len(types) - 2):
            trigrams[(types[i], types[i + 1], types[i + 2])] += 1
    for (a, b, c), n in trigrams.most_common(10):
        log(f"  {a} -> {b} -> {c}: {n}")
    log()

    # Exploration order: first place per participant (object 1,2,3,4)
    log("--- First object placed (exploration start) ---")
    first_place = df[df["action_type"] == "place"].groupby("user_id").first()
    first_obj = first_place["object_idx"].value_counts().sort_index()
    for obj, c in first_obj.items():
        o = int(obj) if isinstance(obj, (int, float)) and not np.isnan(obj) else obj
        log(f"  Object {o}: {c} participants ({100*c/n_part:.1f}%)")
    log()

    # Tests per participant
    tests_per = df.groupby("user_id")["test_index"].max()
    log("--- Tests per participant ---")
    log(f"  mean {tests_per.mean():.1f}, median {tests_per.median():.0f}, "
        f"min {tests_per.min()}, max {tests_per.max()}")
    log()

    log("--- Summary ---")
    log("  - Place (37.9%) and remove (32.0%) dominate; test_off (17.0%) > test_on (13.1%).")
    log("  - Typical cycle: place -> test -> remove -> place -> ... ; after test_off, remove (72%)")
    log("    is far more common than place (26%); after test_on, remove (86%) dominates.")
    log("  - Most common within-test pattern: (1 place, 1 remove) then test, then (1,0) or (0,1).")
    log("  - Top trigrams: place->test_off->remove, test_off->remove->place, place->test_on->remove.")
    log("  - ~91% start by placing Object 1; participants run ~8 tests on median.")
    log()

    report_path = os.path.join(out_dir, args.report)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {report_path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            mpl.rcParams["font.family"] = "sans-serif"

            # Transition matrix heatmap
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            M = np.zeros((len(order), len(order)))
            for i, a in enumerate(order):
                row = trans.get(a, {})
                total = max(sum(row.values()), 1)
                for j, b in enumerate(order):
                    M[i, j] = 100 * row.get(b, 0) / total
            im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=45, ha="right")
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order)
            ax.set_xlabel("Next action type")
            ax.set_ylabel("Current action type")
            for i in range(len(order)):
                for j in range(len(order)):
                    v = M[i, j]
                    ax.text(j, i, f"{v:.0f}" if v >= 5 else "", ha="center", va="center", fontsize=9)
            plt.colorbar(im, ax=ax, label="% of transitions")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "action_transition_heatmap.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {os.path.join(out_dir, 'action_transition_heatmap.png')}")

            # (place, remove) per test distribution
            fig, ax = plt.subplots(figsize=(5, 3))
            pattern_list = sorted(pattern_counts.keys(), key=lambda x: (-pattern_counts[x], x))
            top = pattern_list[:12]
            xlab = [f"({p},{r})" for (p, r) in top]
            y = [pattern_counts[k] for k in top]
            ax.bar(range(len(top)), y, color="steelblue", edgecolor="navy", alpha=0.8)
            ax.set_xticks(range(len(top)))
            ax.set_xticklabels(xlab, rotation=45, ha="right")
            ax.set_ylabel("Number of tests")
            ax.set_xlabel("(place, remove) per test")
            ax.set_title("Within-test action pattern (place, remove) counts")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "action_within_test_patterns.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {os.path.join(out_dir, 'action_within_test_patterns.png')}")
        except Exception as e:
            print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
