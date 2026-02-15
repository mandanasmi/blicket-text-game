"""
Creative visualizations for participants who got the rule right but object identification wrong.
Uses main_game_data_rule_ok_object_wrong_3.csv.
"""

import argparse
import ast
import re
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]


def _parse_list(s):
    if pd.isna(s) or s == "" or s == "[]":
        return []
    try:
        out = ast.literal_eval(s)
        return list(out) if isinstance(out, (list, tuple)) else []
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Visualize rule-ok object-wrong participants.")
    parser.add_argument("--input", default="main_game_data_rule_ok_object_wrong_3.csv", help="Input CSV")
    parser.add_argument("--output-dir", default=".", help="Output directory for PNGs")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["user_objs"] = df["user_object_identification"].map(_parse_list)
    df["truth_objs"] = df["ground_truth_objects"].map(_parse_list)
    df["user_set"] = df["user_objs"].map(lambda x: set(int(i) for i in x))
    df["truth_set"] = df["truth_objs"].map(lambda x: set(int(i) for i in x))
    df["hits"] = df.apply(lambda r: len(r["user_set"] & r["truth_set"]), axis=1)
    df["missed"] = df.apply(lambda r: len(r["truth_set"] - r["user_set"]), axis=1)
    df["false_pos"] = df.apply(lambda r: len(r["user_set"] - r["truth_set"]), axis=1)
    df["short_id"] = df["user_id"].str[-8:]

    n = len(df)
    outdir = args.output_dir.rstrip("/")

    # -------------------------------------------------------------------------
    # 1. Object-overlap breakdown: hits / missed / false positives per participant
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(n)
    w = 0.25
    hits = df["hits"].values
    missed = df["missed"].values
    fp = df["false_pos"].values
    colors_rule = ["#2a9d8f" if r == "conjunctive" else "#e76f51" for r in df["ground_truth_rule"]]

    ax.bar(x - w, hits, w, label="Hits (user & truth)", color="#2ecc71", edgecolor="#333", linewidth=0.8)
    ax.bar(x, missed, w, label="Missed (truth only)", color="#e74c3c", edgecolor="#333", linewidth=0.8)
    ax.bar(x + w, fp, w, label="False positives (user only)", color="#f39c12", edgecolor="#333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["short_id"], fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Count (objects)")
    ax.set_xlabel("Participant (last 8 chars)")
    ax.set_title("Object identification breakdown: rule correct, object wrong\n(hits / missed / false positives)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(4, hits.max() + missed.max() + fp.max() + 0.5))
    plt.tight_layout()
    fig.savefig(f"{outdir}/rule_ok_object_wrong_overlap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outdir}/rule_ok_object_wrong_overlap.png")

    # -------------------------------------------------------------------------
    # 2. Participant x object heatmap (user selection vs ground truth)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    objs = [0, 1, 2, 3]
    # Rows: participants. Cols: objects. 1 = user said blicket, 0.5 = truth blicket, 0 = neither
    # We use: 2 = both, 1 = user only (FP), 0.5 = truth only (missed), 0 = neither
    H = np.zeros((n, 4))
    for i, (_, row) in enumerate(df.iterrows()):
        u, t = row["user_set"], row["truth_set"]
        for j, o in enumerate(objs):
            in_u, in_t = o in u, o in t
            if in_u and in_t:
                H[i, j] = 2
            elif in_u:
                H[i, j] = 1
            elif in_t:
                H[i, j] = 0.5
            else:
                H[i, j] = 0
    im = ax.imshow(H, aspect="auto", cmap="RdYlGn", vmin=0, vmax=2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Obj 1", "Obj 2", "Obj 3", "Obj 4"])
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{s} ({r})" for s, r in zip(df["short_id"], df["ground_truth_rule"])], fontsize=9)
    ax.set_xlabel("Object")
    ax.set_ylabel("Participant (ground-truth rule)")
    ax.set_title("User vs truth: green = both, yellow = user only (FP), red = truth only (missed)")
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 0.5, 1, 2], shrink=0.8)
    cbar.ax.set_yticklabels(["neither", "truth only\n(missed)", "user only\n(FP)", "both"])
    plt.tight_layout()
    fig.savefig(f"{outdir}/rule_ok_object_wrong_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outdir}/rule_ok_object_wrong_heatmap.png")

    # -------------------------------------------------------------------------
    # 3. Rule-inference word frequency by rule type
    # -------------------------------------------------------------------------
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "is", "are", "it", "this", "that", "for", "with", "be", "by", "at", "as", "if", "when", "must", "can", "will", "need", "have", "has", "do", "does", "i", "my", "me", "but", "not", "no", "none", "all", "only", "just", "so", "then", "based", "think", "don't", "didn't", "was", "were"}
    def tokenize(t):
        if pd.isna(t) or not isinstance(t, str):
            return []
        t = re.sub(r"[^\w\s]", " ", t.lower())
        return [w for w in t.split() if len(w) > 1 and w not in stop]

    cj = df[df["ground_truth_rule"] == "conjunctive"]
    dj = df[df["ground_truth_rule"] == "disjunctive"]
    cj_words = []
    for t in cj["rule_inference_text"].dropna():
        cj_words.extend(tokenize(t))
    dj_words = []
    for t in dj["rule_inference_text"].dropna():
        dj_words.extend(tokenize(t))
    cj_cnt = Counter(cj_words)
    dj_cnt = Counter(dj_words)
    all_words = set(cj_cnt.keys()) | set(dj_cnt.keys())
    top = sorted(all_words, key=lambda w: cj_cnt[w] + dj_cnt[w], reverse=True)[:20]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(len(top))
    w = 0.35
    cj_vals = [cj_cnt[w] for w in top]
    dj_vals = [dj_cnt[w] for w in top]
    ax.bar(x - w / 2, cj_vals, w, label="Conjunctive", color="#2a9d8f", edgecolor="#333", linewidth=0.8)
    ax.bar(x + w / 2, dj_vals, w, label="Disjunctive", color="#e76f51", edgecolor="#333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Frequency in rule-inference text")
    ax.set_xlabel("Word")
    ax.set_title("Rule-inference wording: top words by ground-truth rule type\n(rule correct, object wrong)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{outdir}/rule_ok_object_wrong_word_freq.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outdir}/rule_ok_object_wrong_word_freq.png")

    # -------------------------------------------------------------------------
    # 4. Participant cards: rule choice, truncated rule text, user vs truth objects
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    axes = axes.flatten()
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        txt = str(row["rule_inference_text"])[:120] + ("..." if len(str(row["rule_inference_text"])) > 120 else "")
        ax.set_axis_off()
        rule = row["ground_truth_rule"]
        choice = row["rule_choice"]
        u, t = row["user_set"], row["truth_set"]
        num_tests = row["num_tests"]
        hits, missed, fp = row["hits"], row["missed"], row["false_pos"]

        ax.text(0.5, 0.95, row["short_id"], ha="center", fontsize=11, fontweight="bold")
        ax.text(0.5, 0.88, f"rule: {rule} | choice: {choice}", ha="center", fontsize=9, color="#555")
        ax.text(0.5, 0.78, f"tests: {num_tests} | hits: {hits} missed: {missed} FP: {fp}", ha="center", fontsize=8, color="#666")
        ax.text(0.05, 0.68, "Objects:", fontsize=8, fontweight="bold")
        ax.text(0.05, 0.58, f"  user: {sorted(u)}", fontsize=8, color="#333")
        ax.text(0.05, 0.48, f"  truth: {sorted(t)}", fontsize=8, color="#333")
        ax.text(0.05, 0.35, "Rule text:", fontsize=8, fontweight="bold")
        # Truncate to ~100 chars, allow two short lines
        txt = (txt[:100] + "...") if len(txt) > 100 else txt
        ax.text(0.05, 0.05, txt, fontsize=7, va="top", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#ccc", lw=1))

    for j in range(idx + 1, len(axes)):
        axes[j].set_axis_off()
    fig.suptitle("Rule correct, object wrong: participant snapshots\n(rule type, choice, object user vs truth, rule-inference text)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(f"{outdir}/rule_ok_object_wrong_cards.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outdir}/rule_ok_object_wrong_cards.png")

    # -------------------------------------------------------------------------
    # 5. Scatter: num_tests vs object hits, colored by rule type
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for rule, color in [("conjunctive", "#2a9d8f"), ("disjunctive", "#e76f51")]:
        sub = df[df["ground_truth_rule"] == rule]
        ax.scatter(sub["num_tests"], sub["hits"], c=color, s=80, alpha=0.8, edgecolor="#333", linewidth=1, label=rule.capitalize())
        for _, r in sub.iterrows():
            ax.annotate(r["short_id"], (r["num_tests"], r["hits"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Number of tests")
    ax.set_ylabel("Object hits (user ∩ truth)")
    ax.set_title("Rule correct, object wrong: testing effort vs object overlap")
    ax.legend()
    ax.set_ylim(-0.5, max(df["hits"].max() + 0.5, 1))
    plt.tight_layout()
    fig.savefig(f"{outdir}/rule_ok_object_wrong_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outdir}/rule_ok_object_wrong_scatter.png")

    print("Done.")


if __name__ == "__main__":
    main()
