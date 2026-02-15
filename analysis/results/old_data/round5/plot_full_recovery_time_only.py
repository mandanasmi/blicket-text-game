"""
Grouped bar chart: Full Causal Recovery % (both objects + rule correct), plus
Avg Total Time, Avg Num Tests, Avg Time Per Test for that subset.
Same style as accuracy_by_rule_type_grouped_3.png.
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]


def _bool(s):
    v = s if hasattr(s, "item") else s
    return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")


def main():
    parser = argparse.ArgumentParser(description="Full-recovery grouped plot with accuracy + time.")
    parser.add_argument("--input", default="main_game_data_full_recovery_3.csv", help="Full-recovery CSV (time metrics)")
    parser.add_argument("--input-full", default="main_game_data_with_prior_experience_3_no_prior.csv", help="Full CSV (for recovery %)")
    parser.add_argument("--output", default="accuracy_by_rule_type_grouped_full_recovery.png", help="Output PNG")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df_full = pd.read_csv(args.input_full)
    df_full["obj_ok"] = df_full["object_identification_correct"].map(_bool)
    df_full["rule_ok"] = df_full["rule_choice_correct"].map(_bool)
    df_full["full_ok"] = df_full["obj_ok"] & df_full["rule_ok"]

    rule_types = ["Conjunctive", "Disjunctive"]
    recovery_acc = []
    recovery_se = []
    recovery_k = []
    recovery_n = []
    avg_times = []
    avg_time_per_test = []
    avg_num_tests = []
    avg_times_se = []
    avg_time_per_test_se = []
    avg_num_tests_se = []

    for rt in ["conjunctive", "disjunctive"]:
        sub_full = df_full[df_full["ground_truth_rule"] == rt]
        n = len(sub_full)
        k = int(sub_full["full_ok"].sum())
        p = (k / n * 100) if n else 0
        se = 100 * np.sqrt((k / n) * (1 - k / n) / n) if n else 0
        recovery_acc.append(p)
        recovery_se.append(se)
        recovery_k.append(k)
        recovery_n.append(n)

        sub = df[df["ground_truth_rule"] == rt]
        time_data = sub["total_round_time_seconds"].dropna()
        avg_t = time_data.mean() if len(time_data) else 0
        se_t = (time_data.std(ddof=1) / np.sqrt(len(time_data))) if len(time_data) > 1 else 0
        avg_times.append(avg_t)
        avg_times_se.append(se_t)

        sub = sub.copy()
        sub["tpt"] = sub["total_test_time_seconds"] / sub["num_tests"]
        valid = sub[sub["num_tests"] > 0]["tpt"].dropna()
        avg_tpt = valid.mean() if len(valid) else 0
        se_tpt = (valid.std(ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else 0
        avg_time_per_test.append(avg_tpt)
        avg_time_per_test_se.append(se_tpt)

        nt = sub[sub["num_tests"] > 0]["num_tests"].dropna()
        avg_nt = nt.mean() if len(nt) else 0
        se_nt = (nt.std(ddof=1) / np.sqrt(len(nt))) if len(nt) > 1 else 0
        avg_num_tests.append(avg_nt)
        avg_num_tests_se.append(se_nt)

    fig, ax1 = plt.subplots(1, 1, figsize=(9, 5))
    x = np.arange(len(rule_types))
    width = 0.16

    ax2 = ax1.twinx()
    ax3 = ax2.twinx()
    ax3.spines["right"].set_position(("outward", 50))

    bars0 = ax1.bar(
        x - width * 0.5,
        recovery_acc,
        width,
        yerr=recovery_se,
        capsize=4,
        color="#2a9d8f",
        alpha=0.85,
        edgecolor="#333333",
        linewidth=1.2,
        error_kw={"color": "#333333", "linewidth": 1.2},
    )
    bars3 = ax2.bar(
        x + width * 0.5,
        avg_times,
        width,
        yerr=avg_times_se,
        capsize=4,
        color="#f77f00",
        alpha=0.85,
        edgecolor="#333333",
        linewidth=1.2,
        error_kw={"color": "#333333", "linewidth": 1.2},
    )
    bars5 = ax3.bar(
        x + width * 1.5,
        avg_num_tests,
        width,
        yerr=avg_num_tests_se,
        capsize=4,
        color="#8b5cf6",
        alpha=0.85,
        edgecolor="#333333",
        linewidth=1.2,
        error_kw={"color": "#333333", "linewidth": 1.2},
    )
    bars4 = ax2.bar(
        x + width * 2.5,
        avg_time_per_test,
        width,
        yerr=avg_time_per_test_se,
        capsize=4,
        color="#fcbf49",
        alpha=0.85,
        edgecolor="#333333",
        linewidth=1.2,
        error_kw={"color": "#333333", "linewidth": 1.2},
    )

    ax1.set_xlabel("Rule Type", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold", color="#333333")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rule_types, fontsize=13)
    ax1.set_ylim(0, 110)
    ax1.axhline(100, color="#6a3d9a", linestyle="--", linewidth=1.5, alpha=0.5, label="Perfect (100%)")
    ax1.grid(False)
    ax1.yaxis.tick_left()
    ax1.spines["right"].set_visible(False)

    max_time = max(avg_times) if avg_times else 100
    ax2.set_ylim(0, max_time * 1.15)
    ax2.set_ylabel("Time (seconds)", fontsize=14, fontweight="bold", color="#f77f00")
    ax2.tick_params(axis="y", labelcolor="#f77f00")
    ax2.yaxis.tick_right()
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(True)

    max_nt = max(avg_num_tests) if avg_num_tests else 1
    ax3.set_ylim(0, max_nt * 1.2)
    ax3.set_yticks([])
    ax3.set_ylabel("")
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)

    legend_handles = [
        mlines.Line2D([0], [0], color="#6a3d9a", linestyle="--", linewidth=1.5, alpha=0.5, label="Perfect (100%)"),
        mpatches.Patch(facecolor="#2a9d8f", edgecolor="#333333", linewidth=1.5, label="Full hypothesis"),
        mpatches.Patch(facecolor="#f77f00", edgecolor="#333333", linewidth=1.5, label="Avg Total Time"),
        mpatches.Patch(facecolor="#8b5cf6", edgecolor="#333333", linewidth=1.5, label="Avg Num Tests"),
        mpatches.Patch(facecolor="#fcbf49", edgecolor="#333333", linewidth=1.5, label="Avg Time Per Test"),
    ]
    ax1.legend(handles=legend_handles, fontsize=11, loc="upper left", bbox_to_anchor=(1.12, 1), frameon=True, fancybox=True, shadow=True)

    pad_time = max(avg_times) * 0.02 if avg_times else 5
    pad_tpt = max(avg_times) * 0.02 if avg_times else 0.5
    pad_nt = max(avg_num_tests) * 0.01 if avg_num_tests else 0.05

    for bar, acc, k, n, se in zip(bars0, recovery_acc, recovery_k, recovery_n, recovery_se):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + se + 2,
            f"{acc:.1f}%\n({k}/{n})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, val, se in zip(bars3, avg_times, avg_times_se):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + se + pad_time,
            f"{val:.0f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#f77f00",
            fontweight="bold",
        )
    for bar, val, se in zip(bars4, avg_time_per_test, avg_time_per_test_se):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + se + pad_tpt,
            f"{val:.1f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#f77f00",
            fontweight="bold",
        )
    for bar, val, se in zip(bars5, avg_num_tests, avg_num_tests_se):
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + se + pad_nt + 0.4,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#8b5cf6",
            fontweight="bold",
        )

    ax1.set_title("Causal Discovery: Objects + Rule", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout(pad=0.25, rect=[0, 0, 0.9, 1])
    plt.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
