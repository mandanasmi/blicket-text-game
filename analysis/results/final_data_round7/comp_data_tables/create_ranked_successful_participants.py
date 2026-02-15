"""
Create ranked_successful_participants.csv and table from participants_both_correct.csv.
Sorts by num_tests (ascending) and adds rank. Outputs CSV and table PNG.
"""

import argparse
import ast

import matplotlib.pyplot as plt
import pandas as pd


def _to_1based(s):
    if pd.isna(s) or s == "" or s == "[]":
        return "[]"
    try:
        out = ast.literal_eval(s)
        lst = list(out) if isinstance(out, (list, tuple)) else []
        return "[" + ", ".join(str(int(x) + 1) for x in lst) + "]"
    except Exception:
        return str(s)


def _wrap(s, width=85):
    if not s or pd.isna(s):
        return ""
    s = str(s).strip()
    out = []
    while s:
        if len(s) <= width:
            out.append(s)
            break
        i = s.rfind(" ", 0, width + 1)
        if i <= 0:
            i = width
        out.append(s[:i].strip())
        s = s[i:].strip()
    return "\n".join(out)


def build_table_df(df):
    df = df.copy()
    df["user_id_short"] = df["user_id"].str[-8:]
    df["user_objects_1based"] = df["user_object_identification"].map(_to_1based)
    df["truth_objects_1based"] = df["ground_truth_objects"].map(_to_1based)
    df["rule_text"] = df["rule_inference_text"].fillna("").astype(str).apply(_wrap)

    def _fmt(x):
        try:
            v = float(x)
            return f"{v:.1f}" if pd.notna(v) else ""
        except (TypeError, ValueError):
            return ""

    df["test_time_display"] = df["total_test_time_seconds"].apply(_fmt) if "total_test_time_seconds" in df.columns else ""
    df["round_time_display"] = df["total_round_time_seconds"].apply(_fmt) if "total_round_time_seconds" in df.columns else ""

    tab = df[["rank", "user_id_short", "num_tests", "action_history_length", "test_time_display", "round_time_display",
              "user_objects_1based", "truth_objects_1based", "rule_choice", "ground_truth_rule", "rule_text"]].copy()
    tab.columns = ["rank", "user_id", "num_tests", "num_actions", "test_time", "round_time", "user_objects", "true_objects",
                   "rule_choice", "true_rule", "rule_text"]
    return tab


def render_table(ax, tab):
    ax.axis("off")
    if len(tab) == 0:
        ax.text(0.5, 0.5, "No participants", ha="center", va="center", fontsize=12)
        return
    col_widths = [0.04, 0.05, 0.05, 0.05, 0.06, 0.06, 0.05, 0.05, 0.06, 0.06, 0.49]
    tbl = ax.table(
        cellText=tab.values,
        colLabels=tab.columns,
        loc="center",
        cellLoc="left",
        colColours=["#e0e0e0"] * len(tab.columns),
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.15, 2.2)
    for j in range(len(tab.columns)):
        cell = tbl[(0, j)]
        cell.get_text().set_wrap(True)
        cell.get_text().set_fontsize(8)


def process_and_save(df, out_csv, out_png, d):
    """Sort by num_tests, add rank, save CSV and table PNG."""
    if len(df) == 0:
        print(f"Skipping {out_csv}: no participants")
        return
    df = df.sort_values("num_tests", ascending=True).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    cols = ["rank"] + [c for c in df.columns if c != "rank"]
    df[cols].to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df)} rows)")

    tab = build_table_df(df)
    n_rows = len(tab) + 1
    fig_h = max(3, min(12, 1.2 + 0.28 * n_rows))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    render_table(ax, tab)
    plt.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99)
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.01)
    plt.close()
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Create ranked successful participants CSV and table.")
    parser.add_argument("--input", default="participants_both_correct.csv", help="Input CSV (in comp_data_tables)")
    parser.add_argument("--dir", default=".", help="Directory for input/output (default: comp_data_tables)")
    args = parser.parse_args()
    d = args.dir.rstrip("/")

    path = f"{d}/{args.input}"
    df = pd.read_csv(path)

    # All participants
    process_and_save(
        df.copy(),
        f"{d}/ranked_successful_participants.csv",
        f"{d}/table_ranked_successful_participants.png",
        d,
    )

    # Conjunctive only
    conj = df[df["ground_truth_rule"].str.lower() == "conjunctive"].copy()
    process_and_save(
        conj,
        f"{d}/ranked_successful_conjunctive_participants.csv",
        f"{d}/table_ranked_successful_conjunctive_participants.png",
        d,
    )

    # Disjunctive only
    disj = df[df["ground_truth_rule"].str.lower() == "disjunctive"].copy()
    process_and_save(
        disj,
        f"{d}/ranked_successful_disjunctive_participants.csv",
        f"{d}/table_ranked_successful_disjunctive_participants.png",
        d,
    )


if __name__ == "__main__":
    main()
