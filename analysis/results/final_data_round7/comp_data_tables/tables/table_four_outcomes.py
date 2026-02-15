"""
Create table figures from the four-outcome CSVs (same style as round5 table_rule_ok_object_wrong).
Outputs: table_rule_wrong_objects_correct.png, table_rule_correct_objects_wrong.png,
         table_both_wrong.png, table_both_correct.png
"""

import argparse
import ast

import matplotlib.pyplot as plt
import pandas as pd


def _to_1based(s):
    """Parse '[0,1,2]' or '[]' and return '[1,2,3]' (1-based indices)."""
    if pd.isna(s) or s == "" or s == "[]":
        return "[]"
    try:
        out = ast.literal_eval(s)
        lst = list(out) if isinstance(out, (list, tuple)) else []
        return "[" + ", ".join(str(int(x) + 1) for x in lst) + "]"
    except Exception:
        return str(s)


def _wrap(s, width=85):
    """Wrap text at word boundaries for table display."""
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
    """Transform raw CSV into table DataFrame with action_history_length and total_test_time_seconds."""
    df = df.copy()
    df["user_id_short"] = df["user_id"].str[-8:]
    df["user_objects_1based"] = df["user_object_identification"].map(_to_1based)
    df["truth_objects_1based"] = df["ground_truth_objects"].map(_to_1based)
    df["rule_text"] = df["rule_inference_text"].fillna("").astype(str).apply(_wrap)
    if "action_history_length" not in df.columns:
        df["action_history_length"] = ""
    def _fmt(x):
        try:
            v = float(x)
            return f"{v:.1f}" if pd.notna(v) else ""
        except (TypeError, ValueError):
            return ""
    if "total_test_time_seconds" not in df.columns:
        df["test_time_display"] = ""
    else:
        df["test_time_display"] = df["total_test_time_seconds"].apply(_fmt)
    if "total_round_time_seconds" not in df.columns:
        df["round_time_display"] = ""
    else:
        df["round_time_display"] = df["total_round_time_seconds"].apply(_fmt)
    tab = df[["user_id_short", "num_tests", "action_history_length", "test_time_display", "round_time_display",
              "user_objects_1based", "truth_objects_1based", "rule_choice", "ground_truth_rule", "rule_text"]].copy()
    tab.columns = ["user_id", "num_tests", "num_actions", "test_time", "round_time", "user_objects", "true_objects",
                   "rule_choice", "true_rule", "rule_text"]
    return tab


def render_table(ax, tab):
    """Draw matplotlib table from tab DataFrame."""
    ax.axis("off")
    if len(tab) == 0:
        ax.text(0.5, 0.5, "No participants", ha="center", va="center", fontsize=12)
        return
    col_widths = [0.05, 0.05, 0.05, 0.06, 0.06, 0.05, 0.05, 0.06, 0.06, 0.52]
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


def main():
    parser = argparse.ArgumentParser(description="Table figures for four outcome CSVs.")
    parser.add_argument("--csv-dir", default=".", help="Directory containing outcome CSVs")
    parser.add_argument("--out-dir", default=".", help="Directory for output PNGs")
    args = parser.parse_args()
    d = args.csv_dir.rstrip("/")
    out = args.out_dir.rstrip("/")

    configs = [
        ("participants_rule_wrong_objects_correct.csv", "table_rule_wrong_objects_correct.png"),
        ("participants_rule_correct_objects_wrong.csv", "table_rule_correct_objects_wrong.png"),
        ("participants_both_wrong.csv", "table_both_wrong.png"),
        ("participants_both_correct.csv", "table_both_correct.png"),
    ]

    for csv_name, png_name in configs:
        path = f"{d}/{csv_name}"
        df = pd.read_csv(path)
        tab = build_table_df(df)
        n_rows = len(tab) + 1
        fig_h = max(3, min(10, 1.2 + 0.28 * n_rows))
        fig, ax = plt.subplots(figsize=(14, fig_h))
        render_table(ax, tab)
        plt.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99)
        out_path = f"{out}/{png_name}"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.01)
        plt.close()
        print(f"Saved: {out_path} ({len(tab)} rows)")


if __name__ == "__main__":
    main()
