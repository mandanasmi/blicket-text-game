"""
Create table figure for best_performers.csv (similar style to table_both_correct.png)
Includes action_history_length
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


def build_table_df(df):
    """Transform raw CSV into table DataFrame."""
    df = df.copy()
    df["user_id_short"] = df["participant_id"].str[-8:]
    df["user_objects_1based"] = df["chosen_objects"].map(_to_1based)
    df["truth_objects_1based"] = df["true_blicket_objects"].map(_to_1based)
    
    # Capitalize rule names
    df["rule_choice_cap"] = df["chosen_rule"].str.capitalize()
    df["true_rule_cap"] = df["true_rule"].str.capitalize()
    
    cols = ["user_id_short", "num_tests_before_qa", "action_history_length", 
            "user_objects_1based", "truth_objects_1based",
            "rule_choice_cap", "true_rule_cap"]
    tab = df[cols].copy()
    tab.columns = ["user_id", "num_tests", "action_length", "user_objects", 
                   "true_objects", "rule_choice", "true_rule"]
    return tab


def render_table(ax, tab):
    """Draw matplotlib table from tab DataFrame."""
    ax.axis("off")
    if len(tab) == 0:
        ax.text(0.5, 0.5, "No participants", ha="center", va="center", fontsize=12)
        return
    # Column widths: user_id, num_tests, action_length, user_objects, true_objects, rule_choice, true_rule
    col_widths = [0.12, 0.07, 0.08, 0.10, 0.10, 0.08, 0.08]
    
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
    tbl.scale(1.15, 3.8)


def main():
    parser = argparse.ArgumentParser(description="Table figure for best performers CSV.")
    parser.add_argument("--input", default="../../../correlations/best_performers.csv", help="Input CSV file (default: from round7/correlations)")
    parser.add_argument("--output", default="table_best_performers.png", help="Output PNG file")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"ERROR: {args.input} not found!")
        exit(1)

    print(f"Loaded {len(df)} participants")
    
    tab = build_table_df(df)
    fig, ax = plt.subplots(figsize=(14, 10))
    render_table(ax, tab)
    plt.subplots_adjust(top=0.96, bottom=0.02, left=0.02, right=0.98)
    plt.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"Saved: {args.output} ({len(tab)} rows)")


if __name__ == "__main__":
    main()
