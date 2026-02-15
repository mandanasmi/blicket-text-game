"""
Create a table figure from main_game_data_rule_ok_object_wrong_3.csv and save as PNG.
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


def _wrap(s, width=72):
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


def main():
    parser = argparse.ArgumentParser(description="Table of rule-ok object-wrong participants.")
    parser.add_argument("--input", default="main_game_data_rule_ok_object_wrong_3.csv", help="Input CSV")
    parser.add_argument("--output", default="main_game_data_rule_ok_object_wrong.png", help="Output PNG")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["user_id_short"] = df["user_id"].str[-8:]
    df["user_objects_1based"] = df["user_object_identification"].map(_to_1based)
    df["truth_objects_1based"] = df["ground_truth_objects"].map(_to_1based)
    df["rule_text"] = df["rule_inference_text"].fillna("").astype(str).apply(_wrap)

    cols = ["user_id_short", "num_tests", "user_objects_1based", "truth_objects_1based",
            "rule_choice", "ground_truth_rule", "rule_text"]
    tab = df[cols].copy()
    tab.columns = ["user_id", "num_tests", "user_objects", "true_objects", "rule_choice", "true_rule", "rule_text"]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")
    col_widths = [0.10, 0.06, 0.08, 0.08, 0.07, 0.07, 0.49]  # num_tests slightly larger
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
    plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98)
    plt.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
