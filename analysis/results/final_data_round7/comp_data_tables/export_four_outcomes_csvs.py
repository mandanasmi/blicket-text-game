"""
Export CSVs of participants by accuracy outcome:
1. Rule wrong, objects correct
2. Rule correct, objects wrong
3. Rule wrong, objects wrong
4. Both correct (for completeness)
Uses same data and filtering as plot_accuracy_four_outcomes_stacked (no prior experience).
"""

import argparse
import pandas as pd


def _bool(s):
    v = s if hasattr(s, "item") else s
    return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")


def main():
    parser = argparse.ArgumentParser(description="Export CSVs by four rule/object outcomes.")
    parser.add_argument("--input", default="../main_game_data_with_prior_experience.csv", help="Input CSV (default: from round7)")
    parser.add_argument("--out-dir", default=".", help="Directory for output CSVs (default: comp_data_tables)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[df["has_prior_experience"] == False].copy()
    df["obj_ok"] = df["object_identification_correct"].map(_bool)
    df["rule_ok"] = df["rule_choice_correct"].map(_bool)

    df["rule_wrong_obj_ok"] = ~df["rule_ok"] & df["obj_ok"]
    df["rule_ok_obj_wrong"] = df["rule_ok"] & ~df["obj_ok"]
    df["both_wrong"] = ~df["rule_ok"] & ~df["obj_ok"]
    df["both_ok"] = df["rule_ok"] & df["obj_ok"]

    drop_cols = ["obj_ok", "rule_ok", "rule_wrong_obj_ok", "rule_ok_obj_wrong", "both_wrong", "both_ok"]
    base = df.drop(columns=drop_cols)

    outputs = [
        ("participants_rule_wrong_objects_correct.csv", "rule_wrong_obj_ok"),
        ("participants_rule_correct_objects_wrong.csv", "rule_ok_obj_wrong"),
        ("participants_both_wrong.csv", "both_wrong"),
        ("participants_both_correct.csv", "both_ok"),
    ]
    for fname, key in outputs:
        sub = df[df[key]]
        out = base.loc[sub.index].copy() if len(sub) else base.head(0)
        path = f"{args.out_dir.rstrip('/')}/{fname}"
        out.to_csv(path, index=False)
        print(f"Saved: {path} ({len(out)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
