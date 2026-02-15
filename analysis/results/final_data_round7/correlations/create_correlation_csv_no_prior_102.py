"""
Create a CSV that matches human_active_data_no_prior_experience.json (102 participants).
Filters comprehensive_correlation_data.csv to only include participant_ids that appear
in the JSON. Use this CSV when you need the same 102 participants as the JSON.
Output: comprehensive_correlation_data_no_prior_102.csv
"""

import argparse
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Create 102-participant CSV matching human_active_data_no_prior_experience.json")
    parser.add_argument("--json", default="../human_active_data_no_prior_experience.json", help="JSON with 102 participants (default: from round7)")
    parser.add_argument("--csv", default="comprehensive_correlation_data.csv", help="Full correlation CSV (in correlations/)")
    parser.add_argument("--output", default="comprehensive_correlation_data_no_prior_102.csv", help="Output CSV (102 rows)")
    args = parser.parse_args()

    with open(args.json, "r") as f:
        data = json.load(f)
    json_ids = set(data.keys())

    df = pd.read_csv(args.csv)
    df["participant_id"] = df["participant_id"].astype(str)
    df_filtered = df[df["participant_id"].isin(json_ids)].copy()

    df_filtered.to_csv(args.output, index=False)
    n = len(df_filtered)
    n_conj = (df_filtered["true_rule"].str.strip().str.lower() == "conjunctive").sum()
    n_disj = (df_filtered["true_rule"].str.strip().str.lower() == "disjunctive").sum()
    print(f"Saved: {args.output}")
    print(f"  Rows: {n} (matches JSON: {n == len(json_ids)})")
    print(f"  Conjunctive: {n_conj}, Disjunctive: {n_disj}")


if __name__ == "__main__":
    main()
