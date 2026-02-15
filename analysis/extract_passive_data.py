"""
Extract and analyze passive data from passive app responses.
Passive data is stored in data_txt_history/ directory as JSON files.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_passive_responses(data_dir: Path):
    """Load all JSON response files from data_txt_history directory."""
    if not data_dir.exists():
        return []
    
    responses = []
    for json_file in sorted(data_dir.glob("responses_*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                data["filename"] = json_file.name
                responses.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read {json_file}: {e}")
    
    return responses


def analyze_passive_data(responses):
    """Analyze passive responses and compute accuracy metrics."""
    if not responses:
        print("No passive data found.")
        return None
    
    print(f"Found {len(responses)} passive response files.")
    
    # Extract relevant data
    rows = []
    for resp in responses:
        rule_type = resp.get("rule_type", "").strip().lower()
        blicket_classifications = resp.get("blicket_classifications", {})
        
        # Count how many objects were classified as blickets
        num_blickets = sum(1 for v in blicket_classifications.values() if v == "Yes")
        
        rows.append({
            "filename": resp.get("filename", "unknown"),
            "timestamp": resp.get("timestamp", ""),
            "num_objects": resp.get("num_objects", 0),
            "num_steps": resp.get("num_steps", 0),
            "rule_type": rule_type,
            "rule_hypothesis": resp.get("rule_hypothesis", ""),
            "num_blickets": num_blickets,
            "blicket_classifications": blicket_classifications,
        })
    
    df = pd.DataFrame(rows)
    
    print("\n" + "=" * 80)
    print("PASSIVE DATA SUMMARY")
    print("=" * 80)
    print(f"\nTotal responses: {len(df)}")
    print(f"\nRule type distribution:")
    print(df["rule_type"].value_counts())
    print(f"\nNumber of blickets identified (mean): {df['num_blickets'].mean():.2f}")
    print(f"Number of blickets identified (std): {df['num_blickets'].std():.2f}")
    
    return df


def save_to_csv(df: pd.DataFrame, output_path: Path):
    """Save analyzed data to CSV."""
    if df is None or df.empty:
        print("No data to save.")
        return
    
    # Flatten blicket_classifications for CSV
    csv_rows = []
    for _, row in df.iterrows():
        csv_row = {
            "filename": row["filename"],
            "timestamp": row["timestamp"],
            "num_objects": row["num_objects"],
            "num_steps": row["num_steps"],
            "rule_type": row["rule_type"],
            "rule_hypothesis": row["rule_hypothesis"],
            "num_blickets": row["num_blickets"],
        }
        # Add individual object classifications
        blickets = row["blicket_classifications"]
        if isinstance(blickets, dict):
            for key, value in sorted(blickets.items()):
                csv_row[f"object_{key}"] = value
        
        csv_rows.append(csv_row)
    
    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract and analyze passive data from passive app")
    parser.add_argument(
        "--data-dir",
        default="data_txt_history",
        help="Directory containing passive response JSON files (default: data_txt_history)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: analysis/results/passive_data.csv)",
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else project_root / args.data_dir
    output_path = Path(args.output) if args.output else project_root / "analysis" / "results" / "passive_data.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading passive data from: {data_dir}")
    responses = load_passive_responses(data_dir)
    
    df = analyze_passive_data(responses)
    
    if df is not None:
        save_to_csv(df, output_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
