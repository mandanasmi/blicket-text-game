#!/usr/bin/env python3
"""
Compute accuracy for o3-mini and o4-mini models from llm_data results.jsonl files.
Reports all-correct accuracy (num_correct == num_questions) by model and reasoning_effort.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def normalize_model(raw: str) -> str:
    """Normalize model name to o3-mini or o4-mini."""
    raw = (raw or "").lower()
    if "o4-mini" in raw or "o4_mini" in raw:
        return "o4-mini"
    if "o3-mini" in raw or "o3_mini" in raw:
        return "o3-mini"
    return None


def main():
    parser = argparse.ArgumentParser(description="Compute o3-mini and o4-mini accuracy from llm_data")
    parser.add_argument(
        "--llm-data",
        default="/tmp/llm_data",
        help="Path to llm_data directory (default: /tmp/llm_data)",
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=None,
        help="Filter to this number of objects only (default: all). Use 4 for 4-object trials.",
    )
    args = parser.parse_args()

    llm_data_dir = Path(args.llm_data)
    if not llm_data_dir.exists():
        print(f"ERROR: llm_data not found at {llm_data_dir}")
        return 1

    rows = []
    for results_path in llm_data_dir.rglob("results.jsonl"):
        config_path = results_path.parent / ".hydra" / "config.yaml"
        if not config_path.exists():
            continue

        with open(config_path) as f:
            config = yaml.safe_load(f)
        if not config:
            continue

        agent = config.get("agent") or {}
        raw_model = agent.get("model", "")
        model = normalize_model(raw_model)
        if model is None:
            continue

        reasoning_effort = (agent.get("reasoning_effort") or "").strip()
        if not reasoning_effort:
            reasoning_effort = "default"

        env = config.get("env_kwargs") or {}
        rule = env.get("rule", "unknown")
        num_objects = env.get("num_objects", 0)

        with open(results_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                num_correct = row.get("num_correct", 0)
                num_questions = row.get("num_questions", 1)
                all_correct = 1.0 if num_correct == num_questions else 0.0
                rows.append({
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "rule": rule,
                    "num_objects": num_objects,
                    "all_correct": all_correct,
                    "num_correct": num_correct,
                    "num_questions": num_questions,
                })

    if not rows:
        print("No o3-mini or o4-mini data found.")
        return 0

    df = pd.DataFrame(rows)

    if args.num_objects is not None:
        df = df[df["num_objects"] == args.num_objects].copy()
        if df.empty:
            print(f"No o3-mini or o4-mini data found with num_objects={args.num_objects}.")
            return 0
        print(f"Filtered to num_objects={args.num_objects}")

    print("=" * 70)
    print(f"LLM ACCURACY: o3-mini and o4-mini" + (f" ({args.num_objects} objects)" if args.num_objects else ""))
    print("=" * 70)
    print(f"\nTotal trials: {len(df)}")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Reasoning effort: {df['reasoning_effort'].unique().tolist()}")
    print(f"Rules: {df['rule'].unique().tolist()}")
    print(f"Num objects: {df['num_objects'].unique().tolist()}")

    # Overall by model
    print("\n--- Overall accuracy (all trials) ---")
    for model in ["o3-mini", "o4-mini"]:
        sub = df[df["model"] == model]
        if len(sub) == 0:
            print(f"  {model}: No data")
            continue
        mean = sub["all_correct"].mean()
        std = sub["all_correct"].std()
        n = len(sub)
        se = std / np.sqrt(n) if n > 1 else 0.0
        print(f"  {model}: {mean:.2%} (M +/- SE: {mean:.3f} +/- {se:.3f}, n={n})")

    # By model and reasoning_effort
    print("\n--- By model and reasoning effort ---")
    agg = df.groupby(["model", "reasoning_effort"]).agg({
        "all_correct": ["mean", "std", "count"],
    }).reset_index()
    agg.columns = ["model", "reasoning_effort", "mean", "std", "count"]
    agg["se"] = agg["std"] / np.sqrt(agg["count"])

    for _, r in agg.iterrows():
        print(f"  {r['model']} ({r['reasoning_effort']}): {r['mean']:.2%} (n={int(r['count'])}), SE={r['se']:.3f}")

    # By model, reasoning_effort, rule (if we have conjunctive/disjunctive)
    if "conjunctive" in df["rule"].values or "disjunctive" in df["rule"].values:
        print("\n--- By model, reasoning effort, and rule ---")
        agg2 = df[df["rule"].isin(["conjunctive", "disjunctive"])].groupby(
            ["model", "reasoning_effort", "rule"]
        ).agg({"all_correct": ["mean", "std", "count"]}).reset_index()
        agg2.columns = ["model", "reasoning_effort", "rule", "mean", "std", "count"]
        agg2["se"] = agg2["std"] / np.sqrt(agg2["count"])
        for _, r in agg2.iterrows():
            print(f"  {r['model']} ({r['reasoning_effort']}), {r['rule']}: {r['mean']:.2%} (n={int(r['count'])})")

        # Save by-rule CSV for plotting
        suffix = f"_{args.num_objects}obj" if args.num_objects is not None else ""
        out_by_rule = Path(__file__).parent / f"llm_o3_o4_accuracy_by_rule{suffix}.csv"
        agg2.to_csv(out_by_rule, index=False)
        print(f"\nSaved by-rule summary to: {out_by_rule}")

    # Save to CSV
    suffix = f"_{args.num_objects}obj" if args.num_objects is not None else ""
    out_csv = Path(__file__).parent / f"llm_o3_o4_accuracy{suffix}.csv"
    agg.to_csv(out_csv, index=False)
    print(f"Saved summary to: {out_csv}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
