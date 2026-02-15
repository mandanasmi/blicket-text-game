#!/usr/bin/env python3
"""
Compute number of tests required to reach N hypothesis remaining = 1 for LLMs.
Uses action_log_trial-*.jsonl, 4 objects, by model and rule.
Output: mean and SE for disjunctive and conjunctive, saved to CSV.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add llm_analysis to path for hypothesis_helper
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hypothesis_helper as hp_helper

MODEL_DISPLAY = {
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
    "gpt-4o": "gpt-4o",
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "ollama/gemma3:27b": "gemma3:27b",
    "ollama/qwq": "qwq",
}
MODEL_ORDER = ["deepseek-reasoner", "gpt-4o", "deepseek-chat", "gpt-4o-mini", "gemma3:27b", "qwq"]

PATH_WCS = [
    "2025.03.26/144924/*",
    "2025.03.17/230803/*",
    "2025.03.14/010645/*",
    "2025.03.14/010942/*",
    "2025.03.14/195735/*",
    "2025.03.14/200242/*",
    "2025.03.14/201822/*",
    "2025.03.14/202836/*",
    "2025.03.14/203513/*",
    "2025.03.14/203725/*",
    "2025.03.14/203810/*",
    "2025.03.16/170513/*",
    "2025.03.16/170649/*",
    "2025.03.16/170800/*",
    "2025.03.16/171457/*",
    "2025.03.16/171607/*",
    "2025.03.16/171730/*",
    "2025.03.24/213007/*",
    "2025.03.24/214357/*",
    "2025.03.25/002055/*",
    "2025.03.25/002205/*",
    "2025.03.25/002257/*",
    "2025.03.23/165130/*",
    "2025.03.24/165351/*",
    "2025.03.24/165605/*",
    "2025.03.25/002758/*",
    "2025.03.25/165819/*",
    "2025.03.25/165942/*",
    "2025.03.25/171404/*",
    "2025.03.23/165243/*",
    "2025.03.24/165513/*",
    "2025.03.25/003419/*",
    "2025.03.25/164038/*",
    "2025.03.25/164214/*",
    "2025.03.25/001402/*",
    "2025.03.23/204923/*",
    "2025.03.25/170432/*",
]


def _dedupe_states(data):
    """Keep only rows where state changed from previous (unique state transitions)."""
    if len(data) <= 1:
        return data
    out = [data[0]]
    for i in range(1, len(data)):
        if data[i] != data[i - 1]:
            out.append(data[i])
    return out


def tests_to_one_hypothesis(state_traj):
    """
    Return number of tests (1-based) when N hypothesis remaining first reaches 1.
    state_traj: list of [obj0_on,...,objN_on, machine_lit] per step.
    Returns np.nan if never reaches 1.
    """
    state_traj = np.asarray(state_traj)
    if state_traj.size == 0:
        return np.nan
    for t in range(state_traj.shape[0]):
        n = hp_helper.compute_num_valid_hypothesis(state_traj[: t + 1])
        if n == 1:
            return t + 1  # 1-based number of tests
    return np.nan


def main():
    parser = argparse.ArgumentParser(description="Compute tests to 1 hypothesis for LLMs (4 objects)")
    parser.add_argument("--llm-data", default="/tmp/llm_data", help="Path to llm_data directory")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    llm_data_dir = Path(args.llm_data)
    if not llm_data_dir.exists():
        print(f"ERROR: llm_data not found at {llm_data_dir}")
        return 1

    rows = []
    for wc in PATH_WCS:
        for exp_dir in llm_data_dir.glob(wc):
            if not exp_dir.is_dir():
                continue
            config_path = exp_dir / ".hydra" / "config.yaml"
            if not config_path.exists():
                continue
            with open(config_path) as f:
                config = yaml.safe_load(f)
            if not config:
                continue
            env = config.get("env_kwargs") or {}
            if env.get("num_objects") != 4:
                continue
            rule = env.get("rule", "unknown")
            if rule not in ("conjunctive", "disjunctive"):
                continue
            raw_model = (config.get("agent") or {}).get("model", "")
            model = MODEL_DISPLAY.get(raw_model)
            if model is None:
                continue

            for log_path in sorted(exp_dir.glob("action_log_trial-*.jsonl")):
                data = []
                with open(log_path) as f:
                    for line in f:
                        try:
                            cur = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if "question" in cur.get("prompt", "").lower():
                            continue
                        gs = cur.get("game_state")
                        if not gs or "true_state" not in gs:
                            continue
                        ts = gs["true_state"]
                        if isinstance(ts, list) and len(ts) >= 5:
                            vals = []
                            for v in ts:
                                if isinstance(v, bool):
                                    vals.append(v)
                                elif isinstance(v, str) and v.lower() in ("true", "false"):
                                    vals.append(v.lower() == "true")
                                else:
                                    vals.append(bool(v))
                            if len(vals) == 5:
                                data.append(vals)
                data = _dedupe_states(data)
                if not data:
                    continue
                t2one = tests_to_one_hypothesis(data)
                rows.append({"model": model, "rule": rule, "tests_to_one": t2one, "file": str(log_path)})

    if not rows:
        print("No LLM action log data found (4 objects).")
        return 0

    df = pd.DataFrame(rows)
    # Drop trials that never reached 1 (optional: keep them as NaN for mean)
    df_valid = df[df["tests_to_one"].notna()]

    agg = df_valid.groupby(["model", "rule"]).agg(
        mean=("tests_to_one", "mean"),
        std=("tests_to_one", "std"),
        count=("tests_to_one", "count"),
    ).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    agg.loc[agg["count"] == 1, "se"] = 0

    out_path = Path(args.output) if args.output else Path(__file__).parent / "llm_tests_to_one_hypothesis_4obj.csv"
    agg.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(agg.to_string())
    return 0


if __name__ == "__main__":
    exit(main())
