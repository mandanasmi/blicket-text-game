"""
Compute number of tests per LLM trial from action_log_trial-*.jsonl (4 objects).
Uses the same LLM set as active_adults_vs_llm_object_accuracy.png (4-object runs only).
num_tests = number of distinct machine observations per trial: from game_state.unique_state_visited
on the last log line when present, else from counting distinct game_state.true_state (deduped).
Output: llm_num_tests_per_trial_4obj.csv (per trial) and llm_num_tests_by_rule_4obj.csv (aggregated).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Same models as in active_adults_vs_llm_object_accuracy.png (4-object runs).
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


def _dedupe_states(data):
    if len(data) <= 1:
        return data
    out = [data[0]]
    for i in range(1, len(data)):
        if data[i] != data[i - 1]:
            out.append(data[i])
    return out


def _num_tests_from_log(log_path: Path) -> int | None:
    """Extract num_tests for one trial: unique_state_visited from last line, or count distinct true_state."""
    last_gs = None
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
            if not gs:
                continue
            last_gs = gs
            if "true_state" not in gs:
                continue
            ts = gs["true_state"]
            if not isinstance(ts, list) or len(ts) < 5:
                continue
            vals = []
            for v in ts[:5]:
                if isinstance(v, bool):
                    vals.append(v)
                elif isinstance(v, str) and v.lower() in ("true", "false"):
                    vals.append(v.lower() == "true")
                else:
                    vals.append(bool(v))
            if len(vals) == 5:
                data.append(vals)
    if last_gs is not None and "unique_state_visited" in last_gs:
        try:
            return int(last_gs["unique_state_visited"])
        except (TypeError, ValueError):
            pass
    if not data:
        return None
    data = _dedupe_states(data)
    return len(data)


def main():
    parser = argparse.ArgumentParser(description="Compute num_tests per LLM trial (4 objects, same models as figure)")
    parser.add_argument("--llm-data", default=None, help="Path to llm_data directory (default: script_dir/llm_data)")
    parser.add_argument("--out-trials", default="llm_num_tests_per_trial_4obj.csv", help="Output CSV per trial")
    parser.add_argument("--out-agg", default="llm_num_tests_by_rule_4obj.csv", help="Output CSV aggregated by model, rule")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    llm_data_dir = Path(args.llm_data) if args.llm_data else (script_dir / "llm_data")
    if not llm_data_dir.exists():
        print(f"ERROR: llm_data not found at {llm_data_dir}")
        return 1

    rows = []
    # Discover runs that have action logs (same data source as figure: 4-object, conjunctive/disjunctive).
    for log_path in sorted(llm_data_dir.rglob("action_log_trial-*.jsonl")):
        exp_dir = log_path.parent
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

        num_tests = _num_tests_from_log(log_path)
        if num_tests is None:
            continue
        rows.append({"model": model, "rule": rule, "num_tests": num_tests, "file": str(log_path)})

    if not rows:
        print("No LLM action log data found (4 objects).")
        return 0

    df = pd.DataFrame(rows)
    out_trials = script_dir / args.out_trials
    out_agg = script_dir / args.out_agg
    df.to_csv(out_trials, index=False)
    print(f"Saved: {out_trials} ({len(df)} trials)")

    agg = df.groupby(["model", "rule"]).agg(
        mean=("num_tests", "mean"),
        std=("num_tests", "std"),
        count=("num_tests", "count"),
    ).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    agg.loc[agg["count"] == 1, "se"] = 0
    agg.to_csv(out_agg, index=False)
    print(f"Saved: {out_agg}")
    print(agg.to_string())
    return 0


if __name__ == "__main__":
    exit(main())
