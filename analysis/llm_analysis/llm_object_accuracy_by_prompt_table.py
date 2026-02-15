"""
Table of LLM object identification accuracy by model and prompt type.

Accuracy = proportion of trials in which all objects are correctly identified
(num_correct == num_questions). Uses 4-object trials only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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
PROMPT_ORDER = ["default", "cot", "react", "reflexion"]


def get_prompt_type(agent: dict) -> str | None:
    target = agent.get("_target_", "") or ""
    if "CoTPromptAgent" in target:
        return "cot"
    if "ReflexionPromptsAgent" in target:
        return "reflexion"
    if "PromptsAgent" in target:
        react = agent.get("react", False)
        return "react" if react else "default"
    return None


def load_llm_by_model_prompt(llm_data_dir: Path):
    """LLM trials: all_correct = 1 if num_correct == num_questions (all objects correct)."""
    rows = []
    for results_path in llm_data_dir.rglob("results.jsonl"):
        config_path = results_path.parent / ".hydra" / "config.yaml"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            config = yaml.safe_load(f)
        if not config:
            continue
        env = config.get("env_kwargs") or {}
        if env.get("num_objects") != 4:
            continue
        rule = env.get("rule", "")
        if rule not in ("conjunctive", "disjunctive"):
            continue
        agent = config.get("agent") or {}
        raw_model = agent.get("model", "")
        model = MODEL_DISPLAY.get(raw_model)
        if model is None:
            continue
        prompt_type = get_prompt_type(agent)
        if prompt_type is None:
            continue
        with open(results_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                nc = row.get("num_correct", 0)
                nq = row.get("num_questions", 1)
                all_correct = 1.0 if nc == nq else 0.0
                rows.append({
                    "model": model,
                    "prompt_type": prompt_type,
                    "rule": rule,
                    "all_correct": all_correct,
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="LLM object identification accuracy by model and prompt")
    parser.add_argument("--llm-data", default=None, help="Path to llm_data directory")
    parser.add_argument("--output", default="llm_object_accuracy_by_prompt.csv", help="Output CSV")
    parser.add_argument("--by-rule", action="store_true", help="Include conjunctive/disjunctive columns")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    llm_data_dir = Path(args.llm_data) if args.llm_data else script_dir / "llm_data"
    output_path = script_dir / args.output

    if not llm_data_dir.exists():
        print(f"ERROR: llm_data not found at {llm_data_dir}")
        return 1

    print("Loading LLM data...")
    df = load_llm_by_model_prompt(llm_data_dir)
    if df.empty:
        print("No 4-object LLM data found.")
        return 0

    if args.by_rule:
        agg = (
            df.groupby(["model", "prompt_type", "rule"])["all_correct"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["se"] = agg["std"] / np.sqrt(agg["count"])
        # Pivot to wide: one row per (model, prompt_type), columns conjunctive, disjunctive, overall
        wide = agg.pivot_table(
            index=["model", "prompt_type"],
            columns="rule",
            values="mean",
            aggfunc="first",
        ).reset_index()
        if "conjunctive" in wide.columns and "disjunctive" in wide.columns:
            wide["overall"] = (wide["conjunctive"] + wide["disjunctive"]) / 2
        counts = df.groupby(["model", "prompt_type", "rule"]).size().unstack(fill_value=0)
        wide = wide.merge(
            counts.add_suffix("_n").reset_index(),
            on=["model", "prompt_type"],
            how="left",
        )
        # Reorder columns
        cols = ["model", "prompt_type"]
        if "conjunctive" in wide.columns:
            cols.append("conjunctive")
        if "disjunctive" in wide.columns:
            cols.append("disjunctive")
        if "overall" in wide.columns:
            cols.append("overall")
        for c in wide.columns:
            if c not in cols and c.endswith("_n"):
                cols.append(c)
        wide = wide[[c for c in cols if c in wide.columns]]
        table_df = wide
    else:
        agg = (
            df.groupby(["model", "prompt_type"])["all_correct"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["se"] = agg["std"] / np.sqrt(agg["count"])
        table_df = agg.rename(columns={"mean": "accuracy", "count": "n_trials"})

    # Sort: model order, then prompt order, then by accuracy desc
    model_order = [m for m in MODEL_ORDER if m in table_df["model"].unique()]
    prompt_order = [p for p in PROMPT_ORDER if p in table_df["prompt_type"].unique()]
    table_df["_model_ord"] = table_df["model"].apply(lambda x: model_order.index(x) if x in model_order else 999)
    table_df["_prompt_ord"] = table_df["prompt_type"].apply(lambda p: prompt_order.index(p) if p in prompt_order else 999)
    sort_col = "overall" if "overall" in table_df.columns else ("accuracy" if "accuracy" in table_df.columns else "conjunctive")
    if sort_col in table_df.columns:
        table_df = table_df.sort_values(["_model_ord", "_prompt_ord", sort_col], ascending=[True, True, False])
    else:
        table_df = table_df.sort_values(["_model_ord", "_prompt_ord"])
    table_df = table_df.drop(columns=[c for c in ["_model_ord", "_prompt_ord"] if c in table_df.columns])

    table_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}\n")

    # Pretty print
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    print(table_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    exit(main())
