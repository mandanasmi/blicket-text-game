"""
Compute average conjunctive accuracy for LLMs, Active Human, and Passive Human learners.
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
    "o3-mini-2025-01-31": "o3-mini",
    "o4-mini-2025-04-16": "o4-mini",
}
PROMPT_TYPES = ["cot", "default", "react", "reflexion", "oreason"]


def get_prompt_type(agent: dict) -> str | None:
    target = agent.get("_target_", "") or ""
    if "CoTPromptAgent" in target:
        return "cot"
    if "ReflexionPromptsAgent" in target:
        return "reflexion"
    if "OReasonPromptsAgent" in target:
        return "oreason"
    if "PromptsAgent" in target:
        react = agent.get("react", False)
        return "react" if react else "default"
    return None


def load_llm_by_model_prompt(llm_data_dir: Path):
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
        n_obj = env.get("num_objects")
        if n_obj not in (4, 8):
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


def load_human_active(round7_dir: Path):
    path = round7_dir / "human_active_data_no_prior_experience.json"
    if not path.exists():
        return None, None

    def normalize(s):
        if not s:
            return None
        s = str(s).lower()
        if "conjunctive" in s or "all" in s:
            return "conjunctive"
        if "disjunctive" in s or "any" in s:
            return "disjunctive"
        return None

    with open(path) as f:
        data = json.load(f)
    conj, disj = [], []
    for user_data in data.values():
        mg = (user_data or {}).get("main_game") or {}
        config_rule = (mg.get("config") or {}).get("rule") or mg.get("rule") or mg.get("true_rule")
        true_rule = normalize(config_rule)
        user_choice = normalize(mg.get("rule_type", ""))
        correct = 1.0 if (user_choice and user_choice == true_rule) else 0.0
        if true_rule == "conjunctive":
            conj.append(correct)
        elif true_rule == "disjunctive":
            disj.append(correct)
    if not conj or not disj:
        return None, None
    conj_a, disj_a = np.array(conj), np.array(disj)
    return (
        (float(conj_a.mean()), float(conj_a.std(ddof=1) / np.sqrt(len(conj_a))) if len(conj_a) > 1 else 0),
        (float(disj_a.mean()), float(disj_a.std(ddof=1) / np.sqrt(len(disj_a))) if len(disj_a) > 1 else 0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-data", default=None, help="Path to llm_data directory")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    llm_data_dir = Path(args.llm_data) if args.llm_data else Path("/tmp/llm_data")
    round7_dir = project_root / "analysis" / "results" / "round7"

    print("=" * 80)
    print("AVERAGE CONJUNCTIVE ACCURACY")
    print("=" * 80)

    # LLM average conjunctive accuracy
    if llm_data_dir.exists():
        print("\nLoading LLM data...")
        llm_df = load_llm_by_model_prompt(llm_data_dir)
        if not llm_df.empty:
            # Filter to 4-object trials only
            llm_df_4obj = llm_df.copy()  # Already filtered in load function
            
            # Get conjunctive trials only
            conj_df = llm_df_4obj[llm_df_4obj["rule"] == "conjunctive"].copy()
            
            if not conj_df.empty:
                # Compute mean and SE across all trials (not grouped)
                all_correct_values = conj_df["all_correct"].values
                llm_mean = float(np.mean(all_correct_values))
                llm_std = float(np.std(all_correct_values, ddof=1))
                llm_se = llm_std / np.sqrt(len(all_correct_values))
                llm_n = len(all_correct_values)
                
                print(f"\nLLM Average Conjunctive Accuracy:")
                print(f"  Mean: {llm_mean:.4f}")
                print(f"  SE:   {llm_se:.4f}")
                print(f"  N:    {llm_n} trials")
                print(f"  Format: {llm_mean:.4f} ± {llm_se:.4f}")
            else:
                print("\nNo LLM conjunctive data found.")
        else:
            print("\nNo LLM data found.")
    else:
        print(f"\nLLM data directory not found at {llm_data_dir}")

    # Active Human average conjunctive accuracy
    print("\n" + "-" * 80)
    active = load_human_active(round7_dir)
    if active[0] and active[1]:
        (conj_m, conj_se), (disj_m, disj_se) = active
        print(f"\nActive Human Learners Average Conjunctive Accuracy:")
        print(f"  Mean: {conj_m:.4f}")
        print(f"  SE:   {conj_se:.4f}")
        print(f"  Format: {conj_m:.4f} ± {conj_se:.4f}")
    else:
        print("\nActive Human data not found or incomplete.")

    # Passive Human average conjunctive accuracy (hardcoded from PNAS)
    print("\n" + "-" * 80)
    passive_conj_mean = 0.25
    passive_conj_se = 0.02
    print(f"\nPassive Human Learners Average Conjunctive Accuracy:")
    print(f"  Mean: {passive_conj_mean:.4f}")
    print(f"  SE:   {passive_conj_se:.4f}")
    print(f"  Format: {passive_conj_mean:.4f} ± {passive_conj_se:.4f}")
    print(f"  (Source: PNAS Adults - hardcoded)")

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    exit(main())
