"""
Extract exploration efficiency metrics for Human (active, no prior) vs LLM,
by rule (conjunctive vs disjunctive). Compare: tests to 1 hypothesis, number of tests,
and (human only) action history length.

Data:
  - Human: human_active_data_no_prior_experience.json + comprehensive_correlation_data_no_prior_102.csv (round7)
  - LLM: llm_tests_to_one_hypothesis_4obj.csv (from compute_tests_to_one_hypothesis.py)
        + optional: action_log_trial-*.jsonl for num tests per trial

Outputs: exploration_efficiency_human_vs_llm.csv, exploration_efficiency_summary.md
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hypothesis_helper as hp_helper

NUM_OBJECTS = 4
PATH_WCS = [
    "2025.03.26/144924/*", "2025.03.17/230803/*", "2025.03.14/010645/*", "2025.03.14/010942/*",
    "2025.03.14/195735/*", "2025.03.14/200242/*", "2025.03.14/201822/*", "2025.03.14/202836/*",
    "2025.03.14/203513/*", "2025.03.14/203725/*", "2025.03.14/203810/*", "2025.03.16/170513/*",
    "2025.03.16/170649/*", "2025.03.16/170800/*", "2025.03.16/171457/*", "2025.03.16/171607/*",
    "2025.03.16/171730/*", "2025.03.24/213007/*", "2025.03.24/214357/*", "2025.03.25/002055/*",
    "2025.03.25/002205/*", "2025.03.25/002257/*", "2025.03.23/165130/*", "2025.03.24/165351/*",
    "2025.03.24/165605/*", "2025.03.25/002758/*", "2025.03.25/165819/*", "2025.03.25/165942/*",
    "2025.03.25/171404/*", "2025.03.23/165243/*", "2025.03.24/165513/*", "2025.03.25/003419/*",
    "2025.03.25/164038/*", "2025.03.25/164214/*", "2025.03.25/001402/*", "2025.03.23/204923/*",
    "2025.03.25/170432/*",
]
MODEL_DISPLAY = {
    "deepseek-chat": "deepseek-chat", "deepseek-reasoner": "deepseek-reasoner",
    "gpt-4o": "gpt-4o", "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini", "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "ollama/gemma3:27b": "gemma3:27b", "ollama/qwq": "qwq",
}


def _get_true_rule(rd):
    out = rd.get("true_rule") or (rd.get("config") or {}).get("rule") or rd.get("rule")
    return (out or "").strip().lower()


def _has_prior(entry):
    comp = entry.get("comprehension") or {}
    sg = (comp if isinstance(comp, dict) else {}).get("similar_game_experience") or {}
    if not isinstance(sg, dict):
        return None
    a = (sg.get("answer") or "").strip().lower()
    if "yes" in a:
        return True
    if "no" in a:
        return False
    return None


def state_history_to_traj(sh):
    """Convert human state_history to state_traj format [obj0_on,...,obj3_on, machine_lit] for hypothesis_helper."""
    traj = []
    for ent in sh:
        ob = ent.get("objects_on_machine")
        lit = ent.get("machine_lit")
        if ob is None or lit is None:
            continue
        o = set(ob) if isinstance(ob, (list, tuple)) else set()
        row = [1 if i in o else 0 for i in range(NUM_OBJECTS)]
        row.append(1 if lit else 0)
        traj.append(row)
    return traj


def tests_to_one_hypothesis(state_traj):
    """Return 1-based number of tests when N hypothesis remaining first reaches 1, or np.nan."""
    if not state_traj:
        return np.nan
    state_traj = np.asarray(state_traj)
    for t in range(state_traj.shape[0]):
        n = hp_helper.compute_num_valid_hypothesis(state_traj[: t + 1])
        if n == 1:
            return t + 1
    return np.nan


def load_human_exploration(json_path: Path, csv_path: Path):
    """Load human tests_to_one, num_tests, action_history_length by rule (no prior, 102 participants)."""
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for pid, entry in data.items():
        if not isinstance(entry, dict) or _has_prior(entry) is True:
            continue
        mg = entry.get("main_game") or {}
        if not isinstance(mg, dict):
            continue
        rounds = (
            [(k, mg[k]) for k in mg if isinstance(k, str) and k.startswith("round_") and isinstance(mg.get(k), dict)]
            if any(isinstance(k, str) and k.startswith("round_") for k in mg)
            else [("main", mg)]
        )
        for _rk, rd in rounds:
            rule = _get_true_rule(rd)
            if rule not in ("conjunctive", "disjunctive"):
                continue
            sh = rd.get("state_history") or []
            if not isinstance(sh, list) or len(sh) < 1:
                continue
            traj = state_history_to_traj(sh)
            t2one = tests_to_one_hypothesis(traj)
            num_tests = len(traj)
            action_len = len(rd.get("action_history") or [])
            rows.append({"participant_id": pid, "rule": rule, "tests_to_one": t2one, "num_tests": num_tests, "action_history_length": action_len})
            break
    df = pd.DataFrame(rows)

    # Merge CSV for num_unique_tests if available
    if csv_path.exists():
        csv_df = pd.read_csv(csv_path)
        csv_df["true_rule"] = csv_df["true_rule"].str.strip().str.lower()
        csv_df = csv_df[["participant_id", "true_rule", "num_unique_tests", "action_history_length"]].rename(columns={"true_rule": "rule", "num_unique_tests": "num_unique_tests_csv"})
        df = df.merge(csv_df, on=["participant_id", "rule"], how="left", suffixes=("", "_csv"))
        if "action_history_length_csv" in df.columns:
            df["action_history_length"] = df["action_history_length"].fillna(df["action_history_length_csv"])
        if "num_unique_tests_csv" in df.columns:
            df["num_tests"] = df["num_tests"].fillna(df["num_unique_tests_csv"])

    return df


def load_llm_tests_to_one(csv_path: Path):
    """Load LLM tests_to_one by model and rule from precomputed CSV."""
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["rule"] = df["rule"].str.strip().str.lower()
    return df


def main():
    parser = argparse.ArgumentParser(description="Human vs LLM exploration efficiency by rule")
    parser.add_argument("--human-json", default=None, help="Human JSON (round7 no prior)")
    parser.add_argument("--human-csv", default=None, help="Human CSV (round7 102 participants)")
    parser.add_argument("--llm-csv", default=None, help="LLM tests_to_one CSV (4 obj)")
    parser.add_argument("--out-csv", default="exploration_efficiency_human_vs_llm.csv", help="Output CSV")
    parser.add_argument("--out-md", default="exploration_efficiency_summary.md", help="Output summary markdown")
    parser.add_argument("--plot", action="store_true", help="Save bar chart: human exploration by rule")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    round7 = project_root / "analysis" / "results" / "round7"

    human_json = Path(args.human_json) if args.human_json else round7 / "human_active_data_no_prior_experience.json"
    human_csv = Path(args.human_csv) if args.human_csv else round7 / "comprehensive_correlation_data_no_prior_102.csv"
    llm_csv = Path(args.llm_csv) if args.llm_csv else script_dir / "llm_tests_to_one_hypothesis_4obj.csv"
    out_csv = script_dir / args.out_csv
    out_md = script_dir / args.out_md

    # --- Human ---
    if not human_json.exists():
        print(f"Human JSON not found: {human_json}")
        human_df = pd.DataFrame()
    else:
        human_df = load_human_exploration(human_json, human_csv)
        print(f"Human: {len(human_df)} participants")

    # --- LLM ---
    llm_df = load_llm_tests_to_one(llm_csv)
    if not llm_df.empty:
        print(f"LLM: {len(llm_df)} rows (by model x rule)")

    # --- Aggregate Human by rule ---
    out_rows = []
    if not human_df.empty:
        for rule in ["conjunctive", "disjunctive"]:
            sub = human_df[human_df["rule"] == rule]
            if len(sub) == 0:
                continue
            t2one = sub["tests_to_one"].dropna()
            out_rows.append({
                "source": "human",
                "rule": rule,
                "n": len(sub),
                "tests_to_one_mean": t2one.mean() if len(t2one) else np.nan,
                "tests_to_one_se": t2one.sem() if len(t2one) > 1 else 0,
                "tests_to_one_n_reached": int(t2one.notna().sum()),
                "num_tests_mean": sub["num_tests"].mean(),
                "num_tests_se": sub["num_tests"].sem() if len(sub) > 1 else 0,
                "action_history_length_mean": sub["action_history_length"].mean(),
                "action_history_length_se": sub["action_history_length"].sem() if len(sub) > 1 else 0,
            })

    # --- Aggregate LLM by rule (across models) ---
    if not llm_df.empty:
        for rule in ["conjunctive", "disjunctive"]:
            sub = llm_df[llm_df["rule"] == rule]
            if len(sub) == 0:
                continue
            # Weighted mean across models (by count)
            total_count = sub["count"].sum()
            if total_count == 0:
                continue
            mean_weighted = (sub["mean"] * sub["count"]).sum() / total_count
            # Approx SE across trials: pool std and n
            out_rows.append({
                "source": "llm",
                "rule": rule,
                "n": int(total_count),
                "tests_to_one_mean": mean_weighted,
                "tests_to_one_se": sub["se"].mean(),  # rough
                "tests_to_one_n_reached": int(total_count),
                "num_tests_mean": np.nan,
                "num_tests_se": np.nan,
                "action_history_length_mean": np.nan,
                "action_history_length_se": np.nan,
            })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # --- Summary markdown ---
    lines = [
        "# Exploration efficiency: Human (active, no prior) vs LLM",
        "",
        "## Data",
        f"- Human: `{human_json.name}` (n=102), `{human_csv.name}`",
        f"- LLM: `{llm_csv.name}` (tests to 1 hypothesis, 4 objects)",
        "",
        "## Metrics by rule",
        "",
    ]
    if not out_df.empty:
        for _, r in out_df.iterrows():
            s = r["source"]
            rule = r["rule"]
            n = r["n"]
            t2one = r["tests_to_one_mean"]
            t2one_se = r["tests_to_one_se"]
            n_reach = r.get("tests_to_one_n_reached", "")
            lines.append(f"### {s.capitalize()} – {rule}")
            lines.append(f"- N: {n}")
            lines.append(f"- Tests to 1 hypothesis: mean = {t2one:.2f}, SE = {t2one_se:.2f} (trials that reached 1: {n_reach})")
            if pd.notna(r.get("num_tests_mean")):
                lines.append(f"- Num tests: mean = {r['num_tests_mean']:.1f}, SE = {r['num_tests_se']:.1f}")
            if pd.notna(r.get("action_history_length_mean")):
                lines.append(f"- Action history length: mean = {r['action_history_length_mean']:.1f}, SE = {r['action_history_length_se']:.1f}")
            lines.append("")
        lines.append("## Interpretation")
        lines.append("- **Tests to 1 hypothesis**: number of tests until only one hypothesis remains (lower = more efficient).")
        lines.append("- **Num tests**: total tests performed before Q&A.")
        lines.append("- **Action history length**: total actions (place/remove/test) before Q&A (human only).")
        lines.append("- LLM trials are aggregated across models; human is 51 conjunctive + 51 disjunctive (no prior).")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {out_md}")

    # --- Optional bar chart: human exploration by rule ---
    if args.plot and not out_df.empty:
        human_out = out_df[out_df["source"] == "human"]
        if len(human_out) >= 2:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            mpl.rcParams["font.family"] = "sans-serif"
            fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
            x = np.arange(2)
            w = 0.5
            labels = [r.capitalize() for r in ["conjunctive", "disjunctive"]]
            colors = ["#2a9d9f", "#e76f51"]
            # Num tests
            ax = axes[0]
            means = [float(human_out[human_out["rule"] == r]["num_tests_mean"].iloc[0]) for r in ["conjunctive", "disjunctive"]]
            ses = [float(human_out[human_out["rule"] == r]["num_tests_se"].iloc[0]) for r in ["conjunctive", "disjunctive"]]
            ax.bar(x, means, w, yerr=ses, capsize=4, color=colors, alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel("Number of tests")
            ax.set_title("Human: tests before Q&A by rule")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            # Action history length
            ax = axes[1]
            means = [float(human_out[human_out["rule"] == r]["action_history_length_mean"].iloc[0]) for r in ["conjunctive", "disjunctive"]]
            ses = [float(human_out[human_out["rule"] == r]["action_history_length_se"].iloc[0]) for r in ["conjunctive", "disjunctive"]]
            ax.bar(x, means, w, yerr=ses, capsize=4, color=colors, alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel("Action history length")
            ax.set_title("Human: actions before Q&A by rule")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            plot_path = script_dir / "exploration_efficiency_human_by_rule.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
            plt.close()
            print(f"Saved: {plot_path}")

    return 0


if __name__ == "__main__":
    exit(main())
