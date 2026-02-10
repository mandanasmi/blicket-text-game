"""
Compute average number of actions, tests, total time, and time to success
for conjunctive vs disjunctive participants. Output as table with SE.

Uses human_active_data_no_prior_experience.json.
"""

import json
import numpy as np
from datetime import datetime


def normalize_rule(s):
    """Normalize rule string to 'conjunctive' or 'disjunctive'."""
    if not s:
        return None
    s = str(s).lower()
    if 'conjunctive' in s or 'all' in s:
        return 'conjunctive'
    if 'disjunctive' in s or 'any' in s:
        return 'disjunctive'
    return None


def parse_duration_iso(start_str, end_str):
    """Parse start/end ISO strings and return duration in seconds. Returns None if parse fails."""
    try:
        start = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
        return (end - start).total_seconds()
    except Exception:
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute actions, tests, time, exploration time by rule type.")
    parser.add_argument("--input", default="../human_active_data_no_prior_experience.json", help="Input JSON (default: from round7)")
    args = parser.parse_args()
    print(f"Loading {args.input} ...")
    with open(args.input, "r") as f:
        data = json.load(f)

    records = {'conjunctive': [], 'disjunctive': []}

    for user_id, user_data in data.items():
        if 'main_game' not in user_data:
            continue
        mg = user_data['main_game']
        config_rule = (mg.get('config') or {}).get('rule') or mg.get('rule') or mg.get('true_rule')
        true_rule = normalize_rule(config_rule)
        if true_rule not in ('conjunctive', 'disjunctive'):
            continue

        # Number of actions
        ah = mg.get('action_history') or []
        num_actions = len(ah) if isinstance(ah, list) else mg.get('action_history_length', 0)

        # Number of tests
        tt = mg.get('test_timings') or []
        num_tests = len(tt) if isinstance(tt, list) else 0

        # Total time (seconds)
        total_time = mg.get('total_test_time_seconds')
        if total_time is None and isinstance(tt, list) and tt:
            total_time = sum(
                float(t.get('time_since_previous_seconds') or 0)
                for t in tt if isinstance(t, dict)
            )
        if total_time is None:
            total_time = parse_duration_iso(mg.get('start_time'), mg.get('end_time'))
        if total_time is None:
            total_time = np.nan
        else:
            total_time = float(total_time)

        # Exploration time (seconds): round start to last test click
        exploration_time = None
        if isinstance(tt, list) and len(tt) > 0:
            last_t = tt[-1]
            if isinstance(last_t, dict):
                exploration_time = last_t.get('time_since_start_seconds')
        if exploration_time is None:
            exploration_time = np.nan
        else:
            exploration_time = float(exploration_time)

        # Object and rule correctness
        true_blickets = set(mg.get('true_blicket_indices', []))
        user_blickets = set(mg.get('user_chosen_blickets', []))
        obj_ok = true_blickets == user_blickets
        user_choice = normalize_rule(mg.get('rule_type', ''))
        rule_ok = user_choice == true_rule if user_choice else False
        both_ok = obj_ok and rule_ok

        records[true_rule].append({
            'num_actions': num_actions,
            'num_tests': num_tests,
            'total_time': total_time,
            'exploration_time': exploration_time,
            'both_ok': both_ok,
        })

    # Max tests allowed in the game (from round config horizon; UI shows "X out of 16")
    MAX_TESTS_ALLOWED = 16

    # Compute mean and SE by rule type
    def mean_se(arr):
        arr = np.array([x for x in arr if np.isfinite(x)])
        n = len(arr)
        if n == 0:
            return np.nan, np.nan
        m = np.mean(arr)
        se = np.std(arr, ddof=1) / np.sqrt(n) if n > 1 else 0.0
        return m, se

    results = {}
    for rt in ['conjunctive', 'disjunctive']:
        recs = records[rt]
        n = len(recs)
        if n == 0:
            results[rt] = {'n': 0}
            continue

        actions = [r['num_actions'] for r in recs]
        tests = [r['num_tests'] for r in recs]
        n_maxed = sum(1 for r in recs if r['num_tests'] >= MAX_TESTS_ALLOWED)
        n_maxed_both_ok = sum(1 for r in recs if r['num_tests'] >= MAX_TESTS_ALLOWED and r['both_ok'])
        total_times = [r['total_time'] for r in recs]
        time_per_test = [r['total_time'] / r['num_tests'] for r in recs
                        if r['num_tests'] > 0 and np.isfinite(r['total_time'])]
        actions_per_test = [r['num_actions'] / r['num_tests'] for r in recs if r['num_tests'] > 0]
        exploration_times = [r['exploration_time'] for r in recs]
        time_to_success = [r['total_time'] for r in recs if r['both_ok'] and np.isfinite(r['total_time'])]
        tests_to_success = [r['num_tests'] for r in recs if r['both_ok']]

        m_actions, se_actions = mean_se(actions)
        m_tests, se_tests = mean_se(tests)
        m_total, se_total = mean_se(total_times)
        m_tpt, se_tpt = mean_se(time_per_test)
        m_apt, se_apt = mean_se(actions_per_test)
        m_exp, se_exp = mean_se(exploration_times)
        m_tts, se_tts = mean_se(time_to_success)
        m_tts_tests, se_tts_tests = mean_se(tests_to_success)

        results[rt] = {
            'n': n,
            'n_success': len(time_to_success),
            'max_tests_allowed': MAX_TESTS_ALLOWED,
            'n_maxed_out': n_maxed,
            'n_maxed_out_both_ok': n_maxed_both_ok,
            'pct_maxed_out': 100.0 * n_maxed / n if n else 0,
            'mean_actions': m_actions, 'se_actions': se_actions,
            'mean_tests': m_tests, 'se_tests': se_tests,
            'mean_total_time': m_total, 'se_total_time': se_total,
            'mean_time_per_test': m_tpt, 'se_time_per_test': se_tpt,
            'mean_actions_per_test': m_apt, 'se_actions_per_test': se_apt,
            'mean_exploration_time': m_exp, 'se_exploration_time': se_exp,
            'mean_time_to_success': m_tts, 'se_time_to_success': se_tts,
            'mean_tests_to_success': m_tts_tests, 'se_tests_to_success': se_tts_tests,
        }

    # Print table
    print("\n" + "=" * 90)
    print("ACTIONS, TESTS, AND TIME BY RULE TYPE (human_active_data_no_prior_experience.json)")
    print("=" * 90)
    print(f"\n{'Metric':<30} {'Conjunctive':<25} {'Disjunctive':<25}")
    print("-" * 90)

    for label, key in [
        ('Number of actions (M +/- SE)', ('mean_actions', 'se_actions')),
        ('Actions per test (M +/- SE)', ('mean_actions_per_test', 'se_actions_per_test')),
        ('# Tests (all participants) (M +/- SE)', ('mean_tests', 'se_tests')),
        ('# Tests to Success (successful only) (M +/- SE)', ('mean_tests_to_success', 'se_tests_to_success')),
        ('Total Test Time, s (M +/- SE)', ('mean_total_time', 'se_total_time')),
        ('Time per test, s (M +/- SE)', ('mean_time_per_test', 'se_time_per_test')),
        ('Exploration Time, s (M +/- SE)', ('mean_exploration_time', 'se_exploration_time')),
        ('Time to success, s (M +/- SE)*', ('mean_time_to_success', 'se_time_to_success')),
    ]:
        m_c, se_c = results['conjunctive'].get(key[0], np.nan), results['conjunctive'].get(key[1], np.nan)
        m_d, se_d = results['disjunctive'].get(key[0], np.nan), results['disjunctive'].get(key[1], np.nan)
        conj_str = f"{m_c:.2f} +/- {se_c:.2f}" if np.isfinite(m_c) else "N/A"
        disj_str = f"{m_d:.2f} +/- {se_d:.2f}" if np.isfinite(m_d) else "N/A"
        print(f"{label:<30} {conj_str:<25} {disj_str:<25}")

    print(f"{'N maxed out tests (' + str(MAX_TESTS_ALLOWED) + ') (count)':<30} {results['conjunctive']['n_maxed_out']:<25} {results['disjunctive']['n_maxed_out']:<25}")
    print(f"{'N maxed out & both correct':<30} {results['conjunctive']['n_maxed_out_both_ok']:<25} {results['disjunctive']['n_maxed_out_both_ok']:<25}")
    print(f"{'N maxed out tests (%)':<30} {results['conjunctive']['pct_maxed_out']:.1f}%{'':<20} {results['disjunctive']['pct_maxed_out']:.1f}%")
    print("-" * 90)
    print(f"{'N participants':<30} {results['conjunctive']['n']:<25} {results['disjunctive']['n']:<25}")
    print(f"{'N successful (both correct)':<30} {results['conjunctive']['n_success']:<25} {results['disjunctive']['n_success']:<25}")
    print("\n* Time to success = among participants who got both objects and rule correct.")
    print(f"* Max tests allowed in game = {MAX_TESTS_ALLOWED}.")
    print("=" * 90)

    # Save table to file
    out_path = 'actions_tests_time_by_rule_type.txt'
    with open(out_path, 'w') as f:
        f.write("ACTIONS, TESTS, AND TIME BY RULE TYPE\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"{'Metric':<30} {'Conjunctive':<25} {'Disjunctive':<25}\n")
        f.write("-" * 90 + "\n")
        for label, key in [
            ('Number of actions (M +/- SE)', ('mean_actions', 'se_actions')),
            ('Actions per test (M +/- SE)', ('mean_actions_per_test', 'se_actions_per_test')),
            ('# Tests (all participants) (M +/- SE)', ('mean_tests', 'se_tests')),
            ('# Tests to Success (successful only) (M +/- SE)', ('mean_tests_to_success', 'se_tests_to_success')),
            ('Total Test Time, s (M +/- SE)', ('mean_total_time', 'se_total_time')),
            ('Time per test, s (M +/- SE)', ('mean_time_per_test', 'se_time_per_test')),
            ('Exploration Time, s (M +/- SE)', ('mean_exploration_time', 'se_exploration_time')),
            ('Time to success, s (M +/- SE)*', ('mean_time_to_success', 'se_time_to_success')),
        ]:
            m_c, se_c = results['conjunctive'].get(key[0], np.nan), results['conjunctive'].get(key[1], np.nan)
            m_d, se_d = results['disjunctive'].get(key[0], np.nan), results['disjunctive'].get(key[1], np.nan)
            conj_str = f"{m_c:.2f} +/- {se_c:.2f}" if np.isfinite(m_c) else "N/A"
            disj_str = f"{m_d:.2f} +/- {se_d:.2f}" if np.isfinite(m_d) else "N/A"
            f.write(f"{label:<30} {conj_str:<25} {disj_str:<25}\n")
        f.write(f"{'N maxed out tests (' + str(MAX_TESTS_ALLOWED) + ') (count)':<30} {results['conjunctive']['n_maxed_out']:<25} {results['disjunctive']['n_maxed_out']:<25}\n")
        f.write(f"{'N maxed out & both correct':<30} {results['conjunctive']['n_maxed_out_both_ok']:<25} {results['disjunctive']['n_maxed_out_both_ok']:<25}\n")
        f.write(f"{'N maxed out tests (%)':<30} {results['conjunctive']['pct_maxed_out']:.1f}%{'':<20} {results['disjunctive']['pct_maxed_out']:.1f}%\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'N participants':<30} {results['conjunctive']['n']:<25} {results['disjunctive']['n']:<25}\n")
        f.write(f"{'N successful (both correct)':<30} {results['conjunctive']['n_success']:<25} {results['disjunctive']['n_success']:<25}\n")
        f.write("\n* Time to success = among participants who got both objects and rule correct.\n")
        f.write(f"* Max tests allowed in game = {MAX_TESTS_ALLOWED}.\n")
    print(f"\nTable saved to: {out_path}")

    # LaTeX table
    latex_path = 'actions_tests_time_by_rule_type_latex.txt'
    with open(latex_path, 'w') as f:
        f.write(r"    Metric ($M \pm SE$) & Conjunctive & Disjunctive \\" + "\n")
        f.write(r"    \hline" + "\n")
        f.write(f"    \\# Actions & ${results['conjunctive']['mean_actions']:.2f} \\pm {results['conjunctive']['se_actions']:.2f}$ & ${results['disjunctive']['mean_actions']:.2f} \\pm {results['disjunctive']['se_actions']:.2f}$ \\\\\n")
        f.write(f"    Actions per test & ${results['conjunctive']['mean_actions_per_test']:.2f} \\pm {results['conjunctive']['se_actions_per_test']:.2f}$ & ${results['disjunctive']['mean_actions_per_test']:.2f} \\pm {results['disjunctive']['se_actions_per_test']:.2f}$ \\\\\n")
        f.write(f"    \\# Tests (all) & ${results['conjunctive']['mean_tests']:.2f} \\pm {results['conjunctive']['se_tests']:.2f}$ & ${results['disjunctive']['mean_tests']:.2f} \\pm {results['disjunctive']['se_tests']:.2f}$ \\\\\n")
        f.write(f"    Max tests allowed & {MAX_TESTS_ALLOWED} & {MAX_TESTS_ALLOWED} \\\\\n")
        f.write(f"    $N$ maxed out (count) & {results['conjunctive']['n_maxed_out']} & {results['disjunctive']['n_maxed_out']} \\\\\n")
        f.write(f"    $N$ maxed out \\& both correct & {results['conjunctive']['n_maxed_out_both_ok']} & {results['disjunctive']['n_maxed_out_both_ok']} \\\\\n")
        f.write(f"    $N$ maxed out (\\%) & {results['conjunctive']['pct_maxed_out']:.1f}\\% & {results['disjunctive']['pct_maxed_out']:.1f}\\% \\\\\n")
        f.write(f"    Total Test Time (seconds) & ${results['conjunctive']['mean_total_time']:.2f} \\pm {results['conjunctive']['se_total_time']:.2f}$ & ${results['disjunctive']['mean_total_time']:.2f} \\pm {results['disjunctive']['se_total_time']:.2f}$ \\\\\n")
        f.write(f"    Time per test (seconds) & ${results['conjunctive']['mean_time_per_test']:.2f} \\pm {results['conjunctive']['se_time_per_test']:.2f}$ & ${results['disjunctive']['mean_time_per_test']:.2f} \\pm {results['disjunctive']['se_time_per_test']:.2f}$ \\\\\n")
        f.write(f"    Exploration Time (seconds) & ${results['conjunctive']['mean_exploration_time']:.2f} \\pm {results['conjunctive']['se_exploration_time']:.2f}$ & ${results['disjunctive']['mean_exploration_time']:.2f} \\pm {results['disjunctive']['se_exploration_time']:.2f}$ \\\\\n")
        f.write(f"    \\# Tests to Succ. (successful only) & ${results['conjunctive']['mean_tests_to_success']:.2f} \\pm {results['conjunctive']['se_tests_to_success']:.2f}$ & ${results['disjunctive']['mean_tests_to_success']:.2f} \\pm {results['disjunctive']['se_tests_to_success']:.2f}$ \\\\\n")
        f.write(f"    Time to Succ. (seconds) & ${results['conjunctive']['mean_time_to_success']:.2f} \\pm {results['conjunctive']['se_time_to_success']:.2f}$ & ${results['disjunctive']['mean_time_to_success']:.2f} \\pm {results['disjunctive']['se_time_to_success']:.2f}$ \\\\\n")
        f.write(r"    \hline" + "\n")
        f.write(f"    $N$ Participants & {results['conjunctive']['n']} & {results['disjunctive']['n']} \\\\\n")
        f.write(f"    $N$ Successful Participants & {results['conjunctive']['n_success']} & {results['disjunctive']['n_success']} \\\\\n")
    print(f"LaTeX table saved to: {latex_path}")

    # Also save CSV for paper
    csv_path = 'actions_tests_time_by_rule_type.csv'
    with open(csv_path, 'w') as f:
        f.write("metric,conjunctive_mean,conjunctive_se,disjunctive_mean,disjunctive_se\n")
        f.write(f"num_actions,{results['conjunctive']['mean_actions']:.2f},{results['conjunctive']['se_actions']:.2f},{results['disjunctive']['mean_actions']:.2f},{results['disjunctive']['se_actions']:.2f}\n")
        f.write(f"actions_per_test,{results['conjunctive']['mean_actions_per_test']:.2f},{results['conjunctive']['se_actions_per_test']:.2f},{results['disjunctive']['mean_actions_per_test']:.2f},{results['disjunctive']['se_actions_per_test']:.2f}\n")
        f.write(f"num_tests,{results['conjunctive']['mean_tests']:.2f},{results['conjunctive']['se_tests']:.2f},{results['disjunctive']['mean_tests']:.2f},{results['disjunctive']['se_tests']:.2f}\n")
        f.write(f"max_tests_allowed,{MAX_TESTS_ALLOWED},0,{MAX_TESTS_ALLOWED},0\n")
        f.write(f"n_maxed_out,{results['conjunctive']['n_maxed_out']},0,{results['disjunctive']['n_maxed_out']},0\n")
        f.write(f"n_maxed_out_both_ok,{results['conjunctive']['n_maxed_out_both_ok']},0,{results['disjunctive']['n_maxed_out_both_ok']},0\n")
        f.write(f"pct_maxed_out,{results['conjunctive']['pct_maxed_out']:.1f},0,{results['disjunctive']['pct_maxed_out']:.1f},0\n")
        f.write(f"total_time_s,{results['conjunctive']['mean_total_time']:.2f},{results['conjunctive']['se_total_time']:.2f},{results['disjunctive']['mean_total_time']:.2f},{results['disjunctive']['se_total_time']:.2f}\n")
        f.write(f"time_per_test_s,{results['conjunctive']['mean_time_per_test']:.2f},{results['conjunctive']['se_time_per_test']:.2f},{results['disjunctive']['mean_time_per_test']:.2f},{results['disjunctive']['se_time_per_test']:.2f}\n")
        f.write(f"exploration_time_s,{results['conjunctive']['mean_exploration_time']:.2f},{results['conjunctive']['se_exploration_time']:.2f},{results['disjunctive']['mean_exploration_time']:.2f},{results['disjunctive']['se_exploration_time']:.2f}\n")
        f.write(f"tests_to_success,{results['conjunctive']['mean_tests_to_success']:.2f},{results['conjunctive']['se_tests_to_success']:.2f},{results['disjunctive']['mean_tests_to_success']:.2f},{results['disjunctive']['se_tests_to_success']:.2f}\n")
        f.write(f"time_to_success_s,{results['conjunctive']['mean_time_to_success']:.2f},{results['conjunctive']['se_time_to_success']:.2f},{results['disjunctive']['mean_time_to_success']:.2f},{results['disjunctive']['se_time_to_success']:.2f}\n")
    print(f"CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
