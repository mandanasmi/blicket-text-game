import json
import numpy as np
from datetime import datetime

with open('new-data-history.json') as f:
    data = json.load(f)

results = []

for pid, record in data.items():
    if not isinstance(record, dict):
        continue
    case = record.get('disjunctive_case_data')
    if not case:
        continue

    # skip non-prolific-ID entries (e.g. "Upgrade:", URLs)
    if not pid[0].isdigit():
        continue

    demographics = record.get('demographics', {})
    prolific_id = demographics.get('prolific_id', pid)

    training_time = sum(case.get('time_per_training_step_seconds', []))
    test_time     = sum(case.get('time_per_test_step_seconds', []))
    step_time     = training_time + test_time

    # total session time from created_at to completed_at
    created_at  = record.get('created_at')
    completed_at = record.get('completed_at')
    if created_at and completed_at:
        t0 = datetime.fromisoformat(created_at)
        t1 = datetime.fromisoformat(completed_at)
        session_s = (t1 - t0).total_seconds()
    else:
        session_s = 0

    obj_answers = case.get('object_answers', {})

    results.append({
        'prolific_id':  prolific_id,
        'training_s':   round(training_time, 2),
        'test_s':       round(test_time, 2),
        'step_s':       round(step_time, 2),
        'session_s':    round(session_s, 2),
        'session_min':  round(session_s / 60, 2),
        'obj4':         obj_answers.get('object_4', ''),
        'obj5':         obj_answers.get('object_5', ''),
        'obj6':         obj_answers.get('object_6', ''),
    })

# print per participant
print(f"{'Prolific ID':<30} {'Training(s)':>12} {'Test(s)':>10} {'Steps(s)':>10} {'Session(s)':>11} {'Session(min)':>13}")
print("-" * 100)
for r in results:
    print(f"{r['prolific_id']:<30} {r['training_s']:>12} {r['test_s']:>10} {r['step_s']:>10} {r['session_s']:>11} {r['session_min']:>13}")

# summary stats
sessions = [r['session_s'] for r in results]
print(f"\nN = {len(sessions)}")
print(f"Mean session time : {np.mean(sessions):.1f}s  ({np.mean(sessions)/60:.2f} min)")
print(f"Median            : {np.median(sessions):.1f}s  ({np.median(sessions)/60:.2f} min)")
print(f"Min / Max         : {min(sessions):.1f}s / {max(sessions):.1f}s")
print(f"SD                : {np.std(sessions, ddof=1):.1f}s")

# proportion saying "Yes" to objects 4, 5, 6 for session >= 60s
filtered = [r for r in results if r['session_s'] >= 60]
n_filtered = len(filtered)
print(f"\n{'='*60}")
print(f"Proportion saying 'Yes' (session >= 60s, N={n_filtered})")
print(f"{'='*60}")
for obj in ['obj4', 'obj5', 'obj6']:
    yes_count = sum(1 for r in filtered if r[obj] == 'Yes')
    label = obj.replace('obj', 'Object ')
    print(f"  {label}: {yes_count}/{n_filtered} = {yes_count/n_filtered:.2%}")