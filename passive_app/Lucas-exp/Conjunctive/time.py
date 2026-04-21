import json
import numpy as np

with open('data.json') as f:
    data = json.load(f)

results = []

for pid, record in data.items():
    if record.get('status') != 'completed':
        continue
    case = record.get('conjunctive_case_data') or record.get('disjunctive_case_data')
    if not case:
        continue

    # skip malformed entries (non-prolific IDs)
    demographics = record.get('demographics', {})
    prolific_id = demographics.get('prolific_id', '')
    if not prolific_id or len(prolific_id) > 30:
        continue

    training_time = sum(case.get('time_per_training_step_seconds', []))
    test_time     = sum(case.get('time_per_test_step_seconds', []))
    total_time    = training_time + test_time
    response_time = case.get('response_time_seconds', 0)  # time on test questions only

    results.append({
        'prolific_id':    prolific_id,
        'training_s':     round(training_time, 2),
        'test_s':         round(test_time, 2),
        'total_s':        round(total_time, 2),
        'total_min':      round(total_time / 60, 2),
        'response_time_s': round(response_time, 2),
    })

# print per participant
print(f"{'Prolific ID':<30} {'Training(s)':>12} {'Test(s)':>10} {'Total(s)':>10}")
print("-" * 76)
for r in results:
    print(f"{r['prolific_id']:<30} {r['training_s']:>12} {r['test_s']:>10} {r['total_s']:>10}")

# summary stats
totals = [r['total_s'] for r in results]
print(f"\nN = {len(totals)}")
print(f"Mean total time : {np.mean(totals):.1f}s  ({np.mean(totals)/60:.2f} min)")
print(f"Median          : {np.median(totals):.1f}s  ({np.median(totals)/60:.2f} min)")
print(f"Min / Max       : {min(totals):.1f}s / {max(totals):.1f}s")
print(f"SD              : {np.std(totals, ddof=1):.1f}s")