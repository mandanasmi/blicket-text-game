import json
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.use('Agg')

with open('data.json') as f:
    data = json.load(f)

results = []
for pid, record in data.items():
    if record.get('status') != 'completed':
        continue
    case = record.get('conjunctive_case_data') or record.get('disjunctive_case_data')
    if not case:
        continue
    prolific_id = record.get('demographics', {}).get('prolific_id', '')
    if not prolific_id or len(prolific_id) > 30:
        continue
    tr = sum(case.get('time_per_training_step_seconds', []))
    te = sum(case.get('time_per_test_step_seconds', []))
    results.append({'pid': pid[:8], 'training': tr, 'test': te, 'total': tr + te})

results.sort(key=lambda x: x['total'])
n        = len(results)
pids     = [r['pid'] for r in results]
training = [r['training'] for r in results]
test     = [r['test'] for r in results]
totals   = [r['total'] for r in results]
mean_t   = np.mean(totals)
median_t = np.median(totals)

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- plot 1: stacked bar ---
fig1, ax1 = plt.subplots(figsize=(8, max(6, n * 0.28)))
y = np.arange(n)
ax1.barh(y, training, color='#0077B6', label='Training', height=0.7)
ax1.barh(y, test, left=training, color='#FF7F00', label='Test', height=0.7)
ax1.axvline(mean_t,   color='#CC5500', linestyle='--', linewidth=1.5, label=f'Mean {mean_t:.0f}s')
ax1.axvline(median_t, color='#444',    linestyle=':',  linewidth=1.5, label=f'Median {median_t:.0f}s')
ax1.set_yticks(y)
ax1.set_yticklabels(pids, fontsize=8)
ax1.set_xlabel('Time (seconds)')
ax1.set_title(f'Session time per participant (n={n})', fontsize=12)
ax1.spines[['top', 'right']].set_visible(False)
ax1.legend(fontsize=8)
plt.tight_layout()
out1 = os.path.join(script_dir, 'session_times_bars.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to {out1}")

# --- plot 2: histogram ---
bins = list(range(0, 320, 20))  # 0, 20, 40, ..., 300
bin_labels = [f'{b}–{b+20}s' for b in bins[:-1]]

fig2, ax2 = plt.subplots(figsize=(9, 4))
counts, _, patches = ax2.hist(totals, bins=bins, color='#0077B6', edgecolor='#03045E', linewidth=0.5)
ax2.axvline(mean_t,   color='#CC5500', linestyle='--', linewidth=1.5, label=f'Mean {mean_t:.0f}s')
ax2.axvline(median_t, color='#444',    linestyle=':',  linewidth=1.5, label=f'Median {median_t:.0f}s')

ax2.set_xticks([b + 10 for b in bins[:-1]])
ax2.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Count')
ax2.set_title('Distribution of total session times', fontsize=12)
ax2.spines[['top', 'right']].set_visible(False)
ax2.legend(fontsize=9)
plt.tight_layout()
out2 = os.path.join(script_dir, 'session_times_hist.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to {out2}")