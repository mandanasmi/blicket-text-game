import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

labels = ['Object 4', 'Object 5', 'Object 6']

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data.json')
with open(data_path) as f:
    data = json.load(f)

o4, o5, o6 = [], [], []
for pid, record in data.items():
    if not isinstance(record, dict):
        continue
    if record.get('status') != 'completed':
        continue
    case = record.get('conjunctive_case_data')
    if not case:
        continue
    if not pid[0].isdigit():
        continue

    # filter by session time >= 60s
    created_at = record.get('created_at')
    completed_at = record.get('completed_at')
    if created_at and completed_at:
        session_s = (datetime.fromisoformat(completed_at) - datetime.fromisoformat(created_at)).total_seconds()
    else:
        session_s = 0
    if session_s < 60:
        continue

    answers = case.get('nexiom_object_answers', {})
    o4.append(1 if answers.get('object_4') == 'Yes' else 0)
    o5.append(1 if answers.get('object_5') == 'Yes' else 0)
    o6.append(1 if answers.get('object_6') == 'Yes' else 0)

n = len(o4)
yes_counts = [sum(o4), sum(o5), sum(o6)]

means = [sum(v)/n for v in [o4, o5, o6]]
sems  = [math.sqrt(m * (1 - m) / n) for m in means]

fig, ax = plt.subplots(figsize=(6, 4))

x = np.arange(len(labels))
bars = ax.bar(x, means, width=0.45, color='#CC5500', edgecolor='#C1440E',
              linewidth=1, zorder=3)

ax.errorbar(x, means, yerr=sems, fmt='none', color='#FFB347',
            capsize=6, capthick=2, elinewidth=2, zorder=4)

for i, (count, mean, sem) in enumerate(zip(yes_counts, means, sems)):
    ax.text(x[i], mean + sem + 0.03,
            f'{mean:.3f} ± {sem:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Proportion "Yes"')
ax.set_ylim(0, 1)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)

ax.set_title(f'Adults, Conjunctive — session ≥ 60s (n={n})', fontsize=12)

plt.tight_layout()

plt.savefig(os.path.join(script_dir, 'nexiom_chart_longer_that_60s.png'), dpi=150, bbox_inches='tight')

plt.show()
