"""
Generate grouped bar chart showing ONLY object identification and rule inference
accuracy by rule type (Conjunctive vs Disjunctive). No time or test-count metrics.
Uses human_active_data_no_prior_experience.json (51 conjunctive, 51 disjunctive).
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_style("white")
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']

parser = argparse.ArgumentParser(description='Generate accuracy-only grouped plot by rule type.')
parser.add_argument('--input', default='../human_active_data_no_prior_experience.json', help='Input JSON file (default: from round7)')
parser.add_argument('--output', default='accuracy_by_rule_type_accuracy_only.png', help='Output PNG file')
args = parser.parse_args()

# Load data from JSON
print("Loading", args.input, "...")
with open(args.input, 'r') as f:
    data = json.load(f)

# Extract object identification and rule inference correctness per rule type
# Rule accuracy: main_game.config.rule (ground truth) vs main_game.rule_type (user choice)
# Both normalized to conjunctive/disjunctive; match -> accuracy = 1
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

records = { 'conjunctive': {'obj_correct': [], 'rule_correct': [], 'both_correct': []}, 'disjunctive': {'obj_correct': [], 'rule_correct': [], 'both_correct': []} }

for user_id, user_data in data.items():
    if 'main_game' not in user_data:
        continue
    mg = user_data['main_game']
    # Ground truth from config.rule; fallback to rule/true_rule
    config_rule = (mg.get('config') or {}).get('rule') or mg.get('rule') or mg.get('true_rule')
    true_rule = normalize_rule(config_rule)
    if true_rule not in ('conjunctive', 'disjunctive'):
        continue
    true_blickets = set(mg.get('true_blicket_indices', []))
    user_blickets = set(mg.get('user_chosen_blickets', []))
    obj_ok = 1.0 if true_blickets == user_blickets else 0.0
    # User choice from rule_type (contains "Conjunctive (...)" or "Disjunctive (...)")
    user_choice = normalize_rule(mg.get('rule_type', ''))
    rule_ok = 1.0 if (user_choice and user_choice == true_rule) else 0.0
    both_ok = 1.0 if (obj_ok and rule_ok) else 0.0
    records[true_rule]['obj_correct'].append(obj_ok)
    records[true_rule]['rule_correct'].append(rule_ok)
    records[true_rule]['both_correct'].append(both_ok)

# Compute accuracies and SE by rule type (including full hypothesis: both objects and rule correct)
rule_types = ['Conjunctive', 'Disjunctive']
object_accuracies = []
rule_accuracies = []
full_hypothesis_accuracies = []
object_acc_se = []
rule_acc_se = []
full_hypothesis_se = []
counts = []  # (obj_correct, n, rule_correct, n, both_correct, n) per rule type

for rule_type in ['conjunctive', 'disjunctive']:
    obj_vals = np.array(records[rule_type]['obj_correct'])
    rule_vals = np.array(records[rule_type]['rule_correct'])
    both_vals = np.array(records[rule_type]['both_correct'])
    n = len(obj_vals)
    if n == 0:
        object_accuracies.append(0)
        rule_accuracies.append(0)
        full_hypothesis_accuracies.append(0)
        object_acc_se.append(0)
        rule_acc_se.append(0)
        full_hypothesis_se.append(0)
        counts.append((0, 0, 0, 0, 0, 0))
        continue
    obj_acc = obj_vals.mean()
    rule_acc = rule_vals.mean()
    both_acc = both_vals.mean()
    object_accuracies.append(obj_acc)
    rule_accuracies.append(rule_acc)
    full_hypothesis_accuracies.append(both_acc)
    se_obj = obj_vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0
    se_rule = rule_vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0
    se_both = both_vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0
    object_acc_se.append(se_obj)
    rule_acc_se.append(se_rule)
    full_hypothesis_se.append(se_both)
    counts.append((int(obj_vals.sum()), n, int(rule_vals.sum()), n, int(both_vals.sum()), n))

# Create grouped bar chart (accuracy only: object, rule, full hypothesis)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

x = np.array([0.0, 0.45])
width = 0.11

# Bar style: solid fills, no outlines, error bars with caps (minimal clean look)
bars1 = ax.bar(x - width * 1.5, object_accuracies, width, yerr=object_acc_se, capsize=4,
               color='#2a9d8f', alpha=0.9, edgecolor='none',
               error_kw={'color': '#333333', 'linewidth': 1, 'capthick': 1})
bars2 = ax.bar(x - width * 0.5, rule_accuracies, width, yerr=rule_acc_se, capsize=4,
               color='#e76f51', alpha=0.9, edgecolor='none',
               error_kw={'color': '#333333', 'linewidth': 1, 'capthick': 1})
bars3 = ax.bar(x + width * 0.5, full_hypothesis_accuracies, width, yerr=full_hypothesis_se, capsize=4,
               color='#6a3d9a', alpha=0.9, edgecolor='none',
               error_kw={'color': '#333333', 'linewidth': 1, 'capthick': 1})

ax.set_xlabel('Rule Type', fontsize=10, color='#333333')
ax.set_ylabel('Accuracy', fontsize=10, color='#333333')
# Center x-axis labels under the red bar (Rule Inference)
ax.set_xticks(x - width * 0.5)
ax.set_xticklabels(rule_types, fontsize=10)
ax.set_ylim(0, 1.3)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # last label 1.0; ylim 1.3 leaves room for legend
ax.axhline(1.0, color='#6a3d9a', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect (1.0)')
ax.grid(False)

legend_handles = [
    mlines.Line2D([0], [0], color='#6a3d9a', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect (1.0)'),
    mpatches.Patch(facecolor='#2a9d8f', edgecolor='none', label='Object Identification'),
    mpatches.Patch(facecolor='#e76f51', edgecolor='none', label='Rule Inference'),
    mpatches.Patch(facecolor='#6a3d9a', edgecolor='none', label='Full Hypothesis (objects + rule)'),
]
# Place legend in top left
ax.legend(handles=legend_handles, fontsize=10, loc='upper left', frameon=True, fancybox=False, shadow=False)

# Value labels (percentage and count) using precomputed counts
# counts[i] = (obj_correct, n, rule_correct, n, both_correct, n)
for bars, accuracies, se_vals, correct_idx in [
    (bars1, object_accuracies, object_acc_se, 0),   # obj_correct
    (bars2, rule_accuracies, rule_acc_se, 2),      # rule_correct
    (bars3, full_hypothesis_accuracies, full_hypothesis_se, 4),  # both_correct
]:
    for i, (bar, acc, se) in enumerate(zip(bars, accuracies, se_vals)):
        height = bar.get_height()
        c = counts[i]
        correct = c[correct_idx]
        total = c[1]
        y_pos = height + se + 0.035
        ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                f'{acc:.2f} (±{se:.2f})\n({correct}/{total})',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(args.output, dpi=400, bbox_inches='tight', pad_inches=0.03)
pdf_output = args.output.replace('.png', '.pdf')
plt.savefig(pdf_output, bbox_inches='tight', pad_inches=0.03)
print("Saved:", args.output)
print("Saved:", pdf_output)
plt.close()
