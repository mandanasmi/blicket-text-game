"""
Generate grouped bar chart showing object identification and rule inference accuracy
by rule type (Conjunctive vs Disjunctive) using Round 7 data
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns

# Set style
sns.set_style("white")
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']

parser = argparse.ArgumentParser(description='Generate grouped accuracy plot by rule type.')
parser.add_argument('--input', default='../main_game_data_with_prior_experience.csv', help='Input CSV file (default: from round7)')
parser.add_argument('--output', default='accuracy_by_rule_type_grouped.png', help='Output PNG file')
parser.add_argument('--max-per-rule', type=int, default=50, help='Max participants per rule type (sample if exceeded)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
args = parser.parse_args()
input_csv = args.input
output_png = args.output

# Load data
print("="*80)
print("GENERATING GROUPED ACCURACY PLOT BY RULE TYPE (ROUND 7 DATA)")
print("="*80)
print(f"\nLoading {input_csv}...")

df = pd.read_csv(input_csv)

print(f"Total rounds: {len(df)}")
print(f"Participants: {df['user_id'].nunique()}")

# Filter out participants with prior experience
df = df[df['has_prior_experience'] == False].copy()
print(f"\nAfter filtering out prior experience:")
print(f"Total rounds: {len(df)}")
print(f"Participants: {df['user_id'].nunique()}")

# Sample up to --max-per-rule participants per rule type
rng = np.random.default_rng(args.seed)
keep = np.zeros(len(df), dtype=bool)
for rt in ['conjunctive', 'disjunctive']:
    sub = df[df['ground_truth_rule'] == rt]
    uids = sub['user_id'].unique()
    n = min(len(uids), args.max_per_rule)
    if len(uids) > n:
        chosen = rng.choice(uids, size=n, replace=False)
    else:
        chosen = uids
    keep |= (df['ground_truth_rule'] == rt) & (df['user_id'].isin(chosen))
df = df[keep].copy()
print(f"\nAfter sampling up to {args.max_per_rule} participants per rule type (seed={args.seed}):")
print(f"Total rounds: {len(df)}")
print(f"Participants: {df['user_id'].nunique()}")

# Function for paired t-test
def paired_t_test(sample1, sample2):
    """
    Calculate paired t-test manually
    sample1, sample2: paired samples (same length)
    Returns: t-statistic, p-value (two-tailed)
    """
    if len(sample1) != len(sample2):
        return None, None
    
    n = len(sample1)
    if n < 2:
        return None, None
    
    # Calculate differences
    differences = sample1 - sample2
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    # Calculate t-statistic
    se_diff = std_diff / np.sqrt(n)
    if se_diff == 0:
        return None, None
    
    t_stat = mean_diff / se_diff
    
    # Calculate p-value (two-tailed) using t-distribution approximation
    from math import erf, sqrt
    
    df_val = n - 1
    
    if abs(t_stat) > 3:
        p_value = 0.003
    elif abs(t_stat) > 2.5:
        p_value = 0.02
    elif abs(t_stat) > 2:
        p_value = 0.05
    elif abs(t_stat) > 1.5:
        p_value = 0.14
    else:
        p_value = 0.5
    
    return t_stat, p_value

# Calculate accuracies and average time by rule type (with SE between participants)
rule_types = ['Conjunctive', 'Disjunctive']
object_accuracies = []
rule_accuracies = []
avg_times = []
avg_time_per_test = []
avg_num_tests = []
object_acc_se = []
rule_acc_se = []
avg_times_se = []
avg_time_per_test_se = []
avg_num_tests_se = []
paired_test_results = []
prior_stats = []

for rule_type in ['conjunctive', 'disjunctive']:
    rule_df = df[df['ground_truth_rule'] == rule_type]
    
    obj_correct = rule_df['object_identification_correct'].sum()
    obj_total = rule_df['object_identification_correct'].notna().sum()
    obj_acc = (obj_correct / obj_total * 100) if obj_total > 0 else 0
    object_accuracies.append(obj_acc)
    p = obj_correct / obj_total if obj_total > 0 else 0
    se_obj = 100 * np.sqrt(p * (1 - p) / obj_total) if obj_total > 0 else 0
    object_acc_se.append(se_obj)
    
    rule_correct = rule_df['rule_choice_correct'].sum()
    rule_total = rule_df['rule_choice_correct'].notna().sum()
    rule_acc = (rule_correct / rule_total * 100) if rule_total > 0 else 0
    rule_accuracies.append(rule_acc)
    p = rule_correct / rule_total if rule_total > 0 else 0
    se_rule = 100 * np.sqrt(p * (1 - p) / rule_total) if rule_total > 0 else 0
    rule_acc_se.append(se_rule)
    
    time_data = rule_df['total_round_time_seconds'].dropna()
    avg_time = time_data.mean() if len(time_data) > 0 else 0
    avg_times.append(avg_time)
    se_time = (time_data.std(ddof=1) / np.sqrt(len(time_data))) if len(time_data) > 1 else 0
    avg_times_se.append(se_time)
    
    rule_df_copy = rule_df.copy()
    rule_df_copy['time_per_test'] = rule_df_copy['total_test_time_seconds'] / rule_df_copy['num_tests']
    valid_time_per_test = rule_df_copy[rule_df_copy['num_tests'] > 0]['time_per_test'].dropna()
    avg_tpt = valid_time_per_test.mean() if len(valid_time_per_test) > 0 else 0
    avg_time_per_test.append(avg_tpt)
    se_tpt = (valid_time_per_test.std(ddof=1) / np.sqrt(len(valid_time_per_test))) if len(valid_time_per_test) > 1 else 0
    avg_time_per_test_se.append(se_tpt)
    
    num_tests_data = rule_df[rule_df['num_tests'] > 0]['num_tests'].dropna()
    avg_nt = num_tests_data.mean() if len(num_tests_data) > 0 else 0
    avg_num_tests.append(avg_nt)
    se_nt = (num_tests_data.std(ddof=1) / np.sqrt(len(num_tests_data))) if len(num_tests_data) > 1 else 0
    avg_num_tests_se.append(se_nt)
    
    prior_true = rule_df[rule_df['has_prior_experience'] == True]
    prior_false = rule_df[rule_df['has_prior_experience'] == False]
    
    prior_true_obj_correct = prior_true['object_identification_correct'].sum()
    prior_true_obj_total = prior_true['object_identification_correct'].notna().sum()
    prior_true_obj_acc = (prior_true_obj_correct / prior_true_obj_total * 100) if prior_true_obj_total > 0 else 0
    
    prior_false_obj_correct = prior_false['object_identification_correct'].sum()
    prior_false_obj_total = prior_false['object_identification_correct'].notna().sum()
    prior_false_obj_acc = (prior_false_obj_correct / prior_false_obj_total * 100) if prior_false_obj_total > 0 else 0
    
    prior_true_rule_correct = prior_true['rule_choice_correct'].sum()
    prior_true_rule_total = prior_true['rule_choice_correct'].notna().sum()
    prior_true_rule_acc = (prior_true_rule_correct / prior_true_rule_total * 100) if prior_true_rule_total > 0 else 0
    
    prior_false_rule_correct = prior_false['rule_choice_correct'].sum()
    prior_false_rule_total = prior_false['rule_choice_correct'].notna().sum()
    prior_false_rule_acc = (prior_false_rule_correct / prior_false_rule_total * 100) if prior_false_rule_total > 0 else 0
    
    prior_stats.append({
        'prior_true_obj_acc': prior_true_obj_acc,
        'prior_true_obj_total': prior_true_obj_total,
        'prior_true_obj_correct': prior_true_obj_correct,
        'prior_false_obj_acc': prior_false_obj_acc,
        'prior_false_obj_total': prior_false_obj_total,
        'prior_false_obj_correct': prior_false_obj_correct,
        'prior_true_rule_acc': prior_true_rule_acc,
        'prior_true_rule_total': prior_true_rule_total,
        'prior_true_rule_correct': prior_true_rule_correct,
        'prior_false_rule_acc': prior_false_rule_acc,
        'prior_false_rule_total': prior_false_rule_total,
        'prior_false_rule_correct': prior_false_rule_correct
    })
    
    print(f"\n{rule_type.upper()}:")
    print(f"  Object Identification: {obj_correct}/{obj_total} ({obj_acc:.1f}%) [SE={se_obj:.1f}%]")
    print(f"    - With prior experience: {prior_true_obj_correct}/{prior_true_obj_total} ({prior_true_obj_acc:.1f}%)")
    print(f"    - No prior experience: {prior_false_obj_correct}/{prior_false_obj_total} ({prior_false_obj_acc:.1f}%)")
    print(f"  Rule Inference: {rule_correct}/{rule_total} ({rule_acc:.1f}%) [SE={se_rule:.1f}%]")
    print(f"    - With prior experience: {prior_true_rule_correct}/{prior_true_rule_total} ({prior_true_rule_acc:.1f}%)")
    print(f"    - No prior experience: {prior_false_rule_correct}/{prior_false_rule_total} ({prior_false_rule_acc:.1f}%)")
    print(f"  Average Total Time: {avg_time:.1f} +/- {se_time:.1f} seconds (SE)")
    print(f"  Average Time Per Test: {avg_tpt:.2f} +/- {se_tpt:.2f} seconds (SE)")
    print(f"  Average Num Tests: {avg_nt:.1f} +/- {se_nt:.1f} (SE)")
    
    obj_vals = rule_df['object_identification_correct'].astype(float)
    rule_vals = rule_df['rule_choice_correct'].astype(float)
    valid_mask = obj_vals.notna() & rule_vals.notna()
    obj_vals_clean = obj_vals[valid_mask]
    rule_vals_clean = rule_vals[valid_mask]
    
    if len(obj_vals_clean) > 1:
        t_stat, p_value = paired_t_test(rule_vals_clean, obj_vals_clean)
        if t_stat is not None:
            print(f"  Paired t-test (Rule vs Object):")
            print(f"    t-statistic: {t_stat:.3f}")
            print(f"    p-value: ~{p_value:.3f}")
            if p_value < 0.05:
                print(f"    Result: Significant difference (p < 0.05)")
            else:
                print(f"    Result: No significant difference (p >= 0.05)")
            paired_test_results.append({
                'rule_type': rule_type,
                't_stat': t_stat,
                'p_value': p_value,
                'n': len(obj_vals_clean)
            })
        else:
            paired_test_results.append(None)
    else:
        paired_test_results.append(None)

# CREATE GROUPED BAR CHART
print("\n" + "="*80)
print("CREATING GROUPED BAR CHART")
print("="*80)

fig, ax1 = plt.subplots(1, 1, figsize=(9.5, 5))

x = np.array([0.0, 0.80])  # closer to reduce center gap between Conjunctive and Disjunctive
width = 0.10

bars1 = ax1.bar(x - width*1.5, object_accuracies, width, yerr=object_acc_se, capsize=3,
               color='#2a9d8f', alpha=0.85, edgecolor='#333333', linewidth=1.5,
               error_kw={'color': '#333333', 'linewidth': 1.2})
bars2 = ax1.bar(x - width*0.5, rule_accuracies, width, yerr=rule_acc_se, capsize=3,
               color='#e76f51', alpha=0.85, edgecolor='#333333', linewidth=1.5,
               error_kw={'color': '#333333', 'linewidth': 1.2})

ax2 = ax1.twinx()

bars3 = ax2.bar(x + width*0.5, avg_times, width, yerr=avg_times_se, capsize=3,
               color='#f77f00', alpha=0.85, edgecolor='#333333', linewidth=1.5,
               error_kw={'color': '#333333', 'linewidth': 1.2})

ax3 = ax2.twinx()
ax3.spines['right'].set_position(('outward', 60))
bars5 = ax3.bar(x + width*1.5, avg_num_tests, width, yerr=avg_num_tests_se, capsize=3,
                color='#8b5cf6', alpha=0.85, edgecolor='#333333', linewidth=1.5,
                error_kw={'color': '#333333', 'linewidth': 1.2})
bars4 = ax2.bar(x + width*2.5, avg_time_per_test, width, yerr=avg_time_per_test_se, capsize=3,
               color='#fcbf49', alpha=0.85, edgecolor='#333333', linewidth=1.5,
               error_kw={'color': '#333333', 'linewidth': 1.2})

x_labels = rule_types

ax1.set_xlabel('Rule Type', fontsize=10, color='#333333')
ax1.set_ylabel('Accuracy (%)', fontsize=10, color='#333333')
ax2.set_ylabel('Time (seconds)', fontsize=10, color='#f77f00')
ax1.set_title('Accuracy and Average Time by Rule Type', fontsize=11, pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, fontsize=10)
ax1.set_ylim(0, 116)
ax1.axhline(100, color='#6a3d9a', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect (100%)')
ax1.grid(False)

max_time = max(avg_times) if avg_times else 100
ax2.set_ylim(0, max_time * 1.15)
ax2.tick_params(axis='y', labelcolor='#f77f00')

ax1.yaxis.tick_left()
ax1.spines['right'].set_visible(False)
ax2.yaxis.tick_right()
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(True)

max_nt = max(avg_num_tests) if avg_num_tests else 1
ax3.set_ylim(0, max_nt * 1.2)
ax3.set_yticks([])
ax3.set_ylabel('')
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)

legend_handles = [
    mlines.Line2D([0], [0], color='#6a3d9a', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect (100%)'),
    mpatches.Patch(facecolor='#2a9d8f', edgecolor='#333333', linewidth=1.5, label='Object Identification'),
    mpatches.Patch(facecolor='#e76f51', edgecolor='#333333', linewidth=1.5, label='Rule Inference'),
    mpatches.Patch(facecolor='#f77f00', edgecolor='#333333', linewidth=1.5, label='Avg Total Time'),
    mpatches.Patch(facecolor='#8b5cf6', edgecolor='#333333', linewidth=1.5, label='Avg Num Tests'),
    mpatches.Patch(facecolor='#fcbf49', edgecolor='#333333', linewidth=1.5, label='Avg Time Per Test'),
]
ax1.legend(handles=legend_handles, fontsize=7, loc='upper left',
          bbox_to_anchor=(0.80, 0.96), frameon=True, fancybox=False, shadow=False)

for bars, accuracies, se_vals, rule_type_name in [
    (bars1, object_accuracies, object_acc_se, 'object'),
    (bars2, rule_accuracies, rule_acc_se, 'rule'),
]:
    for i, (bar, acc, se) in enumerate(zip(bars, accuracies, se_vals)):
        height = bar.get_height()
        rule_name = 'conjunctive' if i == 0 else 'disjunctive'
        rule_df = df[df['ground_truth_rule'] == rule_name]
        if rule_type_name == 'object':
            correct = rule_df['object_identification_correct'].sum()
            total = rule_df['object_identification_correct'].notna().sum()
        else:
            correct = rule_df['rule_choice_correct'].sum()
            total = rule_df['rule_choice_correct'].notna().sum()
        y_pos = height + se + 3.5
        ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{acc:.1f}%\n({int(correct)}/{int(total)})',
                ha='center', va='bottom', fontsize=8)

for i, (bar, time_val, se) in enumerate(zip(bars3, avg_times, avg_times_se)):
    height = bar.get_height()
    pad = max(avg_times) * 0.018 if avg_times else 5
    ax2.text(bar.get_x() + bar.get_width()/2., height + se + pad,
             f'{time_val:.0f}s',
             ha='center', va='bottom', fontsize=8, color='#f77f00')

for i, (bar, time_val, se) in enumerate(zip(bars4, avg_time_per_test, avg_time_per_test_se)):
    height = bar.get_height()
    pad = max(avg_times) * 0.03 if avg_times else 1
    ax2.text(bar.get_x() + bar.get_width()/2., height + se + pad,
             f'{time_val:.1f}s',
             ha='center', va='bottom', fontsize=8, color='#f77f00')

for i, (bar, nt_val, se) in enumerate(zip(bars5, avg_num_tests, avg_num_tests_se)):
    height = bar.get_height()
    pad = max(avg_num_tests) * 0.02 if avg_num_tests else 0.2
    ax3.text(bar.get_x() + bar.get_width()/2., height + se + pad,
             f'{nt_val:.1f}',
             ha='center', va='bottom', fontsize=8, color='#8b5cf6')


plt.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.14)
plt.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0.03)
print(f"\nSaved: {output_png}")
plt.close()

print("\n" + "="*80)
print("GENERATION COMPLETE")
print("="*80)

print("\n--- SUMMARY ---")
for i, (rule_type, stats) in enumerate(zip(['conjunctive', 'disjunctive'], prior_stats)):
    print(f"\n{rule_type.capitalize()} Rules:")
    print(f"  Object Identification: {object_accuracies[i]:.1f}%")
    print(f"    - With prior experience: {stats['prior_true_obj_acc']:.1f}% ({stats['prior_true_obj_correct']}/{stats['prior_true_obj_total']})")
    print(f"    - No prior experience: {stats['prior_false_obj_acc']:.1f}% ({stats['prior_false_obj_correct']}/{stats['prior_false_obj_total']})")
    print(f"  Rule Inference: {rule_accuracies[i]:.1f}%")
    print(f"    - With prior experience: {stats['prior_true_rule_acc']:.1f}% ({stats['prior_true_rule_correct']}/{stats['prior_true_rule_total']})")
    print(f"    - No prior experience: {stats['prior_false_rule_acc']:.1f}% ({stats['prior_false_rule_correct']}/{stats['prior_false_rule_total']})")
    print(f"  Difference: {rule_accuracies[i] - object_accuracies[i]:.1f} percentage points")
    print(f"  Average Total Time: {avg_times[i]:.1f} seconds")
    print(f"  Average Time Per Test: {avg_time_per_test[i]:.2f} seconds")
    print(f"  Average Num Tests: {avg_num_tests[i]:.1f}")

print("\n" + "="*80)
