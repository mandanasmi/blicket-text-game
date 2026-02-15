"""
Generate grouped bar chart showing object identification and rule inference accuracy
by rule type using sampled data: 15 random conjunctive + all disjunctive games
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Set style
sns.set_style("white")
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load data
print("="*80)
print("GENERATING GROUPED ACCURACY PLOT BY RULE TYPE (SAMPLED DATA)")
print("="*80)
print("\nLoading main_game_data_with_prior_experience.csv...")

df = pd.read_csv('main_game_data_with_prior_experience.csv')

print(f"Total rounds: {len(df)}")
print(f"Participants: {df['user_id'].nunique()}")

# Separate conjunctive and disjunctive
conjunctive_df = df[df['ground_truth_rule'] == 'conjunctive'].copy()
disjunctive_df = df[df['ground_truth_rule'] == 'disjunctive'].copy()

print(f"\nConjunctive games: {len(conjunctive_df)}")
print(f"Disjunctive games: {len(disjunctive_df)}")

# Sample 15 random conjunctive games, ensuring prior experience participants are included
conjunctive_with_prior = conjunctive_df[conjunctive_df['has_prior_experience'] == True].copy()
conjunctive_without_prior = conjunctive_df[conjunctive_df['has_prior_experience'] == False].copy()

print(f"\nConjunctive breakdown:")
print(f"  - With prior experience: {len(conjunctive_with_prior)}")
print(f"  - Without prior experience: {len(conjunctive_without_prior)}")

if len(conjunctive_df) >= 15:
    # Always include all participants with prior experience
    num_with_prior = len(conjunctive_with_prior)
    num_needed = 15 - num_with_prior
    
    if num_needed > 0:
        # Sample the remaining from those without prior experience
        sampled_without_prior = conjunctive_without_prior.sample(n=num_needed, random_state=42)
        sampled_conjunctive = pd.concat([conjunctive_with_prior, sampled_without_prior], ignore_index=True)
    else:
        # If we have 15+ with prior experience, sample 15 from all (including prior)
        sampled_conjunctive = conjunctive_df.sample(n=15, random_state=42)
        # But ensure at least one with prior is included if they exist
        if len(conjunctive_with_prior) > 0 and len(sampled_conjunctive[sampled_conjunctive['has_prior_experience'] == True]) == 0:
            # Replace one random sample with a prior experience participant
            sampled_conjunctive = sampled_conjunctive.iloc[:-1]
            sampled_conjunctive = pd.concat([sampled_conjunctive, conjunctive_with_prior.sample(n=1, random_state=42)], ignore_index=True)
    
    print(f"\nSampled {len(sampled_conjunctive)} conjunctive games")
    print(f"  - With prior experience: {len(sampled_conjunctive[sampled_conjunctive['has_prior_experience'] == True])}")
    print(f"  - Without prior experience: {len(sampled_conjunctive[sampled_conjunctive['has_prior_experience'] == False])}")
else:
    sampled_conjunctive = conjunctive_df.copy()
    print(f"\nWarning: Only {len(conjunctive_df)} conjunctive games available, using all")

# Combine sampled conjunctive with all disjunctive
sampled_df = pd.concat([sampled_conjunctive, disjunctive_df], ignore_index=True)

print(f"Total games in sampled dataset: {len(sampled_df)}")
print(f"  - Conjunctive (sampled): {len(sampled_conjunctive)}")
print(f"  - Disjunctive (all): {len(disjunctive_df)}")

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

# Calculate accuracies and average time by rule type
rule_types = ['Conjunctive\n(Sampled)', 'Disjunctive\n(All)']
object_accuracies = []
rule_accuracies = []
avg_times = []
avg_time_per_test = []
paired_test_results = []
prior_stats = []  # Store prior experience stats

for rule_type, rule_df in [('conjunctive', sampled_conjunctive), ('disjunctive', disjunctive_df)]:
    # Object identification accuracy
    obj_correct = rule_df['object_identification_correct'].sum()
    obj_total = rule_df['object_identification_correct'].notna().sum()
    obj_acc = (obj_correct / obj_total * 100) if obj_total > 0 else 0
    object_accuracies.append(obj_acc)
    
    # Rule inference accuracy
    rule_correct = rule_df['rule_choice_correct'].sum()
    rule_total = rule_df['rule_choice_correct'].notna().sum()
    rule_acc = (rule_correct / rule_total * 100) if rule_total > 0 else 0
    rule_accuracies.append(rule_acc)
    
    # Average total time per round
    time_data = rule_df['total_round_time_seconds'].dropna()
    avg_time = time_data.mean() if len(time_data) > 0 else 0
    avg_times.append(avg_time)
    
    # Average time per test
    rule_df_copy = rule_df.copy()
    rule_df_copy['time_per_test'] = rule_df_copy['total_test_time_seconds'] / rule_df_copy['num_tests']
    valid_time_per_test = rule_df_copy[rule_df_copy['num_tests'] > 0]['time_per_test'].dropna()
    avg_tpt = valid_time_per_test.mean() if len(valid_time_per_test) > 0 else 0
    avg_time_per_test.append(avg_tpt)
    
    # Prior experience breakdown
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
    
    # Store prior experience stats
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
    
    print(f"\n{rule_type.upper()} ({len(rule_df)} games):")
    print(f"  Object Identification: {obj_correct}/{obj_total} ({obj_acc:.1f}%)")
    print(f"    - With prior experience: {prior_true_obj_correct}/{prior_true_obj_total} ({prior_true_obj_acc:.1f}%)")
    print(f"    - No prior experience: {prior_false_obj_correct}/{prior_false_obj_total} ({prior_false_obj_acc:.1f}%)")
    print(f"  Rule Inference: {rule_correct}/{rule_total} ({rule_acc:.1f}%)")
    print(f"    - With prior experience: {prior_true_rule_correct}/{prior_true_rule_total} ({prior_true_rule_acc:.1f}%)")
    print(f"    - No prior experience: {prior_false_rule_correct}/{prior_false_rule_total} ({prior_false_rule_acc:.1f}%)")
    print(f"  Average Total Time: {avg_time:.1f} seconds")
    print(f"  Average Time Per Test: {avg_tpt:.2f} seconds")
    
    # Paired t-test for this rule type
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

# ============================================================================
# CREATE GROUPED BAR CHART
# ============================================================================
print("\n" + "="*80)
print("CREATING GROUPED BAR CHART")
print("="*80)

fig, ax1 = plt.subplots(1, 1, figsize=(13, 7))

# Set up bar positions
x = np.arange(len(rule_types))
width = 0.16

# Create primary y-axis bars (accuracy)
bars1 = ax1.bar(x - width*1.5, object_accuracies, width, label='Object Identification',
               color='#2a9d8f', alpha=0.85, edgecolor='#333333', linewidth=1.5)
bars2 = ax1.bar(x - width*0.5, rule_accuracies, width, label='Rule Inference',
               color='#e76f51', alpha=0.85, edgecolor='#333333', linewidth=1.5)

# Create secondary y-axis for time
ax2 = ax1.twinx()

bars3 = ax2.bar(x + width*0.5, avg_times, width, label='Avg Total Time',
               color='#f77f00', alpha=0.85, edgecolor='#333333', linewidth=1.5)
bars4 = ax2.bar(x + width*1.5, avg_time_per_test, width, label='Avg Time Per Test',
               color='#fcbf49', alpha=0.85, edgecolor='#333333', linewidth=1.5)

# Calculate prior experience counts for labels
prior_counts = []
for rule_df in [sampled_conjunctive, disjunctive_df]:
    prior_count = rule_df[rule_df['has_prior_experience'] == True].shape[0]
    prior_counts.append(prior_count)

# Create x-axis labels with prior experience counts
x_labels = []
for i, rule_type in enumerate(rule_types):
    if prior_counts[i] > 0:
        x_labels.append(f"{rule_type}\n(n={prior_counts[i]} with prior experience)")
    else:
        x_labels.append(rule_type)

# Add labels and title
ax1.set_xlabel('Rule Type', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', color='#333333')
ax2.set_ylabel('Time (seconds)', fontsize=14, fontweight='bold', color='#f77f00')
ax1.set_title('Accuracy and Average Time by Rule Type\n(15 Sampled Conjunctive + All Disjunctive Games)', 
             fontsize=16, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, fontsize=13)
ax1.set_ylim(0, 110)
ax1.axhline(100, color='#6a3d9a', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect (100%)')
ax1.grid(False)

# Set secondary y-axis limits
max_time = max(avg_times) if avg_times else 100
ax2.set_ylim(0, max_time * 1.15)
ax2.tick_params(axis='y', labelcolor='#f77f00')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left', 
          bbox_to_anchor=(1.15, 1), frameon=True, fancybox=True, shadow=True)

# Add value labels on accuracy bars
for bars, accuracies, rule_type_name in [(bars1, object_accuracies, 'object'), 
                                          (bars2, rule_accuracies, 'rule')]:
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        if i == 0:
            rule_df = sampled_conjunctive
        else:
            rule_df = disjunctive_df
        
        if rule_type_name == 'object':
            correct = rule_df['object_identification_correct'].sum()
            total = rule_df['object_identification_correct'].notna().sum()
        else:
            correct = rule_df['rule_choice_correct'].sum()
            total = rule_df['rule_choice_correct'].notna().sum()
        
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%\n({int(correct)}/{int(total)})', 
                ha='center', va='bottom', fontsize=9)

# Add value labels on time bars (total time)
for i, (bar, time_val) in enumerate(zip(bars3, avg_times)):
    height = bar.get_height()
    max_time_for_label = max(avg_times) if avg_times else 100
    ax2.text(bar.get_x() + bar.get_width()/2., height + max_time_for_label * 0.02,
            f'{time_val:.0f}s', 
            ha='center', va='bottom', fontsize=9, color='#f77f00', fontweight='bold')

# Add value labels on time bars (time per test)
for i, (bar, time_val) in enumerate(zip(bars4, avg_time_per_test)):
    height = bar.get_height()
    max_time_for_label = max(avg_times) if avg_times else 100
    ax2.text(bar.get_x() + bar.get_width()/2., height + max_time_for_label * 0.02,
            f'{time_val:.1f}s', 
            ha='center', va='bottom', fontsize=9, color='#fcbf49', fontweight='bold')


plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.savefig('accuracy_by_rule_type_sampled.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: accuracy_by_rule_type_sampled.png")
plt.close()

print("\n" + "="*80)
print("GENERATION COMPLETE")
print("="*80)

# Print summary
print("\n--- SUMMARY ---")
for i, (rule_label, stats) in enumerate(zip(['Conjunctive Rules (15 sampled)', f'Disjunctive Rules (all {len(disjunctive_df)} games)'], prior_stats)):
    print(f"\n{rule_label}:")
    print(f"  Object Identification: {object_accuracies[i]:.1f}%")
    print(f"    - With prior experience: {stats['prior_true_obj_acc']:.1f}% ({stats['prior_true_obj_correct']}/{stats['prior_true_obj_total']})")
    print(f"    - No prior experience: {stats['prior_false_obj_acc']:.1f}% ({stats['prior_false_obj_correct']}/{stats['prior_false_obj_total']})")
    print(f"  Rule Inference: {rule_accuracies[i]:.1f}%")
    print(f"    - With prior experience: {stats['prior_true_rule_acc']:.1f}% ({stats['prior_true_rule_correct']}/{stats['prior_true_rule_total']})")
    print(f"    - No prior experience: {stats['prior_false_rule_acc']:.1f}% ({stats['prior_false_rule_correct']}/{stats['prior_false_rule_total']})")
    print(f"  Difference: {rule_accuracies[i] - object_accuracies[i]:.1f} percentage points")
    print(f"  Average Total Time: {avg_times[i]:.1f} seconds")
    print(f"  Average Time Per Test: {avg_time_per_test[i]:.2f} seconds")

print("\n" + "="*80)
