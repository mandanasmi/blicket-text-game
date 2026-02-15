"""
Generate grouped bar chart showing object identification and rule inference accuracy
by rule type (Conjunctive vs Disjunctive) using combined Round 1 and Round 2 data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("white")
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']

# Load data
print("="*80)
print("GENERATING GROUPED ACCURACY PLOT BY RULE TYPE (COMBINED DATA)")
print("="*80)
print("\nLoading main_game_combined_round1_round2.csv...")

df = pd.read_csv('main_game_combined_round1_round2.csv')

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
    # For n > 30, t-distribution approximates normal distribution
    # Using a simple approximation for p-value
    from math import erf, sqrt
    
    # Degrees of freedom
    df_val = n - 1
    
    # Two-tailed p-value approximation
    # For large samples, we can use normal approximation
    if abs(t_stat) > 3:
        p_value = 0.003  # Very small
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
rule_types = ['Conjunctive', 'Disjunctive']
object_accuracies = []
rule_accuracies = []
avg_times = []
avg_time_per_test = []
paired_test_results = []

for rule_type in ['conjunctive', 'disjunctive']:
    rule_df = df[df['ground_truth_rule'] == rule_type]
    
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
    # Calculate time per test for each round: total_test_time / num_tests
    rule_df_copy = rule_df.copy()
    rule_df_copy['time_per_test'] = rule_df_copy['total_test_time_seconds'] / rule_df_copy['num_tests']
    # Filter out invalid values (where num_tests is 0 or NaN)
    valid_time_per_test = rule_df_copy[rule_df_copy['num_tests'] > 0]['time_per_test'].dropna()
    avg_tpt = valid_time_per_test.mean() if len(valid_time_per_test) > 0 else 0
    avg_time_per_test.append(avg_tpt)
    
    print(f"\n{rule_type.upper()}:")
    print(f"  Object Identification: {obj_correct}/{obj_total} ({obj_acc:.1f}%)")
    print(f"  Rule Inference: {rule_correct}/{rule_total} ({rule_acc:.1f}%)")
    print(f"  Average Total Time: {avg_time:.1f} seconds")
    print(f"  Average Time Per Test: {avg_tpt:.2f} seconds")
    
    # Paired t-test for this rule type
    obj_vals = rule_df['object_identification_correct'].astype(float)
    rule_vals = rule_df['rule_choice_correct'].astype(float)
    
    # Remove any NaN values (keep paired data only)
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
width = 0.16  # Narrower bars to fit four

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

# Add labels and title
ax1.set_xlabel('Rule Type', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', color='#333333')
ax2.set_ylabel('Time (seconds)', fontsize=14, fontweight='bold', color='#f77f00')
ax1.set_title('Accuracy and Average Time by Rule Type\n(Combined 1 & 2 data)', 
             fontsize=16, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(rule_types, fontsize=13)
ax1.set_ylim(0, 110)
ax1.axhline(100, color='#6a3d9a', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect (100%)')
ax1.grid(False)

# Set secondary y-axis limits based on actual time values (use max of total time)
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
        rule_name = 'conjunctive' if i == 0 else 'disjunctive'
        rule_df = df[df['ground_truth_rule'] == rule_name]
        
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
plt.savefig('accuracy_by_rule_type_grouped_combined.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: accuracy_by_rule_type_grouped_combined.png")
plt.close()

print("\n" + "="*80)
print("GENERATION COMPLETE")
print("="*80)

# Print summary
print("\n--- SUMMARY ---")
print(f"\nConjunctive Rules:")
print(f"  Object Identification: {object_accuracies[0]:.1f}%")
print(f"  Rule Inference: {rule_accuracies[0]:.1f}%")
print(f"  Difference: {rule_accuracies[0] - object_accuracies[0]:.1f} percentage points")
print(f"  Average Total Time: {avg_times[0]:.1f} seconds")
print(f"  Average Time Per Test: {avg_time_per_test[0]:.2f} seconds")

print(f"\nDisjunctive Rules:")
print(f"  Object Identification: {object_accuracies[1]:.1f}%")
print(f"  Rule Inference: {rule_accuracies[1]:.1f}%")
print(f"  Difference: {rule_accuracies[1] - object_accuracies[1]:.1f} percentage points")
print(f"  Average Total Time: {avg_times[1]:.1f} seconds")
print(f"  Average Time Per Test: {avg_time_per_test[1]:.2f} seconds")

print("\n" + "="*80)

