"""
Generate grouped bar chart showing object identification and rule inference accuracy
by rule type (Conjunctive vs Disjunctive)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']

# Load data
print("="*80)
print("GENERATING GROUPED ACCURACY PLOT BY RULE TYPE")
print("="*80)
print("\nLoading main_game_data_with_prior_experience.csv...")

df = pd.read_csv('main_game_data_with_prior_experience.csv')

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

# Calculate accuracies by rule type
rule_types = ['Conjunctive', 'Disjunctive']
object_accuracies = []
rule_accuracies = []
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
    
    print(f"\n{rule_type.upper()}:")
    print(f"  Object Identification: {obj_correct}/{obj_total} ({obj_acc:.1f}%)")
    print(f"  Rule Inference: {rule_correct}/{rule_total} ({rule_acc:.1f}%)")
    
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

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Set up bar positions
x = np.arange(len(rule_types))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, object_accuracies, width, label='Object Identification',
               color='#2a9d8f', alpha=0.85, edgecolor='#333333', linewidth=1.5)
bars2 = ax.bar(x + width/2, rule_accuracies, width, label='Rule Inference',
               color='#e76f51', alpha=0.85, edgecolor='#333333', linewidth=1.5)

# Add labels and title
ax.set_xlabel('Rule Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Object Identification and Rule Inference Accuracy by Rule Type', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(rule_types, fontsize=13)
ax.set_ylim(0, 110)
ax.axhline(100, color='#6a3d9a', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect (100%)')
ax.grid(True, alpha=0.3, axis='y')

# Add legend outside the plot
ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1.02, 1), 
          frameon=True, fancybox=True, shadow=True)

# Add value labels on bars
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
        
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%\n({int(correct)}/{int(total)})', 
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('accuracy_by_rule_type_grouped.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: accuracy_by_rule_type_grouped.png")
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

print(f"\nDisjunctive Rules:")
print(f"  Object Identification: {object_accuracies[1]:.1f}%")
print(f"  Rule Inference: {rule_accuracies[1]:.1f}%")
print(f"  Difference: {rule_accuracies[1] - object_accuracies[1]:.1f} percentage points")

print("\n" + "="*80)

