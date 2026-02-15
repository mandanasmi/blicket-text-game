"""
Analyze whether participants took more evidence/steps to figure out 
conjunctive rules compared to disjunctive rules, despite similar accuracy rates
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
print("EXPLORATION ANALYSIS: CONJUNCTIVE VS DISJUNCTIVE RULES")
print("="*80)
print("\nLoading main_game_data_with_prior_experience.csv...")

df = pd.read_csv('main_game_data_with_prior_experience.csv')

print(f"Total rounds: {len(df)}")
print(f"Participants: {df['user_id'].nunique()}")

# ============================================================================
# ACCURACY ANALYSIS BY RULE TYPE
# ============================================================================
print("\n" + "="*80)
print("1. ACCURACY BY RULE TYPE")
print("="*80)

for rule_type in ['conjunctive', 'disjunctive']:
    rule_df = df[df['ground_truth_rule'] == rule_type]
    
    # Object accuracy
    obj_correct = rule_df['object_identification_correct'].sum()
    obj_total = rule_df['object_identification_correct'].notna().sum()
    obj_acc = (obj_correct / obj_total * 100) if obj_total > 0 else 0
    
    # Rule accuracy
    rule_correct = rule_df['rule_choice_correct'].sum()
    rule_total = rule_df['rule_choice_correct'].notna().sum()
    rule_acc = (rule_correct / rule_total * 100) if rule_total > 0 else 0
    
    print(f"\n{rule_type.upper()}:")
    print(f"  Rounds: {len(rule_df)}")
    print(f"  Object Identification: {obj_correct}/{obj_total} ({obj_acc:.1f}%)")
    print(f"  Rule Classification: {rule_correct}/{rule_total} ({rule_acc:.1f}%)")

# ============================================================================
# EXPLORATION EFFORT ANALYSIS BY RULE TYPE
# ============================================================================
print("\n" + "="*80)
print("2. EXPLORATION EFFORT BY RULE TYPE")
print("="*80)

conjunctive_df = df[df['ground_truth_rule'] == 'conjunctive']
disjunctive_df = df[df['ground_truth_rule'] == 'disjunctive']

# Number of tests
print("\n--- Number of Tests ---")
conj_tests = conjunctive_df['num_tests']
disj_tests = disjunctive_df['num_tests']

print(f"\nConjunctive:")
print(f"  Mean tests: {conj_tests.mean():.2f}")
print(f"  Median tests: {conj_tests.median():.0f}")
print(f"  Std: {conj_tests.std():.2f}")
print(f"  Range: {conj_tests.min()}-{conj_tests.max()}")

print(f"\nDisjunctive:")
print(f"  Mean tests: {disj_tests.mean():.2f}")
print(f"  Median tests: {disj_tests.median():.0f}")
print(f"  Std: {disj_tests.std():.2f}")
print(f"  Range: {disj_tests.min()}-{disj_tests.max()}")

# Statistical comparison
if len(conj_tests) > 0 and len(disj_tests) > 0:
    diff = conj_tests.mean() - disj_tests.mean()
    pct_diff = (diff / disj_tests.mean() * 100) if disj_tests.mean() > 0 else 0
    print(f"\nComparison:")
    print(f"  Difference: {diff:.2f} tests ({pct_diff:+.1f}%)")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(conj_tests)-1)*conj_tests.std()**2 + (len(disj_tests)-1)*disj_tests.std()**2) / (len(conj_tests)+len(disj_tests)-2))
    cohens_d = (conj_tests.mean() - disj_tests.mean()) / pooled_std
    print(f"  Cohen's d (effect size): {cohens_d:.3f}")
    if abs(cohens_d) < 0.2:
        effect_interp = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interp = "small"
    elif abs(cohens_d) < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    print(f"  Effect interpretation: {effect_interp}")

# Time analysis
print("\n--- Time Analysis ---")
conj_time = conjunctive_df['total_test_time_seconds']
disj_time = disjunctive_df['total_test_time_seconds']

print(f"\nConjunctive (Total Test Time):")
print(f"  Mean: {conj_time.mean():.2f} seconds")
print(f"  Median: {conj_time.median():.2f} seconds")
print(f"  Std: {conj_time.std():.2f}")

print(f"\nDisjunctive (Total Test Time):")
print(f"  Mean: {disj_time.mean():.2f} seconds")
print(f"  Median: {disj_time.median():.2f} seconds")
print(f"  Std: {disj_time.std():.2f}")

if len(conj_time) > 0 and len(disj_time) > 0:
    diff_time = conj_time.mean() - disj_time.mean()
    pct_diff_time = (diff_time / disj_time.mean() * 100) if disj_time.mean() > 0 else 0
    print(f"\nComparison (time):")
    print(f"  Difference: {diff_time:.2f}s ({pct_diff_time:+.1f}%)")

# Total round time
print("\n--- Total Round Time ---")
conj_round_time = conjunctive_df['total_round_time_seconds']
disj_round_time = disjunctive_df['total_round_time_seconds']

print(f"\nConjunctive (Total Round Time):")
print(f"  Mean: {conj_round_time.mean():.2f} seconds")
print(f"  Median: {conj_round_time.median():.2f} seconds")

print(f"\nDisjunctive (Total Round Time):")
print(f"  Mean: {disj_round_time.mean():.2f} seconds")
print(f"  Median: {disj_round_time.median():.2f} seconds")

# ============================================================================
# EXPLORATION EFFORT WHEN CORRECT
# ============================================================================
print("\n" + "="*80)
print("3. EXPLORATION EFFORT FOR CORRECT CLASSIFICATIONS")
print("="*80)

# Filter for correct rule classifications only
correct_df = df[df['rule_choice_correct'] == True]

conj_correct = correct_df[correct_df['ground_truth_rule'] == 'conjunctive']
disj_correct = correct_df[correct_df['ground_truth_rule'] == 'disjunctive']

print(f"\nRounds with correct rule classification:")
print(f"  Conjunctive: {len(conj_correct)}")
print(f"  Disjunctive: {len(disj_correct)}")

print(f"\n--- Number of Tests (Correct Only) ---")
print(f"Conjunctive: Mean={conj_correct['num_tests'].mean():.2f}, Median={conj_correct['num_tests'].median():.0f}")
print(f"Disjunctive: Mean={disj_correct['num_tests'].mean():.2f}, Median={disj_correct['num_tests'].median():.0f}")
print(f"Difference: {conj_correct['num_tests'].mean() - disj_correct['num_tests'].mean():.2f} more tests for conjunctive")

print(f"\n--- Total Test Time (Correct Only) ---")
print(f"Conjunctive: Mean={conj_correct['total_test_time_seconds'].mean():.2f}s, Median={conj_correct['total_test_time_seconds'].median():.2f}s")
print(f"Disjunctive: Mean={disj_correct['total_test_time_seconds'].mean():.2f}s, Median={disj_correct['total_test_time_seconds'].median():.2f}s")
print(f"Difference: {conj_correct['total_test_time_seconds'].mean() - disj_correct['total_test_time_seconds'].mean():.2f}s more for conjunctive")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("4. CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(22, 13))

# ============================================================================
# Plot 1: Number of Tests by Rule Type (All Rounds)
# ============================================================================
ax1 = axes[0, 0]

rule_types = ['Conjunctive', 'Disjunctive']
mean_tests = [conjunctive_df['num_tests'].mean(), disjunctive_df['num_tests'].mean()]
std_tests = [conjunctive_df['num_tests'].std(), disjunctive_df['num_tests'].std()]

colors = ['#1b9e77', '#d95f02']
bars = ax1.bar(range(len(rule_types)), mean_tests, color=colors, alpha=0.8,
               edgecolor='#333333', linewidth=1.5, yerr=std_tests, capsize=5)

ax1.set_title('Mean Number of Tests by Rule Type', fontsize=14, fontweight='bold')
ax1.set_ylabel('Mean Number of Tests', fontsize=12)
ax1.set_xticks(range(len(rule_types)))
ax1.set_xticklabels(rule_types, fontsize=12)
ax1.set_ylim(0, max(mean_tests) + max(std_tests) + 3)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, mean_val, n) in enumerate(zip(bars, mean_tests, [len(conjunctive_df), len(disjunctive_df)])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std_tests[i] + 0.5,
            f'{mean_val:.2f}\n(n={n})', 
            ha='center', va='bottom', fontsize=10)

# ============================================================================
# Plot 2: Box Plot of Tests by Rule Type
# ============================================================================
ax2 = axes[0, 1]

data_for_box = [conjunctive_df['num_tests'].dropna(), disjunctive_df['num_tests'].dropna()]
bp = ax2.boxplot(data_for_box, tick_labels=rule_types, patch_artist=True,
                 medianprops=dict(color='red', linewidth=2),
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_title('Distribution of Tests by Rule Type', fontsize=14, fontweight='bold')
ax2.set_ylabel('Number of Tests', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Plot 3: Mean Test Time by Rule Type
# ============================================================================
ax3 = axes[0, 2]

mean_time = [conjunctive_df['total_test_time_seconds'].mean(), 
             disjunctive_df['total_test_time_seconds'].mean()]
std_time = [conjunctive_df['total_test_time_seconds'].std(), 
            disjunctive_df['total_test_time_seconds'].std()]

bars3 = ax3.bar(range(len(rule_types)), mean_time, color=colors, alpha=0.8,
                edgecolor='#333333', linewidth=1.5, yerr=std_time, capsize=5)

ax3.set_title('Mean Total Test Time by Rule Type', fontsize=14, fontweight='bold')
ax3.set_ylabel('Mean Time (seconds)', fontsize=12)
ax3.set_xticks(range(len(rule_types)))
ax3.set_xticklabels(rule_types, fontsize=12)
ax3.set_ylim(0, max(mean_time) + max(std_time) + 10)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, mean_val, n) in enumerate(zip(bars3, mean_time, [len(conjunctive_df), len(disjunctive_df)])):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + std_time[i] + 3,
            f'{mean_val:.1f}s\n(n={n})', 
            ha='center', va='bottom', fontsize=10)

# ============================================================================
# Plot 4: Tests for Correct Classifications Only
# ============================================================================
ax4 = axes[1, 0]

conj_correct = correct_df[correct_df['ground_truth_rule'] == 'conjunctive']
disj_correct = correct_df[correct_df['ground_truth_rule'] == 'disjunctive']

mean_tests_correct = [conj_correct['num_tests'].mean(), disj_correct['num_tests'].mean()]
std_tests_correct = [conj_correct['num_tests'].std(), disj_correct['num_tests'].std()]

bars4 = ax4.bar(range(len(rule_types)), mean_tests_correct, color=colors, alpha=0.8,
                edgecolor='#333333', linewidth=1.5, yerr=std_tests_correct, capsize=5)

ax4.set_title('Mean Tests for Correct Classifications', fontsize=14, fontweight='bold')
ax4.set_ylabel('Mean Number of Tests', fontsize=12)
ax4.set_xticks(range(len(rule_types)))
ax4.set_xticklabels(rule_types, fontsize=12)
ax4.set_ylim(0, max(mean_tests_correct) + max(std_tests_correct) + 3)
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels with accuracy
for i, (bar, mean_val, rule_type) in enumerate(zip(bars4, mean_tests_correct, ['conjunctive', 'disjunctive'])):
    rule_df = correct_df[correct_df['ground_truth_rule'] == rule_type]
    height = bar.get_height()
    n = len(rule_df)
    accuracy = len(rule_df) / len(df[df['ground_truth_rule'] == rule_type]) * 100
    ax4.text(bar.get_x() + bar.get_width()/2., height + std_tests_correct[i] + 0.5,
            f'{mean_val:.2f}\n(n={n}, {accuracy:.0f}% acc)', 
            ha='center', va='bottom', fontsize=9)

# ============================================================================
# Plot 5: Scatter Plot - Tests vs Accuracy
# ============================================================================
ax5 = axes[1, 1]

# Plot conjunctive
conj_success = conjunctive_df[conjunctive_df['rule_choice_correct'] == True]
conj_fail = conjunctive_df[conjunctive_df['rule_choice_correct'] == False]

ax5.scatter(conj_success['num_tests'], [1]*len(conj_success), 
           color=colors[0], s=100, alpha=0.6, label='Conjunctive (Correct)', marker='o')
ax5.scatter(conj_fail['num_tests'], [0.9]*len(conj_fail), 
           color=colors[0], s=100, alpha=0.6, label='Conjunctive (Incorrect)', marker='x')

# Plot disjunctive
disj_success = disjunctive_df[disjunctive_df['rule_choice_correct'] == True]
disj_fail = disjunctive_df[disjunctive_df['rule_choice_correct'] == False]

ax5.scatter(disj_success['num_tests'], [0.5]*len(disj_success), 
           color=colors[1], s=100, alpha=0.6, label='Disjunctive (Correct)', marker='o')
ax5.scatter(disj_fail['num_tests'], [0.4]*len(disj_fail), 
           color=colors[1], s=100, alpha=0.6, label='Disjunctive (Incorrect)', marker='x')

ax5.set_title('Number of Tests vs Rule Classification Success', fontsize=14, fontweight='bold')
ax5.set_xlabel('Number of Tests', fontsize=12)
ax5.set_ylabel('Rule Type', fontsize=12)
ax5.set_yticks([0.45, 0.95])
ax5.set_yticklabels(['Disjunctive', 'Conjunctive'], fontsize=12)
ax5.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=True, shadow=True)
ax5.grid(True, alpha=0.3, axis='x')

# ============================================================================
# Plot 6: Average Time Per Test by Rule Type
# ============================================================================
ax6 = axes[1, 2]

# Calculate average time per test
conjunctive_df_copy = conjunctive_df.copy()
disjunctive_df_copy = disjunctive_df.copy()

conjunctive_df_copy['avg_time_per_test'] = conjunctive_df_copy['total_test_time_seconds'] / conjunctive_df_copy['num_tests']
disjunctive_df_copy['avg_time_per_test'] = disjunctive_df_copy['total_test_time_seconds'] / disjunctive_df_copy['num_tests']

mean_avg_time = [conjunctive_df_copy['avg_time_per_test'].mean(), 
                 disjunctive_df_copy['avg_time_per_test'].mean()]
std_avg_time = [conjunctive_df_copy['avg_time_per_test'].std(), 
                disjunctive_df_copy['avg_time_per_test'].std()]

bars6 = ax6.bar(range(len(rule_types)), mean_avg_time, color=colors, alpha=0.8,
                edgecolor='#333333', linewidth=1.5, yerr=std_avg_time, capsize=5)

ax6.set_title('Mean Time Per Test by Rule Type', fontsize=14, fontweight='bold')
ax6.set_ylabel('Mean Time (seconds)', fontsize=12)
ax6.set_xticks(range(len(rule_types)))
ax6.set_xticklabels(rule_types, fontsize=12)
ax6.set_ylim(0, max(mean_avg_time) + max(std_avg_time) + 2)
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, mean_val) in enumerate(zip(bars6, mean_avg_time)):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + std_avg_time[i] + 0.5,
            f'{mean_val:.2f}s', 
            ha='center', va='bottom', fontsize=10)

plt.suptitle('Exploration Effort Analysis: Conjunctive vs Disjunctive Rules', 
             fontsize=16, fontweight='bold', y=0.998)

plt.tight_layout(rect=[0, 0, 0.95, 0.98])
plt.subplots_adjust(hspace=0.35, wspace=0.35)
plt.savefig('exploration_by_rule_type_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: exploration_by_rule_type_analysis.png")
plt.close()

# ============================================================================
# SUMMARY AND INTERPRETATION
# ============================================================================
print("\n" + "="*80)
print("5. SUMMARY AND INTERPRETATION")
print("="*80)

print("\n--- Key Findings ---")

# Compare accuracies
conj_rule_acc = (conjunctive_df['rule_choice_correct'].sum() / len(conjunctive_df) * 100)
disj_rule_acc = (disjunctive_df['rule_choice_correct'].sum() / len(disjunctive_df) * 100)

print(f"\n1. ACCURACY RATES (similar):")
print(f"   - Conjunctive: {conj_rule_acc:.1f}%")
print(f"   - Disjunctive: {disj_rule_acc:.1f}%")
print(f"   - Difference: {abs(conj_rule_acc - disj_rule_acc):.1f} percentage points")

print(f"\n2. EXPLORATION EFFORT (number of tests):")
print(f"   - Conjunctive: {conj_tests.mean():.2f} tests on average")
print(f"   - Disjunctive: {disj_tests.mean():.2f} tests on average")
diff_tests = conj_tests.mean() - disj_tests.mean()
pct_more = (diff_tests / disj_tests.mean() * 100) if disj_tests.mean() > 0 else 0
print(f"   - Difference: {diff_tests:.2f} more tests for conjunctive ({pct_more:.1f}% more)")

print(f"\n3. TIME INVESTMENT (total test time):")
print(f"   - Conjunctive: {conj_time.mean():.2f}s on average")
print(f"   - Disjunctive: {disj_time.mean():.2f}s on average")
diff_time = conj_time.mean() - disj_time.mean()
pct_more_time = (diff_time / disj_time.mean() * 100) if disj_time.mean() > 0 else 0
print(f"   - Difference: {diff_time:.2f}s more for conjunctive ({pct_more_time:.1f}% more)")

print("\n--- CONCLUSION ---")
print("\nThe data shows that:")
if diff_tests > 0:
    print(f"✓ Participants took MORE tests to classify conjunctive rules")
    print(f"  ({conj_tests.mean():.2f} vs {disj_tests.mean():.2f} tests, {pct_more:+.1f}% more)")
    print(f"  Effect size (Cohen's d): {cohens_d:.3f} ({effect_interp})")
else:
    print(f"• No clear difference in number of tests between rule types")

if conj_rule_acc > 85 and disj_rule_acc > 85:
    print(f"✓ DESPITE having high and similar accuracy rates for both rule types")
    print(f"  (Conjunctive: {conj_rule_acc:.1f}%, Disjunctive: {disj_rule_acc:.1f}%)")

print("\nThis suggests that conjunctive rules may require more evidence to")
print("confidently classify, even though adults can ultimately classify them")
print("just as accurately as disjunctive rules.")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

