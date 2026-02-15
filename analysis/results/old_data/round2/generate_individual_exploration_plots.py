"""
Generate separate plots for each panel of the exploration analysis
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
print("GENERATING INDIVIDUAL EXPLORATION PLOTS")
print("="*80)
print("\nLoading main_game_data_with_prior_experience.csv...")

df = pd.read_csv('main_game_data_with_prior_experience.csv')

print(f"Total rounds: {len(df)}")
print(f"Participants: {df['user_id'].nunique()}")

# Prepare data
conjunctive_df = df[df['ground_truth_rule'] == 'conjunctive']
disjunctive_df = df[df['ground_truth_rule'] == 'disjunctive']
correct_df = df[df['rule_choice_correct'] == True]

rule_types = ['Conjunctive', 'Disjunctive']
colors = ['#1b9e77', '#d95f02']

# ============================================================================
# PLOT 1: Mean Number of Tests by Rule Type
# ============================================================================
print("\n1. Generating: plot1_mean_tests_by_rule_type.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

mean_tests = [conjunctive_df['num_tests'].mean(), disjunctive_df['num_tests'].mean()]
std_tests = [conjunctive_df['num_tests'].std(), disjunctive_df['num_tests'].std()]

bars = ax.bar(range(len(rule_types)), mean_tests, color=colors, alpha=0.8,
              edgecolor='#333333', linewidth=1.5, yerr=std_tests, capsize=5)

ax.set_title('Mean Number of Tests by Rule Type', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Mean Number of Tests', fontsize=13)
ax.set_xticks(range(len(rule_types)))
ax.set_xticklabels(rule_types, fontsize=13)
ax.set_ylim(0, max(mean_tests) + max(std_tests) + 3)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, mean_val, n) in enumerate(zip(bars, mean_tests, [len(conjunctive_df), len(disjunctive_df)])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std_tests[i] + 0.5,
            f'{mean_val:.2f}\n(n={n})', 
            ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('plot1_mean_tests_by_rule_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved")

# ============================================================================
# PLOT 2: Box Plot of Tests by Rule Type
# ============================================================================
print("2. Generating: plot2_boxplot_tests_by_rule_type.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

data_for_box = [conjunctive_df['num_tests'].dropna(), disjunctive_df['num_tests'].dropna()]
bp = ax.boxplot(data_for_box, tick_labels=rule_types, patch_artist=True,
                medianprops=dict(color='red', linewidth=2),
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('Distribution of Tests by Rule Type', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Number of Tests', fontsize=13)
ax.set_xticklabels(rule_types, fontsize=13)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plot2_boxplot_tests_by_rule_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved")

# ============================================================================
# PLOT 3: Mean Total Test Time by Rule Type
# ============================================================================
print("3. Generating: plot3_mean_test_time_by_rule_type.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

mean_time = [conjunctive_df['total_test_time_seconds'].mean(), 
             disjunctive_df['total_test_time_seconds'].mean()]
std_time = [conjunctive_df['total_test_time_seconds'].std(), 
            disjunctive_df['total_test_time_seconds'].std()]

bars = ax.bar(range(len(rule_types)), mean_time, color=colors, alpha=0.8,
              edgecolor='#333333', linewidth=1.5, yerr=std_time, capsize=5)

ax.set_title('Mean Total Test Time by Rule Type', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Mean Time (seconds)', fontsize=13)
ax.set_xticks(range(len(rule_types)))
ax.set_xticklabels(rule_types, fontsize=13)
ax.set_ylim(0, max(mean_time) + max(std_time) + 10)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, mean_val, n) in enumerate(zip(bars, mean_time, [len(conjunctive_df), len(disjunctive_df)])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std_time[i] + 3,
            f'{mean_val:.1f}s\n(n={n})', 
            ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('plot3_mean_test_time_by_rule_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved")

# ============================================================================
# PLOT 4: Tests for Correct Classifications Only
# ============================================================================
print("4. Generating: plot4_tests_correct_classifications.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

conj_correct = correct_df[correct_df['ground_truth_rule'] == 'conjunctive']
disj_correct = correct_df[correct_df['ground_truth_rule'] == 'disjunctive']

mean_tests_correct = [conj_correct['num_tests'].mean(), disj_correct['num_tests'].mean()]
std_tests_correct = [conj_correct['num_tests'].std(), disj_correct['num_tests'].std()]

bars = ax.bar(range(len(rule_types)), mean_tests_correct, color=colors, alpha=0.8,
              edgecolor='#333333', linewidth=1.5, yerr=std_tests_correct, capsize=5)

ax.set_title('Mean Tests for Correct Classifications', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Mean Number of Tests', fontsize=13)
ax.set_xticks(range(len(rule_types)))
ax.set_xticklabels(rule_types, fontsize=13)
ax.set_ylim(0, max(mean_tests_correct) + max(std_tests_correct) + 3)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels with accuracy
for i, (bar, mean_val, rule_type) in enumerate(zip(bars, mean_tests_correct, ['conjunctive', 'disjunctive'])):
    rule_df = correct_df[correct_df['ground_truth_rule'] == rule_type]
    height = bar.get_height()
    n = len(rule_df)
    accuracy = len(rule_df) / len(df[df['ground_truth_rule'] == rule_type]) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height + std_tests_correct[i] + 0.5,
            f'{mean_val:.2f}\n(n={n}, {accuracy:.0f}% acc)', 
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('plot4_tests_correct_classifications.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved")

# ============================================================================
# PLOT 5: Scatter Plot - Tests vs Accuracy
# ============================================================================
print("5. Generating: plot5_tests_vs_accuracy_scatter.png")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot conjunctive
conj_success = conjunctive_df[conjunctive_df['rule_choice_correct'] == True]
conj_fail = conjunctive_df[conjunctive_df['rule_choice_correct'] == False]

ax.scatter(conj_success['num_tests'], [1]*len(conj_success), 
          color=colors[0], s=120, alpha=0.6, label='Conjunctive (Correct)', marker='o', edgecolors='black', linewidths=1)
ax.scatter(conj_fail['num_tests'], [0.9]*len(conj_fail), 
          color=colors[0], s=120, alpha=0.6, label='Conjunctive (Incorrect)', marker='x', linewidths=2)

# Plot disjunctive
disj_success = disjunctive_df[disjunctive_df['rule_choice_correct'] == True]
disj_fail = disjunctive_df[disjunctive_df['rule_choice_correct'] == False]

ax.scatter(disj_success['num_tests'], [0.5]*len(disj_success), 
          color=colors[1], s=120, alpha=0.6, label='Disjunctive (Correct)', marker='o', edgecolors='black', linewidths=1)
ax.scatter(disj_fail['num_tests'], [0.4]*len(disj_fail), 
          color=colors[1], s=120, alpha=0.6, label='Disjunctive (Incorrect)', marker='x', linewidths=2)

ax.set_title('Number of Tests vs Rule Classification Success', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Number of Tests', fontsize=13)
ax.set_ylabel('Rule Type', fontsize=13)
ax.set_yticks([0.45, 0.95])
ax.set_yticklabels(['Disjunctive', 'Conjunctive'], fontsize=13)
ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('plot5_tests_vs_accuracy_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved")

# ============================================================================
# PLOT 6: Average Time Per Test by Rule Type
# ============================================================================
print("6. Generating: plot6_avg_time_per_test_by_rule_type.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Calculate average time per test
conjunctive_df_copy = conjunctive_df.copy()
disjunctive_df_copy = disjunctive_df.copy()

conjunctive_df_copy['avg_time_per_test'] = conjunctive_df_copy['total_test_time_seconds'] / conjunctive_df_copy['num_tests']
disjunctive_df_copy['avg_time_per_test'] = disjunctive_df_copy['total_test_time_seconds'] / disjunctive_df_copy['num_tests']

mean_avg_time = [conjunctive_df_copy['avg_time_per_test'].mean(), 
                 disjunctive_df_copy['avg_time_per_test'].mean()]
std_avg_time = [conjunctive_df_copy['avg_time_per_test'].std(), 
                disjunctive_df_copy['avg_time_per_test'].std()]

bars = ax.bar(range(len(rule_types)), mean_avg_time, color=colors, alpha=0.8,
              edgecolor='#333333', linewidth=1.5, yerr=std_avg_time, capsize=5)

ax.set_title('Mean Time Per Test by Rule Type', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Mean Time (seconds)', fontsize=13)
ax.set_xticks(range(len(rule_types)))
ax.set_xticklabels(rule_types, fontsize=13)
ax.set_ylim(0, max(mean_avg_time) + max(std_avg_time) + 2)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, mean_val) in enumerate(zip(bars, mean_avg_time)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std_avg_time[i] + 0.5,
            f'{mean_val:.2f}s', 
            ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('plot6_avg_time_per_test_by_rule_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved")

print("\n" + "="*80)
print("ALL INDIVIDUAL PLOTS GENERATED")
print("="*80)
print("\nGenerated files:")
print("  1. plot1_mean_tests_by_rule_type.png")
print("  2. plot2_boxplot_tests_by_rule_type.png")
print("  3. plot3_mean_test_time_by_rule_type.png")
print("  4. plot4_tests_correct_classifications.png")
print("  5. plot5_tests_vs_accuracy_scatter.png")
print("  6. plot6_avg_time_per_test_by_rule_type.png")
print("="*80)

