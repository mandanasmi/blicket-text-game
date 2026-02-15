"""
Visualize main game analysis:
1. Exploration per round for users
2. Understanding of conjunctive vs disjunctive
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")

# Load data
rounds_df = pd.read_csv('results/3_main_game_rounds.csv')
rule_df = pd.read_csv('results/4_rule_inference.csv')

print(f"Loaded {rounds_df['participant_id'].nunique()} participants")
print(f"Total rounds: {len(rounds_df)}")

# Create figure for both visualizations
fig = plt.figure(figsize=(18, 12))

# ============================================================================
# VISUALIZATION 1: EXPLORATION PER ROUND
# ============================================================================

# Create subplot grid
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1a. Tests per round - stacked bar by participant
ax1 = fig.add_subplot(gs[0, :2])

# Pivot data for stacked bar
pivot_data = rounds_df.pivot(index='participant_id', columns='round', values='num_tests_round')
pivot_data.index = [pid[:8] + '...' for pid in pivot_data.index]

pivot_data.plot(kind='bar', ax=ax1, stacked=False, color=['#1976D2', '#388E3C', '#F57C00'], 
                alpha=0.8, edgecolor='black', linewidth=1)
ax1.set_title('Number of Tests Per Round by Participant', fontsize=14, fontweight='bold')
ax1.set_xlabel('Participant', fontsize=12)
ax1.set_ylabel('Number of Tests', fontsize=12)
ax1.legend(title='Round', labels=['Round 1', 'Round 2', 'Round 3'])
ax1.grid(True, alpha=0.3, axis='y')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 1b. Average tests by round
ax2 = fig.add_subplot(gs[0, 2])

round_averages = rounds_df.groupby('round')['num_tests_round'].agg(['mean', 'std'])
rounds = round_averages.index
x_pos = np.arange(len(rounds))

bars = ax2.bar(x_pos, round_averages['mean'], yerr=round_averages['std'], 
               capsize=5, color=['#1976D2', '#388E3C', '#F57C00'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_title('Average Tests by Round', fontsize=14, fontweight='bold')
ax2.set_xlabel('Round', fontsize=12)
ax2.set_ylabel('Average Tests', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'Round {r}' for r in rounds])
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + round_averages['std'].iloc[i] + 0.3,
            f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 1c. Distribution of tests across all rounds
ax3 = fig.add_subplot(gs[1, :])

# Create histogram
ax3.hist(rounds_df['num_tests_round'], bins=range(0, int(rounds_df['num_tests_round'].max())+2), 
         edgecolor='black', color='#2196F3', alpha=0.7, rwidth=0.8)
ax3.axvline(rounds_df['num_tests_round'].mean(), color='red', linestyle='--', linewidth=2.5, 
            label=f'Mean: {rounds_df["num_tests_round"].mean():.2f}')
ax3.axvline(rounds_df['num_tests_round'].median(), color='green', linestyle='--', linewidth=2.5, 
            label=f'Median: {rounds_df["num_tests_round"].median():.1f}')
ax3.axvline(4, color='orange', linestyle='--', linewidth=2.5, label='Threshold: 4 tests')
ax3.set_title('Distribution of Number of Tests (All Rounds)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Number of Tests', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')

# Add stats box
stats_text = f"Mean: {rounds_df['num_tests_round'].mean():.2f}\nMedian: {rounds_df['num_tests_round'].median():.0f}\nStd: {rounds_df['num_tests_round'].std():.2f}\nRange: {rounds_df['num_tests_round'].min():.0f}-{rounds_df['num_tests_round'].max():.0f}"
ax3.text(0.98, 0.97, stats_text, transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# ============================================================================
# VISUALIZATION 2: CONJUNCTIVE VS DISJUNCTIVE UNDERSTANDING
# ============================================================================

# 2a. Overall checkbox accuracy
ax4 = fig.add_subplot(gs[2, 0])

checkbox_match = rule_df['checkbox_matches_truth'].value_counts()
colors_pie = ['#4CAF50', '#F44336']
labels_pie = ['Correct', 'Incorrect']
actual_vals = [checkbox_match.get(True, 0), checkbox_match.get(False, 0)]
actual_colors = [colors_pie[0] if actual_vals[0] > 0 else None, colors_pie[1] if actual_vals[1] > 0 else None]
actual_colors = [c for c in actual_colors if c is not None]
actual_labels = [l for i, l in enumerate(labels_pie) if actual_vals[i] > 0]
actual_vals = [v for v in actual_vals if v > 0]

wedges, texts, autotexts = ax4.pie(actual_vals, labels=actual_labels, autopct='%1.1f%%',
                                     colors=actual_colors, startangle=90,
                                     textprops={'fontsize': 12, 'fontweight': 'bold'})
ax4.set_title('Checkbox Selection Accuracy\n(All Rounds)', fontsize=14, fontweight='bold')

# Add count in center
total_checkbox = checkbox_match.sum()
correct_checkbox = checkbox_match.get(True, 0)
ax4.text(0, 0, f'{correct_checkbox}/{total_checkbox}', ha='center', va='center', 
         fontsize=18, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 2b. Accuracy by true rule type
ax5 = fig.add_subplot(gs[2, 1])

# Calculate accuracy by true rule
rule_accuracy = rule_df.groupby('ground_truth_rule')['checkbox_matches_truth'].agg(['sum', 'count'])
rule_accuracy['accuracy'] = rule_accuracy['sum'] / rule_accuracy['count'] * 100

x_pos = np.arange(len(rule_accuracy))
bars = ax5.bar(x_pos, rule_accuracy['accuracy'], color=['#2196F3', '#FF9800'], 
               alpha=0.7, edgecolor='black', linewidth=1.5)
ax5.set_title('Accuracy by True Rule Type', fontsize=14, fontweight='bold')
ax5.set_xlabel('Ground Truth Rule', fontsize=12)
ax5.set_ylabel('Accuracy (%)', fontsize=12)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(rule_accuracy.index, rotation=0)
ax5.set_ylim(0, 100)
ax5.axhline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Chance (50%)')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%\n(n={int(rule_accuracy.iloc[i]["count"])})',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# 2c. Confusion matrix
ax6 = fig.add_subplot(gs[2, 2])

# Create confusion matrix
confusion_df = rule_df[rule_df['checkbox_matches_truth'].notna()].copy()
if len(confusion_df) > 0:
    confusion_matrix = pd.crosstab(
        confusion_df['ground_truth_rule'], 
        confusion_df['user_checkbox_selection']
    )
    
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Count'}, ax=ax6, linewidths=1, linecolor='black')
    ax6.set_title('Confusion Matrix\n(Truth vs User Selection)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('User Selection', fontsize=12)
    ax6.set_ylabel('Ground Truth', fontsize=12)
else:
    ax6.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax6.transAxes)

plt.suptitle('Main Game Analysis: Exploration and Rule Understanding', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('main_game_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: main_game_visualization.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("VISUALIZATION SUMMARY")
print("="*80)

print("\n--- EXPLORATION PER ROUND ---")
print(f"Average tests per round: {rounds_df['num_tests_round'].mean():.2f}")
print(f"Median tests per round: {rounds_df['num_tests_round'].median():.1f}")
print(f"Range: {rounds_df['num_tests_round'].min():.0f} - {rounds_df['num_tests_round'].max():.0f}")
print(f"Rounds with >4 tests: {(rounds_df['num_tests_round'] > 4).sum()}/{len(rounds_df)} ({(rounds_df['num_tests_round'] > 4).sum()/len(rounds_df)*100:.1f}%)")

print("\n--- RULE UNDERSTANDING ---")
correct = rule_df['checkbox_matches_truth'].sum()
total = rule_df['checkbox_matches_truth'].notna().sum()
print(f"Overall accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

for rule_type in rule_df['ground_truth_rule'].unique():
    if pd.notna(rule_type):
        subset = rule_df[rule_df['ground_truth_rule'] == rule_type]
        correct_subset = subset['checkbox_matches_truth'].sum()
        total_subset = subset['checkbox_matches_truth'].notna().sum()
        print(f"  {rule_type}: {correct_subset}/{total_subset} ({correct_subset/total_subset*100:.1f}%)")

print("\n" + "="*80)

