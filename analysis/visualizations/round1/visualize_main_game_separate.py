"""
Visualize main game analysis as separate PNG files
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
print(f"Total rounds: {len(rounds_df)}\n")

# ============================================================================
# PLOT 1: Tests per round by participant
# ============================================================================
print("1. Creating tests per round by participant...")

fig, ax = plt.subplots(figsize=(14, 8))

pivot_data = rounds_df.pivot(index='participant_id', columns='round', values='num_tests_round')
pivot_data.index = [pid[:8] + '...' for pid in pivot_data.index]

pivot_data.plot(kind='bar', ax=ax, stacked=False, color=['#1976D2', '#388E3C', '#F57C00'], 
                alpha=0.8, edgecolor='black', linewidth=1)
ax.set_title('Number of Tests Per Round by Participant', fontsize=16, fontweight='bold')
ax.set_xlabel('Participant', fontsize=13)
ax.set_ylabel('Number of Tests', fontsize=13)
ax.legend(title='', labels=['Round 1', 'Round 2', 'Round 3'], fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('plot1_tests_per_round_by_participant.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: plot1_tests_per_round_by_participant.png")
plt.close()

# ============================================================================
# PLOT 2: Average tests by round
# ============================================================================
print("2. Creating average tests by round...")

fig, ax = plt.subplots(figsize=(10, 7))

round_averages = rounds_df.groupby('round')['num_tests_round'].agg(['mean', 'std'])
rounds = round_averages.index
x_pos = np.arange(len(rounds))

bars = ax.bar(x_pos, round_averages['mean'], yerr=round_averages['std'], 
              capsize=5, color=['#1976D2', '#388E3C', '#F57C00'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_title('Average Number of Tests by Round', fontsize=16, fontweight='bold')
ax.set_xlabel('', fontsize=13)
ax.set_ylabel('Average Tests', fontsize=13)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Round {r}' for r in rounds], fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + round_averages['std'].iloc[i] + 0.3,
            f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('plot2_average_tests_by_round.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: plot2_average_tests_by_round.png")
plt.close()

# ============================================================================
# PLOT 3: Distribution of tests
# ============================================================================
print("3. Creating test distribution...")

fig, ax = plt.subplots(figsize=(12, 7))

ax.hist(rounds_df['num_tests_round'], bins=range(0, int(rounds_df['num_tests_round'].max())+2), 
        edgecolor='black', color='#2196F3', alpha=0.7, rwidth=0.8)
ax.axvline(rounds_df['num_tests_round'].mean(), color='red', linestyle='--', linewidth=2.5, 
           label=f'Mean: {rounds_df["num_tests_round"].mean():.2f}')
ax.axvline(rounds_df['num_tests_round'].median(), color='green', linestyle='--', linewidth=2.5, 
           label=f'Median: {rounds_df["num_tests_round"].median():.1f}')
ax.axvline(4, color='orange', linestyle='--', linewidth=2.5, label='Threshold: 4 tests')
ax.set_title('Distribution of Number of Tests (All Rounds)', fontsize=16, fontweight='bold')
ax.set_xlabel('Number of Tests', fontsize=13)
ax.set_ylabel('Frequency', fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Add stats box
stats_text = f"Mean: {rounds_df['num_tests_round'].mean():.2f}\nMedian: {rounds_df['num_tests_round'].median():.0f}\nStd: {rounds_df['num_tests_round'].std():.2f}\nRange: {rounds_df['num_tests_round'].min():.0f}-{rounds_df['num_tests_round'].max():.0f}"
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('plot3_test_distribution.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: plot3_test_distribution.png")
plt.close()

# ============================================================================
# PLOT 4: Overall checkbox accuracy (pie chart)
# ============================================================================
print("4. Creating overall accuracy pie chart...")

fig, ax = plt.subplots(figsize=(10, 8))

checkbox_match = rule_df['checkbox_matches_truth'].value_counts()
colors_pie = ['#4CAF50', '#F44336']
labels_pie = ['Correct', 'Incorrect']
actual_vals = [checkbox_match.get(True, 0), checkbox_match.get(False, 0)]
actual_colors = [colors_pie[0] if actual_vals[0] > 0 else None, colors_pie[1] if actual_vals[1] > 0 else None]
actual_colors = [c for c in actual_colors if c is not None]
actual_labels = [l for i, l in enumerate(labels_pie) if actual_vals[i] > 0]
actual_vals = [v for v in actual_vals if v > 0]

wedges, texts, autotexts = ax.pie(actual_vals, labels=actual_labels, autopct='%1.1f%%',
                                    colors=actual_colors, startangle=90,
                                    textprops={'fontsize': 14, 'fontweight': 'bold'},
                                    explode=[0.05] * len(actual_vals))
ax.set_title('Rule Type Classification Accuracy (All Rounds)', fontsize=16, fontweight='bold')

# Add count in center
total_checkbox = checkbox_match.sum()
correct_checkbox = checkbox_match.get(True, 0)
ax.text(0, 0, f'{correct_checkbox}/{total_checkbox}\nCorrect', ha='center', va='center', 
        fontsize=18, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig('plot4_overall_rule_accuracy.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: plot4_overall_rule_accuracy.png")
plt.close()

# ============================================================================
# PLOT 5: Accuracy by true rule type
# ============================================================================
print("5. Creating accuracy by rule type...")

fig, ax = plt.subplots(figsize=(10, 7))

rule_accuracy = rule_df.groupby('ground_truth_rule')['checkbox_matches_truth'].agg(['sum', 'count'])
rule_accuracy['accuracy'] = rule_accuracy['sum'] / rule_accuracy['count'] * 100

x_pos = np.arange(len(rule_accuracy))
bars = ax.bar(x_pos, rule_accuracy['accuracy'], color=['#2196F3', '#FF9800'], 
              alpha=0.7, edgecolor='black', linewidth=2)
ax.set_title('Rule Classification Accuracy by True Rule Type', fontsize=16, fontweight='bold')
ax.set_xlabel('Ground Truth Rule', fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_xticks(x_pos)
ax.set_xticklabels([r.capitalize() for r in rule_accuracy.index], rotation=0, fontsize=12)
ax.set_ylim(0, 105)
ax.axhline(50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Chance (50%)')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%\n(n={int(rule_accuracy.iloc[i]["count"])})',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('plot5_accuracy_by_rule_type.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: plot5_accuracy_by_rule_type.png")
plt.close()

# ============================================================================
# PLOT 6: Confusion matrix
# ============================================================================
print("6. Creating confusion matrix...")

fig, ax = plt.subplots(figsize=(10, 8))

confusion_df = rule_df[rule_df['checkbox_matches_truth'].notna()].copy()
confusion_matrix = pd.crosstab(
    confusion_df['ground_truth_rule'], 
    confusion_df['user_checkbox_selection']
)

sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Count'}, ax=ax, linewidths=2, linecolor='black',
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
ax.set_title('Confusion Matrix: True Rule vs User Selection', fontsize=16, fontweight='bold')
ax.set_xlabel('User Selection', fontsize=13)
ax.set_ylabel('Ground Truth', fontsize=13)
ax.set_xticklabels([l.get_text().capitalize() for l in ax.get_xticklabels()], fontsize=12)
ax.set_yticklabels([l.get_text().capitalize() for l in ax.get_yticklabels()], fontsize=12, rotation=0)

plt.tight_layout()
plt.savefig('plot6_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: plot6_confusion_matrix.png")
plt.close()

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE")
print("="*80)
print("\nGenerated 6 separate PNG files:")
print("  1. plot1_tests_per_round_by_participant.png")
print("  2. plot2_average_tests_by_round.png")
print("  3. plot3_test_distribution.png")
print("  4. plot4_overall_rule_accuracy.png")
print("  5. plot5_accuracy_by_rule_type.png")
print("  6. plot6_confusion_matrix.png")
print("\n" + "="*80)

