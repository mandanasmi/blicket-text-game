"""
Visualize relationship between exploration per round and accuracy in main game
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")

# Load data
main_df = pd.read_csv('results/3_main_game_rounds.csv')

print("="*80)
print("ROUND EXPLORATION vs ACCURACY ANALYSIS")
print("="*80)

# Convert True/False to numeric for correlation
main_df['objects_match_numeric'] = main_df['objects_match'].astype(int) * 100  # Convert to percentage
main_df['rule_match_numeric'] = main_df['rule_match'].astype(int) * 100  # Convert to percentage

print(f"\nTotal rounds analyzed: {len(main_df)}")
print(f"Participants: {main_df['participant_id'].nunique()}")
print(f"Rounds per participant: {main_df.groupby('participant_id')['round'].count().mean():.1f}")

# Calculate correlations using numpy
def pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient"""
    x_arr = np.array(x)
    y_arr = np.array(y)
    n = len(x_arr)
    sum_x = np.sum(x_arr)
    sum_y = np.sum(y_arr)
    sum_xy = np.sum(x_arr * y_arr)
    sum_x2 = np.sum(x_arr ** 2)
    sum_y2 = np.sum(y_arr ** 2)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
    
    if denominator == 0:
        return 0.0
    
    r = numerator / denominator
    return r

# Overall correlations
corr_obj_overall = pearson_correlation(main_df['num_tests_round'], main_df['objects_match_numeric'])
corr_rule_overall = pearson_correlation(main_df['num_tests_round'], main_df['rule_match_numeric'])

print(f"\n--- Overall Correlations (All Rounds) ---")
print(f"Exploration vs Object Identification Accuracy: r = {corr_obj_overall:.3f}")
print(f"Exploration vs Rule Choice Accuracy: r = {corr_rule_overall:.3f}")

# Correlations by round
print(f"\n--- Correlations by Round ---")
for round_num in sorted(main_df['round'].unique()):
    round_data = main_df[main_df['round'] == round_num]
    corr_obj_round = pearson_correlation(round_data['num_tests_round'], round_data['objects_match_numeric'])
    corr_rule_round = pearson_correlation(round_data['num_tests_round'], round_data['rule_match_numeric'])
    print(f"Round {round_num}:")
    print(f"  Object Accuracy: r = {corr_obj_round:.3f} (n={len(round_data)})")
    print(f"  Rule Accuracy: r = {corr_rule_round:.3f} (n={len(round_data)})")

# Add jitter only to x-axis to prevent overlapping points
# No y-axis jitter so 100% points are exactly on the 100 line
np.random.seed(42)
x_jitter = np.random.normal(0, 0.3, len(main_df))

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 18))

# Plot 1: Exploration vs Object Accuracy (Overall)
ax1 = axes[0, 0]
x1_jittered = main_df['num_tests_round'] + x_jitter
scatter1 = ax1.scatter(x1_jittered, main_df['objects_match_numeric'], 
                       s=80, alpha=0.6, c='#2196F3', edgecolors='black', linewidth=1, zorder=3)

# Add trend line
z1 = np.polyfit(main_df['num_tests_round'], main_df['objects_match_numeric'], 1)
p1 = np.poly1d(z1)
x_trend1 = np.linspace(main_df['num_tests_round'].min(), main_df['num_tests_round'].max(), 100)
ax1.plot(x_trend1, p1(x_trend1), 
         "r--", alpha=0.6, linewidth=2, label=f'Trend line (r={corr_obj_overall:.3f})', zorder=2)

ax1.set_title('Exploration vs Object Identification Accuracy (All Rounds)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Number of Tests in Round', fontsize=12)
ax1.set_ylabel('Object Identification Accuracy (%)', fontsize=12)
ax1.set_ylim(-10, 110)
ax1.set_xlim(-1, max(main_df['num_tests_round']) + 1)
ax1.set_yticks([0, 50, 100])
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))

# Plot 2: Exploration vs Rule Accuracy (Overall)
ax2 = axes[0, 1]
x2_jittered = main_df['num_tests_round'] + x_jitter
scatter2 = ax2.scatter(x2_jittered, main_df['rule_match_numeric'], 
                       s=80, alpha=0.6, c='#FF9800', edgecolors='black', linewidth=1, zorder=3)

# Add trend line
z2 = np.polyfit(main_df['num_tests_round'], main_df['rule_match_numeric'], 1)
p2 = np.poly1d(z2)
x_trend2 = np.linspace(main_df['num_tests_round'].min(), main_df['num_tests_round'].max(), 100)
ax2.plot(x_trend2, p2(x_trend2), 
         "r--", alpha=0.6, linewidth=2, label=f'Trend line (r={corr_rule_overall:.3f})', zorder=2)

ax2.set_title('Exploration vs Rule Choice Accuracy (All Rounds)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Number of Tests in Round', fontsize=12)
ax2.set_ylabel('Rule Choice Accuracy (%)', fontsize=12)
ax2.set_ylim(-10, 110)
ax2.set_xlim(-1, max(main_df['num_tests_round']) + 1)
ax2.set_yticks([0, 50, 100])
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))

# Plot 3: Exploration vs Object Accuracy by Round
ax3 = axes[1, 0]
colors_rounds = ['#4CAF50', '#2196F3', '#FF9800']
for i, round_num in enumerate(sorted(main_df['round'].unique())):
    round_data = main_df[main_df['round'] == round_num]
    x_jitter_round = np.random.normal(0, 0.25, len(round_data))
    x3_jittered = round_data['num_tests_round'] + x_jitter_round
    
    corr_round = pearson_correlation(round_data['num_tests_round'], round_data['objects_match_numeric'])
    ax3.scatter(x3_jittered, round_data['objects_match_numeric'], 
                s=80, alpha=0.6, c=colors_rounds[i], edgecolors='black', linewidth=1, 
                label=f'Round {round_num} (r={corr_round:.3f})', zorder=3)
    
    # Add trend line for this round
    if len(round_data) > 1:
        z3 = np.polyfit(round_data['num_tests_round'], round_data['objects_match_numeric'], 1)
        p3 = np.poly1d(z3)
        x_trend3 = np.linspace(round_data['num_tests_round'].min(), round_data['num_tests_round'].max(), 100)
        ax3.plot(x_trend3, p3(x_trend3), 
                "--", alpha=0.5, linewidth=1.5, color=colors_rounds[i], zorder=2)

ax3.set_title('Exploration vs Object Identification Accuracy (By Round)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Number of Tests in Round', fontsize=12)
ax3.set_ylabel('Object Identification Accuracy (%)', fontsize=12)
ax3.set_ylim(-10, 110)
ax3.set_xlim(-1, max(main_df['num_tests_round']) + 1)
ax3.set_yticks([0, 50, 100])
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))

# Plot 4: Exploration vs Rule Accuracy by Round
ax4 = axes[1, 1]
for i, round_num in enumerate(sorted(main_df['round'].unique())):
    round_data = main_df[main_df['round'] == round_num]
    x_jitter_round = np.random.normal(0, 0.25, len(round_data))
    x4_jittered = round_data['num_tests_round'] + x_jitter_round
    
    corr_round = pearson_correlation(round_data['num_tests_round'], round_data['rule_match_numeric'])
    ax4.scatter(x4_jittered, round_data['rule_match_numeric'], 
                s=80, alpha=0.6, c=colors_rounds[i], edgecolors='black', linewidth=1, 
                label=f'Round {round_num} (r={corr_round:.3f})', zorder=3)
    
    # Add trend line for this round
    if len(round_data) > 1:
        z4 = np.polyfit(round_data['num_tests_round'], round_data['rule_match_numeric'], 1)
        p4 = np.poly1d(z4)
        x_trend4 = np.linspace(round_data['num_tests_round'].min(), round_data['num_tests_round'].max(), 100)
        ax4.plot(x_trend4, p4(x_trend4), 
                "--", alpha=0.5, linewidth=1.5, color=colors_rounds[i], zorder=2)

ax4.set_title('Exploration vs Rule Choice Accuracy (By Round)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Number of Tests in Round', fontsize=12)
ax4.set_ylabel('Rule Choice Accuracy (%)', fontsize=12)
ax4.set_ylim(-10, 110)
ax4.set_xlim(-1, max(main_df['num_tests_round']) + 1)
ax4.set_yticks([0, 50, 100])
ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax4.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))

plt.suptitle('Relationship: Main Game Round Exploration vs Accuracy', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 0.88, 0.98])
plt.savefig('plot9_round_exploration_vs_accuracy.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plot9_round_exploration_vs_accuracy.png")
plt.close()

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nExploration (Tests per round):")
print(f"  Mean: {main_df['num_tests_round'].mean():.1f}")
print(f"  Median: {main_df['num_tests_round'].median():.1f}")
print(f"  Range: {main_df['num_tests_round'].min():.0f} - {main_df['num_tests_round'].max():.0f}")

print(f"\nObject Identification Accuracy:")
print(f"  Mean: {main_df['objects_match_numeric'].mean():.1f}%")
print(f"  Correct: {main_df['objects_match'].sum()}/{len(main_df)} ({main_df['objects_match'].sum()/len(main_df)*100:.1f}%)")

print(f"\nRule Choice Accuracy:")
print(f"  Mean: {main_df['rule_match_numeric'].mean():.1f}%")
print(f"  Correct: {main_df['rule_match'].sum()}/{len(main_df)} ({main_df['rule_match'].sum()/len(main_df)*100:.1f}%)")

print("\n" + "="*80)

