"""
Visualize relationship between comprehension exploration and main game success
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")

# Load data
comp_df = pd.read_csv('results/1_comprehension_phase.csv')
main_df = pd.read_csv('results/3_main_game_rounds.csv')
accuracy_df = pd.read_csv('results/5_per_user_rule_accuracy.csv')

print("="*80)
print("EXPLORATION vs SUCCESS ANALYSIS")
print("="*80)

# Merge data
# Get comprehension exploration
comp_exploration = comp_df[['participant_id', 'num_tests']].copy()
comp_exploration.columns = ['participant_id', 'comprehension_tests']

# Get main game success metrics
# Option 1: Rule accuracy
rule_accuracy = accuracy_df[['participant_id', 'checkbox_accuracy_percent']].copy()
rule_accuracy.columns = ['participant_id', 'rule_accuracy']

# Option 2: Object identification accuracy
object_accuracy = main_df.groupby('participant_id').agg({
    'objects_match': lambda x: x.notna().sum() > 0 and (x.sum() / x.notna().sum() * 100)
}).reset_index()
object_accuracy.columns = ['participant_id', 'object_accuracy']

# Merge all
merged = comp_exploration.merge(rule_accuracy, on='participant_id')
merged = merged.merge(object_accuracy, on='participant_id')

print(f"\nParticipants in analysis: {len(merged)}")
print(f"\nComprehension tests range: {merged['comprehension_tests'].min():.0f} - {merged['comprehension_tests'].max():.0f}")
print(f"Rule accuracy range: {merged['rule_accuracy'].min():.1f}% - {merged['rule_accuracy'].max():.1f}%")
print(f"Object accuracy range: {merged['object_accuracy'].min():.1f}% - {merged['object_accuracy'].max():.1f}%")

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
        return 0.0, 0.0
    
    r = numerator / denominator
    return r, 0.0  # Return 0.0 for p-value as placeholder

corr_rule_r, corr_rule_p = pearson_correlation(merged['comprehension_tests'], merged['rule_accuracy'])
corr_object_r, corr_object_p = pearson_correlation(merged['comprehension_tests'], merged['object_accuracy'])

print(f"\n--- Correlation Analysis ---")
print(f"Exploration vs Rule Accuracy: r = {corr_rule_r:.3f}")
print(f"Exploration vs Object Accuracy: r = {corr_object_r:.3f}")

# Add jitter to prevent overlapping points
np.random.seed(42)  # For reproducibility
x_jitter = np.random.normal(0, 0.08, len(merged))
y_jitter_rule = np.random.normal(0, 1.5, len(merged))
y_jitter_object = np.random.normal(0, 1.5, len(merged))

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Exploration vs Rule Accuracy
ax1 = axes[0]

# Create scatter plot with jitter to show all points
x1_jittered = merged['comprehension_tests'] + x_jitter
y1_jittered = merged['rule_accuracy'] + y_jitter_rule
scatter1 = ax1.scatter(x1_jittered, y1_jittered, 
                       s=80, alpha=0.6, c='#2196F3', edgecolors='black', linewidth=1, zorder=3)

# Add trend line
z1 = np.polyfit(merged['comprehension_tests'], merged['rule_accuracy'], 1)
p1 = np.poly1d(z1)
x_trend = np.linspace(merged['comprehension_tests'].min(), merged['comprehension_tests'].max(), 100)
ax1.plot(x_trend, p1(x_trend), 
         "r--", alpha=0.6, linewidth=2, label=f'Trend line (r={corr_rule_r:.3f})', zorder=2)

ax1.set_title('Comprehension Exploration vs Rule Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Number of Tests in Comprehension Phase', fontsize=12)
ax1.set_ylabel('Rule Classification Accuracy (%)', fontsize=12)
ax1.set_ylim(60, 110)
ax1.set_xlim(0.5, max(merged['comprehension_tests']) + 0.5)
ax1.set_xticks(range(1, max(merged['comprehension_tests']) + 1))
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.legend(fontsize=10, loc='lower right')

# Add correlation text
ax1.text(0.98, 0.98, f'r = {corr_rule_r:.3f}', 
         transform=ax1.transAxes, fontsize=11, fontweight='bold',
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

# Plot 2: Exploration vs Object Identification Accuracy
ax2 = axes[1]

# Create scatter plot with jitter to show all points
x2_jittered = merged['comprehension_tests'] + x_jitter
y2_jittered = merged['object_accuracy'] + y_jitter_object
scatter2 = ax2.scatter(x2_jittered, y2_jittered, 
                       s=80, alpha=0.6, c='#FF9800', edgecolors='black', linewidth=1, zorder=3)

# Add trend line
z2 = np.polyfit(merged['comprehension_tests'], merged['object_accuracy'], 1)
p2 = np.poly1d(z2)
x_trend2 = np.linspace(merged['comprehension_tests'].min(), merged['comprehension_tests'].max(), 100)
ax2.plot(x_trend2, p2(x_trend2), 
         "r--", alpha=0.6, linewidth=2, label=f'Trend line (r={corr_object_r:.3f})', zorder=2)

ax2.set_title('Comprehension Exploration vs Object Identification Accuracy', fontsize=14, fontweight='bold')
ax2.set_xlabel('Number of Tests in Comprehension Phase', fontsize=12)
ax2.set_ylabel('Object Identification Accuracy (%)', fontsize=12)
ax2.set_ylim(60, 110)
ax2.set_xlim(0.5, max(merged['comprehension_tests']) + 0.5)
ax2.set_xticks(range(1, max(merged['comprehension_tests']) + 1))
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.legend(fontsize=10, loc='lower right')

# Add correlation text
ax2.text(0.98, 0.98, f'r = {corr_object_r:.3f}', 
         transform=ax2.transAxes, fontsize=11, fontweight='bold',
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

plt.suptitle('Relationship: Comprehension Exploration vs Main Game Success', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('plot8_exploration_vs_success.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plot8_exploration_vs_success.png")
plt.close()

# Print detailed summary
print("\n" + "="*80)
print("SUMMARY BY EXPLORATION LEVEL")
print("="*80)

# Categorize by exploration
merged['exploration_level'] = pd.cut(merged['comprehension_tests'], 
                                      bins=[0, 3, 4, 6], 
                                      labels=['Low (≤3)', 'Medium (4)', 'High (≥5)'])

for level in ['Low (≤3)', 'Medium (4)', 'High (≥5)']:
    subset = merged[merged['exploration_level'] == level]
    if len(subset) > 0:
        print(f"\n{level} Explorers ({len(subset)} participants):")
        print(f"  Average Rule Accuracy: {subset['rule_accuracy'].mean():.1f}%")
        print(f"  Average Object Accuracy: {subset['object_accuracy'].mean():.1f}%")
        print(f"  Average Comprehension Tests: {subset['comprehension_tests'].mean():.1f}")

print("\n" + "="*80)

