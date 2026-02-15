"""
Visualize comprehension phase data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# Load data
df = pd.read_csv('results/1_comprehension_phase.csv')

print(f"Loaded {len(df)} participants")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Number of tests per participant
ax1 = axes[0, 0]
participant_labels = [pid[:8] + '...' for pid in df['participant_id']]
colors = ['#4CAF50' if match else '#F44336' for match in df['objects_match']]
bars = ax1.bar(range(len(df)), df['num_tests'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_title('Number of Tests Per Participant (Comprehension Phase)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Participant', fontsize=12)
ax1.set_ylabel('Number of Tests', fontsize=12)
ax1.set_xticks(range(len(df)))
ax1.set_xticklabels(participant_labels, rotation=45, ha='right', fontsize=9)
ax1.axhline(df['num_tests'].mean(), color='blue', linestyle='--', linewidth=2, 
            label=f'Mean: {df["num_tests"].mean():.1f}', alpha=0.7)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, df['num_tests'])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{int(val)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2. Match distribution
ax2 = axes[0, 1]
match_counts = df['objects_match'].value_counts()
colors_pie = ['#4CAF50', '#F44336']  # Green for True, Red for False
labels = ['Correct Match', 'Incorrect Match']
actual_labels = []
actual_colors = []
values = []
for val in [True, False]:
    if val in match_counts.index:
        values.append(match_counts[val])
        actual_labels.append(labels[0] if val else labels[1])
        actual_colors.append(colors_pie[0] if val else colors_pie[1])

wedges, texts, autotexts = ax2.pie(values, labels=actual_labels, autopct='%1.1f%%',
                                     colors=actual_colors, startangle=90,
                                     textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('Object Selection Accuracy', fontsize=14, fontweight='bold')

# Add count in center
total = len(df)
correct = match_counts.get(True, 0)
ax2.text(0, 0, f'{correct}/{total}\nCorrect', ha='center', va='center', 
         fontsize=16, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 3. Test distribution histogram
ax3 = axes[1, 0]
test_counts = df['num_tests'].value_counts().sort_index()
ax3.bar(test_counts.index, test_counts.values, color='#2196F3', alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_title('Distribution of Number of Tests', fontsize=14, fontweight='bold')
ax3.set_xlabel('Number of Tests', fontsize=12)
ax3.set_ylabel('Number of Participants', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for x, y in zip(test_counts.index, test_counts.values):
    ax3.text(x, y + 0.1, str(y), ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add statistics text
stats_text = f"Mean: {df['num_tests'].mean():.1f}\nMedian: {df['num_tests'].median():.0f}\nRange: {df['num_tests'].min():.0f}-{df['num_tests'].max():.0f}"
ax3.text(0.98, 0.97, stats_text, transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Ground truth vs User choice comparison
ax4 = axes[1, 1]
# Parse the object lists
ground_truth_objects = df['ground_truth_objects'].apply(eval)
user_chosen_objects = df['user_chosen_objects'].apply(eval)

# Count object frequencies
object_counts = {'Ground Truth': {}, 'User Choice': {}}
for objects in ground_truth_objects:
    for obj in objects:
        object_counts['Ground Truth'][obj] = object_counts['Ground Truth'].get(obj, 0) + 1

for objects in user_chosen_objects:
    for obj in objects:
        object_counts['User Choice'][obj] = object_counts['User Choice'].get(obj, 0) + 1

# Create grouped bar chart
objects = sorted(set(list(object_counts['Ground Truth'].keys()) + list(object_counts['User Choice'].keys())))
x = np.arange(len(objects))
width = 0.35

truth_values = [object_counts['Ground Truth'].get(obj, 0) for obj in objects]
user_values = [object_counts['User Choice'].get(obj, 0) for obj in objects]

bars1 = ax4.bar(x - width/2, truth_values, width, label='Ground Truth', color='#FF9800', alpha=0.7, edgecolor='black')
bars2 = ax4.bar(x + width/2, user_values, width, label='User Selected', color='#4CAF50', alpha=0.7, edgecolor='black')

ax4.set_title('Object Selection Frequency (Object A=0, B=1, C=2)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Object ID', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels([f'Object {obj}' for obj in objects])
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('comprehension_phase_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: comprehension_phase_visualization.png")
plt.close()

# Print summary statistics
print("\n" + "="*60)
print("COMPREHENSION PHASE SUMMARY")
print("="*60)
print(f"Total Participants: {len(df)}")
print(f"Average Tests: {df['num_tests'].mean():.1f}")
print(f"Test Range: {df['num_tests'].min():.0f} - {df['num_tests'].max():.0f}")
print(f"Correct Matches: {match_counts.get(True, 0)}/{len(df)} ({match_counts.get(True, 0)/len(df)*100:.1f}%)")
print(f"Incorrect Matches: {match_counts.get(False, 0)}/{len(df)} ({match_counts.get(False, 0)/len(df)*100:.1f}%)")
print("="*60)

