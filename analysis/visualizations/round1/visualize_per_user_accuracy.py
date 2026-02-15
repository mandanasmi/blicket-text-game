"""
Visualize per-user rule inference accuracy
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")

# Load data
df = pd.read_csv('results/5_per_user_rule_accuracy.csv')

print(f"Loaded {len(df)} participants\n")

# ============================================================================
# Create visualization
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Create bars
participant_labels = [pid[:8] + '...' for pid in df['participant_id']]
# Convert accuracy percentage string to determine if 100%
colors = ['#4CAF50' if (pd.notna(acc) and acc == '100.0%') else '#FFC107' for acc in df['accuracy']]

bars = ax.bar(range(len(df)), df['checkbox_accuracy_percent'], color=colors, 
              alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_title('Rule Inference Accuracy Per Participant (All 3 Rounds)', fontsize=16, fontweight='bold')
ax.set_xlabel('Participant', fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(participant_labels, rotation=45, ha='right', fontsize=10)
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val, correct, total) in enumerate(zip(bars, df['checkbox_accuracy_percent'], 
                                                     df['checkbox_correct'], df['checkbox_total'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val:.0f}%\n({int(correct)}/{int(total)})', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('plot7_per_user_rule_accuracy.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plot7_per_user_rule_accuracy.png")
plt.close()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total Participants: {len(df)}")
print(f"Average Accuracy: {df['checkbox_accuracy_percent'].mean():.1f}%")
print(f"Median Accuracy: {df['checkbox_accuracy_percent'].median():.1f}%")
# Count participants with 100% accuracy
perfect_count = (df['checkbox_accuracy_percent'] == 100.0).sum()
print(f"Perfect Score (100%): {perfect_count} participants")
print(f"High Performance (>90%): {(df['checkbox_accuracy_percent'] > 90).sum()} participants")
print(f"Above Chance (>50%): {(df['checkbox_accuracy_percent'] > 50).sum()} participants")
print("="*80)

