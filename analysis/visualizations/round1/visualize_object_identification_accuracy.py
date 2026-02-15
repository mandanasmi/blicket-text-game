"""
Visualize object identification accuracy during main game from CSV 3
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from datetime import datetime

# Set style
sns.set_style("whitegrid")
# Use a distinctive serif font for the entire figure
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']

# Load data
main_df = pd.read_csv('results/3_main_game_rounds.csv')

print("="*80)
print("OBJECT IDENTIFICATION ACCURACY - MAIN GAME")
print("="*80)

# Convert True/False to numeric for calculations
main_df['objects_match_numeric'] = main_df['objects_match'].astype(int) * 100

print(f"\nTotal rounds analyzed: {len(main_df)}")
print(f"Participants: {main_df['participant_id'].nunique()}")

# Calculate overall accuracy
total_correct = main_df['objects_match'].sum()
total_rounds = len(main_df)
overall_accuracy = (total_correct / total_rounds) * 100

print(f"\nOverall Object Identification Accuracy: {total_correct}/{total_rounds} ({overall_accuracy:.1f}%)")

# Calculate accuracy by round
accuracy_by_round = main_df.groupby('round').agg({
    'objects_match': ['sum', 'count']
}).reset_index()
accuracy_by_round.columns = ['round', 'correct', 'total']
accuracy_by_round['accuracy_percent'] = (accuracy_by_round['correct'] / accuracy_by_round['total']) * 100

print("\nAccuracy by Round:")
for _, row in accuracy_by_round.iterrows():
    print(f"  Round {int(row['round'])}: {int(row['correct'])}/{int(row['total'])} ({row['accuracy_percent']:.1f}%)")

# Create figure with subplots (3 plots in a row - horizontal)
fig, axes = plt.subplots(1, 3, figsize=(22, 8))

# Plot 1: Overall Accuracy (Bar Chart)
ax1 = axes[0]
match_counts = main_df['objects_match'].value_counts()
labels = ['Correct', 'Incorrect']
colors_pie = ['#1b9e77', '#d95f02']  # custom, distinctive palette
sizes = [match_counts.get(True, 0), match_counts.get(False, 0)]

# Draw bar chart
x_pos = np.arange(len(labels))
bars1 = ax1.bar(
    x_pos,
    sizes,
    color=colors_pie,
    alpha=0.9,
    edgecolor='#333333',
    linewidth=1.2
)
ax1.set_title('Overall Object Identification Accuracy', fontsize=14)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_ylim(0, max(sizes) + max(2, int(0.1 * max(sizes))))
ax1.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width()/2.,
        height + 0.5,
        f'{int(height)}',
        ha='center',
        va='bottom',
        fontsize=11,
        # no bold
    )

# Add compact summary box inside plot
summary_text = f'{overall_accuracy:.1f}% overall\n({total_correct}/{total_rounds})'
ax1.text(
    0.98, 0.95, summary_text,
    transform=ax1.transAxes,
    ha='right',
    va='top',
    fontsize=11,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#cccccc')
)

# Plot 2: Accuracy by Round (Bar Chart)
ax2 = axes[1]
rounds = sorted(accuracy_by_round['round'].astype(int))
accuracies = [accuracy_by_round[accuracy_by_round['round'] == r]['accuracy_percent'].values[0] for r in rounds]
correct_counts = [int(accuracy_by_round[accuracy_by_round['round'] == r]['correct'].values[0]) for r in rounds]
total_counts = [int(accuracy_by_round[accuracy_by_round['round'] == r]['total'].values[0]) for r in rounds]

bars = ax2.bar(
    range(len(rounds)),
    accuracies,
    color=['#1b9e77', '#d95f02', '#7570b3'],  # custom trio
    alpha=0.8,
    edgecolor='#333333',
    linewidth=1.5
)

ax2.set_title('Object Identification Accuracy by Round', fontsize=14)
# Removed x-axis label per request
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_xticks(range(len(rounds)))
ax2.set_xticklabels([f'Round {r}' for r in rounds], fontsize=12)
ax2.set_ylim(0, 110)
ax2.axhline(100, color='#6a3d9a', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect (100%)')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(fontsize=10)

# Add value labels on bars
for i, (bar, acc, correct, total) in enumerate(zip(bars, accuracies, correct_counts, total_counts)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{acc:.1f}%\n({correct}/{total})', 
            ha='center', va='bottom', fontsize=11)

# Plot 3: Per-Participant Accuracy
ax3 = axes[2]
participant_accuracy = main_df.groupby('participant_id').agg({
    'objects_match': ['sum', 'count']
}).reset_index()
participant_accuracy.columns = ['participant_id', 'correct', 'total']
participant_accuracy['accuracy_percent'] = (participant_accuracy['correct'] / participant_accuracy['total']) * 100

# Optional: load ages from firebase_data.json to sort participants (younger -> older)
ages_map = {}
try:
    with open('firebase_data.json', 'r') as f:
        all_data = json.load(f)
    current_year = datetime.now().year
    for pid, pdata in all_data.items():
        if not isinstance(pdata, dict):
            continue
        age_val = None
        # Try several locations/keys
        candidates = [
            pdata.get('age'),
            pdata.get('demographics', {}).get('age') if isinstance(pdata.get('demographics'), dict) else None,
            pdata.get('profile', {}).get('age') if isinstance(pdata.get('profile'), dict) else None,
            pdata.get('prolific_info', {}).get('age') if isinstance(pdata.get('prolific_info'), dict) else None,
            pdata.get('prolific', {}).get('age') if isinstance(pdata.get('prolific'), dict) else None,
            pdata.get('age_years'),
        ]
        # Birth year options
        birth_year_candidates = [
            pdata.get('birth_year'),
            pdata.get('demographics', {}).get('birth_year') if isinstance(pdata.get('demographics'), dict) else None,
            pdata.get('profile', {}).get('birth_year') if isinstance(pdata.get('profile'), dict) else None,
        ]
        # Parse direct age
        for c in candidates:
            if c is None:
                continue
            try:
                if isinstance(c, (int, float)):
                    age_val = float(c)
                    break
                if isinstance(c, str):
                    s = c.strip()
                    if '-' in s:  # e.g., "18-24"
                        low = s.split('-', 1)[0].strip()
                        age_val = float(low)
                        break
                    age_val = float(s)
                    break
            except Exception:
                pass
        # Parse birth year if age not found
        if age_val is None:
            for by in birth_year_candidates:
                try:
                    if by is None:
                        continue
                    by_int = int(str(by).strip())
                    if 1900 < by_int <= current_year:
                        age_val = float(current_year - by_int)
                        break
                except Exception:
                    pass
        if age_val is not None:
            ages_map[pid] = age_val
except Exception:
    # If file missing or invalid, leave ages_map empty
    ages_map = {}

# Merge ages into participant_accuracy if available
if ages_map:
    participant_accuracy['age'] = participant_accuracy['participant_id'].map(ages_map)
    # Sort by age (ascending). If age missing, place at end by using large fill value
    participant_accuracy['age_sort_key'] = participant_accuracy['age'].fillna(10_000)
    participant_accuracy = participant_accuracy.sort_values(['age_sort_key', 'accuracy_percent'], ascending=[True, False])
    participant_accuracy = participant_accuracy.drop(columns=['age_sort_key'])
else:
    # Default order by accuracy if ages not available
    participant_accuracy = participant_accuracy.sort_values('accuracy_percent', ascending=False)

participant_labels = [pid[:8] + '...' for pid in participant_accuracy['participant_id']]
# Distinctive colors: teal for perfect, amber for others
colors = ['#2a9d8f' if acc == 100 else '#e9c46a' for acc in participant_accuracy['accuracy_percent']]

bars3 = ax3.bar(
    range(len(participant_accuracy)),
    participant_accuracy['accuracy_percent'],
    color=colors,
    alpha=0.9,
    edgecolor='#333333',
    linewidth=1.2
)

ax3.set_title('Object Identification Accuracy Per Participant', fontsize=14)
ax3.set_xlabel('Participant', fontsize=12)
ax3.set_ylabel('Accuracy (%)', fontsize=12)
ax3.set_xticks(range(len(participant_accuracy)))
ax3.set_xticklabels(participant_labels, rotation=45, ha='right', fontsize=10)
ax3.set_ylim(0, 110)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, acc, correct, total) in enumerate(zip(bars3, participant_accuracy['accuracy_percent'], 
                                                     participant_accuracy['correct'], 
                                                     participant_accuracy['total'])):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{acc:.0f}%\n({int(correct)}/{int(total)})', 
            ha='center', va='bottom', fontsize=9)

plt.suptitle('Object Identification Accuracy Analysis - Main Game', 
             fontsize=16, y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('plot10_object_identification_accuracy.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plot10_object_identification_accuracy.png")
plt.close()

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nOverall Accuracy: {overall_accuracy:.1f}%")
print(f"Correct: {total_correct}/{total_rounds}")
print(f"\nBy Round:")
for _, row in accuracy_by_round.iterrows():
    print(f"  Round {int(row['round'])}: {row['accuracy_percent']:.1f}%")
print(f"\nParticipants with 100% accuracy: {(participant_accuracy['accuracy_percent'] == 100).sum()}/{len(participant_accuracy)}")
print(f"Average accuracy per participant: {participant_accuracy['accuracy_percent'].mean():.1f}%")
print("="*80)

