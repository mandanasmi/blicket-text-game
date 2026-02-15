"""
Visualize improvement from Round 1 to Round 3 in object classification and rule inference
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']

# Load data
main_df = pd.read_csv('results/3_main_game_rounds.csv')
rule_df = pd.read_csv('results/4_rule_inference.csv')

print("="*80)
print("VISUALIZING ROUND 1 TO ROUND 3 IMPROVEMENT")
print("="*80)

# Calculate accuracy by round for object classification
obj_accuracy_by_round = []
for round_num in [1, 2, 3]:
    round_data = main_df[main_df['round'] == round_num]
    correct = round_data['objects_match'].sum()
    total = len(round_data)
    accuracy = (correct / total * 100) if total > 0 else 0
    obj_accuracy_by_round.append({
        'round': round_num,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    })

obj_round_df = pd.DataFrame(obj_accuracy_by_round)

# Calculate accuracy by round for rule inference
rule_accuracy_by_round = []
for round_num in [1, 2, 3]:
    round_data = rule_df[rule_df['round'] == round_num]
    correct = round_data['checkbox_matches_truth'].sum()
    total = len(round_data)
    accuracy = (correct / total * 100) if total > 0 else 0
    rule_accuracy_by_round.append({
        'round': round_num,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    })

rule_round_df = pd.DataFrame(rule_accuracy_by_round)

# Prepare per-participant comparison data
participant_obj_comparisons = []
for participant_id in main_df['participant_id'].unique():
    p_round1 = main_df[(main_df['participant_id'] == participant_id) & (main_df['round'] == 1)]
    p_round3 = main_df[(main_df['participant_id'] == participant_id) & (main_df['round'] == 3)]
    
    if len(p_round1) > 0 and len(p_round3) > 0:
        r1_correct = bool(p_round1['objects_match'].iloc[0])
        r3_correct = bool(p_round3['objects_match'].iloc[0])
        participant_obj_comparisons.append({
            'participant_id': participant_id,
            'round1': 100 if r1_correct else 0,
            'round3': 100 if r3_correct else 0,
            'improvement': int(r3_correct) - int(r1_correct)
        })

participant_rule_comparisons = []
for participant_id in rule_df['participant_id'].unique():
    p_round1 = rule_df[(rule_df['participant_id'] == participant_id) & (rule_df['round'] == 1)]
    p_round3 = rule_df[(rule_df['participant_id'] == participant_id) & (rule_df['round'] == 3)]
    
    if len(p_round1) > 0 and len(p_round3) > 0:
        r1_correct = bool(p_round1['checkbox_matches_truth'].iloc[0])
        r3_correct = bool(p_round3['checkbox_matches_truth'].iloc[0])
        participant_rule_comparisons.append({
            'participant_id': participant_id,
            'round1': 100 if r1_correct else 0,
            'round3': 100 if r3_correct else 0,
            'improvement': int(r3_correct) - int(r1_correct)
        })

obj_comp_df = pd.DataFrame(participant_obj_comparisons)
rule_comp_df = pd.DataFrame(participant_rule_comparisons)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2)

# ============================================================================
# Plot 1: Accuracy Trends Across All Rounds (Line Plot)
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])

# Plot both metrics
ax1.plot(obj_round_df['round'], obj_round_df['accuracy'], 
         'o-', linewidth=2.5, markersize=10, color='#2E86AB', label='Object Classification', zorder=3)
ax1.plot(rule_round_df['round'], rule_round_df['accuracy'], 
         's-', linewidth=2.5, markersize=10, color='#A23B72', label='Rule Inference', zorder=3)

# Add value labels on points
for _, row in obj_round_df.iterrows():
    ax1.text(row['round'], row['accuracy'] + 2, f"{row['accuracy']:.1f}%", 
             ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2E86AB')
for _, row in rule_round_df.iterrows():
    ax1.text(row['round'], row['accuracy'] + 2, f"{row['accuracy']:.1f}%", 
             ha='center', va='bottom', fontsize=11, fontweight='bold', color='#A23B72')

ax1.set_xlabel('Round', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Accuracy Trends Across Rounds', fontsize=15, fontweight='bold', pad=5)
ax1.set_xticks([1, 2, 3])
ax1.tick_params(axis='both', which='major', labelsize=11)
ax1.set_ylim(0, 110)
ax1.legend(loc='upper right', fontsize=12, framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# ============================================================================
# Plot 2: Round 1 vs Round 3 Comparison - Object Classification
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])

r1_obj_acc = obj_round_df[obj_round_df['round'] == 1]['accuracy'].iloc[0]
r3_obj_acc = obj_round_df[obj_round_df['round'] == 3]['accuracy'].iloc[0]
r1_obj_correct = obj_round_df[obj_round_df['round'] == 1]['correct'].iloc[0]
r1_obj_total = obj_round_df[obj_round_df['round'] == 1]['total'].iloc[0]
r3_obj_correct = obj_round_df[obj_round_df['round'] == 3]['correct'].iloc[0]
r3_obj_total = obj_round_df[obj_round_df['round'] == 3]['total'].iloc[0]

bars2 = ax2.bar(['Round 1', 'Round 3'], [r1_obj_acc, r3_obj_acc], 
                color=['#4A90A4', '#A23B72'], alpha=0.8, edgecolor='#333333', linewidth=1.5)

# Add value labels
for i, (bar, acc, correct, total) in enumerate(zip(bars2, [r1_obj_acc, r3_obj_acc], 
                                                    [r1_obj_correct, r3_obj_correct],
                                                    [r1_obj_total, r3_obj_total])):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.1f}%\n({int(correct)}/{int(total)})',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add change indicator
change = r3_obj_acc - r1_obj_acc
change_color = '#d95f02' if change < 0 else '#1b9e77'
change_symbol = '↓' if change < 0 else '↑'
ax2.text(0.5, max(r1_obj_acc, r3_obj_acc) + 8,
         f'{change_symbol} {abs(change):.1f} pp',
         ha='center', fontsize=12, fontweight='bold', color=change_color,
         transform=ax2.get_xaxis_transform())

ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Object Classification: Round 1 vs Round 3', fontsize=13, fontweight='bold', pad=10)
ax2.tick_params(axis='both', which='major', labelsize=11)
ax2.set_ylim(0, 110)
ax2.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# ============================================================================
# Plot 3: Round 1 vs Round 3 Comparison - Rule Inference
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])

r1_rule_acc = rule_round_df[rule_round_df['round'] == 1]['accuracy'].iloc[0]
r3_rule_acc = rule_round_df[rule_round_df['round'] == 3]['accuracy'].iloc[0]
r1_rule_correct = rule_round_df[rule_round_df['round'] == 1]['correct'].iloc[0]
r1_rule_total = rule_round_df[rule_round_df['round'] == 1]['total'].iloc[0]
r3_rule_correct = rule_round_df[rule_round_df['round'] == 3]['correct'].iloc[0]
r3_rule_total = rule_round_df[rule_round_df['round'] == 3]['total'].iloc[0]

bars3 = ax3.bar(['Round 1', 'Round 3'], [r1_rule_acc, r3_rule_acc], 
                color=['#4A90A4', '#A23B72'], alpha=0.8, edgecolor='#333333', linewidth=1.5)

# Add value labels
for i, (bar, acc, correct, total) in enumerate(zip(bars3, [r1_rule_acc, r3_rule_acc], 
                                                    [r1_rule_correct, r3_rule_correct],
                                                    [r1_rule_total, r3_rule_total])):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.1f}%\n({int(correct)}/{int(total)})',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add change indicator
change_rule = r3_rule_acc - r1_rule_acc
change_color_rule = '#d95f02' if change_rule < 0 else '#1b9e77'
change_symbol_rule = '↓' if change_rule < 0 else '↑'
ax3.text(0.5, max(r1_rule_acc, r3_rule_acc) + 8,
         f'{change_symbol_rule} {abs(change_rule):.1f} pp',
         ha='center', fontsize=12, fontweight='bold', color=change_color_rule,
         transform=ax3.get_xaxis_transform())

ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax3.set_title('Rule Inference: Round 1 vs Round 3', fontsize=13, fontweight='bold', pad=10)
ax3.tick_params(axis='both', which='major', labelsize=11)
ax3.set_ylim(0, 110)
ax3.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# ============================================================================
# Plot 4: Per-Participant Changes - Object Classification
# ============================================================================
ax4 = fig.add_subplot(gs[2, 0])

# Sort by improvement for better visualization
obj_comp_df_sorted = obj_comp_df.sort_values('improvement')
x_pos = np.arange(len(obj_comp_df_sorted))

# Plot individual participant lines
for i, row in obj_comp_df_sorted.iterrows():
    color = '#1b9e77' if row['improvement'] > 0 else '#d95f02' if row['improvement'] < 0 else '#7570b3'
    alpha = 0.6 if row['improvement'] == 0 else 0.8
    ax4.plot([0, 1], [row['round1'], row['round3']], 
             'o-', color=color, alpha=alpha, linewidth=1.5, markersize=4, zorder=2)

# Plot means with error bars
r1_mean = obj_comp_df_sorted['round1'].mean()
r1_std = obj_comp_df_sorted['round1'].std()
r3_mean = obj_comp_df_sorted['round3'].mean()
r3_std = obj_comp_df_sorted['round3'].std()

ax4.errorbar([0], [r1_mean], yerr=r1_std, fmt='o', color='#2E86AB', 
             markersize=12, capsize=8, capthick=2, linewidth=2.5, 
             label=f'Round 1 (mean)', zorder=3)
ax4.errorbar([1], [r3_mean], yerr=r3_std, fmt='o', color='#A23B72', 
             markersize=12, capsize=8, capthick=2, linewidth=2.5, 
             label=f'Round 3 (mean)', zorder=3)

ax4.set_xticks([0, 1])
ax4.set_xticklabels(['Round 1', 'Round 3'])
ax4.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax4.set_title('Per-Participant Changes: Object Classification', fontsize=13, fontweight='bold', pad=10)
ax4.tick_params(axis='both', which='major', labelsize=11)
ax4.set_ylim(-5, 105)
ax4.legend(loc='best', fontsize=10, framealpha=0.95)
ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add summary text
improved_obj = (obj_comp_df_sorted['improvement'] > 0).sum()
declined_obj = (obj_comp_df_sorted['improvement'] < 0).sum()
same_obj = (obj_comp_df_sorted['improvement'] == 0).sum()
summary_text = f'Improved: {improved_obj} | Declined: {declined_obj} | Same: {same_obj}'
ax4.text(0.5, -10, summary_text, ha='center', fontsize=10, 
         transform=ax4.get_xaxis_transform(), style='italic')

# ============================================================================
# Plot 5: Per-Participant Changes - Rule Inference
# ============================================================================
ax5 = fig.add_subplot(gs[2, 1])

# Sort by improvement
rule_comp_df_sorted = rule_comp_df.sort_values('improvement')
x_pos_rule = np.arange(len(rule_comp_df_sorted))

# Plot individual participant lines
for i, row in rule_comp_df_sorted.iterrows():
    color = '#1b9e77' if row['improvement'] > 0 else '#d95f02' if row['improvement'] < 0 else '#7570b3'
    alpha = 0.6 if row['improvement'] == 0 else 0.8
    ax5.plot([0, 1], [row['round1'], row['round3']], 
             'o-', color=color, alpha=alpha, linewidth=1.5, markersize=4, zorder=2)

# Plot means with error bars
r1_rule_mean = rule_comp_df_sorted['round1'].mean()
r1_rule_std = rule_comp_df_sorted['round1'].std()
r3_rule_mean = rule_comp_df_sorted['round3'].mean()
r3_rule_std = rule_comp_df_sorted['round3'].std()

ax5.errorbar([0], [r1_rule_mean], yerr=r1_rule_std, fmt='o', color='#2E86AB', 
             markersize=12, capsize=8, capthick=2, linewidth=2.5, 
             label=f'Round 1 (mean)', zorder=3)
ax5.errorbar([1], [r3_rule_mean], yerr=r3_rule_std, fmt='o', color='#A23B72', 
             markersize=12, capsize=8, capthick=2, linewidth=2.5, 
             label=f'Round 3 (mean)', zorder=3)

ax5.set_xticks([0, 1])
ax5.set_xticklabels(['Round 1', 'Round 3'])
ax5.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax5.set_title('Per-Participant Changes: Rule Inference', fontsize=13, fontweight='bold', pad=10)
ax5.tick_params(axis='both', which='major', labelsize=11)
ax5.set_ylim(-5, 105)
ax5.legend(loc='best', fontsize=10, framealpha=0.95)
ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add summary text
improved_rule_viz = (rule_comp_df_sorted['improvement'] > 0).sum()
declined_rule_viz = (rule_comp_df_sorted['improvement'] < 0).sum()
same_rule_viz = (rule_comp_df_sorted['improvement'] == 0).sum()
summary_text_rule = f'Improved: {improved_rule_viz} | Declined: {declined_rule_viz} | Same: {same_rule_viz}'
ax5.text(0.5, -10, summary_text_rule, ha='center', fontsize=10, 
         transform=ax5.get_xaxis_transform(), style='italic')

# ============================================================================
# Save figure
# ============================================================================
# Adjust spacing first
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.30, wspace=0.3)

# Add suptitle after adjusting
plt.suptitle('Round 1 to Round 3 Improvement Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

os.makedirs('images', exist_ok=True)
output_path = 'images/round_improvement_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.02)
print(f"\n✓ Saved visualization to: {output_path}")

plt.close()

print("\n" + "="*80)
print("Visualization complete!")
print("="*80)

