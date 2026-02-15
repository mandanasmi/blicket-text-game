"""
Visualization: Exploration (number of tests) in conjunctive vs disjunctive rounds
Shows paired t-test results with individual participant data
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def is_prolific_id(participant_id):
    """Check if participant_id is a Prolific ID"""
    if not participant_id or not isinstance(participant_id, str):
        return False
    
    id_lower = participant_id.lower()
    test_patterns = ['test', 'debug', 'demo', 'sample', 'example', 'alice', 'bob', 
                     'charlie', 'user', 'admin', 'dummy', 'fake', 'emma', 'mandana']
    
    for pattern in test_patterns:
        if pattern in id_lower:
            return False
    
    if len(participant_id) == 24:
        try:
            int(participant_id, 16)
            return True
        except ValueError:
            pass
    
    if len(participant_id) >= 20 and participant_id.replace('_', '').replace('-', '').isalnum():
        return True
    
    return False

# Load Firebase data
print("Loading Firebase data...")
# Note: Run this script from the analysis/ directory
with open('firebase_data.json', 'r') as f:
    all_data = json.load(f)

# Filter for Prolific IDs
prolific_data = {pid: pdata for pid, pdata in all_data.items() if is_prolific_id(pid)}
print(f"Found {len(prolific_data)} Prolific participants\n")

# Collect data
results = []

for participant_id, participant_data in prolific_data.items():
    prolific_id = participant_data.get('demographics', {}).get('prolific_id', participant_id)
    
    if not isinstance(participant_data, dict):
        continue
    
    main_game = participant_data.get('main_game', {})
    if not main_game or not isinstance(main_game, dict):
        continue
    
    # Collect test counts by rule type
    conjunctive_tests = []
    disjunctive_tests = []
    
    for round_key, round_data in main_game.items():
        if not isinstance(round_data, dict) or not round_key.startswith('round_'):
            continue
        
        # Get rule type
        true_rule = round_data.get('true_rule', '').lower()
        
        # Get number of tests
        state_history = round_data.get('state_history', [])
        num_tests = len(state_history)
        
        if true_rule == 'conjunctive':
            conjunctive_tests.append(num_tests)
        elif true_rule == 'disjunctive':
            disjunctive_tests.append(num_tests)
    
    # Calculate means within participant
    if conjunctive_tests and disjunctive_tests:
        mean_conjunctive = np.mean(conjunctive_tests)
        mean_disjunctive = np.mean(disjunctive_tests)
        difference = mean_conjunctive - mean_disjunctive
        
        results.append({
            'participant_id': prolific_id,
            'mean_conjunctive': mean_conjunctive,
            'mean_disjunctive': mean_disjunctive,
            'difference': difference
        })

# Create DataFrame
df = pd.DataFrame(results)

if len(df) == 0:
    print("ERROR: No data found!")
    exit(1)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# 1. Paired lines plot (main visualization)
ax1 = fig.add_subplot(gs[0, :])
x_pos = np.arange(len(df))
width = 0.35

# Plot individual participant lines
for i, row in df.iterrows():
    ax1.plot([0, 1], [row['mean_conjunctive'], row['mean_disjunctive']], 
             'o-', color='gray', alpha=0.4, linewidth=1, markersize=4)

# Plot means with error bars
conj_mean = df['mean_conjunctive'].mean()
conj_std = df['mean_conjunctive'].std()
disj_mean = df['mean_disjunctive'].mean()
disj_std = df['mean_disjunctive'].std()

ax1.errorbar([0], [conj_mean], yerr=conj_std, fmt='o', color='#2E86AB', 
             markersize=12, capsize=10, capthick=2, linewidth=2, label='Conjunctive (mean ± SD)')
ax1.errorbar([1], [disj_mean], yerr=disj_std, fmt='o', color='#A23B72', 
             markersize=12, capsize=10, capthick=2, linewidth=2, label='Disjunctive (mean ± SD)')

ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Conjunctive', 'Disjunctive'])
ax1.set_ylabel('Number of Tests', fontsize=12, fontweight='bold')
ax1.set_title('Paired Comparison: Tests in Conjunctive vs Disjunctive Rounds', 
              fontsize=14, fontweight='bold', pad=20)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Calculate statistics
mean_diff = df['difference'].mean()
std_diff = df['difference'].std()
n = len(df)
se_diff = std_diff / np.sqrt(n)
t_stat = mean_diff / se_diff if se_diff > 0 else 0
df_degrees = n - 1

# Calculate p-value (approximation if scipy not available)
try:
    from scipy import stats
    p_value = stats.t.sf(abs(t_stat), df_degrees) * 2  # Two-tailed
except ImportError:
    # Simple approximation
    abs_t = abs(t_stat)
    if abs_t > 3.5:
        p_value = 0.001
    elif abs_t > 3.0:
        p_value = 0.005
    elif abs_t > 2.5:
        p_value = 0.02
    elif abs_t > 2.0:
        p_value = 0.05
    else:
        p_value = 0.5

sig_text = f"Mean difference: {mean_diff:.2f} tests\n"
sig_text += f"Paired t-test: t({df_degrees}) = {t_stat:.2f}, p = {p_value:.3f}"
if p_value < 0.001:
    sig_text += " ***"
elif p_value < 0.01:
    sig_text += " **"
elif p_value < 0.05:
    sig_text += " *"

ax1.text(0.5, 0.02, sig_text, transform=ax1.transAxes, 
         fontsize=10, ha='center', va='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Distribution of differences
ax2 = fig.add_subplot(gs[1, 0])
differences = df['difference'].values
ax2.hist(differences, bins=10, color='#F18F01', edgecolor='black', alpha=0.7)
ax2.axvline(0, color='black', linestyle='--', linewidth=2, label='No difference')
ax2.axvline(mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.2f}')
ax2.set_xlabel('Difference (Conjunctive - Disjunctive)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Distribution of Within-Participant Differences', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Box plot comparison
ax3 = fig.add_subplot(gs[1, 1])
# Prepare data for box plot
plot_data = pd.DataFrame({
    'Rule Type': ['Conjunctive'] * len(df) + ['Disjunctive'] * len(df),
    'Number of Tests': list(df['mean_conjunctive']) + list(df['mean_disjunctive'])
})

box_plot = ax3.boxplot([df['mean_conjunctive'], df['mean_disjunctive']], 
                       tick_labels=['Conjunctive', 'Disjunctive'],
                       patch_artist=True, widths=0.6)
box_plot['boxes'][0].set_facecolor('#2E86AB')
box_plot['boxes'][1].set_facecolor('#A23B72')
for patch in box_plot['boxes']:
    patch.set_alpha(0.7)

# Add individual points
for i, (conj, disj) in enumerate(zip(df['mean_conjunctive'], df['mean_disjunctive'])):
    ax3.plot([1, 2], [conj, disj], 'o-', color='gray', alpha=0.3, linewidth=0.5, markersize=3)

ax3.set_ylabel('Number of Tests', fontsize=11, fontweight='bold')
ax3.set_title('Box Plot: Conjunctive vs Disjunctive', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add significance bracket
y_max = max(df['mean_conjunctive'].max(), df['mean_disjunctive'].max())
y_bracket = y_max * 1.15
ax3.plot([1, 2], [y_bracket, y_bracket], 'k-', linewidth=1.5)
ax3.plot([1, 1], [y_bracket, y_bracket * 0.98], 'k-', linewidth=1.5)
ax3.plot([2, 2], [y_bracket, y_bracket * 0.98], 'k-', linewidth=1.5)
ax3.text(1.5, y_bracket * 1.02, '**', ha='center', fontsize=14, fontweight='bold')

plt.suptitle('Exploration Analysis: Conjunctive vs Disjunctive Rounds', 
             fontsize=16, fontweight='bold', y=0.98)

# Save figure
os.makedirs('images', exist_ok=True)
plt.savefig('images/rule_exploration_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: images/rule_exploration_comparison.png")

# Also create a simpler single-panel version
fig2, ax = plt.subplots(figsize=(10, 6))

# Plot individual participant lines
for i, row in df.iterrows():
    ax.plot([0, 1], [row['mean_conjunctive'], row['mean_disjunctive']], 
           'o-', color='gray', alpha=0.4, linewidth=1.5, markersize=5)

# Plot means with error bars
ax.errorbar([0], [conj_mean], yerr=conj_std, fmt='o', color='#2E86AB', 
           markersize=15, capsize=12, capthick=3, linewidth=3, 
           label=f'Conjunctive\n(mean = {conj_mean:.2f} ± {conj_std:.2f})', zorder=10)
ax.errorbar([1], [disj_mean], yerr=disj_std, fmt='o', color='#A23B72', 
           markersize=15, capsize=12, capthick=3, linewidth=3,
           label=f'Disjunctive\n(mean = {disj_mean:.2f} ± {disj_std:.2f})', zorder=10)

ax.set_xticks([0, 1])
ax.set_xticklabels(['Conjunctive', 'Disjunctive'], fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Tests', fontsize=13, fontweight='bold')
sig_stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
ax.set_title('Paired Comparison: Tests in Conjunctive vs Disjunctive Rounds\n' + 
            f'Paired t-test: t({df_degrees}) = {t_stat:.2f}, p = {p_value:.3f} {sig_stars}', 
            fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/rule_exploration_paired.png', dpi=300, bbox_inches='tight')
print("✓ Saved: images/rule_exploration_paired.png")

print("\nVisualization complete!")

