"""
Visualize object identification accuracy and rule inference accuracy by rule type
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']

def is_prolific_id(participant_id):
    """Check if participant_id is a valid Prolific ID"""
    if not participant_id or not isinstance(participant_id, str):
        return False
    
    id_lower = participant_id.lower()
    test_patterns = ['test', 'debug', 'demo', 'sample', 'example', 'alice', 'bob', 
                     'charlie', 'user', 'admin', 'dummy', 'fake', 'emma', 'mandana']
    
    for pattern in test_patterns:
        if pattern in id_lower:
            return False
    
    if len(participant_id) <= 2 and participant_id.isdigit():
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

def get_prolific_id(entry):
    """Extract prolific_id from an entry"""
    if 'demographics' in entry and isinstance(entry['demographics'], dict):
        if 'prolific_id' in entry['demographics']:
            return entry['demographics']['prolific_id']
    
    if 'config' in entry and isinstance(entry['config'], dict):
        if 'demographics' in entry['config'] and isinstance(entry['config']['demographics'], dict):
            if 'prolific_id' in entry['config']['demographics']:
                return entry['config']['demographics']['prolific_id']
    
    return None

def parse_rule_type(rule_type_str):
    """Parse rule type from string to 'conjunctive' or 'disjunctive'"""
    if not rule_type_str or not isinstance(rule_type_str, str):
        return None
    
    rule_lower = rule_type_str.lower()
    if 'conjunctive' in rule_lower:
        return 'conjunctive'
    elif 'disjunctive' in rule_lower:
        return 'disjunctive'
    return None

def get_user_chosen_blickets(round_data):
    """Extract user's chosen blickets from round data"""
    # Try user_chosen_blickets first
    if 'user_chosen_blickets' in round_data:
        chosen = round_data['user_chosen_blickets']
        if isinstance(chosen, list):
            return sorted(chosen)
    
    # Try blicket_classifications
    if 'blicket_classifications' in round_data:
        classifications = round_data['blicket_classifications']
        if isinstance(classifications, dict):
            chosen = []
            for key, value in classifications.items():
                if value == "Yes":
                    # Extract index from key like "object_0" or "object_1"
                    try:
                        idx = int(key.split('_')[1])
                        chosen.append(idx)
                    except (ValueError, IndexError):
                        pass
            return sorted(chosen)
    
    return None

# Load data
print("="*80)
print("ACCURACY VISUALIZATION BY RULE TYPE")
print("="*80)
print("\nLoading firebase_data2_with_prior_experience.json...")

try:
    with open('firebase_data2_with_prior_experience.json', 'r') as f:
        all_data = json.load(f)
except FileNotFoundError:
    print("ERROR: firebase_data2_with_prior_experience.json not found!")
    exit(1)
except json.JSONDecodeError as e:
    print(f"ERROR: Invalid JSON - {e}")
    exit(1)

print(f"Total entries loaded: {len(all_data)}")

# Filter for valid Prolific IDs
prolific_data = {}
for entry_id, entry_data in all_data.items():
    prolific_id = get_prolific_id(entry_data)
    if prolific_id and is_prolific_id(prolific_id):
        prolific_data[entry_id] = entry_data

print(f"Valid Prolific participants: {len(prolific_data)}")

# Collect data
results = []

for participant_id, participant_data in prolific_data.items():
    if not isinstance(participant_data, dict):
        continue
    
    prolific_id = get_prolific_id(participant_data)
    if not prolific_id:
        prolific_id = participant_id
    
    main_game = participant_data.get('main_game', {})
    if not main_game or not isinstance(main_game, dict):
        continue
    
    for round_key, round_data in main_game.items():
        if not isinstance(round_data, dict) or not round_key.startswith('round_'):
            continue
        
        round_number = round_data.get('round_number', 0)
        if round_number == 0:
            continue
        
        # Get ground truth
        true_blickets = round_data.get('true_blicket_indices', [])
        true_rule = round_data.get('true_rule', '')
        
        # Get user's choices
        user_chosen_blickets = get_user_chosen_blickets(round_data)
        rule_type_str = round_data.get('rule_type', '')
        user_rule = parse_rule_type(rule_type_str)
        
        # Calculate object identification accuracy
        object_accuracy = None
        if true_blickets is not None and user_chosen_blickets is not None:
            true_set = set(true_blickets) if isinstance(true_blickets, list) else set()
            user_set = set(user_chosen_blickets) if isinstance(user_chosen_blickets, list) else set()
            object_accuracy = (true_set == user_set)
        
        # Calculate rule inference accuracy
        rule_accuracy = None
        if true_rule and user_rule:
            rule_accuracy = (true_rule.lower() == user_rule.lower())
        
        results.append({
            'participant_id': prolific_id,
            'round_number': round_number,
            'true_rule': true_rule,
            'user_rule': user_rule,
            'object_accuracy': object_accuracy,
            'rule_accuracy': rule_accuracy
        })

# Create DataFrame
df = pd.DataFrame(results)

if len(df) == 0:
    print("\nERROR: No valid round data found!")
    exit(1)

print(f"\nTotal rounds analyzed: {len(df)}")
print(f"Unique participants: {df['participant_id'].nunique()}")

# Filter out rows with missing rule type
df_with_rules = df[df['true_rule'].notna() & (df['true_rule'] != '')].copy()

print(f"\nRounds with rule type information: {len(df_with_rules)}")

# ============================================================================
# OBJECT IDENTIFICATION ACCURACY
# ============================================================================
print("\n" + "="*80)
print("OBJECT IDENTIFICATION ACCURACY")
print("="*80)

# Overall accuracy
object_acc_valid = df_with_rules['object_accuracy'].notna()
object_correct = df_with_rules['object_accuracy'].sum()
object_total = object_acc_valid.sum()
overall_object_acc = (object_correct / object_total * 100) if object_total > 0 else 0

print(f"\nOverall Object Identification Accuracy: {object_correct}/{object_total} ({overall_object_acc:.1f}%)")

# Accuracy by rule type
print(f"\n--- Accuracy by Rule Type ---")
object_acc_by_rule = df_with_rules.groupby('true_rule').agg({
    'object_accuracy': ['sum', 'count']
}).reset_index()
object_acc_by_rule.columns = ['rule_type', 'correct', 'total']
object_acc_by_rule['accuracy_percent'] = (object_acc_by_rule['correct'] / object_acc_by_rule['total'] * 100).round(1)

for _, row in object_acc_by_rule.iterrows():
    print(f"{row['rule_type'].capitalize()}: {int(row['correct'])}/{int(row['total'])} ({row['accuracy_percent']:.1f}%)")

# ============================================================================
# RULE INFERENCE ACCURACY
# ============================================================================
print("\n" + "="*80)
print("RULE INFERENCE ACCURACY")
print("="*80)

# Overall accuracy
rule_acc_valid = df_with_rules['rule_accuracy'].notna()
rule_correct = df_with_rules['rule_accuracy'].sum()
rule_total = rule_acc_valid.sum()
overall_rule_acc = (rule_correct / rule_total * 100) if rule_total > 0 else 0

print(f"\nOverall Rule Inference Accuracy: {rule_correct}/{rule_total} ({overall_rule_acc:.1f}%)")

# Accuracy by rule type
print(f"\n--- Accuracy by Rule Type ---")
rule_acc_by_rule = df_with_rules.groupby('true_rule').agg({
    'rule_accuracy': ['sum', 'count']
}).reset_index()
rule_acc_by_rule.columns = ['rule_type', 'correct', 'total']
rule_acc_by_rule['accuracy_percent'] = (rule_acc_by_rule['correct'] / rule_acc_by_rule['total'] * 100).round(1)

for _, row in rule_acc_by_rule.iterrows():
    print(f"{row['rule_type'].capitalize()}: {int(row['correct'])}/{int(row['total'])} ({row['accuracy_percent']:.1f}%)")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# ============================================================================
# Plot 1: Object Identification Accuracy by Rule Type
# ============================================================================
ax1 = axes[0, 0]

rule_types = object_acc_by_rule['rule_type'].tolist()
accuracies = object_acc_by_rule['accuracy_percent'].tolist()
correct_counts = object_acc_by_rule['correct'].astype(int).tolist()
total_counts = object_acc_by_rule['total'].astype(int).tolist()

# Capitalize rule type labels for display
rule_labels = [r.capitalize() for r in rule_types]

# Use distinct colors for each rule type
colors = ['#1b9e77', '#d95f02']  # teal and orange
if len(rule_types) > 2:
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e'][:len(rule_types)]

bars1 = ax1.bar(
    range(len(rule_types)),
    accuracies,
    color=colors,
    alpha=0.8,
    edgecolor='#333333',
    linewidth=1.5
)

ax1.set_title('Object Identification Accuracy by Rule Type', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_xticks(range(len(rule_types)))
ax1.set_xticklabels(rule_labels, fontsize=12)
ax1.set_ylim(0, 110)
ax1.axhline(100, color='#6a3d9a', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect (100%)')
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend(fontsize=10)

# Add value labels on bars
for i, (bar, acc, correct, total) in enumerate(zip(bars1, accuracies, correct_counts, total_counts)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{acc:.1f}%\n({correct}/{total})', 
            ha='center', va='bottom', fontsize=11)

# ============================================================================
# Plot 2: Rule Inference Accuracy by Rule Type
# ============================================================================
ax2 = axes[0, 1]

rule_types_rule = rule_acc_by_rule['rule_type'].tolist()
accuracies_rule = rule_acc_by_rule['accuracy_percent'].tolist()
correct_counts_rule = rule_acc_by_rule['correct'].astype(int).tolist()
total_counts_rule = rule_acc_by_rule['total'].astype(int).tolist()

# Capitalize rule type labels for display
rule_labels_rule = [r.capitalize() for r in rule_types_rule]

bars2 = ax2.bar(
    range(len(rule_types_rule)),
    accuracies_rule,
    color=colors[:len(rule_types_rule)],
    alpha=0.8,
    edgecolor='#333333',
    linewidth=1.5
)

ax2.set_title('Rule Inference Accuracy by Rule Type', fontsize=14, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_xticks(range(len(rule_types_rule)))
ax2.set_xticklabels(rule_labels_rule, fontsize=12)
ax2.set_ylim(0, 110)
ax2.axhline(100, color='#6a3d9a', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect (100%)')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(fontsize=10)

# Add value labels on bars
for i, (bar, acc, correct, total) in enumerate(zip(bars2, accuracies_rule, correct_counts_rule, total_counts_rule)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{acc:.1f}%\n({correct}/{total})', 
            ha='center', va='bottom', fontsize=11)

# ============================================================================
# Plot 3: Overall Object Identification Accuracy (Bar Chart)
# ============================================================================
ax3 = axes[1, 0]

match_counts = df_with_rules['object_accuracy'].value_counts()
labels = ['Correct', 'Incorrect']
colors_pie = ['#1b9e77', '#d95f02']
sizes = [match_counts.get(True, 0), match_counts.get(False, 0)]

x_pos = np.arange(len(labels))
bars3 = ax3.bar(
    x_pos,
    sizes,
    color=colors_pie,
    alpha=0.9,
    edgecolor='#333333',
    linewidth=1.2
)
ax3.set_title('Overall Object Identification Accuracy', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(labels, fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_ylim(0, max(sizes) + max(2, int(0.1 * max(sizes))))
ax3.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Value labels on bars
for bar in bars3:
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width()/2.,
        height + 0.5,
        f'{int(height)}',
        ha='center',
        va='bottom',
        fontsize=11,
    )

# Add summary box
summary_text = f'{overall_object_acc:.1f}% overall\n({object_correct}/{object_total})'
ax3.text(
    0.98, 0.95, summary_text,
    transform=ax3.transAxes,
    ha='right',
    va='top',
    fontsize=11,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#cccccc')
)

# ============================================================================
# Plot 4: Overall Rule Inference Accuracy (Bar Chart)
# ============================================================================
ax4 = axes[1, 1]

rule_match_counts = df_with_rules['rule_accuracy'].value_counts()
labels_rule = ['Correct', 'Incorrect']
colors_pie_rule = ['#1b9e77', '#d95f02']
sizes_rule = [rule_match_counts.get(True, 0), rule_match_counts.get(False, 0)]

x_pos_rule = np.arange(len(labels_rule))
bars4 = ax4.bar(
    x_pos_rule,
    sizes_rule,
    color=colors_pie_rule,
    alpha=0.9,
    edgecolor='#333333',
    linewidth=1.2
)
ax4.set_title('Overall Rule Inference Accuracy', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos_rule)
ax4.set_xticklabels(labels_rule, fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_ylim(0, max(sizes_rule) + max(2, int(0.1 * max(sizes_rule))))
ax4.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Value labels on bars
for bar in bars4:
    height = bar.get_height()
    ax4.text(
        bar.get_x() + bar.get_width()/2.,
        height + 0.5,
        f'{int(height)}',
        ha='center',
        va='bottom',
        fontsize=11,
    )

# Add summary box
summary_text_rule = f'{overall_rule_acc:.1f}% overall\n({rule_correct}/{rule_total})'
ax4.text(
    0.98, 0.95, summary_text_rule,
    transform=ax4.transAxes,
    ha='right',
    va='top',
    fontsize=11,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#cccccc')
)

# ============================================================================
# Finalize and Save
# ============================================================================
plt.suptitle('Accuracy Analysis: Object Identification and Rule Inference by Rule Type', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('images/accuracy_by_rule_type.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: images/accuracy_by_rule_type.png")
plt.close()

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)

