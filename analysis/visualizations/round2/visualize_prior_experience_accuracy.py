"""
Visualize accuracy for users with prior experience question answered
Shows object identification and rule inference accuracy across all users and rounds
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

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

def get_user_chosen_blickets(round_data):
    """Extract user's chosen blickets from round data"""
    if 'user_chosen_blickets' in round_data:
        chosen = round_data['user_chosen_blickets']
        if isinstance(chosen, list):
            return sorted(chosen)
    if 'blicket_classifications' in round_data:
        classifications = round_data['blicket_classifications']
        if isinstance(classifications, dict):
            chosen = []
            for key, value in classifications.items():
                if value == "Yes":
                    try:
                        idx = int(key.split('_')[1])
                        chosen.append(idx)
                    except (ValueError, IndexError):
                        pass
            return sorted(chosen) if chosen else None
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

# Load the filtered JSON file
print("="*80)
print("ACCURACY VISUALIZATION: ALL USERS OVER ALL ROUNDS")
print("="*80)
print("\nLoading firebase_data2_with_prior_experience.json...")

with open('firebase_data2_with_prior_experience.json', 'r') as f:
    data = json.load(f)

print(f"Total entries loaded: {len(data)}")

# Collect accuracy data
results = []

for entry_id, entry_data in data.items():
    if not isinstance(entry_data, dict):
        continue
    
    prolific_id = get_prolific_id(entry_data)
    if not prolific_id:
        prolific_id = entry_id
    
    main_game = entry_data.get('main_game', {})
    if not isinstance(main_game, dict):
        continue
    
    for round_key, round_data in main_game.items():
        if not isinstance(round_data, dict) or not round_key.startswith('round_'):
            continue
        
        round_number = round_data.get('round_number', 0)
        if round_number == 0:
            continue
        
        # Get ground truth
        true_blickets = round_data.get('true_blicket_indices', [])
        true_rule = round_data.get('true_rule', '').lower()
        
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
            'object_accuracy': object_accuracy,
            'rule_accuracy': rule_accuracy,
            'true_rule': true_rule if true_rule in ['conjunctive', 'disjunctive'] else None
        })

# Create DataFrame
df = pd.DataFrame(results)

if len(df) == 0:
    print("\nERROR: No valid round data found!")
    exit(1)

print(f"\nTotal rounds analyzed: {len(df)}")
print(f"Unique participants: {df['participant_id'].nunique()}")

# Calculate per-user accuracies
user_accuracy = df.groupby('participant_id').agg({
    'object_accuracy': ['sum', 'count'],
    'rule_accuracy': ['sum', 'count']
}).reset_index()

user_accuracy.columns = ['participant_id', 'object_correct', 'object_total', 'rule_correct', 'rule_total']
user_accuracy['object_accuracy_pct'] = (user_accuracy['object_correct'] / user_accuracy['object_total'] * 100).round(1)
user_accuracy['rule_accuracy_pct'] = (user_accuracy['rule_correct'] / user_accuracy['rule_total'] * 100).round(1)

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Create figure
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

# 1. Object identification accuracy by user
ax1 = fig.add_subplot(gs[0, 0])
user_accuracy_sorted = user_accuracy.sort_values('object_accuracy_pct', ascending=True)
participant_labels = [pid[:8] + '...' for pid in user_accuracy_sorted['participant_id']]
bars = ax1.barh(range(len(user_accuracy_sorted)), user_accuracy_sorted['object_accuracy_pct'],
                color='#2196F3', alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_yticks(range(len(user_accuracy_sorted)))
ax1.set_yticklabels(participant_labels, fontsize=9)
ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Participant', fontsize=12, fontweight='bold')
ax1.set_title('Object Identification Accuracy by Participant', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 100)
ax1.grid(True, alpha=0.3, axis='x')
# Add value labels
for i, (bar, row) in enumerate(zip(bars, user_accuracy_sorted.itertuples())):
    width = bar.get_width()
    ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
            f'{width:.1f}% ({int(row.object_correct)}/{int(row.object_total)})',
            ha='left', va='center', fontsize=9, fontweight='bold')

# 2. Rule inference accuracy by user
ax2 = fig.add_subplot(gs[0, 1])
user_accuracy_sorted_rule = user_accuracy.sort_values('rule_accuracy_pct', ascending=True)
participant_labels_rule = [pid[:8] + '...' for pid in user_accuracy_sorted_rule['participant_id']]
bars = ax2.barh(range(len(user_accuracy_sorted_rule)), user_accuracy_sorted_rule['rule_accuracy_pct'],
                color='#FF9800', alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_yticks(range(len(user_accuracy_sorted_rule)))
ax2.set_yticklabels(participant_labels_rule, fontsize=9)
ax2.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Participant', fontsize=12, fontweight='bold')
ax2.set_title('Rule Inference Accuracy by Participant', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 100)
ax2.grid(True, alpha=0.3, axis='x')
# Add value labels
for i, (bar, row) in enumerate(zip(bars, user_accuracy_sorted_rule.itertuples())):
    width = bar.get_width()
    ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
            f'{width:.1f}% ({int(row.rule_correct)}/{int(row.rule_total)})',
            ha='left', va='center', fontsize=9, fontweight='bold')

# 3. Accuracy by round - Object Identification
ax3 = fig.add_subplot(gs[1, 0])
round_obj_acc = df.groupby('round_number')['object_accuracy'].agg(['sum', 'count'])
round_obj_acc['accuracy_pct'] = (round_obj_acc['sum'] / round_obj_acc['count'] * 100).round(1)
rounds = round_obj_acc.index
x_pos = np.arange(len(rounds))
bars = ax3.bar(x_pos, round_obj_acc['accuracy_pct'], color='#2196F3', alpha=0.7, 
               edgecolor='black', linewidth=1.5)
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'Round {r}' for r in rounds])
ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax3.set_title('Object Identification Accuracy by Round', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 100)
ax3.grid(True, alpha=0.3, axis='y')
# Add value labels
for i, (bar, row) in enumerate(zip(bars, round_obj_acc.itertuples())):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%\n(n={int(row.count)})',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 4. Accuracy by round - Rule Inference
ax4 = fig.add_subplot(gs[1, 1])
round_rule_acc = df.groupby('round_number')['rule_accuracy'].agg(['sum', 'count'])
round_rule_acc['accuracy_pct'] = (round_rule_acc['sum'] / round_rule_acc['count'] * 100).round(1)
rounds = round_rule_acc.index
x_pos = np.arange(len(rounds))
bars = ax4.bar(x_pos, round_rule_acc['accuracy_pct'], color='#FF9800', alpha=0.7, 
               edgecolor='black', linewidth=1.5)
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'Round {r}' for r in rounds])
ax4.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax4.set_title('Rule Inference Accuracy by Round', fontsize=14, fontweight='bold')
ax4.set_ylim(0, 100)
ax4.grid(True, alpha=0.3, axis='y')
# Add value labels
for i, (bar, row) in enumerate(zip(bars, round_rule_acc.itertuples())):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%\n(n={int(row.count)})',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 5. Combined accuracy comparison
ax5 = fig.add_subplot(gs[2, :])
# Prepare data for grouped bar chart
x_pos = np.arange(len(user_accuracy))
width = 0.35
bars1 = ax5.bar(x_pos - width/2, user_accuracy['object_accuracy_pct'], width,
                label='Object Identification', color='#2196F3', alpha=0.7, 
                edgecolor='black', linewidth=1.5)
bars2 = ax5.bar(x_pos + width/2, user_accuracy['rule_accuracy_pct'], width,
                label='Rule Inference', color='#FF9800', alpha=0.7, 
                edgecolor='black', linewidth=1.5)
ax5.set_xticks(x_pos)
ax5.set_xticklabels([pid[:8] + '...' for pid in user_accuracy['participant_id']], 
                    rotation=45, ha='right', fontsize=9)
ax5.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Participant', fontsize=12, fontweight='bold')
ax5.set_title('Accuracy Comparison: Object Identification vs Rule Inference', 
              fontsize=14, fontweight='bold')
ax5.set_ylim(0, 100)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Object Identification Accuracy by Rule Type
ax6 = fig.add_subplot(gs[3, 0])
rule_obj_acc = df[df['true_rule'].notna()].groupby('true_rule')['object_accuracy'].agg(['sum', 'count'])
rule_obj_acc['accuracy_pct'] = (rule_obj_acc['sum'] / rule_obj_acc['count'] * 100).round(1)
rules = rule_obj_acc.index
x_pos = np.arange(len(rules))
colors_rule = ['#1976D2', '#FF9800']  # Blue for conjunctive, Orange for disjunctive
bars = ax6.bar(x_pos, rule_obj_acc['accuracy_pct'], color=colors_rule[:len(rules)], 
               alpha=0.7, edgecolor='black', linewidth=1.5)
ax6.set_xticks(x_pos)
ax6.set_xticklabels([r.capitalize() for r in rules], fontsize=12)
ax6.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax6.set_title('Object Identification Accuracy\nby Rule Type', fontsize=14, fontweight='bold')
ax6.set_ylim(0, 100)
ax6.grid(True, alpha=0.3, axis='y')
# Add value labels
for i, (bar, row) in enumerate(zip(bars, rule_obj_acc.itertuples())):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%\n(n={int(row.count)})',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 7. Rule Inference Accuracy by Rule Type
ax7 = fig.add_subplot(gs[3, 1])
rule_rule_acc = df[df['true_rule'].notna()].groupby('true_rule')['rule_accuracy'].agg(['sum', 'count'])
rule_rule_acc['accuracy_pct'] = (rule_rule_acc['sum'] / rule_rule_acc['count'] * 100).round(1)
rules = rule_rule_acc.index
x_pos = np.arange(len(rules))
bars = ax7.bar(x_pos, rule_rule_acc['accuracy_pct'], color=colors_rule[:len(rules)], 
               alpha=0.7, edgecolor='black', linewidth=1.5)
ax7.set_xticks(x_pos)
ax7.set_xticklabels([r.capitalize() for r in rules], fontsize=12)
ax7.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax7.set_title('Rule Inference Accuracy\nby Rule Type', fontsize=14, fontweight='bold')
ax7.set_ylim(0, 100)
ax7.grid(True, alpha=0.3, axis='y')
# Add value labels
for i, (bar, row) in enumerate(zip(bars, rule_rule_acc.itertuples())):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%\n(n={int(row.count)})',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Accuracy Analysis: All Users Over All Rounds\n(Users with Prior Experience Question Answered)', 
             fontsize=16, fontweight='bold', y=0.995)

os.makedirs('images', exist_ok=True)
plt.tight_layout()
plt.savefig('images/accuracy_all_users_all_rounds.png', dpi=300, bbox_inches='tight')
print("\nSaved: images/accuracy_all_users_all_rounds.png")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nTotal Participants: {len(user_accuracy)}")
print(f"Total Rounds Analyzed: {len(df)}")

print("\nObject Identification Accuracy:")
overall_obj_acc = df['object_accuracy'].sum() / df['object_accuracy'].notna().sum() * 100
print(f"  Overall: {overall_obj_acc:.1f}% ({df['object_accuracy'].sum()}/{df['object_accuracy'].notna().sum()})")
print(f"  Mean per participant: {user_accuracy['object_accuracy_pct'].mean():.1f}%")
print(f"  Range: {user_accuracy['object_accuracy_pct'].min():.1f}% - {user_accuracy['object_accuracy_pct'].max():.1f}%")

print("\nRule Inference Accuracy:")
overall_rule_acc = df['rule_accuracy'].sum() / df['rule_accuracy'].notna().sum() * 100
print(f"  Overall: {overall_rule_acc:.1f}% ({df['rule_accuracy'].sum()}/{df['rule_accuracy'].notna().sum()})")
print(f"  Mean per participant: {user_accuracy['rule_accuracy_pct'].mean():.1f}%")
print(f"  Range: {user_accuracy['rule_accuracy_pct'].min():.1f}% - {user_accuracy['rule_accuracy_pct'].max():.1f}%")

print("\nAccuracy by Rule Type:")
df_with_rule = df[df['true_rule'].notna()]
if len(df_with_rule) > 0:
    print("\n  Object Identification Accuracy by Rule Type:")
    for rule_type in ['conjunctive', 'disjunctive']:
        rule_data = df_with_rule[df_with_rule['true_rule'] == rule_type]
        if len(rule_data) > 0:
            rule_obj_acc = rule_data['object_accuracy'].sum() / rule_data['object_accuracy'].notna().sum() * 100
            print(f"    {rule_type.capitalize()}: {rule_obj_acc:.1f}% ({rule_data['object_accuracy'].sum()}/{rule_data['object_accuracy'].notna().sum()})")
    
    print("\n  Rule Inference Accuracy by Rule Type:")
    for rule_type in ['conjunctive', 'disjunctive']:
        rule_data = df_with_rule[df_with_rule['true_rule'] == rule_type]
        if len(rule_data) > 0:
            rule_rule_acc = rule_data['rule_accuracy'].sum() / rule_data['rule_accuracy'].notna().sum() * 100
            print(f"    {rule_type.capitalize()}: {rule_rule_acc:.1f}% ({rule_data['rule_accuracy'].sum()}/{rule_data['rule_accuracy'].notna().sum()})")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)

plt.close()

