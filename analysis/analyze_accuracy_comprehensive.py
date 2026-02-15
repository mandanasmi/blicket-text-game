"""
Comprehensive analysis of object identification and rule inference accuracy
over users and rounds, including prior game experience information
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict

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

def has_prior_experience(participant_data):
    """Check if participant has prior game experience"""
    comprehension = participant_data.get('comprehension', {})
    if not isinstance(comprehension, dict):
        return None
    
    similar_game = comprehension.get('similar_game_experience', {})
    if not isinstance(similar_game, dict):
        return None
    
    answer = similar_game.get('answer', '')
    if isinstance(answer, str):
        answer_lower = answer.lower()
        if 'yes' in answer_lower:
            return True
        elif 'no' in answer_lower:
            return False
    
    return None

# Load data
print("="*80)
print("COMPREHENSIVE ACCURACY ANALYSIS")
print("="*80)
print("\nLoading firebase_data2.json...")

try:
    with open('firebase_data2.json', 'r') as f:
        all_data = json.load(f)
except FileNotFoundError:
    print("ERROR: firebase_data2.json not found!")
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
prior_experience_data = []

for participant_id, participant_data in prolific_data.items():
    if not isinstance(participant_data, dict):
        continue
    
    prolific_id = get_prolific_id(participant_data)
    if not prolific_id:
        prolific_id = participant_id
    
    # Check prior experience
    has_prior = has_prior_experience(participant_data)
    prior_experience_data.append({
        'participant_id': prolific_id,
        'has_prior_experience': has_prior
    })
    
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
            'true_blickets': str(sorted(true_blickets)) if true_blickets else '[]',
            'user_chosen_blickets': str(sorted(user_chosen_blickets)) if user_chosen_blickets else '[]',
            'true_rule': true_rule,
            'user_rule': user_rule,
            'object_accuracy': object_accuracy,
            'rule_accuracy': rule_accuracy,
            'has_prior_experience': has_prior
        })

# Create DataFrames
df = pd.DataFrame(results)
prior_df = pd.DataFrame(prior_experience_data)

if len(df) == 0:
    print("\nERROR: No valid round data found!")
    exit(1)

print(f"\nTotal rounds analyzed: {len(df)}")
print(f"Unique participants: {df['participant_id'].nunique()}")

# ============================================================================
# PRIOR EXPERIENCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("1. PRIOR GAME EXPERIENCE")
print("="*80)

prior_counts = prior_df['has_prior_experience'].value_counts()
total_with_answer = prior_df['has_prior_experience'].notna().sum()

print(f"\nTotal participants: {len(prior_df)}")
print(f"Participants with answered question: {total_with_answer}")

if True in prior_counts.index:
    yes_count = prior_counts[True]
    print(f"  - Have prior experience: {yes_count} ({yes_count/total_with_answer*100:.1f}%)")

if False in prior_counts.index:
    no_count = prior_counts[False]
    print(f"  - No prior experience: {no_count} ({no_count/total_with_answer*100:.1f}%)")

if None in prior_counts.index:
    unknown_count = prior_counts[None]
    print(f"  - Unknown/No answer: {unknown_count}")

# ============================================================================
# OBJECT IDENTIFICATION ACCURACY
# ============================================================================
print("\n" + "="*80)
print("2. OBJECT IDENTIFICATION ACCURACY")
print("="*80)

# Overall accuracy
object_acc_valid = df['object_accuracy'].notna()
object_correct = df['object_accuracy'].sum()
object_total = object_acc_valid.sum()
overall_object_acc = (object_correct / object_total * 100) if object_total > 0 else 0

print(f"\n--- Overall Accuracy ---")
print(f"Correct: {object_correct}/{object_total} ({overall_object_acc:.1f}%)")

# Accuracy by round
print(f"\n--- Accuracy by Round ---")
for round_num in sorted(df['round_number'].unique()):
    round_data = df[df['round_number'] == round_num]
    round_valid = round_data['object_accuracy'].notna()
    round_correct = round_data['object_accuracy'].sum()
    round_total = round_valid.sum()
    round_acc = (round_correct / round_total * 100) if round_total > 0 else 0
    print(f"Round {round_num}: {round_correct}/{round_total} ({round_acc:.1f}%)")

# Accuracy by participant
print(f"\n--- Accuracy by Participant ---")
participant_object_acc = df.groupby('participant_id').agg({
    'object_accuracy': ['sum', 'count']
}).reset_index()
participant_object_acc.columns = ['participant_id', 'correct', 'total']
participant_object_acc['accuracy_pct'] = (participant_object_acc['correct'] / participant_object_acc['total'] * 100).round(1)

print(f"\nPer-participant object identification accuracy:")
print(participant_object_acc.sort_values('accuracy_pct', ascending=False).to_string(index=False))

mean_participant_acc = participant_object_acc['accuracy_pct'].mean()
print(f"\nMean per-participant accuracy: {mean_participant_acc:.1f}%")
print(f"Std per-participant accuracy: {participant_object_acc['accuracy_pct'].std():.1f}%")

# Accuracy by prior experience
print(f"\n--- Accuracy by Prior Experience ---")
for has_prior in [True, False]:
    prior_data = df[df['has_prior_experience'] == has_prior]
    prior_valid = prior_data['object_accuracy'].notna()
    prior_correct = prior_data['object_accuracy'].sum()
    prior_total = prior_valid.sum()
    prior_acc = (prior_correct / prior_total * 100) if prior_total > 0 else 0
    label = "Has prior experience" if has_prior else "No prior experience"
    print(f"{label}: {prior_correct}/{prior_total} ({prior_acc:.1f}%)")

# ============================================================================
# RULE INFERENCE ACCURACY
# ============================================================================
print("\n" + "="*80)
print("3. RULE INFERENCE ACCURACY")
print("="*80)

# Overall accuracy
rule_acc_valid = df['rule_accuracy'].notna()
rule_correct = df['rule_accuracy'].sum()
rule_total = rule_acc_valid.sum()
overall_rule_acc = (rule_correct / rule_total * 100) if rule_total > 0 else 0

print(f"\n--- Overall Accuracy ---")
print(f"Correct: {rule_correct}/{rule_total} ({overall_rule_acc:.1f}%)")

# Accuracy by round
print(f"\n--- Accuracy by Round ---")
for round_num in sorted(df['round_number'].unique()):
    round_data = df[df['round_number'] == round_num]
    round_valid = round_data['rule_accuracy'].notna()
    round_correct = round_data['rule_accuracy'].sum()
    round_total = round_valid.sum()
    round_acc = (round_correct / round_total * 100) if round_total > 0 else 0
    print(f"Round {round_num}: {round_correct}/{round_total} ({round_acc:.1f}%)")

# Accuracy by participant
print(f"\n--- Accuracy by Participant ---")
participant_rule_acc = df.groupby('participant_id').agg({
    'rule_accuracy': ['sum', 'count']
}).reset_index()
participant_rule_acc.columns = ['participant_id', 'correct', 'total']
participant_rule_acc['accuracy_pct'] = (participant_rule_acc['correct'] / participant_rule_acc['total'] * 100).round(1)

print(f"\nPer-participant rule inference accuracy:")
print(participant_rule_acc.sort_values('accuracy_pct', ascending=False).to_string(index=False))

mean_participant_rule_acc = participant_rule_acc['accuracy_pct'].mean()
print(f"\nMean per-participant accuracy: {mean_participant_rule_acc:.1f}%")
print(f"Std per-participant accuracy: {participant_rule_acc['accuracy_pct'].std():.1f}%")

# Accuracy by prior experience
print(f"\n--- Accuracy by Prior Experience ---")
for has_prior in [True, False]:
    prior_data = df[df['has_prior_experience'] == has_prior]
    prior_valid = prior_data['rule_accuracy'].notna()
    prior_correct = prior_data['rule_accuracy'].sum()
    prior_total = prior_valid.sum()
    prior_acc = (prior_correct / prior_total * 100) if prior_total > 0 else 0
    label = "Has prior experience" if has_prior else "No prior experience"
    print(f"{label}: {prior_correct}/{prior_total} ({prior_acc:.1f}%)")

# ============================================================================
# COMBINED ACCURACY (Both correct)
# ============================================================================
print("\n" + "="*80)
print("4. COMBINED ACCURACY (Both Object and Rule Correct)")
print("="*80)

df['both_correct'] = df['object_accuracy'] & df['rule_accuracy']
both_valid = df['both_correct'].notna()
both_correct = df['both_correct'].sum()
both_total = both_valid.sum()
overall_both_acc = (both_correct / both_total * 100) if both_total > 0 else 0

print(f"\n--- Overall Combined Accuracy ---")
print(f"Both correct: {both_correct}/{both_total} ({overall_both_acc:.1f}%)")

# Combined accuracy by round
print(f"\n--- Combined Accuracy by Round ---")
for round_num in sorted(df['round_number'].unique()):
    round_data = df[df['round_number'] == round_num]
    round_valid = round_data['both_correct'].notna()
    round_correct = round_data['both_correct'].sum()
    round_total = round_valid.sum()
    round_acc = (round_correct / round_total * 100) if round_total > 0 else 0
    print(f"Round {round_num}: {round_correct}/{round_total} ({round_acc:.1f}%)")

# ============================================================================
# SAVE RESULTS TO CSV
# ============================================================================
print("\n" + "="*80)
print("5. SAVING RESULTS")
print("="*80)

# Save detailed results
df.to_csv('results/accuracy_detailed.csv', index=False)
print("Saved: results/accuracy_detailed.csv")

# Save per-participant summary
participant_summary = pd.merge(
    participant_object_acc.rename(columns={'correct': 'object_correct', 'total': 'object_total', 'accuracy_pct': 'object_accuracy_pct'}),
    participant_rule_acc.rename(columns={'correct': 'rule_correct', 'total': 'rule_total', 'accuracy_pct': 'rule_accuracy_pct'}),
    on='participant_id'
)

# Add prior experience
prior_summary = prior_df.groupby('participant_id')['has_prior_experience'].first().reset_index()
participant_summary = pd.merge(participant_summary, prior_summary, on='participant_id', how='left')

# Calculate combined accuracy per participant
participant_both = df.groupby('participant_id').agg({
    'both_correct': ['sum', 'count']
}).reset_index()
participant_both.columns = ['participant_id', 'both_correct_count', 'both_total']
participant_both['both_accuracy_pct'] = (participant_both['both_correct_count'] / participant_both['both_total'] * 100).round(1)

participant_summary = pd.merge(participant_summary, participant_both, on='participant_id', how='left')

participant_summary.to_csv('results/accuracy_per_participant.csv', index=False)
print("Saved: results/accuracy_per_participant.csv")

# Save prior experience summary
prior_df.to_csv('results/prior_experience.csv', index=False)
print("Saved: results/prior_experience.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

