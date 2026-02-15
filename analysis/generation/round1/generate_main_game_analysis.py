"""
Generate main game analysis CSV with per-round and total exploration data
"""

import json
import pandas as pd
import numpy as np

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
    
    return False


# Load Firebase data
print("Loading Firebase data...")
with open('firebase_data.json', 'r') as f:
    all_data = json.load(f)

# Filter for Prolific IDs
prolific_data = {pid: pdata for pid, pdata in all_data.items() if is_prolific_id(pid)}
print(f"Found {len(prolific_data)} Prolific participants\n")

print("="*80)
print("MAIN GAME ANALYSIS")
print("="*80)

results = []

for participant_id, participant_data in prolific_data.items():
    if not isinstance(participant_data, dict):
        continue
    
    main_game = participant_data.get('main_game', {})
    if not main_game or not isinstance(main_game, dict):
        continue
    
    # Collect data for each round
    round_data_list = []
    
    for round_key, round_data in main_game.items():
        if not isinstance(round_data, dict) or not round_key.startswith('round_'):
            continue
        
        round_number = round_data.get('round_number', 0)
        
        # Get number of tests
        state_history = round_data.get('state_history', [])
        num_tests = len(state_history)
        
        # Get action sequence
        user_test_actions = round_data.get('user_test_actions', [])
        action_sequence = []
        
        if user_test_actions:
            for action in user_test_actions:
                if isinstance(action, dict):
                    action_type = action.get('action_type', '')
                    if action_type == 'test':
                        objects = action.get('objects_tested', [])
                        action_sequence.append(f"Test:{objects}")
                    else:
                        obj_idx = action.get('object_index', '')
                        action_sequence.append(f"{action_type}:obj{obj_idx}")
        
        action_sequence_str = " | ".join(action_sequence) if action_sequence else ""
        
        # Get ground truth
        true_blickets = round_data.get('true_blicket_indices', [])
        true_rule = round_data.get('true_rule', '')
        
        # Get user selections
        user_chosen_blickets = round_data.get('user_chosen_blickets', [])
        
        # Check if matches
        objects_match = set(true_blickets) == set(user_chosen_blickets) if true_blickets and user_chosen_blickets is not None else None
        
        # Get rule information
        rule_type = round_data.get('rule_type', '')
        user_rule_type = None
        if rule_type:
            if 'conjunctive' in rule_type.lower():
                user_rule_type = 'conjunctive'
            elif 'disjunctive' in rule_type.lower():
                user_rule_type = 'disjunctive'
        
        rule_match = (user_rule_type == true_rule.lower()) if user_rule_type and true_rule else None
        
        round_data_list.append({
            'round': round_number,
            'num_tests': num_tests,
            'action_sequence': action_sequence_str,
            'user_chosen_objects': str(sorted(user_chosen_blickets)) if user_chosen_blickets else '[]',
            'ground_truth_objects': str(sorted(true_blickets)) if true_blickets else '[]',
            'objects_match': objects_match,
            'ground_truth_rule': true_rule,
            'user_rule_type': user_rule_type,
            'rule_match': rule_match
        })
    
    # Sort by round number
    round_data_list = sorted(round_data_list, key=lambda x: x['round'])
    
    # Calculate totals
    total_tests = sum(rd['num_tests'] for rd in round_data_list)
    
    # Add data for each round
    for rd in round_data_list:
        results.append({
            'participant_id': participant_id,
            'round': rd['round'],
            'num_tests_round': rd['num_tests'],
            'total_tests_all_rounds': total_tests,
            'action_sequence': rd['action_sequence'],
            'user_chosen_objects': rd['user_chosen_objects'],
            'ground_truth_objects': rd['ground_truth_objects'],
            'objects_match': rd['objects_match'],
            'ground_truth_rule': rd['ground_truth_rule'],
            'user_rule_type': rd['user_rule_type'],
            'rule_match': rd['rule_match']
        })

# Create DataFrame
df = pd.DataFrame(results)

# Save to CSV
df.to_csv('results/3_main_game_rounds.csv', index=False)

print(f"\nTotal rows: {len(df)} (one per participant per round)")
print(f"Participants with data: {df['participant_id'].nunique()}")

# Summary statistics
print("\n--- Exploration Statistics ---")
print(f"Average tests per round: {df['num_tests_round'].mean():.2f}")
print(f"Median tests per round: {df['num_tests_round'].median():.1f}")
print(f"Range: {df['num_tests_round'].min():.0f} - {df['num_tests_round'].max():.0f}")

# By round
print("\n--- By Round ---")
for round_num in sorted(df['round'].unique()):
    round_df = df[df['round'] == round_num]
    print(f"\nRound {round_num}:")
    print(f"  Participants: {len(round_df)}")
    print(f"  Average tests: {round_df['num_tests_round'].mean():.2f}")
    print(f"  Object match rate: {round_df['objects_match'].sum()}/{round_df['objects_match'].notna().sum()} ({round_df['objects_match'].sum()/round_df['objects_match'].notna().sum()*100:.1f}%)")
    print(f"  Rule match rate: {round_df['rule_match'].sum()}/{round_df['rule_match'].notna().sum()} ({round_df['rule_match'].sum()/round_df['rule_match'].notna().sum()*100:.1f}%)")

# Tested more than 4 times per round
tested_more_than_4 = (df['num_tests_round'] > 4).sum()
print(f"\n--- Testing Threshold ---")
print(f"Rounds with >4 tests: {tested_more_than_4}/{len(df)} ({tested_more_than_4/len(df)*100:.1f}%)")

print("\n✓ Saved: results/3_main_game_rounds.csv")
print("\nColumns:")
print("  - participant_id: Prolific ID")
print("  - round: Round number (1, 2, or 3)")
print("  - num_tests_round: Number of tests in this specific round")
print("  - total_tests_all_rounds: Total tests across all rounds for this participant")
print("  - action_sequence: Sequence of test actions in this round")
print("  - user_chosen_objects: Objects selected as Nexioms")
print("  - ground_truth_objects: True Nexiom objects")
print("  - objects_match: True/False if selection matches ground truth")
print("  - ground_truth_rule: True rule type (conjunctive/disjunctive)")
print("  - user_rule_type: User's rule classification")
print("  - rule_match: True/False if rule classification matches")

print("\n" + "="*80)

