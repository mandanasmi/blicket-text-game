"""
Combine main game data from round 1 and round 2 into a single CSV
Similar to main_game_comprehensive_data.csv but without prior_experience column
"""

import json
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to import helper functions
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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

def extract_round_data(json_file_path, round_label):
    """Extract main game data from a JSON file"""
    print(f"\nProcessing {round_label}...")
    print(f"Loading {json_file_path}...")
    
    try:
        with open(json_file_path, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {json_file_path} not found!")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON - {e}")
        return pd.DataFrame()
    
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
            
            # Extract test timings
            test_timings = round_data.get('test_timings', [])
            num_tests = len(test_timings)
            
            # Extract time taken per test (time_since_previous_seconds)
            time_per_test = []
            if isinstance(test_timings, list):
                for timing in test_timings:
                    if isinstance(timing, dict):
                        time_since_prev = timing.get('time_since_previous_seconds', None)
                        if time_since_prev is not None:
                            time_per_test.append(round(time_since_prev, 3))
            
            # Total time taken for testing (sum of time_per_test)
            total_test_time = sum(time_per_test) if time_per_test else None
            
            # Total time for the round
            total_round_time = round_data.get('total_time_seconds', None)
            
            # Get ground truth
            true_blickets = round_data.get('true_blicket_indices', [])
            true_rule = round_data.get('true_rule', '')
            
            # Get user's choices
            user_chosen_blickets = get_user_chosen_blickets(round_data)
            
            # Get rule hypothesis (written text)
            rule_hypothesis = round_data.get('rule_hypothesis', '')
            
            # Get rule choice (checkbox selection)
            rule_type_str = round_data.get('rule_type', '')
            user_rule_choice = parse_rule_type(rule_type_str)
            
            # Calculate accuracy
            object_accuracy = None
            if true_blickets is not None and user_chosen_blickets is not None:
                true_set = set(true_blickets) if isinstance(true_blickets, list) else set()
                user_set = set(user_chosen_blickets) if isinstance(user_chosen_blickets, list) else set()
                object_accuracy = (true_set == user_set)
            
            rule_accuracy = None
            if true_rule and user_rule_choice:
                rule_accuracy = (true_rule.lower() == user_rule_choice.lower())
            
            results.append({
                'user_id': prolific_id,
                'round_number': round_number,
                'num_tests': num_tests,
                'time_per_test_seconds': str(time_per_test),  # Store as string for CSV
                'total_test_time_seconds': round(total_test_time, 3) if total_test_time else None,
                'total_round_time_seconds': round(total_round_time, 3) if total_round_time else None,
                'user_object_identification': str(sorted(user_chosen_blickets)) if user_chosen_blickets else '[]',
                'ground_truth_objects': str(sorted(true_blickets)) if true_blickets else '[]',
                'object_identification_correct': object_accuracy,
                'rule_inference_text': rule_hypothesis,
                'rule_choice': user_rule_choice if user_rule_choice else '',
                'ground_truth_rule': true_rule,
                'rule_choice_correct': rule_accuracy,
                'round_label': round_label  # Add label to track which round
            })
    
    return pd.DataFrame(results)

# Main execution
print("="*80)
print("COMBINING ROUND 1 AND ROUND 2 DATA")
print("="*80)

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

# File paths
round1_json = os.path.join(base_dir, 'round1', 'firebase_data1.json')
round2_json = os.path.join(base_dir, 'round2', 'firebase_data2.json')

# If round2 JSON doesn't exist, try the one with prior experience
if not os.path.exists(round2_json):
    round2_json = os.path.join(base_dir, 'round2', 'firebase_data2_with_prior_experience.json')

# Extract data from both rounds
df_round1 = extract_round_data(round1_json, 'Round 1')
df_round2 = extract_round_data(round2_json, 'Round 2')

if len(df_round1) == 0 and len(df_round2) == 0:
    print("\nERROR: No valid round data found!")
    exit(1)

# Combine dataframes
df_combined = pd.concat([df_round1, df_round2], ignore_index=True)

# Remove the round_label column (it was just for tracking)
if 'round_label' in df_combined.columns:
    df_combined = df_combined.drop(columns=['round_label'])

# Remove duplicates based on user_id and round_number (keep first occurrence)
rows_before = len(df_combined)
print(f"\nBefore deduplication: {rows_before} rows")
df_combined = df_combined.drop_duplicates(subset=['user_id', 'round_number'], keep='first')
rows_after = len(df_combined)
print(f"After deduplication: {rows_after} rows")
print(f"Removed {rows_before - rows_after} duplicate rows")

# Sort by user_id and round_number
df_combined = df_combined.sort_values(['user_id', 'round_number']).reset_index(drop=True)

print("\n" + "="*80)
print("COMBINED DATA SUMMARY")
print("="*80)
print(f"\nTotal rounds: {len(df_combined)}")
print(f"Unique participants: {df_combined['user_id'].nunique()}")
print(f"Participants from Round 1: {df_round1['user_id'].nunique() if len(df_round1) > 0 else 0}")
print(f"Participants from Round 2: {df_round2['user_id'].nunique() if len(df_round2) > 0 else 0}")

# Display sample data
print("\n" + "="*80)
print("SAMPLE DATA (first 5 rows)")
print("="*80)
print(df_combined.head().to_string())

# Calculate summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nTotal tests across all rounds: {df_combined['num_tests'].sum()}")
print(f"Average tests per round: {df_combined['num_tests'].mean():.2f}")
print(f"Median tests per round: {df_combined['num_tests'].median():.0f}")

print(f"\nObject identification accuracy: {df_combined['object_identification_correct'].sum()}/{df_combined['object_identification_correct'].notna().sum()} ({df_combined['object_identification_correct'].mean()*100:.1f}%)")
print(f"Rule choice accuracy: {df_combined['rule_choice_correct'].sum()}/{df_combined['rule_choice_correct'].notna().sum()} ({df_combined['rule_choice_correct'].mean()*100:.1f}%)")

# Accuracy by round
print(f"\nAccuracy by Round:")
for round_num in sorted(df_combined['round_number'].unique()):
    round_df = df_combined[df_combined['round_number'] == round_num]
    obj_acc = round_df['object_identification_correct'].mean() * 100
    rule_acc = round_df['rule_choice_correct'].mean() * 100
    print(f"  Round {round_num}: Object={obj_acc:.1f}%, Rule={rule_acc:.1f}%")

# Save to CSV
output_file = os.path.join(script_dir, 'main_game_combined_round1_round2.csv')
df_combined.to_csv(output_file, index=False)
print("\n" + "="*80)
print(f"✓ Saved: {output_file}")
print("="*80)

print("\nColumns in CSV:")
for i, col in enumerate(df_combined.columns, 1):
    print(f"  {i}. {col}")

print("\n" + "="*80)
print("COMBINATION COMPLETE")
print("="*80)

