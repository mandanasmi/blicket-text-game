"""
Extract main game data from data.json (round4)
and create a CSV file with comprehensive test and accuracy information
INCLUDING prior experience information
"""

import json
import pandas as pd
import numpy as np

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

def parse_rule_type(rule_type_str):
    """Parse rule type from string to 'conjunctive' or 'disjunctive'"""
    if not rule_type_str or not isinstance(rule_type_str, str):
        return None
    
    rule_lower = rule_type_str.lower()
    if 'conjunctive' in rule_lower:
        return 'conjunctive'
    elif 'disjunctive' in rule_lower or 'any' in rule_lower:
        return 'disjunctive'
    return None

def get_user_chosen_blickets(round_data):
    """Extract user's chosen blickets from round data"""
    # Try user_chosen_blickets first
    if 'user_chosen_blickets' in round_data:
        chosen = round_data['user_chosen_blickets']
        if isinstance(chosen, list):
            # Filter out None values
            chosen = [x for x in chosen if x is not None]
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
print("EXTRACTING MAIN GAME DATA TO CSV (WITH PRIOR EXPERIENCE) - ROUND 4")
print("="*80)
print("\nLoading data.json...")

try:
    with open('data.json', 'r') as f:
        all_data = json.load(f)
except FileNotFoundError:
    print("ERROR: data.json not found!")
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
    
    # Check prior experience
    has_prior = has_prior_experience(participant_data)
    
    # In round4, main_game is a single object, not nested rounds
    main_game = participant_data.get('main_game', {})
    if not main_game or not isinstance(main_game, dict):
        continue
    
    # Check if main_game has rounds or is a single game
    # First check if it has nested rounds (like round2 structure)
    has_nested_rounds = False
    for key in main_game.keys():
        if isinstance(key, str) and key.startswith('round_'):
            has_nested_rounds = True
            break
    
    if has_nested_rounds:
        # Process nested rounds (like round2 structure)
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
                'has_prior_experience': has_prior,
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
                'rule_choice_correct': rule_accuracy
            })
    else:
        # Process single main_game (round4 structure)
        # Treat as round_number = 1
        round_data = main_game
        round_number = 1
        
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
            'has_prior_experience': has_prior,
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
            'rule_choice_correct': rule_accuracy
        })

# Create DataFrame
df = pd.DataFrame(results)

if len(df) == 0:
    print("\nERROR: No valid round data found!")
    exit(1)

# Sort by user_id and round_number
df = df.sort_values(['user_id', 'round_number']).reset_index(drop=True)

print(f"\nTotal rounds extracted: {len(df)}")
print(f"Unique participants: {df['user_id'].nunique()}")

# Prior experience summary
print("\n" + "="*80)
print("PRIOR EXPERIENCE SUMMARY")
print("="*80)
prior_counts = df.groupby('user_id')['has_prior_experience'].first().value_counts()
total_participants = df['user_id'].nunique()
print(f"Total participants: {total_participants}")
if True in prior_counts.index:
    yes_count = prior_counts[True]
    print(f"  - Has prior experience: {yes_count} ({yes_count/total_participants*100:.1f}%)")
if False in prior_counts.index:
    no_count = prior_counts[False]
    print(f"  - No prior experience: {no_count} ({no_count/total_participants*100:.1f}%)")

# Display sample data
print("\n" + "="*80)
print("SAMPLE DATA (first 5 rows)")
print("="*80)
# Display with selected columns for readability
display_cols = ['user_id', 'has_prior_experience', 'round_number', 'num_tests', 
                'object_identification_correct', 'rule_choice_correct']
print(df[display_cols].head(10).to_string())

# Calculate some summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nTotal tests across all rounds: {df['num_tests'].sum()}")
print(f"Average tests per round: {df['num_tests'].mean():.2f}")
print(f"Median tests per round: {df['num_tests'].median():.0f}")
print(f"Min tests per round: {df['num_tests'].min()}")
print(f"Max tests per round: {df['num_tests'].max()}")

print(f"\nObject identification accuracy: {df['object_identification_correct'].sum()}/{df['object_identification_correct'].notna().sum()} ({df['object_identification_correct'].mean()*100:.1f}%)")
print(f"Rule choice accuracy: {df['rule_choice_correct'].sum()}/{df['rule_choice_correct'].notna().sum()} ({df['rule_choice_correct'].mean()*100:.1f}%)")

# Accuracy by round
print(f"\nAccuracy by Round:")
for round_num in sorted(df['round_number'].unique()):
    round_df = df[df['round_number'] == round_num]
    obj_acc = round_df['object_identification_correct'].mean() * 100
    rule_acc = round_df['rule_choice_correct'].mean() * 100
    print(f"  Round {round_num}: Object={obj_acc:.1f}%, Rule={rule_acc:.1f}%")

# Accuracy by rule type
print(f"\nAccuracy by Ground Truth Rule:")
for rule_type in ['conjunctive', 'disjunctive']:
    rule_df = df[df['ground_truth_rule'] == rule_type]
    if len(rule_df) > 0:
        obj_acc = rule_df['object_identification_correct'].mean() * 100
        rule_acc = rule_df['rule_choice_correct'].mean() * 100
        print(f"  {rule_type.capitalize()}: Object={obj_acc:.1f}%, Rule={rule_acc:.1f}%")

# Accuracy by prior experience
print(f"\nAccuracy by Prior Experience:")
for has_prior in [True, False]:
    prior_df = df[df['has_prior_experience'] == has_prior]
    if len(prior_df) > 0:
        obj_acc = prior_df['object_identification_correct'].mean() * 100
        rule_acc = prior_df['rule_choice_correct'].mean() * 100
        label = "Has prior experience" if has_prior else "No prior experience"
        print(f"  {label}: Object={obj_acc:.1f}%, Rule={rule_acc:.1f}%")

# Save to CSV
output_file = 'main_game_data_with_prior_experience.csv'
df.to_csv(output_file, index=False)
print("\n" + "="*80)
print(f"✓ Saved: {output_file}")
print("="*80)

print("\nColumns in CSV:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print("\n" + "="*80)
print("EXTRACTION COMPLETE")
print("="*80)
