"""
Generate comprehensive CSV from firebase_data2.json
Extracts comprehension phase data and main game rounds 1-3 data for each participant
"""

import json
import pandas as pd
import numpy as np
import os


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
            return sorted(chosen) if chosen else None
    
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


def extract_comprehension_data(participant_data):
    """Extract comprehension phase data"""
    comprehension = participant_data.get('comprehension', {})
    if not isinstance(comprehension, dict):
        return {
            'num_tests': None,
            'blicket_correct': None,
            'prior_experience': None
        }
    
    # Number of tests
    state_history = comprehension.get('state_history', [])
    num_tests = len(state_history) if isinstance(state_history, list) else None
    
    # Check if blicket correct
    true_blicket_indices = comprehension.get('true_blicket_indices', [])
    blicket_classifications = comprehension.get('blicket_classifications', {})
    
    user_chosen = []
    if isinstance(blicket_classifications, dict):
        for obj_key, classification in blicket_classifications.items():
            if classification == "Yes":
                try:
                    obj_idx = int(obj_key.split('_')[1])
                    user_chosen.append(obj_idx)
                except (ValueError, IndexError):
                    pass
    
    blicket_correct = None
    if true_blicket_indices and user_chosen:
        blicket_correct = set(true_blicket_indices) == set(user_chosen)
    
    # Prior experience
    prior_experience = has_prior_experience(participant_data)
    
    return {
        'num_tests': num_tests,
        'blicket_correct': blicket_correct,
        'prior_experience': prior_experience
    }


def extract_round_data(round_data):
    """Extract data for a single round"""
    if not isinstance(round_data, dict):
        return {
            'num_tests': None,
            'test_times': None,
            'total_time': None,
            'true_blickets': None,
            'chosen_blickets': None,
            'inferred_rule': None,
            'chosen_rule': None
        }
    
    # Number of tests
    state_history = round_data.get('state_history', [])
    num_tests = len(state_history) if isinstance(state_history, list) else None
    
    # Test times
    test_timings = round_data.get('test_timings', [])
    test_times_list = []
    if isinstance(test_timings, list):
        for timing in test_timings:
            if isinstance(timing, dict):
                time_val = timing.get('time_since_previous_seconds')
                if time_val is not None:
                    test_times_list.append(str(round(time_val, 3)))
    test_times = ','.join(test_times_list) if test_times_list else None
    
    # Total time
    total_time = round_data.get('total_time_seconds')
    if total_time is not None:
        total_time = round(total_time, 3)
    
    # True blickets
    true_blickets = round_data.get('true_blicket_indices', [])
    true_blickets_str = str(sorted(true_blickets)) if true_blickets else None
    
    # User chosen blickets
    user_chosen_blickets = get_user_chosen_blickets(round_data)
    chosen_blickets_str = str(user_chosen_blickets) if user_chosen_blickets is not None else None
    
    # Inferred rule (user's rule)
    rule_type_str = round_data.get('rule_type', '')
    inferred_rule = parse_rule_type(rule_type_str)
    
    # Chosen rule (same as inferred rule, or could be true_rule)
    chosen_rule = inferred_rule
    
    return {
        'num_tests': num_tests,
        'test_times': test_times,
        'total_time': total_time,
        'true_blickets': true_blickets_str,
        'chosen_blickets': chosen_blickets_str,
        'inferred_rule': inferred_rule,
        'chosen_rule': chosen_rule
    }


def main():
    print("="*80)
    print("COMPREHENSIVE CSV GENERATION")
    print("="*80)
    print("\nLoading firebase_data2.json...")
    
    try:
        with open('firebase_data2.json', 'r') as f:
            content = f.read().strip()
            if not content:
                print("ERROR: firebase_data2.json is empty!")
                exit(1)
            all_data = json.loads(content)
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
    
    # Collect data for each participant
    results = []
    
    for participant_id, participant_data in prolific_data.items():
        if not isinstance(participant_data, dict):
            continue
        
        prolific_id = get_prolific_id(participant_data)
        if not prolific_id:
            prolific_id = participant_id
        
        # Extract comprehension data
        comp_data = extract_comprehension_data(participant_data)
        
        # Extract main game rounds
        main_game = participant_data.get('main_game', {})
        if not isinstance(main_game, dict):
            main_game = {}
        
        # Find rounds by iterating through main_game
        # Rounds can be keyed by 'round_1', 'round_2', etc. or have round_number field
        round1_data = extract_round_data({})
        round2_data = extract_round_data({})
        round3_data = extract_round_data({})
        
        for round_key, round_data in main_game.items():
            if not isinstance(round_data, dict) or not round_key.startswith('round_'):
                continue
            
            round_number = round_data.get('round_number', 0)
            if round_number == 1:
                round1_data = extract_round_data(round_data)
            elif round_number == 2:
                round2_data = extract_round_data(round_data)
            elif round_number == 3:
                round3_data = extract_round_data(round_data)
        
        # Build row
        row = {
            'id': prolific_id,
            'comprehension_num_tests': comp_data['num_tests'],
            'comprehension_blicket_correct': comp_data['blicket_correct'],
            'comprehension_prior_experience': comp_data['prior_experience'],
            'round1_num_tests': round1_data['num_tests'],
            'round1_test_times': round1_data['test_times'],
            'round1_total_time': round1_data['total_time'],
            'round1_true_blickets': round1_data['true_blickets'],
            'round1_chosen_blickets': round1_data['chosen_blickets'],
            'round1_inferred_rule': round1_data['inferred_rule'],
            'round1_chosen_rule': round1_data['chosen_rule'],
            'round2_num_tests': round2_data['num_tests'],
            'round2_test_times': round2_data['test_times'],
            'round2_total_time': round2_data['total_time'],
            'round2_true_blickets': round2_data['true_blickets'],
            'round2_chosen_blickets': round2_data['chosen_blickets'],
            'round2_inferred_rule': round2_data['inferred_rule'],
            'round2_chosen_rule': round2_data['chosen_rule'],
            'round3_num_tests': round3_data['num_tests'],
            'round3_test_times': round3_data['test_times'],
            'round3_total_time': round3_data['total_time'],
            'round3_true_blickets': round3_data['true_blickets'],
            'round3_chosen_blickets': round3_data['chosen_blickets'],
            'round3_inferred_rule': round3_data['inferred_rule'],
            'round3_chosen_rule': round3_data['chosen_rule']
        }
        
        results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("\nERROR: No valid data found!")
        exit(1)
    
    print(f"\nTotal participants: {len(df)}")
    print(f"Participants with comprehension data: {df['comprehension_num_tests'].notna().sum()}")
    print(f"Participants with round 1 data: {df['round1_num_tests'].notna().sum()}")
    print(f"Participants with round 2 data: {df['round2_num_tests'].notna().sum()}")
    print(f"Participants with round 3 data: {df['round3_num_tests'].notna().sum()}")
    
    # Save to CSV
    os.makedirs('results', exist_ok=True)
    output_file = 'results/comprehensive_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
