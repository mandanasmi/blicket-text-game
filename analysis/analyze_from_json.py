"""
Analysis script that works with exported Firebase JSON data
No Firebase credentials needed - just export your data as JSON
"""

import json
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

def num_to_letter(obj_num):
    """Convert object number (0, 1, 2) to letter (A, B, C)"""
    if isinstance(obj_num, (int, float)):
        return chr(65 + int(obj_num))  # 0→A, 1→B, 2→C
    return obj_num

def convert_obj_list_to_letters(obj_list):
    """Convert a list of object numbers to letters"""
    if not obj_list:
        return []
    try:
        # Handle string representation of list
        if isinstance(obj_list, str):
            import ast
            obj_list = ast.literal_eval(obj_list)
        # Convert each number to letter
        return [num_to_letter(obj) for obj in obj_list]
    except:
        return obj_list

def is_prolific_id(participant_id):
    """
    Check if participant_id is a Prolific ID
    Prolific IDs are 24-character hexadecimal strings
    """
    if not participant_id or not isinstance(participant_id, str):
        return False
    
    id_lower = participant_id.lower()
    test_patterns = ['test', 'debug', 'demo', 'sample', 'example', 'alice', 'bob', 
                     'charlie', 'user', 'admin', 'dummy', 'fake']
    
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


def load_json_data(filepath='firebase_data.json'):
    """Load exported Firebase JSON data"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"✓ Loaded data from {filepath}")
        return data
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        print("\nPlease export your data from Firebase:")
        print("1. Go to: https://console.firebase.google.com/u/0/project/nexiom-text-game/database")
        print("2. Click the ⋮ menu → Export JSON")
        print(f"3. Save as: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON file: {e}")
        return None


def analyze_comprehension_correctness(data):
    """Analyze comprehension phase with detailed action sequences"""
    print("\n" + "="*80)
    print("ANALYSIS 1: COMPREHENSION PHASE")
    print("="*80)
    
    results = []
    
    for participant_id, participant_data in data.items():
        if not isinstance(participant_data, dict):
            continue
        
        comprehension = participant_data.get('comprehension', {})
        if not comprehension:
            continue
        
        # Get ground truth
        true_blicket_indices = comprehension.get('true_blicket_indices', [])
        
        # Get number of tests
        state_history = comprehension.get('state_history', [])
        num_tests = len(state_history)
        
        # Get action sequence
        action_history = comprehension.get('action_history', [])
        user_test_actions = comprehension.get('user_test_actions', [])
        
        # Build action sequence string
        action_sequence = []
        if user_test_actions:
            for action in user_test_actions:
                if isinstance(action, dict):
                    action_type = action.get('action_type', '')
                    if action_type == 'test':
                        objects = action.get('objects_tested', [])
                        # Convert numbers to letters
                        objects_letters = [num_to_letter(obj) for obj in objects]
                        action_sequence.append(f"Test:{objects_letters}")
                    else:
                        obj_idx = action.get('object_index', '')
                        action_sequence.append(f"{action_type}:obj{obj_idx}")
                elif isinstance(action, str):
                    action_sequence.append(action)
        
        action_sequence_str = " | ".join(action_sequence) if action_sequence else ""
        # Add trailing | after the last Test
        if action_sequence_str:
            action_sequence_str += " |"
        
        # Get user's object selection
        practice_answer = comprehension.get('practice_blicket_question', {})
        answer_text = practice_answer.get('answer_text', '')
        selected_object = practice_answer.get('selected_object_one_based', None)
        
        # Parse from answer text if needed
        if selected_object is None and answer_text:
            if "Object A" in answer_text:
                selected_object = 1  # A = 1 (1-based)
            elif "Object B" in answer_text:
                selected_object = 2
            elif "Object C" in answer_text:
                selected_object = 3
        
        # Get full classifications
        blicket_classifications = comprehension.get('blicket_classifications', {})
        user_chosen = []
        if blicket_classifications:
            for obj_key, classification in blicket_classifications.items():
                if classification == "Yes":
                    obj_idx = int(obj_key.split('_')[1])
                    user_chosen.append(obj_idx)
        
        # If no full classifications, use selected_object
        if not user_chosen and selected_object is not None:
            user_chosen = [selected_object - 1]  # Convert to 0-based
        
        # Check if matches (using numbers for comparison)
        objects_match = set(user_chosen) == set(true_blicket_indices) if true_blicket_indices else None
        
        # Convert to letters for display
        user_chosen_letters = [num_to_letter(obj) for obj in sorted(user_chosen)] if user_chosen else []
        ground_truth_letters = [num_to_letter(obj) for obj in sorted(true_blicket_indices)] if true_blicket_indices else []
        
        results.append({
            'participant_id': participant_id,
            'num_tests': num_tests,
            'action_sequence': action_sequence_str,
            'user_chosen_objects': str(user_chosen_letters) if user_chosen_letters else '[]',
            'ground_truth_objects': str(ground_truth_letters) if ground_truth_letters else '[]',
            'objects_match': objects_match,
            'practice_answer_text': answer_text
        })
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        print(f"\nTotal participants: {len(df)}")
        print(f"Average tests: {df['num_tests'].mean():.1f}")
        print(f"Test range: {df['num_tests'].min():.0f} - {df['num_tests'].max():.0f}")
        
        correct_count = df['objects_match'].sum()
        total_with_answer = df['objects_match'].notna().sum()
        if total_with_answer > 0:
            print(f"\nObject Selection Correctness:")
            print(f"  Correct: {correct_count}/{total_with_answer} ({correct_count/total_with_answer*100:.1f}%)")
        
        print("\n--- Sample Data (First 3) ---")
        for idx, row in df.head(3).iterrows():
            print(f"\n{row['participant_id'][:10]}...:")
            print(f"  Tests: {row['num_tests']}")
            print(f"  Chose: {row['user_chosen_objects']}, Truth: {row['ground_truth_objects']}, Match: {row['objects_match']}")
    
    return df


def analyze_exploration(data):
    """Analyze exploration behavior"""
    print("\n" + "="*80)
    print("ANALYSIS 2: EXPLORATION IN TEST PHASES")
    print("="*80)
    
    results = []
    
    for participant_id, participant_data in data.items():
        if not isinstance(participant_data, dict):
            continue
        
        comprehension = participant_data.get('comprehension', {})
        if comprehension:
            state_history = comprehension.get('state_history', [])
            num_tests = len(state_history)
            
            results.append({
                'participant_id': participant_id,
                'phase': 'comprehension',
                'round_number': 0,
                'total_tests': num_tests,
                'tested_more_than_4': num_tests > 4
            })
        
        main_game = participant_data.get('main_game', {})
        if main_game and isinstance(main_game, dict):
            for round_key, round_data in main_game.items():
                if not isinstance(round_data, dict) or not round_key.startswith('round_'):
                    continue
                
                round_number = round_data.get('round_number', 0)
                state_history = round_data.get('state_history', [])
                num_tests = len(state_history)
                
                results.append({
                    'participant_id': participant_id,
                    'phase': 'main_experiment',
                    'round_number': round_number,
                    'total_tests': num_tests,
                    'tested_more_than_4': num_tests > 4
                })
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        print(f"\nTotal test sessions: {len(df)}")
        print(f"\nAverage tests: {df['total_tests'].mean():.2f}")
        print(f"Median tests: {df['total_tests'].median():.1f}")
        print(f"Range: {df['total_tests'].min():.0f} - {df['total_tests'].max():.0f}")
        
        tested_more_than_4 = df['tested_more_than_4'].sum()
        print(f"\nSessions with >4 tests: {tested_more_than_4}/{len(df)} ({tested_more_than_4/len(df)*100:.1f}%)")
    
    return df


def analyze_rule_understanding(data):
    """Analyze rule understanding with detailed object selections and rule hypotheses"""
    print("\n" + "="*80)
    print("ANALYSIS 3: RULE UNDERSTANDING")
    print("="*80)
    
    results = []
    
    for participant_id, participant_data in data.items():
        if not isinstance(participant_data, dict):
            continue
        
        main_game = participant_data.get('main_game', {})
        if main_game and isinstance(main_game, dict):
            for round_key, round_data in main_game.items():
                if not isinstance(round_data, dict) or not round_key.startswith('round_'):
                    continue
                
                round_number = round_data.get('round_number', 0)
                true_rule = round_data.get('true_rule', '')
                rule_type = round_data.get('rule_type', '')
                rule_hypothesis = round_data.get('rule_hypothesis', '')
                
                # Parse user's rule type selection
                user_rule_type = None
                if rule_type:
                    if 'conjunctive' in rule_type.lower():
                        user_rule_type = 'conjunctive'
                    elif 'disjunctive' in rule_type.lower():
                        user_rule_type = 'disjunctive'
                
                # Check if rule type is correct
                rule_type_correct = None
                if user_rule_type and true_rule:
                    rule_type_correct = (user_rule_type == true_rule.lower())
                
                # Get object selections
                true_blickets = round_data.get('true_blicket_indices', [])
                user_chosen_blickets = round_data.get('user_chosen_blickets', [])
                
                # Check if objects match
                blickets_correct = None
                if true_blickets is not None and user_chosen_blickets is not None:
                    blickets_correct = set(true_blickets) == set(user_chosen_blickets)
                
                # Get blicket classifications (which objects were marked Yes/No)
                blicket_classifications = round_data.get('blicket_classifications', {})
                blicket_answers = {}
                for i in range(4):  # Assuming max 4 objects
                    key = f'object_{i}'
                    if key in blicket_classifications:
                        blicket_answers[f'object_{i}_selected'] = blicket_classifications[key]
                
                results.append({
                    'participant_id': participant_id,
                    'round_number': round_number,
                    'true_rule': true_rule,
                    'true_blickets': str(sorted(true_blickets)) if true_blickets else '[]',
                    'user_rule_type': user_rule_type,
                    'user_rule_text': rule_type,
                    'user_chosen_blickets': str(sorted(user_chosen_blickets)) if user_chosen_blickets else '[]',
                    'rule_hypothesis': rule_hypothesis,
                    'rule_type_correct': rule_type_correct,
                    'blickets_correct': blickets_correct,
                    **blicket_answers  # Add individual object selections
                })
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        print(f"\nTotal rounds: {len(df)}")
        
        correct_rule = df['rule_type_correct'].sum()
        total_with_rule = df['rule_type_correct'].notna().sum()
        if total_with_rule > 0:
            print(f"\nRule Type Correct: {correct_rule}/{total_with_rule} ({correct_rule/total_with_rule*100:.1f}%)")
        
        correct_blickets = df['blickets_correct'].sum()
        total_with_blickets = df['blickets_correct'].notna().sum()
        if total_with_blickets > 0:
            print(f"Blicket ID Correct: {correct_blickets}/{total_with_blickets} ({correct_blickets/total_with_blickets*100:.1f}%)")
        
        # Show breakdown by rule type
        print("\n--- Performance by True Rule Type ---")
        for true_rule in df['true_rule'].unique():
            if pd.isna(true_rule) or not true_rule:
                continue
            rule_df = df[df['true_rule'] == true_rule]
            correct = rule_df['rule_type_correct'].sum()
            total = rule_df['rule_type_correct'].notna().sum()
            print(f"\n{true_rule}:")
            print(f"  Rounds: {len(rule_df)}")
            if total > 0:
                print(f"  Rule classification correct: {correct}/{total} ({correct/total*100:.1f}%)")
            
            blicket_correct = rule_df['blickets_correct'].sum()
            blicket_total = rule_df['blickets_correct'].notna().sum()
            if blicket_total > 0:
                print(f"  Object identification correct: {blicket_correct}/{blicket_total} ({blicket_correct/blicket_total*100:.1f}%)")
        
        # Show sample comparisons
        print("\n--- Sample Comparisons (First 5 Rounds) ---")
        for idx, row in df.head(5).iterrows():
            print(f"\nParticipant {row['participant_id'][:8]}... - Round {row['round_number']}:")
            print(f"  True: {row['true_rule']} rule with objects {row['true_blickets']}")
            print(f"  User selected: {row['user_chosen_blickets']} as Nexioms")
            print(f"  User classified as: {row['user_rule_type']}")
            print(f"  Objects match: {row['blickets_correct']}, Rule match: {row['rule_type_correct']}")
            if row['rule_hypothesis']:
                print(f"  Hypothesis: {row['rule_hypothesis'][:80]}{'...' if len(str(row['rule_hypothesis'])) > 80 else ''}")
    
    return df


def main():
    print("="*80)
    print("NEXIOM TEXT GAME - ANALYSIS FROM JSON")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    all_data = load_json_data('firebase_data.json')
    
    if not all_data:
        sys.exit(1)
    
    # Filter for Prolific IDs
    print(f"\nTotal entries: {len(all_data)}")
    prolific_data = {pid: pdata for pid, pdata in all_data.items() if is_prolific_id(pid)}
    excluded = len(all_data) - len(prolific_data)
    
    print(f"Prolific participants: {len(prolific_data)}")
    print(f"Excluded (test users): {excluded}")
    
    if excluded > 0:
        excluded_ids = [pid for pid in all_data.keys() if not is_prolific_id(pid)]
        print(f"Excluded IDs: {excluded_ids}")
    
    if len(prolific_data) == 0:
        print("\nERROR: No Prolific participants found!")
        sys.exit(1)
    
    print(f"\nAnalyzing {len(prolific_data)} participants...")
    
    # Run analyses
    comp_df = analyze_comprehension_correctness(prolific_data)
    
    # Save CSV in results folder
    import os
    os.makedirs('results', exist_ok=True)
    
    comp_df.to_csv('results/1_comprehension_phase.csv', index=False)
    
    print("\n" + "="*80)
    print("CSV GENERATED")
    print("="*80)
    print("\nSaved: results/1_comprehension_phase.csv")
    print("\nColumns:")
    print("  - participant_id: Prolific ID")
    print("  - num_tests: Number of times 'Test' was pressed")
    print("  - action_sequence: Sequence of all actions")
    print("  - user_chosen_objects: Objects selected as Nexioms")
    print("  - ground_truth_objects: True Nexiom objects")
    print("  - objects_match: True/False if selection matches ground truth")
    print("  - practice_answer_text: Free-text answer")


if __name__ == "__main__":
    main()

