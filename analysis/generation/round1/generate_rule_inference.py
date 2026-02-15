"""
Generate rule inference analysis CSV for main game
Focus on user's written hypothesis and checkbox selection vs ground truth
"""

import json
import pandas as pd

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
print("RULE INFERENCE ANALYSIS")
print("="*80)

results = []

for participant_id, participant_data in prolific_data.items():
    if not isinstance(participant_data, dict):
        continue
    
    main_game = participant_data.get('main_game', {})
    if not main_game or not isinstance(main_game, dict):
        continue
    
    for round_key, round_data in main_game.items():
        if not isinstance(round_data, dict) or not round_key.startswith('round_'):
            continue
        
        round_number = round_data.get('round_number', 0)
        
        # Get ground truth
        ground_truth_rule = round_data.get('true_rule', '')
        
        # Get user's written hypothesis
        rule_hypothesis = round_data.get('rule_hypothesis', '')
        
        # Get user's checkbox selection
        rule_type_selection = round_data.get('rule_type', '')
        
        # Parse the checkbox selection
        user_rule_checkbox = None
        if rule_type_selection:
            if 'conjunctive' in rule_type_selection.lower():
                user_rule_checkbox = 'conjunctive'
            elif 'disjunctive' in rule_type_selection.lower():
                user_rule_checkbox = 'disjunctive'
        
        # Check if checkbox matches ground truth
        checkbox_matches_truth = None
        if user_rule_checkbox and ground_truth_rule:
            checkbox_matches_truth = (user_rule_checkbox == ground_truth_rule.lower())
        
        results.append({
            'participant_id': participant_id,
            'round': round_number,
            'ground_truth_rule': ground_truth_rule,
            'user_written_hypothesis': rule_hypothesis,
            'user_checkbox_selection': user_rule_checkbox,
            'checkbox_matches_truth': checkbox_matches_truth
        })

# Create DataFrame
df = pd.DataFrame(results)

# Save to CSV
df.to_csv('results/4_rule_inference.csv', index=False)

print(f"\nTotal rows: {len(df)} (one per participant per round)")
print(f"Participants: {df['participant_id'].nunique()}")

# Summary statistics
print("\n--- Checkbox Selection Accuracy ---")
correct_checkbox = df['checkbox_matches_truth'].sum()
total_checkbox = df['checkbox_matches_truth'].notna().sum()
print(f"Correct: {correct_checkbox}/{total_checkbox} ({correct_checkbox/total_checkbox*100:.1f}%)")

# Sample hypotheses
print("\n--- Sample Hypotheses (First 5) ---")
for idx, row in df.head(5).iterrows():
    print(f"\nParticipant {row['participant_id'][:8]}... Round {row['round']}:")
    print(f"  Ground Truth: {row['ground_truth_rule']}")
    print(f"  Checkbox: {row['user_checkbox_selection']}, Match: {row['checkbox_matches_truth']}")
    print(f"  Written: {row['user_written_hypothesis'][:80]}...")

print("\n✓ Saved: results/4_rule_inference.csv")

print("\n" + "="*80)
print("CSV COLUMNS")
print("="*80)
print("  - participant_id: Prolific ID")
print("  - round: Round number (1, 2, 3)")
print("  - ground_truth_rule: True rule (conjunctive/disjunctive)")
print("  - user_written_hypothesis: Free-text hypothesis user typed")
print("  - user_checkbox_selection: What they chose (conjunctive/disjunctive)")
print("  - checkbox_matches_truth: True/False if checkbox matches ground truth")

print("\n" + "="*80)

