"""
Create a filtered JSON file containing only human participants
without prior experience from data.json.
Keeps exactly 51 conjunctive and 51 disjunctive cases (balanced).
"""

import json
import random

# Load the original data
print("Loading data.json...")
with open('data.json', 'r') as f:
    data = json.load(f)

print(f"Total users in data.json: {len(data)}")

# Filter participants
filtered_data = {}
excluded_count = 0
no_comprehension_count = 0
no_main_game_count = 0

for user_id, user_data in data.items():
    # Check if user has comprehension phase with similar_game_experience
    if 'comprehension' not in user_data:
        no_comprehension_count += 1
        continue
    
    comprehension = user_data['comprehension']
    
    # Check if similar_game_experience exists
    if 'similar_game_experience' not in comprehension:
        # No experience info, assume no prior experience
        has_prior = False
    else:
        answer = comprehension['similar_game_experience'].get('answer', 'No')
        has_prior = answer.lower() == 'yes'
    
    # Check if user has main_game data
    if 'main_game' not in user_data:
        no_main_game_count += 1
        continue
    
    # Filter out users with prior experience
    if has_prior:
        excluded_count += 1
        continue
    
    # Include this user
    filtered_data[user_id] = user_data

print(f"\nFiltering results:")
print(f"  Users without comprehension phase: {no_comprehension_count}")
print(f"  Users without main_game data: {no_main_game_count}")
print(f"  Users with prior experience (excluded): {excluded_count}")
print(f"  Users included (no prior experience): {len(filtered_data)}")

# Split by rule type
conj_user_ids = []
disj_user_ids = []

for user_id, user_data in filtered_data.items():
    if 'main_game' not in user_data:
        continue
    main_game = user_data['main_game']
    rule = main_game.get('rule') or main_game.get('true_rule', 'unknown')
    if rule == 'conjunctive':
        conj_user_ids.append(user_id)
    elif rule == 'disjunctive':
        disj_user_ids.append(user_id)

print(f"\nBefore balancing: conjunctive={len(conj_user_ids)}, disjunctive={len(disj_user_ids)}")

# Sample exactly 51 per rule type (fixed seed for reproducibility)
TARGET_PER_RULE = 51
random.seed(42)

conj_selected = random.sample(conj_user_ids, min(TARGET_PER_RULE, len(conj_user_ids)))
disj_selected = random.sample(disj_user_ids, min(TARGET_PER_RULE, len(disj_user_ids)))

balanced_data = {
    uid: filtered_data[uid] for uid in conj_selected + disj_selected
}

# Save balanced data
output_file = 'human_active_data_no_prior_experience.json'
with open(output_file, 'w') as f:
    json.dump(balanced_data, f, indent=2)

print(f"\nSaved balanced data to: {output_file}")
print(f"  Conjunctive: {len(conj_selected)}")
print(f"  Disjunctive: {len(disj_selected)}")
print(f"  Total: {len(balanced_data)}")
