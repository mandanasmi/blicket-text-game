"""
Generate simplified CSV with only rule inference text per user
"""

import pandas as pd

# Load rule inference data
rule_df = pd.read_csv('results/4_rule_inference.csv')

print("="*80)
print("RULE INFERENCE TEXT - SIMPLIFIED VIEW")
print("="*80)

# Create simplified view with just the text
results = []

for idx, row in rule_df.iterrows():
    results.append({
        'participant_id': row['participant_id'],
        'round': row['round'],
        'ground_truth_rule': row['ground_truth_rule'],
        'user_written_hypothesis': row['user_written_hypothesis']
    })

# Create DataFrame
df = pd.DataFrame(results)

# Save to CSV
df.to_csv('results/6_rule_text_only.csv', index=False)

print(f"\nTotal entries: {len(df)}")
print(f"Participants: {df['participant_id'].nunique()}")
print(f"Rounds per participant: {len(df) // df['participant_id'].nunique()}")

print("\n--- Sample Hypotheses ---")
for idx, row in df.head(6).iterrows():
    print(f"\n{row['participant_id'][:10]}... Round {row['round']} ({row['ground_truth_rule']}):")
    print(f"  \"{row['user_written_hypothesis']}\"")

print("\n✓ Saved: results/6_rule_text_only.csv")

print("\n" + "="*80)
print("CSV COLUMNS")
print("="*80)
print("  - participant_id: Prolific ID")
print("  - round: Round number (1, 2, 3)")
print("  - ground_truth_rule: True rule type")
print("  - user_written_hypothesis: What the user typed as their rule explanation")

print("\n" + "="*80)

