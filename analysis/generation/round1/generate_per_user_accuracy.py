"""
Generate per-user rule inference accuracy
"""

import pandas as pd

# Load rule inference data
rule_df = pd.read_csv('results/4_rule_inference.csv')

print("="*80)
print("PER-USER RULE INFERENCE ACCURACY")
print("="*80)

results = []

for participant_id in rule_df['participant_id'].unique():
    user_data = rule_df[rule_df['participant_id'] == participant_id]
    
    # Count rounds
    total_rounds = len(user_data)
    
    # Checkbox accuracy (main metric)
    checkbox_correct = user_data['checkbox_matches_truth'].sum()
    checkbox_total = user_data['checkbox_matches_truth'].notna().sum()
    checkbox_accuracy = (checkbox_correct / checkbox_total * 100) if checkbox_total > 0 else None
    
    # Hypothesis accuracy (text-based inference) - not available in current CSV
    hypothesis_correct = None
    hypothesis_total = 0
    hypothesis_accuracy = None
    
    # Get rule breakdown by round
    round_details = []
    for _, row in user_data.iterrows():
        round_num = row['round']
        truth = row['ground_truth_rule']
        user_choice = row['user_checkbox_selection']
        is_correct = row['checkbox_matches_truth']
        round_details.append(f"R{round_num}:{truth[0].upper()}→{user_choice[0].upper() if user_choice else '?'}:{'Y' if is_correct else 'N'}")
    
    round_summary = " | ".join(round_details)
    
    results.append({
        'participant_id': participant_id,
        'total_rounds': total_rounds,
        'checkbox_correct': checkbox_correct,
        'checkbox_total': checkbox_total,
        'checkbox_accuracy_percent': round(checkbox_accuracy, 1) if checkbox_accuracy is not None else None,
        'hypothesis_correct': hypothesis_correct,
        'hypothesis_total': hypothesis_total,
        'hypothesis_accuracy_percent': round(hypothesis_accuracy, 1) if hypothesis_accuracy is not None else None,
        'round_by_round_summary': round_summary,
        'accuracy': f"{round(checkbox_accuracy, 1)}%" if checkbox_accuracy is not None else None
    })

# Create DataFrame
df = pd.DataFrame(results)

# Sort by accuracy
df = df.sort_values('checkbox_accuracy_percent', ascending=False)

# Save to CSV
df.to_csv('results/5_per_user_rule_accuracy.csv', index=False)

print(f"\nTotal participants: {len(df)}")
print(f"\n--- Overall Statistics ---")
print(f"Average accuracy: {df['checkbox_accuracy_percent'].mean():.1f}%")
print(f"Median accuracy: {df['checkbox_accuracy_percent'].median():.1f}%")
print(f"Range: {df['checkbox_accuracy_percent'].min():.1f}% - {df['checkbox_accuracy_percent'].max():.1f}%")
# Count participants with 100% accuracy
perfect_count = (df['checkbox_accuracy_percent'] == 100.0).sum()
print(f"Participants with 100% accuracy: {perfect_count}/{len(df)}")

print("\n--- Individual Participant Accuracy ---")
for idx, row in df.iterrows():
    print(f"\n{row['participant_id'][:10]}...:")
    print(f"  Accuracy: {row['checkbox_correct']}/{row['checkbox_total']} ({row['checkbox_accuracy_percent']:.1f}%)")
    print(f"  Rounds: {row['round_by_round_summary']}")

print("\n✓ Saved: results/5_per_user_rule_accuracy.csv")

print("\n" + "="*80)
print("CSV COLUMNS")
print("="*80)
print("  - participant_id: Prolific ID")
print("  - total_rounds: Number of rounds completed")
print("  - checkbox_correct: Number of correct rule classifications")
print("  - checkbox_total: Total rounds with rule classification")
print("  - checkbox_accuracy_percent: Percentage correct")
print("  - hypothesis_correct: Correct rules inferred from written text")
print("  - hypothesis_total: Total clear written hypotheses")
print("  - hypothesis_accuracy_percent: Text-based accuracy")
print("  - round_by_round_summary: Summary of each round (R1:C→C:Y)")
print("  - accuracy: Accuracy percentage (e.g., '100.0%', '66.7%')")
print("\n" + "="*80)

