"""
Generate exploration metrics CSV for comprehension phase
"""

import pandas as pd
import numpy as np

# Load the comprehension phase data
df = pd.read_csv('results/1_comprehension_phase.csv')

print("="*80)
print("COMPREHENSION PHASE EXPLORATION ANALYSIS")
print("="*80)

# Calculate metrics
total_participants = len(df)
num_tests = df['num_tests']
average_tests = num_tests.mean()
median_tests = num_tests.median()

# Check who tested more than 4 times
tested_more_than_4 = (num_tests > 4).sum()
percentage_more_than_4 = (tested_more_than_4 / total_participants) * 100

print(f"\nTotal Participants: {total_participants}")
print(f"Average Tests: {average_tests:.2f}")
print(f"Median Tests: {median_tests:.1f}")
print(f"Tested >4 times: {tested_more_than_4}/{total_participants} ({percentage_more_than_4:.1f}%)")
print(f"Range: {num_tests.min():.0f} - {num_tests.max():.0f}")

# Create individual CSV
exploration_data = []
for idx, row in df.iterrows():
    exploration_data.append({
        'participant_id': row['participant_id'],
        'num_tests': row['num_tests'],
        'tested_more_than_4': 'Yes' if row['num_tests'] > 4 else 'No',
        'exploration_level': 'High' if row['num_tests'] >= 5 else ('Medium' if row['num_tests'] >= 3 else 'Low')
    })

exploration_df = pd.DataFrame(exploration_data)

# Save individual participant data
exploration_df.to_csv('results/2_comprehension_exploration.csv', index=False)
print(f"\n✓ Saved: results/2_comprehension_exploration.csv")

# Create summary CSV with overall statistics
summary_data = {
    'metric': [
        'total_participants',
        'average_tests',
        'median_tests',
        'min_tests',
        'max_tests',
        'std_tests',
        'tested_more_than_4_count',
        'tested_more_than_4_percentage',
        'tested_4_or_less_count',
        'tested_4_or_less_percentage'
    ],
    'value': [
        total_participants,
        round(average_tests, 2),
        median_tests,
        num_tests.min(),
        num_tests.max(),
        round(num_tests.std(), 2),
        tested_more_than_4,
        round(percentage_more_than_4, 1),
        total_participants - tested_more_than_4,
        round(100 - percentage_more_than_4, 1)
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('results/2_comprehension_exploration_summary.csv', index=False)
print(f"✓ Saved: results/2_comprehension_exploration_summary.csv")

print("\n" + "="*80)
print("FILES GENERATED")
print("="*80)
print("\n1. results/2_comprehension_exploration.csv")
print("   Columns:")
print("   - participant_id: Prolific ID")
print("   - num_tests: Number of tests performed")
print("   - tested_more_than_4: Yes/No")
print("   - exploration_level: High/Medium/Low")

print("\n2. results/2_comprehension_exploration_summary.csv")
print("   Summary statistics including:")
print("   - Total participants")
print("   - Average and median tests")
print("   - Count and percentage who tested >4 times")

# Show breakdown
print("\n" + "="*80)
print("EXPLORATION BREAKDOWN")
print("="*80)
print(f"\nHigh Explorers (≥5 tests): {(num_tests >= 5).sum()} participants")
print(f"Medium Explorers (3-4 tests): {((num_tests >= 3) & (num_tests < 5)).sum()} participants")
print(f"Low Explorers (<3 tests): {(num_tests < 3).sum()} participants")

print("\n" + "="*80)

