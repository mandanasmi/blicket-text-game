"""
Analysis: Exploration (number of tests) in conjunctive vs disjunctive rounds
Uses paired within-participant analysis with paired t-test
"""

import json
import pandas as pd
import numpy as np
import os

def paired_ttest(x, y):
    """
    Perform paired t-test
    Returns: t-statistic, p-value
    """
    # Try to use scipy if available (recommended)
    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(x, y)
        return t_stat, p_value
    except ImportError:
        # Manual calculation if scipy not available
        # Note: This is a simplified approximation. For accurate p-values, install scipy:
        # pip install scipy
        differences = np.array(x) - np.array(y)
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)  # Sample standard deviation
        
        if std_diff == 0:
            return 0.0, 1.0
        
        se = std_diff / np.sqrt(n)
        t_stat = mean_diff / se
        
        # Simple approximation for p-value
        # For more accurate results, install scipy
        abs_t = abs(t_stat)
        df = n - 1
        
        # Rough approximation - install scipy for exact values
        if abs_t > 3.5:
            p_value = 0.001
        elif abs_t > 3.0:
            p_value = 0.005
        elif abs_t > 2.5:
            p_value = 0.02
        elif abs_t > 2.0:
            p_value = 0.05
        elif abs_t > 1.5:
            p_value = 0.15
        else:
            p_value = 0.5
        
        return t_stat, p_value

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
    
    if len(participant_id) >= 20 and participant_id.replace('_', '').replace('-', '').isalnum():
        return True
    
    return False

# Load Firebase data
print("Loading Firebase data...")
# Note: Run this script from the analysis/ directory
with open('firebase_data.json', 'r') as f:
    all_data = json.load(f)

# Filter for Prolific IDs
prolific_data = {pid: pdata for pid, pdata in all_data.items() if is_prolific_id(pid)}
print(f"Found {len(prolific_data)} Prolific participants\n")

print("="*80)
print("RULE TYPE EXPLORATION ANALYSIS")
print("="*80)
print("\nComparing number of tests in conjunctive vs disjunctive rounds")
print("Using paired within-participant analysis\n")

results = []

for participant_id, participant_data in prolific_data.items():
    prolific_id = participant_data.get('demographics', {}).get('prolific_id', participant_id)
    
    if not isinstance(participant_data, dict):
        continue
    
    main_game = participant_data.get('main_game', {})
    if not main_game or not isinstance(main_game, dict):
        continue
    
    # Collect test counts by rule type
    conjunctive_tests = []
    disjunctive_tests = []
    
    for round_key, round_data in main_game.items():
        if not isinstance(round_data, dict) or not round_key.startswith('round_'):
            continue
        
        # Get rule type
        true_rule = round_data.get('true_rule', '').lower()
        
        # Get number of tests
        state_history = round_data.get('state_history', [])
        num_tests = len(state_history)
        
        if true_rule == 'conjunctive':
            conjunctive_tests.append(num_tests)
        elif true_rule == 'disjunctive':
            disjunctive_tests.append(num_tests)
    
    # Calculate means within participant
    if conjunctive_tests and disjunctive_tests:
        mean_conjunctive = np.mean(conjunctive_tests)
        mean_disjunctive = np.mean(disjunctive_tests)
        difference = mean_conjunctive - mean_disjunctive
        
        # Also calculate ratio for normalization
        ratio = mean_conjunctive / mean_disjunctive if mean_disjunctive > 0 else np.nan
        
        results.append({
            'participant_id': prolific_id,
            'conjunctive_rounds': len(conjunctive_tests),
            'disjunctive_rounds': len(disjunctive_tests),
            'conjunctive_tests': conjunctive_tests,
            'disjunctive_tests': disjunctive_tests,
            'mean_conjunctive': mean_conjunctive,
            'mean_disjunctive': mean_disjunctive,
            'difference': difference,
            'ratio': ratio
        })

# Create DataFrame
df = pd.DataFrame(results)

if len(df) == 0:
    print("ERROR: No data found!")
    exit(1)

# Save detailed results
os.makedirs('results', exist_ok=True)
df.to_csv('results/rule_exploration_comparison.csv', index=False)

print(f"\nTotal participants: {len(df)}")
print(f"\n--- Descriptive Statistics ---")
print(f"Mean tests (conjunctive): {df['mean_conjunctive'].mean():.2f} (SD={df['mean_conjunctive'].std():.2f})")
print(f"Mean tests (disjunctive): {df['mean_disjunctive'].mean():.2f} (SD={df['mean_disjunctive'].std():.2f})")
print(f"Mean difference (conj - disj): {df['difference'].mean():.2f} (SD={df['difference'].std():.2f})")
print(f"Mean ratio (conj/disj): {df['ratio'].mean():.2f} (SD={df['ratio'].std():.2f})")

print(f"\n--- Individual Participant Data ---")
for _, row in df.iterrows():
    print(f"{row['participant_id']}: conj={row['mean_conjunctive']:.1f}, disj={row['mean_disjunctive']:.1f}, diff={row['difference']:.1f}")

# Paired t-test
print(f"\n--- Statistical Analysis ---")
print(f"\nPaired t-test: Conjunctive vs Disjunctive")
print(f"Null hypothesis: Mean difference = 0")

t_stat, p_value = paired_ttest(df['mean_conjunctive'], df['mean_disjunctive'])

print(f"\nt-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {len(df) - 1}")

# Check if using approximation
try:
    from scipy import stats
    scipy_available = True
except ImportError:
    scipy_available = False
    print("\nNote: p-value is approximated. For exact p-values, install scipy: pip install scipy")

if p_value < 0.001:
    significance = "***"
elif p_value < 0.01:
    significance = "**"
elif p_value < 0.05:
    significance = "*"
else:
    significance = "ns"

print(f"\nSignificance: {significance}")

# Effect size (Cohen's d for paired samples)
mean_diff = df['difference'].mean()
std_diff = df['difference'].std()
cohens_d = mean_diff / std_diff if std_diff > 0 else 0

print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")

# Interpretation
if p_value < 0.05:
    if mean_diff > 0:
        print(f"\n✓ Participants took significantly MORE tests in conjunctive rounds")
        print(f"  (mean difference: {mean_diff:.2f} more tests)")
    else:
        print(f"\n✓ Participants took significantly MORE tests in disjunctive rounds")
        print(f"  (mean difference: {abs(mean_diff):.2f} more tests)")
else:
    print(f"\n✗ No significant difference in number of tests between rule types")

# Additional statistics
print(f"\n--- Additional Statistics ---")
print(f"Participants with more tests in conjunctive: {(df['difference'] > 0).sum()}/{len(df)}")
print(f"Participants with more tests in disjunctive: {(df['difference'] < 0).sum()}/{len(df)}")
print(f"Participants with equal tests: {(df['difference'] == 0).sum()}/{len(df)}")

print(f"\n✓ Saved: results/rule_exploration_comparison.csv")
print("\nColumns:")
print("  - participant_id: Prolific ID")
print("  - conjunctive_rounds: Number of conjunctive rounds for this participant")
print("  - disjunctive_rounds: Number of disjunctive rounds for this participant")
print("  - conjunctive_tests: List of test counts for conjunctive rounds")
print("  - disjunctive_tests: List of test counts for disjunctive rounds")
print("  - mean_conjunctive: Mean number of tests in conjunctive rounds")
print("  - mean_disjunctive: Mean number of tests in disjunctive rounds")
print("  - difference: Mean conjunctive - Mean disjunctive")
print("  - ratio: Mean conjunctive / Mean disjunctive")

print("\n" + "="*80)

