"""
Visualize comprehensive data from comprehensive_data.csv
Shows comprehension phase and main game rounds 1-3 data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import json
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# Load data
df = pd.read_csv('results/comprehensive_data.csv')

print("="*80)
print("COMPREHENSIVE DATA VISUALIZATION")
print("="*80)
print(f"\nLoaded {len(df)} participants")

# Helper function to parse list strings
def parse_list_str(s):
    """Parse string representation of list"""
    if pd.isna(s) or s is None:
        return []
    try:
        return ast.literal_eval(str(s))
    except:
        return []

# Create figure with subplots
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

# ============================================================================
# COMPREHENSION PHASE VISUALIZATIONS
# ============================================================================

# 1. Comprehension: Number of tests distribution
ax1 = fig.add_subplot(gs[0, 0])
if df['comprehension_num_tests'].notna().sum() > 0:
    valid_tests = df['comprehension_num_tests'].dropna()
    ax1.hist(valid_tests, bins=range(0, int(valid_tests.max())+2), 
             edgecolor='black', color='#2196F3', alpha=0.7, rwidth=0.8)
    ax1.axvline(valid_tests.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {valid_tests.mean():.2f}')
    ax1.axvline(valid_tests.median(), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {valid_tests.median():.1f}')
    ax1.set_title('Comprehension Phase: Number of Tests', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Tests', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
else:
    ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Comprehension Phase: Number of Tests', fontsize=14, fontweight='bold')

# 2. Comprehension: Blicket correctness
ax2 = fig.add_subplot(gs[0, 1])
if df['comprehension_blicket_correct'].notna().sum() > 0:
    correct_counts = df['comprehension_blicket_correct'].value_counts()
    labels = ['Correct', 'Incorrect']
    colors = ['#4CAF50', '#F44336']
    values = [correct_counts.get(True, 0), correct_counts.get(False, 0)]
    
    wedges, texts, autotexts = ax2.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Comprehension Phase: Blicket Correctness', fontsize=14, fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Comprehension Phase: Blicket Correctness', fontsize=14, fontweight='bold')

# 3. Comprehension: Prior experience
ax3 = fig.add_subplot(gs[0, 2])
if df['comprehension_prior_experience'].notna().sum() > 0:
    exp_counts = df['comprehension_prior_experience'].value_counts()
    labels = ['Has Prior Experience', 'No Prior Experience']
    colors = ['#FF9800', '#9E9E9E']
    values = [exp_counts.get(True, 0), exp_counts.get(False, 0)]
    
    wedges, texts, autotexts = ax3.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax3.set_title('Prior Experience Distribution', fontsize=14, fontweight='bold')
else:
    ax3.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Prior Experience Distribution', fontsize=14, fontweight='bold')

# ============================================================================
# MAIN GAME: NUMBER OF TESTS PER ROUND
# ============================================================================

# 4. Tests per round comparison
ax4 = fig.add_subplot(gs[1, :])
rounds_data = []
for round_num in [1, 2, 3]:
    col = f'round{round_num}_num_tests'
    if col in df.columns:
        valid_data = df[col].dropna()
        for val in valid_data:
            rounds_data.append({'round': f'Round {round_num}', 'num_tests': val})

if rounds_data:
    rounds_df = pd.DataFrame(rounds_data)
    round_averages = rounds_df.groupby('round')['num_tests'].agg(['mean', 'std'])
    rounds = round_averages.index
    x_pos = np.arange(len(rounds))
    
    bars = ax4.bar(x_pos, round_averages['mean'], yerr=round_averages['std'],
                   capsize=5, color=['#1976D2', '#388E3C', '#F57C00'], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_title('Average Number of Tests by Round', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('Average Number of Tests', fontsize=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(rounds)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        std_val = round_averages['std'].iloc[i]
        ax4.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.3,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
else:
    ax4.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Average Number of Tests by Round', fontsize=14, fontweight='bold')

# ============================================================================
# MAIN GAME: TOTAL TIME PER ROUND
# ============================================================================

# 5. Total time per round
ax5 = fig.add_subplot(gs[2, :])
time_data = []
for round_num in [1, 2, 3]:
    col = f'round{round_num}_total_time'
    if col in df.columns:
        valid_data = df[col].dropna()
        for val in valid_data:
            time_data.append({'round': f'Round {round_num}', 'total_time': val})

if time_data:
    time_df = pd.DataFrame(time_data)
    time_averages = time_df.groupby('round')['total_time'].agg(['mean', 'std'])
    rounds = time_averages.index
    x_pos = np.arange(len(rounds))
    
    bars = ax5.bar(x_pos, time_averages['mean'], yerr=time_averages['std'],
                   capsize=5, color=['#1976D2', '#388E3C', '#F57C00'], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.set_title('Average Total Time by Round (seconds)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Round', fontsize=12)
    ax5.set_ylabel('Average Total Time (seconds)', fontsize=12)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(rounds)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        std_val = time_averages['std'].iloc[i]
        ax5.text(bar.get_x() + bar.get_width()/2., height + std_val + 5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
else:
    ax5.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax5.transAxes)
    ax5.set_title('Average Total Time by Round (seconds)', fontsize=14, fontweight='bold')

# ============================================================================
# MAIN GAME: RULE INFERENCE ACCURACY
# ============================================================================

# 6. Rule inference accuracy by round
ax6 = fig.add_subplot(gs[3, 0])
rule_data = []
for round_num in [1, 2, 3]:
    true_col = f'round{round_num}_true_blickets'
    inferred_col = f'round{round_num}_inferred_rule'
    
    # We need to compare inferred_rule with true_rule
    # For now, let's count how many have inferred_rule
    if inferred_col in df.columns:
        valid_rules = df[inferred_col].dropna()
        rule_data.append({'round': f'Round {round_num}', 'has_rule': len(valid_rules)})

if rule_data:
    rule_df = pd.DataFrame(rule_data)
    bars = ax6.bar(range(len(rule_df)), rule_df['has_rule'],
                   color=['#1976D2', '#388E3C', '#F57C00'], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax6.set_title('Participants with Rule Inference by Round', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Round', fontsize=12)
    ax6.set_ylabel('Number of Participants', fontsize=12)
    ax6.set_xticks(range(len(rule_df)))
    ax6.set_xticklabels(rule_df['round'])
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
else:
    ax6.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax6.transAxes)
    ax6.set_title('Participants with Rule Inference by Round', fontsize=14, fontweight='bold')

# 7. Rule type distribution
ax7 = fig.add_subplot(gs[3, 1])
all_rules = []
for round_num in [1, 2, 3]:
    col = f'round{round_num}_inferred_rule'
    if col in df.columns:
        valid_rules = df[col].dropna()
        all_rules.extend(valid_rules.tolist())

if all_rules:
    rule_counts = pd.Series(all_rules).value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax7.bar(range(len(rule_counts)), rule_counts.values,
                   color=colors[:len(rule_counts)], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax7.set_title('Rule Type Distribution (All Rounds)', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Rule Type', fontsize=12)
    ax7.set_ylabel('Frequency', fontsize=12)
    ax7.set_xticks(range(len(rule_counts)))
    ax7.set_xticklabels([r.capitalize() for r in rule_counts.index])
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
else:
    ax7.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax7.transAxes)
    ax7.set_title('Rule Type Distribution (All Rounds)', fontsize=14, fontweight='bold')

# 8. Blicket identification accuracy by round (comparing true vs chosen)
ax8 = fig.add_subplot(gs[3, 2])
accuracy_data = []
for round_num in [1, 2, 3]:
    true_col = f'round{round_num}_true_blickets'
    chosen_col = f'round{round_num}_chosen_blickets'
    
    if true_col in df.columns and chosen_col in df.columns:
        correct = 0
        total = 0
        for idx, row in df.iterrows():
            true_vals = parse_list_str(row[true_col])
            chosen_vals = parse_list_str(row[chosen_col])
            if true_vals and chosen_vals:
                total += 1
                if set(true_vals) == set(chosen_vals):
                    correct += 1
        if total > 0:
            accuracy_data.append({
                'round': f'Round {round_num}',
                'accuracy': (correct / total) * 100,
                'correct': correct,
                'total': total
            })

if accuracy_data:
    acc_df = pd.DataFrame(accuracy_data)
    bars = ax8.bar(range(len(acc_df)), acc_df['accuracy'],
                   color=['#1976D2', '#388E3C', '#F57C00'], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax8.set_title('Blicket Identification Accuracy by Round', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Round', fontsize=12)
    ax8.set_ylabel('Accuracy (%)', fontsize=12)
    ax8.set_xticks(range(len(acc_df)))
    ax8.set_xticklabels(acc_df['round'])
    ax8.set_ylim(0, 100)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, row) in enumerate(zip(bars, acc_df.itertuples())):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%\n({row.correct}/{row.total})', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
else:
    ax8.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax8.transAxes)
    ax8.set_title('Blicket Identification Accuracy by Round', fontsize=14, fontweight='bold')

plt.suptitle('Comprehensive Data Analysis: Comprehension Phase and Main Game Rounds', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('images/comprehensive_data_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved: images/comprehensive_data_analysis.png")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Comprehension phase
if df['comprehension_num_tests'].notna().sum() > 0:
    print("\nComprehension Phase:")
    valid_tests = df['comprehension_num_tests'].dropna()
    print(f"  Average tests: {valid_tests.mean():.2f}")
    print(f"  Median tests: {valid_tests.median():.1f}")
    print(f"  Range: {valid_tests.min():.0f} - {valid_tests.max():.0f}")
    
    if df['comprehension_blicket_correct'].notna().sum() > 0:
        correct = df['comprehension_blicket_correct'].sum()
        total = df['comprehension_blicket_correct'].notna().sum()
        print(f"  Blicket correct: {correct}/{total} ({correct/total*100:.1f}%)")

# Main game rounds
for round_num in [1, 2, 3]:
    col = f'round{round_num}_num_tests'
    if col in df.columns and df[col].notna().sum() > 0:
        valid_tests = df[col].dropna()
        print(f"\nRound {round_num}:")
        print(f"  Average tests: {valid_tests.mean():.2f}")
        print(f"  Median tests: {valid_tests.median():.1f}")
        print(f"  Range: {valid_tests.min():.0f} - {valid_tests.max():.0f}")
        
        time_col = f'round{round_num}_total_time'
        if time_col in df.columns and df[time_col].notna().sum() > 0:
            valid_time = df[time_col].dropna()
            print(f"  Average time: {valid_time.mean():.2f} seconds")
            print(f"  Median time: {valid_time.median():.2f} seconds")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)

plt.close()

# ============================================================================
# CONJUNCTIVE VS DISJUNCTIVE ACCURACY VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("LOADING DATA FOR CONJUNCTIVE VS DISJUNCTIVE ACCURACY")
print("="*80)

# Load JSON data to get true_rule information
try:
    with open('firebase_data2.json', 'r') as f:
        json_data = json.load(f)
    print(f"Loaded JSON data with {len(json_data)} entries")
except FileNotFoundError:
    print("WARNING: firebase_data2.json not found. Skipping conjunctive/disjunctive accuracy visualization.")
    json_data = None
except json.JSONDecodeError as e:
    print(f"WARNING: Error parsing JSON: {e}. Skipping conjunctive/disjunctive accuracy visualization.")
    json_data = None

if json_data:
    # Helper functions
    def is_prolific_id(participant_id):
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
        if 'demographics' in entry and isinstance(entry['demographics'], dict):
            if 'prolific_id' in entry['demographics']:
                return entry['demographics']['prolific_id']
        if 'config' in entry and isinstance(entry['config'], dict):
            if 'demographics' in entry['config'] and isinstance(entry['config']['demographics'], dict):
                if 'prolific_id' in entry['config']['demographics']:
                    return entry['config']['demographics']['prolific_id']
        return None

    def get_user_chosen_blickets(round_data):
        if 'user_chosen_blickets' in round_data:
            chosen = round_data['user_chosen_blickets']
            if isinstance(chosen, list):
                return sorted(chosen)
        if 'blicket_classifications' in round_data:
            classifications = round_data['blicket_classifications']
            if isinstance(classifications, dict):
                chosen = []
                for key, value in classifications.items():
                    if value == "Yes":
                        try:
                            idx = int(key.split('_')[1])
                            chosen.append(idx)
                        except (ValueError, IndexError):
                            pass
                return sorted(chosen) if chosen else None
        return None

    def parse_rule_type(rule_type_str):
        if not rule_type_str or not isinstance(rule_type_str, str):
            return None
        rule_lower = rule_type_str.lower()
        if 'conjunctive' in rule_lower:
            return 'conjunctive'
        elif 'disjunctive' in rule_lower:
            return 'disjunctive'
        return None

    # Collect accuracy data by rule type
    accuracy_by_rule = []
    
    for entry_id, entry_data in json_data.items():
        prolific_id = get_prolific_id(entry_data)
        if not prolific_id or not is_prolific_id(prolific_id):
            continue
        
        if not isinstance(entry_data, dict):
            continue
        
        main_game = entry_data.get('main_game', {})
        if not isinstance(main_game, dict):
            continue
        
        for round_key, round_data in main_game.items():
            if not isinstance(round_data, dict) or not round_key.startswith('round_'):
                continue
            
            true_rule = round_data.get('true_rule', '').lower()
            if not true_rule or true_rule not in ['conjunctive', 'disjunctive']:
                continue
            
            true_blickets = round_data.get('true_blicket_indices', [])
            user_chosen_blickets = get_user_chosen_blickets(round_data)
            
            # Calculate object identification accuracy
            object_accuracy = None
            if true_blickets and user_chosen_blickets is not None:
                object_accuracy = set(true_blickets) == set(user_chosen_blickets)
            
            # Calculate rule inference accuracy
            rule_type_str = round_data.get('rule_type', '')
            user_rule = parse_rule_type(rule_type_str)
            rule_accuracy = None
            if user_rule and true_rule:
                rule_accuracy = (user_rule.lower() == true_rule.lower())
            
            accuracy_by_rule.append({
                'true_rule': true_rule,
                'object_accuracy': object_accuracy,
                'rule_accuracy': rule_accuracy
            })
    
    if accuracy_by_rule:
        acc_df = pd.DataFrame(accuracy_by_rule)
        
        # Create visualization
        fig2 = plt.figure(figsize=(16, 10))
        gs2 = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Object identification accuracy by rule type
        ax1 = fig2.add_subplot(gs2[0, 0])
        obj_acc_by_rule = acc_df.groupby('true_rule')['object_accuracy'].agg(['sum', 'count'])
        obj_acc_by_rule['accuracy_pct'] = (obj_acc_by_rule['sum'] / obj_acc_by_rule['count']) * 100
        
        x_pos = np.arange(len(obj_acc_by_rule))
        bars = ax1.bar(x_pos, obj_acc_by_rule['accuracy_pct'], 
                      color=['#1976D2', '#FF9800'], 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_title('Object Identification Accuracy\nby Rule Type', fontsize=14, fontweight='bold')
        ax1.set_xlabel('True Rule Type', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([r.capitalize() for r in obj_acc_by_rule.index])
        ax1.set_ylim(0, 100)
        ax1.axhline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Chance (50%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, row) in enumerate(zip(bars, obj_acc_by_rule.itertuples())):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%\n(n={int(row.count)})',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 2. Rule inference accuracy by rule type
        ax2 = fig2.add_subplot(gs2[0, 1])
        rule_acc_by_rule = acc_df.groupby('true_rule')['rule_accuracy'].agg(['sum', 'count'])
        rule_acc_by_rule['accuracy_pct'] = (rule_acc_by_rule['sum'] / rule_acc_by_rule['count']) * 100
        
        x_pos = np.arange(len(rule_acc_by_rule))
        bars = ax2.bar(x_pos, rule_acc_by_rule['accuracy_pct'], 
                      color=['#1976D2', '#FF9800'], 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_title('Rule Inference Accuracy\nby Rule Type', fontsize=14, fontweight='bold')
        ax2.set_xlabel('True Rule Type', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([r.capitalize() for r in rule_acc_by_rule.index])
        ax2.set_ylim(0, 100)
        ax2.axhline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Chance (50%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, row) in enumerate(zip(bars, rule_acc_by_rule.itertuples())):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%\n(n={int(row.count)})',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 3. Combined accuracy (both correct) by rule type
        ax3 = fig2.add_subplot(gs2[1, 0])
        acc_df['both_correct'] = acc_df['object_accuracy'] & acc_df['rule_accuracy']
        both_acc_by_rule = acc_df.groupby('true_rule')['both_correct'].agg(['sum', 'count'])
        both_acc_by_rule['accuracy_pct'] = (both_acc_by_rule['sum'] / both_acc_by_rule['count']) * 100
        
        x_pos = np.arange(len(both_acc_by_rule))
        bars = ax3.bar(x_pos, both_acc_by_rule['accuracy_pct'], 
                      color=['#1976D2', '#FF9800'], 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.set_title('Combined Accuracy (Both Correct)\nby Rule Type', fontsize=14, fontweight='bold')
        ax3.set_xlabel('True Rule Type', fontsize=12)
        ax3.set_ylabel('Accuracy (%)', fontsize=12)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([r.capitalize() for r in both_acc_by_rule.index])
        ax3.set_ylim(0, 100)
        ax3.axhline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Chance (50%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, row) in enumerate(zip(bars, both_acc_by_rule.itertuples())):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%\n(n={int(row.count)})',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 4. Side-by-side comparison
        ax4 = fig2.add_subplot(gs2[1, 1])
        comparison_data = pd.DataFrame({
            'Rule Type': ['Conjunctive'] * 3 + ['Disjunctive'] * 3,
            'Accuracy Type': ['Object ID', 'Rule Inference', 'Combined'] * 2,
            'Accuracy': [
                obj_acc_by_rule.loc['conjunctive', 'accuracy_pct'],
                rule_acc_by_rule.loc['conjunctive', 'accuracy_pct'],
                both_acc_by_rule.loc['conjunctive', 'accuracy_pct'],
                obj_acc_by_rule.loc['disjunctive', 'accuracy_pct'],
                rule_acc_by_rule.loc['disjunctive', 'accuracy_pct'],
                both_acc_by_rule.loc['disjunctive', 'accuracy_pct']
            ]
        })
        
        x_pos = np.arange(3)
        width = 0.35
        conj_acc = [
            obj_acc_by_rule.loc['conjunctive', 'accuracy_pct'],
            rule_acc_by_rule.loc['conjunctive', 'accuracy_pct'],
            both_acc_by_rule.loc['conjunctive', 'accuracy_pct']
        ]
        disj_acc = [
            obj_acc_by_rule.loc['disjunctive', 'accuracy_pct'],
            rule_acc_by_rule.loc['disjunctive', 'accuracy_pct'],
            both_acc_by_rule.loc['disjunctive', 'accuracy_pct']
        ]
        
        bars1 = ax4.bar(x_pos - width/2, conj_acc, width, label='Conjunctive', 
                       color='#1976D2', alpha=0.7, edgecolor='black', linewidth=1.5)
        bars2 = ax4.bar(x_pos + width/2, disj_acc, width, label='Disjunctive', 
                       color='#FF9800', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax4.set_title('Accuracy Comparison:\nConjunctive vs Disjunctive', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Accuracy Type', fontsize=12)
        ax4.set_ylabel('Accuracy (%)', fontsize=12)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(['Object ID', 'Rule\nInference', 'Combined'])
        ax4.set_ylim(0, 100)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.suptitle('Accuracy Analysis: Conjunctive vs Disjunctive Rules', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        os.makedirs('images', exist_ok=True)
        plt.savefig('images/accuracy_conjunctive_vs_disjunctive.png', dpi=300, bbox_inches='tight')
        print("\nSaved: images/accuracy_conjunctive_vs_disjunctive.png")
        
        # Print statistics
        print("\n" + "="*80)
        print("ACCURACY BY RULE TYPE STATISTICS")
        print("="*80)
        print("\nObject Identification Accuracy:")
        for rule_type in ['conjunctive', 'disjunctive']:
            if rule_type in obj_acc_by_rule.index:
                row = obj_acc_by_rule.loc[rule_type]
                print(f"  {rule_type.capitalize()}: {row['sum']:.0f}/{row['count']:.0f} ({row['accuracy_pct']:.1f}%)")
        
        print("\nRule Inference Accuracy:")
        for rule_type in ['conjunctive', 'disjunctive']:
            if rule_type in rule_acc_by_rule.index:
                row = rule_acc_by_rule.loc[rule_type]
                print(f"  {rule_type.capitalize()}: {row['sum']:.0f}/{row['count']:.0f} ({row['accuracy_pct']:.1f}%)")
        
        print("\nCombined Accuracy (Both Correct):")
        for rule_type in ['conjunctive', 'disjunctive']:
            if rule_type in both_acc_by_rule.index:
                row = both_acc_by_rule.loc[rule_type]
                print(f"  {rule_type.capitalize()}: {row['sum']:.0f}/{row['count']:.0f} ({row['accuracy_pct']:.1f}%)")
        
        plt.close()

