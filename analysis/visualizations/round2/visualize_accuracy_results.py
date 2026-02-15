"""
Create comprehensive figures showing object identification and rule inference accuracy
over users and rounds, including prior experience analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# Set font
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']

def load_data():
    """Load analysis results from CSV files"""
    try:
        detailed_df = pd.read_csv('results/accuracy_detailed.csv')
        participant_df = pd.read_csv('results/accuracy_per_participant.csv')
        prior_df = pd.read_csv('results/prior_experience.csv')
        return detailed_df, participant_df, prior_df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run analyze_accuracy_comprehensive.py first to generate CSV files.")
        return None, None, None

def create_comprehensive_figure(detailed_df, participant_df, prior_df):
    """Create a comprehensive figure with multiple subplots"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # ============================================================================
    # 1. PRIOR EXPERIENCE DISTRIBUTION
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    prior_counts = prior_df['has_prior_experience'].value_counts()
    labels = []
    sizes = []
    colors_pie = ['#2a9d8f', '#e76f51', '#95a5a6']
    
    if True in prior_counts.index:
        labels.append('Has Prior\nExperience')
        sizes.append(prior_counts[True])
    if False in prior_counts.index:
        labels.append('No Prior\nExperience')
        sizes.append(prior_counts[False])
    if None in prior_counts.index:
        labels.append('Unknown')
        sizes.append(prior_counts[None])
    
    if sizes:
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors_pie[:len(sizes)],
                                           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax1.set_title('Prior Game Experience\nDistribution', fontsize=13, fontweight='bold', pad=15)
    
    # ============================================================================
    # 2. OBJECT IDENTIFICATION ACCURACY BY ROUND
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    object_by_round = detailed_df.groupby('round_number').agg({
        'object_accuracy': ['sum', 'count']
    }).reset_index()
    object_by_round.columns = ['round', 'correct', 'total']
    object_by_round['accuracy_pct'] = (object_by_round['correct'] / object_by_round['total'] * 100)
    
    rounds = sorted(object_by_round['round'].astype(int))
    accuracies = [object_by_round[object_by_round['round'] == r]['accuracy_pct'].values[0] for r in rounds]
    correct_counts = [int(object_by_round[object_by_round['round'] == r]['correct'].values[0]) for r in rounds]
    total_counts = [int(object_by_round[object_by_round['round'] == r]['total'].values[0]) for r in rounds]
    
    bars = ax2.bar(range(len(rounds)), accuracies, 
                   color=['#1b9e77', '#d95f02', '#7570b3'], 
                   alpha=0.8, edgecolor='#333333', linewidth=1.5)
    ax2.set_title('Object Identification\nAccuracy by Round', fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_xticks(range(len(rounds)))
    ax2.set_xticklabels([f'Round {r}' for r in rounds], fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, acc, correct, total) in enumerate(zip(bars, accuracies, correct_counts, total_counts)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%\n({correct}/{total})', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ============================================================================
    # 3. RULE INFERENCE ACCURACY BY ROUND
    # ============================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    rule_by_round = detailed_df.groupby('round_number').agg({
        'rule_accuracy': ['sum', 'count']
    }).reset_index()
    rule_by_round.columns = ['round', 'correct', 'total']
    rule_by_round['accuracy_pct'] = (rule_by_round['correct'] / rule_by_round['total'] * 100)
    
    rounds_rule = sorted(rule_by_round['round'].astype(int))
    accuracies_rule = [rule_by_round[rule_by_round['round'] == r]['accuracy_pct'].values[0] for r in rounds_rule]
    correct_counts_rule = [int(rule_by_round[rule_by_round['round'] == r]['correct'].values[0]) for r in rounds_rule]
    total_counts_rule = [int(rule_by_round[rule_by_round['round'] == r]['total'].values[0]) for r in rounds_rule]
    
    bars = ax3.bar(range(len(rounds_rule)), accuracies_rule, 
                   color=['#1b9e77', '#d95f02', '#7570b3'], 
                   alpha=0.8, edgecolor='#333333', linewidth=1.5)
    ax3.set_title('Rule Inference\nAccuracy by Round', fontsize=13, fontweight='bold', pad=15)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_xticks(range(len(rounds_rule)))
    ax3.set_xticklabels([f'Round {r}' for r in rounds_rule], fontsize=11)
    ax3.set_ylim(0, 110)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, acc, correct, total) in enumerate(zip(bars, accuracies_rule, correct_counts_rule, total_counts_rule)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%\n({correct}/{total})', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ============================================================================
    # 4. OBJECT ACCURACY BY PRIOR EXPERIENCE
    # ============================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    
    object_by_prior = detailed_df.groupby('has_prior_experience').agg({
        'object_accuracy': ['sum', 'count']
    }).reset_index()
    object_by_prior.columns = ['has_prior', 'correct', 'total']
    object_by_prior['accuracy_pct'] = (object_by_prior['correct'] / object_by_prior['total'] * 100)
    
    # Filter out None values for plotting
    object_by_prior_plot = object_by_prior[object_by_prior['has_prior'].notna()].copy()
    
    if len(object_by_prior_plot) > 0:
        labels_prior = ['Has Prior\nExperience' if x else 'No Prior\nExperience' 
                        for x in object_by_prior_plot['has_prior']]
        accuracies_prior = object_by_prior_plot['accuracy_pct'].values
        correct_prior = object_by_prior_plot['correct'].values.astype(int)
        total_prior = object_by_prior_plot['total'].values.astype(int)
        
        bars = ax4.bar(range(len(labels_prior)), accuracies_prior, 
                       color=['#2a9d8f', '#e76f51'], 
                       alpha=0.8, edgecolor='#333333', linewidth=1.5)
        ax4.set_title('Object Identification\nAccuracy by Prior Experience', fontsize=13, fontweight='bold', pad=15)
        ax4.set_ylabel('Accuracy (%)', fontsize=12)
        ax4.set_xticks(range(len(labels_prior)))
        ax4.set_xticklabels(labels_prior, fontsize=11)
        ax4.set_ylim(0, 110)
        ax4.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, acc, correct, total) in enumerate(zip(bars, accuracies_prior, correct_prior, total_prior)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{acc:.1f}%\n({correct}/{total})', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ============================================================================
    # 5. RULE ACCURACY BY PRIOR EXPERIENCE
    # ============================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    
    rule_by_prior = detailed_df.groupby('has_prior_experience').agg({
        'rule_accuracy': ['sum', 'count']
    }).reset_index()
    rule_by_prior.columns = ['has_prior', 'correct', 'total']
    rule_by_prior['accuracy_pct'] = (rule_by_prior['correct'] / rule_by_prior['total'] * 100)
    
    # Filter out None values for plotting
    rule_by_prior_plot = rule_by_prior[rule_by_prior['has_prior'].notna()].copy()
    
    if len(rule_by_prior_plot) > 0:
        labels_prior = ['Has Prior\nExperience' if x else 'No Prior\nExperience' 
                        for x in rule_by_prior_plot['has_prior']]
        accuracies_prior = rule_by_prior_plot['accuracy_pct'].values
        correct_prior = rule_by_prior_plot['correct'].values.astype(int)
        total_prior = rule_by_prior_plot['total'].values.astype(int)
        
        bars = ax5.bar(range(len(labels_prior)), accuracies_prior, 
                       color=['#2a9d8f', '#e76f51'], 
                       alpha=0.8, edgecolor='#333333', linewidth=1.5)
        ax5.set_title('Rule Inference\nAccuracy by Prior Experience', fontsize=13, fontweight='bold', pad=15)
        ax5.set_ylabel('Accuracy (%)', fontsize=12)
        ax5.set_xticks(range(len(labels_prior)))
        ax5.set_xticklabels(labels_prior, fontsize=11)
        ax5.set_ylim(0, 110)
        ax5.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, acc, correct, total) in enumerate(zip(bars, accuracies_prior, correct_prior, total_prior)):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{acc:.1f}%\n({correct}/{total})', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ============================================================================
    # 6. PER-PARTICIPANT OBJECT ACCURACY
    # ============================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    
    participant_obj = participant_df.sort_values('object_accuracy_pct', ascending=False)
    participant_labels = [pid[:8] + '...' for pid in participant_obj['participant_id']]
    colors = ['#2a9d8f' if acc == 100 else '#e9c46a' for acc in participant_obj['object_accuracy_pct']]
    
    bars = ax6.bar(range(len(participant_obj)), participant_obj['object_accuracy_pct'],
                   color=colors, alpha=0.9, edgecolor='#333333', linewidth=1.2)
    ax6.set_title('Object Identification\nAccuracy per Participant', fontsize=13, fontweight='bold', pad=15)
    ax6.set_ylabel('Accuracy (%)', fontsize=12)
    ax6.set_xlabel('Participant', fontsize=12)
    ax6.set_xticks(range(len(participant_obj)))
    ax6.set_xticklabels(participant_labels, rotation=45, ha='right', fontsize=9)
    ax6.set_ylim(0, 110)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels (only show if space allows)
    for i, (bar, acc) in enumerate(zip(bars, participant_obj['object_accuracy_pct'])):
        height = bar.get_height()
        if len(participant_obj) <= 15:  # Only label if not too many participants
            ax6.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{acc:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # ============================================================================
    # 7. PER-PARTICIPANT RULE ACCURACY
    # ============================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    
    participant_rule = participant_df.sort_values('rule_accuracy_pct', ascending=False)
    participant_labels = [pid[:8] + '...' for pid in participant_rule['participant_id']]
    colors = ['#2a9d8f' if acc == 100 else '#e9c46a' for acc in participant_rule['rule_accuracy_pct']]
    
    bars = ax7.bar(range(len(participant_rule)), participant_rule['rule_accuracy_pct'],
                   color=colors, alpha=0.9, edgecolor='#333333', linewidth=1.2)
    ax7.set_title('Rule Inference\nAccuracy per Participant', fontsize=13, fontweight='bold', pad=15)
    ax7.set_ylabel('Accuracy (%)', fontsize=12)
    ax7.set_xlabel('Participant', fontsize=12)
    ax7.set_xticks(range(len(participant_rule)))
    ax7.set_xticklabels(participant_labels, rotation=45, ha='right', fontsize=9)
    ax7.set_ylim(0, 110)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add value labels (only show if space allows)
    for i, (bar, acc) in enumerate(zip(bars, participant_rule['rule_accuracy_pct'])):
        height = bar.get_height()
        if len(participant_rule) <= 15:  # Only label if not too many participants
            ax7.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{acc:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # ============================================================================
    # 8. COMBINED ACCURACY (BOTH CORRECT) BY ROUND
    # ============================================================================
    ax8 = fig.add_subplot(gs[2, 1])
    
    detailed_df['both_correct'] = detailed_df['object_accuracy'] & detailed_df['rule_accuracy']
    both_by_round = detailed_df.groupby('round_number').agg({
        'both_correct': ['sum', 'count']
    }).reset_index()
    both_by_round.columns = ['round', 'correct', 'total']
    both_by_round['accuracy_pct'] = (both_by_round['correct'] / both_by_round['total'] * 100)
    
    rounds_both = sorted(both_by_round['round'].astype(int))
    accuracies_both = [both_by_round[both_by_round['round'] == r]['accuracy_pct'].values[0] for r in rounds_both]
    correct_counts_both = [int(both_by_round[both_by_round['round'] == r]['correct'].values[0]) for r in rounds_both]
    total_counts_both = [int(both_by_round[both_by_round['round'] == r]['total'].values[0]) for r in rounds_both]
    
    bars = ax8.bar(range(len(rounds_both)), accuracies_both, 
                   color=['#1b9e77', '#d95f02', '#7570b3'], 
                   alpha=0.8, edgecolor='#333333', linewidth=1.5)
    ax8.set_title('Combined Accuracy\n(Both Object & Rule Correct)\nby Round', fontsize=13, fontweight='bold', pad=15)
    ax8.set_ylabel('Accuracy (%)', fontsize=12)
    ax8.set_xticks(range(len(rounds_both)))
    ax8.set_xticklabels([f'Round {r}' for r in rounds_both], fontsize=11)
    ax8.set_ylim(0, 110)
    ax8.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, acc, correct, total) in enumerate(zip(bars, accuracies_both, correct_counts_both, total_counts_both)):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%\n({correct}/{total})', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ============================================================================
    # 9. ACCURACY COMPARISON: OBJECT vs RULE
    # ============================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    
    # Calculate overall accuracies
    object_overall = (detailed_df['object_accuracy'].sum() / detailed_df['object_accuracy'].notna().sum() * 100)
    rule_overall = (detailed_df['rule_accuracy'].sum() / detailed_df['rule_accuracy'].notna().sum() * 100)
    both_overall = (detailed_df['both_correct'].sum() / detailed_df['both_correct'].notna().sum() * 100)
    
    categories = ['Object\nIdentification', 'Rule\nInference', 'Both\nCorrect']
    accuracies_comp = [object_overall, rule_overall, both_overall]
    
    bars = ax9.bar(range(len(categories)), accuracies_comp, 
                   color=['#1b9e77', '#d95f02', '#7570b3'], 
                   alpha=0.8, edgecolor='#333333', linewidth=1.5)
    ax9.set_title('Overall Accuracy\nComparison', fontsize=13, fontweight='bold', pad=15)
    ax9.set_ylabel('Accuracy (%)', fontsize=12)
    ax9.set_xticks(range(len(categories)))
    ax9.set_xticklabels(categories, fontsize=11)
    ax9.set_ylim(0, 110)
    ax9.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies_comp)):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add overall title
    plt.suptitle('Comprehensive Accuracy Analysis: Object Identification & Rule Inference', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('results/accuracy_comprehensive_figure.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/accuracy_comprehensive_figure.png")
    plt.close()

def create_round_comparison_figure(detailed_df):
    """Create a figure comparing accuracy across rounds"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Object accuracy by round
    ax1 = axes[0]
    object_by_round = detailed_df.groupby('round_number').agg({
        'object_accuracy': ['sum', 'count']
    }).reset_index()
    object_by_round.columns = ['round', 'correct', 'total']
    object_by_round['accuracy_pct'] = (object_by_round['correct'] / object_by_round['total'] * 100)
    
    rounds = sorted(object_by_round['round'].astype(int))
    accuracies = [object_by_round[object_by_round['round'] == r]['accuracy_pct'].values[0] for r in rounds]
    
    ax1.plot(rounds, accuracies, marker='o', linewidth=2.5, markersize=10, 
             color='#1b9e77', label='Object Identification')
    ax1.set_title('Object Identification Accuracy Across Rounds', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_xticks(rounds)
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    for r, acc in zip(rounds, accuracies):
        ax1.text(r, acc + 3, f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Rule accuracy by round
    ax2 = axes[1]
    rule_by_round = detailed_df.groupby('round_number').agg({
        'rule_accuracy': ['sum', 'count']
    }).reset_index()
    rule_by_round.columns = ['round', 'correct', 'total']
    rule_by_round['accuracy_pct'] = (rule_by_round['correct'] / rule_by_round['total'] * 100)
    
    rounds_rule = sorted(rule_by_round['round'].astype(int))
    accuracies_rule = [rule_by_round[rule_by_round['round'] == r]['accuracy_pct'].values[0] for r in rounds_rule]
    
    ax2.plot(rounds_rule, accuracies_rule, marker='s', linewidth=2.5, markersize=10, 
             color='#d95f02', label='Rule Inference')
    ax2.set_title('Rule Inference Accuracy Across Rounds', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_xticks(rounds_rule)
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    for r, acc in zip(rounds_rule, accuracies_rule):
        ax2.text(r, acc + 3, f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/accuracy_by_round_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/accuracy_by_round_comparison.png")
    plt.close()

def main():
    """Main function to create all visualizations"""
    
    print("="*80)
    print("ACCURACY VISUALIZATION")
    print("="*80)
    
    # Load data
    detailed_df, participant_df, prior_df = load_data()
    
    if detailed_df is None:
        return
    
    print(f"\nLoaded data:")
    print(f"  - Detailed rounds: {len(detailed_df)}")
    print(f"  - Participants: {len(participant_df)}")
    print(f"  - Prior experience records: {len(prior_df)}")
    
    # Create comprehensive figure
    print("\n1. Creating comprehensive accuracy figure...")
    create_comprehensive_figure(detailed_df, participant_df, prior_df)
    
    # Create round comparison figure
    print("2. Creating round comparison figure...")
    create_round_comparison_figure(detailed_df)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("\nGenerated figures:")
    print("  - results/accuracy_comprehensive_figure.png")
    print("  - results/accuracy_by_round_comparison.png")
    print("="*80)

if __name__ == "__main__":
    main()

