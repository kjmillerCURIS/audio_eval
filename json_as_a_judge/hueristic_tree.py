#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np

def heuristic_tree(row):
    """
    - Hierarchy: Content > Instruction Following > Voice Quality
    - Includes the "Dominant both_bad" rule and handles missing data.
    """
    content = str(row['prediction_content'])
    instruction = str(row['prediction_if'])
    voice = str(row['prediction_vq'])

    # Handle Missing Data
    if content == 'nan':
        return 'both_bad'
    # "Dominant both_bad" Rule
    if content == 'both_bad' and instruction == 'both_bad':
        return 'both_bad'

    # Original Hierarchy
    valid_winners = ['1', '2']
    if content in valid_winners:
        return content
    if instruction in valid_winners:
        return instruction
    if voice in valid_winners:
        return voice
    
    # Default Tie Handling
    if content == 'both_good':
        return 'both_good'
    
    return 'both_bad'

# --- Main Script ---
try:
    # --- 1. Load and Prepare the Data ---
    df_full = pd.read_csv('json_judge_hcot_fusion.csv')
    total_rows = len(df_full)

    # Define required columns and clean the data
    required_columns = ['prediction_content', 'prediction_if', 'prediction_vq', 'gt_overall', 'gt_content']
    eval_df = df_full.dropna(subset=required_columns).copy()
    
    valid_gt_labels = ['1', '2', 'both_good', 'both_bad']
    eval_df = eval_df[eval_df['gt_overall'].isin(valid_gt_labels)]
    
    # Apply the decision tree to get predictions
    eval_df['tree_prediction'] = eval_df.apply(heuristic_tree, axis=1)

    # --- 2. Calculate All Statistics ---
    print("## Consolidated Performance Analysis ##")
    print("-" * 40)

    # Dropped Rows
    cleaned_rows = len(eval_df)
    dropped_rows = total_rows - cleaned_rows
    print(f"Data Cleaning: Dropped {dropped_rows} of {total_rows} rows due to missing data.")
    print("-" * 40)

    # Overall Metrics
    total_predictions = len(eval_df)
    correct_predictions = (eval_df['tree_prediction'] == eval_df['gt_overall']).sum()
    overall_accuracy = correct_predictions / total_predictions
    
    partial_credit_score = 0.0
    for _, row in eval_df.iterrows():
        if row['tree_prediction'] == row['gt_overall']:
            partial_credit_score += 1.0
        elif row['gt_overall'] in ['both_good', 'both_bad'] and row['tree_prediction'] in ['1', '2']:
            partial_credit_score += 0.5
    partial_credit_accuracy = partial_credit_score / total_predictions

    print("## Overall Performance ##")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print(f"Partial Credit Accuracy: {partial_credit_accuracy:.2%}")
    print("-" * 40)

    # Subset Accuracy Analysis
    print("## Accuracy on Specific Subsets ##")
    
    # When GT Content is a tie
    subset_1 = eval_df[eval_df['gt_content'] == 'both_good']
    acc_1 = (subset_1['tree_prediction'] == subset_1['gt_overall']).sum() / len(subset_1) if not subset_1.empty else 0
    print(f"When Ground Truth Content is 'both_good': {acc_1:.2%} ({len(subset_1)} cases)")
    
    # When GT Overall is a tie
    subset_tie = eval_df[eval_df['gt_overall'].isin(['both_good', 'both_bad'])]
    acc_tie = (subset_tie['tree_prediction'] == subset_tie['gt_overall']).sum() / len(subset_tie) if not subset_tie.empty else 0
    print(f"When Ground Truth Overall is a Tie:       {acc_tie:.2%} ({len(subset_tie)} cases)")

    # When GT Overall is a winner
    subset_winner = eval_df[eval_df['gt_overall'].isin(['1', '2'])]
    acc_winner = (subset_winner['tree_prediction'] == subset_winner['gt_overall']).sum() / len(subset_winner) if not subset_winner.empty else 0
    print(f"When Ground Truth Overall is a Winner:    {acc_winner:.2%} ({len(subset_winner)} cases)")

    # When top predictions conflict
    conflict_condition = ((eval_df['prediction_content'] == '1') & (eval_df['prediction_if'] == '2') |
                        (eval_df['prediction_content'] == '2') & (eval_df['prediction_if'] == '1'))
    subset_3 = eval_df[conflict_condition]
    acc_3 = (subset_3['tree_prediction'] == subset_3['gt_overall']).sum() / len(subset_3) if not subset_3.empty else 0
    print(f"When Top Predictions Conflict:            {acc_3:.2%} ({len(subset_3)} cases)")
    print("-" * 40)
    
    # Final Confusion Matrix
    print("## Final Confusion Matrix ##")
    print("Rows: Tree Prediction, Columns: Ground Truth")
    final_cm = pd.crosstab(eval_df['tree_prediction'], eval_df['gt_overall'])
    print(final_cm)

except FileNotFoundError:
    print("Error: 'json_judge_hcot_fusion.csv' not found.")
except Exception as e:
    print(f"An error occurred: {e}")

