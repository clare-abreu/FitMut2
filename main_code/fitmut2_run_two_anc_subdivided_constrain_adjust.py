#!/usr/bin/env python3
"""
Step 4: Adjust subdivided environment results to satisfy mix constraint.

For each adaptive lineage, adjusts (s1, s2) to be consistent with s_mix:
  log((exp(s1) + exp(s2))/2) = s_mix

Usage:
  python fitmut2_adjust_subdivided.py -e1 results_E1_step2_MutSeq_Result.csv -e2 results_E2_step2_MutSeq_Result.csv -mix results_mix_MutSeq_Result.csv -o results_final.csv
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize

###########################
##### PARSE ARGUMENTS #####
###########################

parser = argparse.ArgumentParser(description='Adjust subdivided results for mix constraint')

parser.add_argument('-e1', '--E1_results', required=True,
                   help='E1 Step 2 results CSV')
parser.add_argument('-e2', '--E2_results', required=True,
                   help='E2 Step 2 results CSV')
parser.add_argument('-mix', '--mix_results', required=True,
                   help='Mix results CSV')
parser.add_argument('-o', '--output', required=True,
                   help='Output CSV file')
parser.add_argument('--threshold', type=float, default=0.3,
                   help='Probability threshold for adaptive lineages (default: 0.3)')

args = parser.parse_args()

###########################
##### LOAD DATA ###########
###########################

results_E1 = pd.read_csv(args.E1_results)
results_E2 = pd.read_csv(args.E2_results)
results_mix = pd.read_csv(args.mix_results)

# Extract fitness values
s1_step2 = results_E1['Fitness'].values
s2_step2 = results_E2['Fitness'].values
s_mix = results_mix['Fitness'].values

# Extract probabilities to identify adaptive lineages
p_adaptive_E1 = results_E1['Probability_Adaptive'].values
p_adaptive_E2 = results_E2['Probability_Adaptive'].values
p_adaptive_mix = results_mix['Probability_Adaptive'].values

# A lineage is adaptive if it's adaptive in E1, E2, OR mix
is_adaptive = ((p_adaptive_E1 > args.threshold) | 
               (p_adaptive_E2 > args.threshold) | 
               (p_adaptive_mix > args.threshold))

num_adaptive = is_adaptive.sum()
num_lineages = len(s1_step2)

print(f"Total lineages: {num_lineages}")
print(f"Adaptive lineages: {num_adaptive}")

###########################
##### ADJUSTMENT ##########
###########################

def adjust_for_constraint(s1_refined, s2_refined, s_mix_observed):
    """
    Adjust (s1, s2) to satisfy mix constraint while staying close to refined values.
    
    minimize: (s1_final - s1_refined)² + (s2_final - s2_refined)²
    subject to: log((exp(s1_final) + exp(s2_final))/2) = s_mix_observed
    """
    
    def objective(params):
        s1_new, s2_new = params
        return (s1_new - s1_refined)**2 + (s2_new - s2_refined)**2
    
    def constraint(params):
        s1_new, s2_new = params
        s_mix_predicted = np.log((np.exp(s1_new) + np.exp(s2_new)) / 2)
        return s_mix_predicted - s_mix_observed
    
    # Starting point
    x0 = [s1_refined, s2_refined]
    
    # Optimization
    result = minimize(
        objective,
        x0=x0,
        constraints={'type': 'eq', 'fun': constraint},
        method='SLSQP'
    )
    
    if result.success:
        return result.x  # [s1_final, s2_final]
    else:
        # If optimization fails, return original values
        print(f"  Warning: Optimization failed for s1={s1_refined:.4f}, s2={s2_refined:.4f}")
        return x0

s1_final = s1_step2.copy()
s2_final = s2_step2.copy()

for i in range(num_lineages):
    if is_adaptive[i]:
        s1_adj, s2_adj = adjust_for_constraint(s1_step2[i], s2_step2[i], s_mix[i])
        s1_final[i] = s1_adj
        s2_final[i] = s2_adj
        
        # Check constraint satisfaction
        s_mix_check = np.log((np.exp(s1_adj) + np.exp(s2_adj)) / 2)
        error = abs(s_mix_check - s_mix[i])
        
        if error > 1e-4:
            print(f"  Lineage {i}: Large constraint error = {error:.6f}")

###########################
##### SAVE RESULTS ########
###########################

# Create output dataframe with both E1 and E2 results
output_df = pd.DataFrame({
    'Lineage_Index': range(num_lineages),
    'Fitness_E1': s1_final,
    'Fitness_E2': s2_final,
    'Fitness_Mix': s_mix,
    'Fitness_E1_Step2': s1_step2,
    'Fitness_E2_Step2': s2_step2,
    'Tau_E1': results_E1['Establishment_Time'].values,
    'Tau_E2': results_E2['Establishment_Time'].values,
    'Tau_Mix': results_mix['Establishment_Time'].values,
    'P_Adaptive_E1': p_adaptive_E1,
    'P_Adaptive_E2': p_adaptive_E2,
    'P_Adaptive_Mix': p_adaptive_mix,
    'Is_Adaptive': is_adaptive
})

output_df.to_csv(args.output, index=False)

print(f"Results saved to {args.output}")

adaptive_df = output_df[output_df['Is_Adaptive']]
if len(adaptive_df) > 0:
    print(f"\nSummary for {len(adaptive_df)} adaptive lineages:")
    print(f"  Mean adjustment in E1: {abs(adaptive_df['Fitness_E1'] - adaptive_df['Fitness_E1_Step2']).mean():.6f}")
    print(f"  Mean adjustment in E2: {abs(adaptive_df['Fitness_E2'] - adaptive_df['Fitness_E2_Step2']).mean():.6f}")
    print(f"  Max adjustment in E1: {abs(adaptive_df['Fitness_E1'] - adaptive_df['Fitness_E1_Step2']).max():.6f}")
    print(f"  Max adjustment in E2: {abs(adaptive_df['Fitness_E2'] - adaptive_df['Fitness_E2_Step2']).max():.6f}")
    
    
    