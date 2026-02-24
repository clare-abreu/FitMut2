#!/usr/bin/env python3
"""
Run FitMut2 on subdivided environment data (E1 or E2).

Step 1: Initial analysis (no other_env_params)
  python fitmut2_run_subdivided.py -i reads_E1.csv -io reads_E2.csv -t time.csv -al labels.csv -ds 0.03 -dt 6.64 -env E1 -o results_E1_step1

Step 2: Refined analysis (with other_env_params and mean fitness)
  python fitmut2_run_subdivided.py -i reads_E1.csv -io reads_E2.csv -t time.csv -al labels.csv -ds 0.03 -dt 6.64 -env E1 -op results_E2_step1_MutSeq_Result.csv -omf results_E2_step1_Mean_fitness_Result.csv -o results_E1_step2
"""

import argparse
import numpy as np
import pandas as pd
from fitmut2_methods_two_anc_subdivided import FitMut_two_anc_sub

###########################
##### PARSE ARGUMENTS #####
###########################

parser = argparse.ArgumentParser(description='FitMut2 for subdivided environments')

# Required arguments
parser.add_argument('-i', '--input', required=True,
                   help='Input read counts CSV (this environment)')
parser.add_argument('-io', '--input_other', required=True,
                   help='Input read counts CSV (other environment)')
parser.add_argument('-t', '--time', required=True,
                   help='Time points CSV (generations and effective cell numbers)')
parser.add_argument('-al', '--ancestor_labels', required=True,
                   help='Ancestor labels CSV (A or B for each lineage)')
parser.add_argument('-ds', '--delta_s', required=True, type=float,
                   help='Fitness difference in this environment (can be negative)')
parser.add_argument('-dt', '--delta_t', required=True, type=float,
                   help='Generations per cycle')
parser.add_argument('-o', '--output', required=True,
                   help='Output file prefix')

# Optional arguments for Step 2
parser.add_argument('-op', '--other_params', default=None,
                   help='Other environment parameters CSV (for Step 2 refinement)')
parser.add_argument('-omf', '--other_mean_fitness', default=None,
                   help='Other environment mean fitness CSV (for Step 2 refinement)')

# Optional parameters
parser.add_argument('-kappa', '--kappa_value', type=float, default=2.5,
                   help='Kappa value for noise model (default: 2.5)')
parser.add_argument('-c', '--noise_c', type=float, default=2e-3,
                   help='Noise per generation (default: 2e-3)')
parser.add_argument('-Ub', '--Ub', type=float, default=1e-8,
                   help='Beneficial mutation rate (default: 1e-8)')
parser.add_argument('--max_iter', type=int, default=100,
                   help='Maximum iterations (default: 100)')
parser.add_argument('--opt_algorithm', default='differential_evolution',
                   choices=['differential_evolution', 'nelder_mead', 'direct_search'],
                   help='Optimization algorithm (default: differential_evolution)')
parser.add_argument('--parallelize', action='store_true',
                   help='Use parallel processing')
parser.add_argument('--save_steps', action='store_true',
                   help='Save intermediate results')

args = parser.parse_args()

# Load data
r_seq = pd.read_csv(args.input, header=None).values
r_seq_other = pd.read_csv(args.input_other, header=None).values
time_data = pd.read_csv(args.time, header=None).values
ancestor_labels = pd.read_csv(args.ancestor_labels, header=None).values.flatten()

t_list = time_data[:, 0]
cell_depth_list = time_data[:, 1]

# Load other environment parameters if provided (Step 2)
other_env_params = None
other_env_mean_fitness = None

if args.other_params is not None and args.other_mean_fitness is not None:
    print("Step 2: Refined analysis")
    step_label = "_step2"
    other_env_params = pd.read_csv(args.other_params)
    mean_fitness_data = pd.read_csv(args.other_mean_fitness)
    # Extract the last iteration's mean fitness
    other_env_mean_fitness = mean_fitness_data.iloc[:, -1].values
elif args.other_params is not None or args.other_mean_fitness is not None:
    raise ValueError("Both -op and -omf must be provided together for Step 2")
else:
    print("Step 1: Initial analysis (self-approximation)")
    step_label = "_step1"
    
# Add step label to output filename
output_filename = args.output + step_label

# Create FitMut object
my_obj = FitMut_two_anc_sub(
            r_seq=r_seq,
            t_list=t_list,
            cell_depth_list=cell_depth_list,
            Ub=args.Ub,
            delta_t=args.delta_t,
            c=args.noise_c,
            opt_algorithm=args.opt_algorithm,
            max_iter_num=args.max_iter,
            parallelize=args.parallelize,
            save_steps=args.save_steps,
            output_filename=output_filename,
            ancestor_labels=ancestor_labels,
            delta_s=args.delta_s,
            kappa_value=args.kappa_value,
            is_subdivided=True,
            r_seq_other=r_seq_other,
            other_env_params=other_env_params,
            other_env_mean_fitness=other_env_mean_fitness
        )

# Run inference
my_obj.main()

print(f"Results saved to {output_filename}_MutSeq_Result.csv")
