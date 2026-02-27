#!/usr/bin/env python3
"""
Run FitMut2 on mix-to-mix trajectory (calculated from E1 and E2 data).

Usage:
  python fitmut2_run_mix.py -i1 reads_E1.csv -i2 reads_E2.csv -t time.csv -al labels.csv -ds1 0.03 -ds2 0.04 -dt 6.64 -o results_mix
"""

import argparse
import numpy as np
import pandas as pd
from fitmut2_methods_two_anc_subdivided import FitMut_two_anc_sub, calculate_delta_s_mix

###########################
##### PARSE ARGUMENTS #####
###########################

parser = argparse.ArgumentParser(description='FitMut2 for mix-to-mix trajectory')

# Required arguments
parser.add_argument('-i1', '--input_E1', required=True,
                   help='Input read counts CSV for E1')
parser.add_argument('-i2', '--input_E2', required=True,
                   help='Input read counts CSV for E2')
parser.add_argument('-t', '--time', required=True,
                   help='Time points CSV (generations and effective cell numbers)')
parser.add_argument('-al', '--ancestor_labels', required=True,
                   help='Ancestor labels CSV (A or B for each lineage)')
parser.add_argument('-ds1', '--delta_s_E1', required=True, type=float,
                   help='Fitness difference in E1 (can be negative)')
parser.add_argument('-ds2', '--delta_s_E2', required=True, type=float,
                   help='Fitness difference in E2 (can be negative)')
parser.add_argument('-dt', '--delta_t', required=True, type=float,
                   help='Generations per cycle')
parser.add_argument('-o', '--output', required=True,
                   help='Output file prefix')

# Optional parameters
parser.add_argument('-kappa', '--kappa_value', type=float, default=2.5,
                   help='Kappa value for noise model (default: 2.5)')
parser.add_argument('-c', '--noise_c', type=float, default=1,
                   help='Noise per generation (default: 2e-3)')
parser.add_argument('-Ub', '--Ub', type=float, default=1e-5,
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

###########################
##### CALCULATE MIX #######
###########################

# Load E1 and E2 data
r_seq_E1 = pd.read_csv(args.input_E1, header=None).values
r_seq_E2 = pd.read_csv(args.input_E2, header=None).values

# Calculate frequencies for each environment
freq_E1 = r_seq_E1 / r_seq_E1.sum(axis=0, keepdims=True)  # Each column sums to 1
freq_E2 = r_seq_E2 / r_seq_E2.sum(axis=0, keepdims=True)  # Each column sums to 1

# Average the frequencies
freq_mix = (freq_E1 + freq_E2) / 2

# Convert back to read counts using average total depth
total_reads_E1 = r_seq_E1.sum(axis=0)
total_reads_E2 = r_seq_E2.sum(axis=0)
total_reads_avg = (total_reads_E1 + total_reads_E2) / 2
r_seq_mix = freq_mix * total_reads_avg

# Load other data
time_data = pd.read_csv(args.time, header=None).values
ancestor_labels = pd.read_csv(args.ancestor_labels, header=None).values.flatten()

t_list = time_data[:, 0]
cell_depth_list = time_data[:, 1]

# Calculate delta_s for mix using the subdivided fitness formula
delta_s_mix = calculate_delta_s_mix(args.delta_s_E1, args.delta_s_E2)

print(f"Delta_s_E1: {args.delta_s_E1:.4f}")
print(f"Delta_s_E2: {args.delta_s_E2:.4f}")
print(f"Delta_s_mix (calculated): {delta_s_mix:.4f}")

# The class will automatically determine the reference ancestor
if delta_s_mix >= 0:
    print("Reference ancestor in mix: A (B is fitter)")
else:
    print("Reference ancestor in mix: B (A is fitter)")

###########################
##### RUN FITMUT2 #########
###########################

# Create FitMut object using subdivided class but with is_subdivided=False
# This allows us to use the reference_ancestor logic without label swapping
my_obj = FitMut_two_anc_sub(
            r_seq=r_seq_mix,
            t_list=t_list,
            cell_depth_list=cell_depth_list,
            Ub=args.Ub,
            delta_t=args.delta_t,
            c=args.noise_c,
            opt_algorithm=args.opt_algorithm,
            max_iter_num=args.max_iter,
            parallelize=args.parallelize,
            save_steps=args.save_steps,
            output_filename=args.output,
            ancestor_labels=ancestor_labels,
            delta_s=delta_s_mix,  # Can be positive or negative
            kappa_value=args.kappa_value,
            is_subdivided=False,  # NOT subdivided - just standard two-ancestor
            r_seq_other=None,
            other_env_params=None,
            other_env_mean_fitness=None
        )

# Run inference
my_obj.main()

print(f"Results saved to {args.output}_MutSeq_Result.csv")