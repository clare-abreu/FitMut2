#!/usr/bin/env python3
"""
Automated workflow for subdivided environment analysis.
Runs Step 1 and Step 2 for both E1 and E2 automatically.

Usage:
  python fitmut2_run_subdivided_workflow.py -i1 reads_E1.csv -i2 reads_E2.csv -t time.csv -al labels.csv -ds1 0.03 -ds2 0.04 -dt 6.64 -o results
"""

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description='Automated FitMut2 subdivided environment workflow')

# Required arguments
parser.add_argument('-i1', '--input_E1', required=True,
                   help='Input read counts CSV for E1')
parser.add_argument('-i2', '--input_E2', required=True,
                   help='Input read counts CSV for E2')
parser.add_argument('-t', '--time', required=True,
                   help='Time points CSV')
parser.add_argument('-al', '--ancestor_labels', required=True,
                   help='Ancestor labels CSV')
parser.add_argument('-ds1', '--delta_s_E1', required=True, type=float,
                   help='Fitness difference in E1')
parser.add_argument('-ds2', '--delta_s_E2', required=True, type=float,
                   help='Fitness difference in E2')
parser.add_argument('-dt', '--delta_t', required=True, type=float,
                   help='Generations per cycle')
parser.add_argument('-o', '--output_prefix', required=True,
                   help='Output prefix (e.g., "results" will create results_E1, results_E2)')

# Optional parameters
parser.add_argument('-kappa', '--kappa_value', type=float, default=2.5,
                   help='Kappa value (default: 2.5)')
parser.add_argument('-c', '--noise_c', type=float, default=2e-3,
                   help='Noise per generation (default: 2e-3)')
parser.add_argument('-Ub', '--Ub', type=float, default=1e-8,
                   help='Beneficial mutation rate (default: 1e-8)')
parser.add_argument('--max_iter', type=int, default=100,
                   help='Maximum iterations (default: 100)')
parser.add_argument('--opt_algorithm', default='differential_evolution',
                   help='Optimization algorithm (default: differential_evolution)')
parser.add_argument('--parallelize', action='store_true',
                   help='Use parallel processing')
parser.add_argument('--save_steps', action='store_true',
                   help='Save intermediate results')

args = parser.parse_args()

# Build common arguments
common_args = [
    '-t', args.time,
    '-al', args.ancestor_labels,
    '-dt', str(args.delta_t),
    '-kappa', str(args.kappa_value),
    '-c', str(args.noise_c),
    '-Ub', str(args.Ub),
    '--max_iter', str(args.max_iter),
    '--opt_algorithm', args.opt_algorithm
]

if args.parallelize:
    common_args.append('--parallelize')
if args.save_steps:
    common_args.append('--save_steps')

output_E1 = f"{args.output_prefix}_E1"
output_E2 = f"{args.output_prefix}_E2"

print("=" * 60)
print("SUBDIVIDED ENVIRONMENT WORKFLOW")
print("=" * 60)

# STEP 1: Run E1 and E2 independently
print("\n### STEP 1: Initial Analysis ###\n")

print("Running E1 Step 1...")
cmd_E1_step1 = [
    'python', 'main_code/fitmut2_run_two_anc_subdivided.py',
    '-i', args.input_E1,
    '-io', args.input_E2,
    '-ds', str(args.delta_s_E1),
    '-o', output_E1
] + common_args

result = subprocess.run(cmd_E1_step1)
if result.returncode != 0:
    print("ERROR: E1 Step 1 failed")
    sys.exit(1)

print("\nRunning E2 Step 1...")
cmd_E2_step1 = [
    'python', 'main_code/fitmut2_run_two_anc_subdivided.py',
    '-i', args.input_E2,
    '-io', args.input_E1,
    '-ds', str(args.delta_s_E2),
    '-o', output_E2
] + common_args

result = subprocess.run(cmd_E2_step1)
if result.returncode != 0:
    print("ERROR: E2 Step 1 failed")
    sys.exit(1)

# STEP 2: Run E1 and E2 with other environment's parameters
print("\n### STEP 2: Refined Analysis ###\n")

print("Running E1 Step 2...")
cmd_E1_step2 = [
    'python', 'main_code/fitmut2_run_two_anc_subdivided.py',
    '-i', args.input_E1,
    '-io', args.input_E2,
    '-ds', str(args.delta_s_E1),
    '-op', f"{output_E2}_step1_MutSeq_Result.csv",
    '-omf', f"{output_E2}_step1_Mean_fitness_Result.csv",
    '-o', output_E1
] + common_args

result = subprocess.run(cmd_E1_step2)
if result.returncode != 0:
    print("ERROR: E1 Step 2 failed")
    sys.exit(1)

print("\nRunning E2 Step 2...")
cmd_E2_step2 = [
    'python', 'main_code/fitmut2_run_two_anc_subdivided.py',
    '-i', args.input_E2,
    '-io', args.input_E1,
    '-ds', str(args.delta_s_E2),
    '-op', f"{output_E1}_step1_MutSeq_Result.csv",
    '-omf', f"{output_E1}_step1_Mean_fitness_Result.csv",
    '-o', output_E2
] + common_args

result = subprocess.run(cmd_E2_step2)
if result.returncode != 0:
    print("ERROR: E2 Step 2 failed")
    sys.exit(1)

print(f"\nStep 1 Results:")
print(f"  E1: {output_E1}_step1_MutSeq_Result.csv")
print(f"  E2: {output_E2}_step1_MutSeq_Result.csv")
print(f"\nStep 2 Results:")
print(f"  E1: {output_E1}_step2_MutSeq_Result.csv")
print(f"  E2: {output_E2}_step2_MutSeq_Result.csv")