#!/usr/bin/env python3
"""
Test subdivided code with is_subdivided=False on non-subdivided data
"""
import sys
sys.path.append('main_code')  # Add main_code to path

import numpy as np
import pandas as pd
from fitmut2_methods_two_anc_subdivided import FitMut_two_anc_sub

# Load non-subdivided data
r_seq = pd.read_csv('test/test_input_two_anc.csv', header=None).values
time_data = pd.read_csv('test/test_time_two_anc.csv', header=None).values
ancestor_labels = pd.read_csv('test/test_ancestor_labels.csv', header=None).values.flatten()

t_list = time_data[:, 0]
cell_depth_list = time_data[:, 1]

# Create FitMut object with is_subdivided=FALSE
print("Running subdivided code with is_subdivided=False...")
fm = FitMut_two_anc_sub(
    r_seq=r_seq,
    t_list=t_list,
    cell_depth_list=cell_depth_list,
    Ub=1e-8,
    delta_t=6.64,
    c=2e-3,
    opt_algorithm='differential_evolution',
    max_iter_num=100,
    parallelize=True,
    save_steps=False,
    output_filename='test_nosub_subdcode_nosubdivided',  # Will save in current directory
    ancestor_labels=ancestor_labels,
    delta_s=0.05,
    kappa_value=2.5,
    is_subdivided=False,  # KEY: Turn off subdivided mode
    r_seq_other=None,
    other_env_params=None,
    other_env_mean_fitness=None
)

# Run
fm.main()

print("Done! Results saved to test_nosub_subdcode_nosubdivided_MutSeq_Result.csv")