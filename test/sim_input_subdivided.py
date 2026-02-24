#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 14:34:12 2026

@author: ca3258
"""


import pandas as pd
import numpy as np
import random
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#%% Functions:
    
def dX_dt_log_multi(t, X, r, K, delta):
# Logistic Generalized Species Number
# This function can simulate any number of species and outputs a time series of species
# abundances. It should be called like this: 
# X_final = solve_ivp(dX_dt_log_multi, t_span, X_initial, args=(r,K,delta,N)), where:
    # X_initial: length-N 1D array of initial abundances, 
    # t_span: 1D array of starting/ending time steps (e.g. [0, num_hours]),
    # r: length-N 1D array of maximum growth rates,
    # delta: the global death rate, and
    # N: the number of species
    # K: total carrying capacity
    
    N = X.sum()
            
    return X * (r * (1 - N / K) - delta)

def calculate_delta_s(r_A, r_B):
    """
    Calculate delta_s = s_B - s_A.
    
    Determines which ancestor is fitter, sets them as reference (s=0),
    then calculates the fitness difference.
    
    Returns positive if B is fitter, negative if A is fitter.
    """
    if r_B > r_A:
        # B is fitter - A is reference
        s_B = (r_B / r_A) - 1
        delta_s = s_B  # Positive
    elif r_A > r_B:
        # A is fitter - B is reference
        s_A = (r_A / r_B) - 1
        delta_s = -s_A  # Negative
    else:
        # Equal fitness
        delta_s = 0
    
    return delta_s


#%% TWO ANCESTORS, SUBDIVIDED

# 2 x 25,000 lineages, 20 timepoints:
L1 = 25000
L2 = 25000
anc_labels = ['A']*L1+['B']*L2
L = L1+L2
num_muts1 = 500
num_muts2 = 500
tmpts = 20
r_anc1_E1 = 0.4
r_anc2_E1 = 0.42
r_anc1_E2 = 0.42
r_anc2_E2 = 0.4

# Calculate fitness differences:
delta_s_E1 = calculate_delta_s(r_anc1_E1, r_anc2_E1)  
delta_s_E2 = calculate_delta_s(r_anc1_E2, r_anc2_E2)  

K = 1e9
DF = 100
mut_fits1_E1 = np.random.uniform(0.405,0.45,num_muts1)
mut_fits2_E1 = np.random.uniform(0.405,0.45,num_muts2)
mut_fits1_E2 = np.random.uniform(0.405,0.45,num_muts1)
mut_fits2_E2 = np.random.uniform(0.405,0.45,num_muts2)
mut_times1 = random.choices(range(1, tmpts-3), k=num_muts1)
mut_times2 = random.choices(range(1, tmpts-3), k=num_muts2)
pre_fits1_E1 = r_anc1_E1*np.ones(L1)
pre_fits2_E1 = r_anc2_E1*np.ones(L2)
pre_fits_E1 = np.ones(L1+L2)
pre_fits_E1[0:L1] = pre_fits1_E1
pre_fits_E1[L1:L1+L2] = pre_fits2_E1
pre_fits1_E2 = r_anc1_E2*np.ones(L1)
pre_fits2_E2 = r_anc2_E2*np.ones(L2)
pre_fits_E2 = np.ones(L1+L2)
pre_fits_E2[0:L1] = pre_fits1_E2
pre_fits_E2[L1:L1+L2] = pre_fits2_E2
t_span = [0,24]
delta = 0

# Start with ancestral fitness for all lineages:
all_fits_E1 = pre_fits_E1
weights_E1 = pre_fits_E1
all_fits_E2 = pre_fits_E2
weights_E2 = pre_fits_E2
weights_comb = (weights_E1 + weights_E2)/2
p_comb = weights_comb / weights_comb.sum() # normalize
n = K/DF
X_initial = np.random.multinomial(n, p_comb)
track_freqs_E1 = np.zeros([tmpts+1,L])
track_freqs_E2 = np.zeros([tmpts+1,L])

track_freqs_E1[0,:] = X_initial/X_initial.sum()
track_freqs_E2[0,:] = X_initial/X_initial.sum()

for k in range(tmpts):
    for j in range(len(mut_times1)):
        if mut_times1[j] == k:
            all_fits_E1[j] = mut_fits1_E1[j]
            all_fits_E2[j] = mut_fits1_E2[j]
    for j in range(len(mut_times2)):
        if mut_times2[j] == k:
            all_fits_E1[L1+j] = mut_fits2_E1[j]
            all_fits_E2[L1+j] = mut_fits2_E2[j]
    
    X_final_E1 = solve_ivp(dX_dt_log_multi, t_span, X_initial, args=(all_fits_E1,K,delta))
    X_final_E2 = solve_ivp(dX_dt_log_multi, t_span, X_initial, args=(all_fits_E2,K,delta))
    
    track_freqs_E1[k+1,:] = X_final_E1.y[:, -1]/X_final_E1.y[:, -1].sum()
    track_freqs_E2[k+1,:] = X_final_E2.y[:, -1]/X_final_E2.y[:, -1].sum()
    
    weights_comb = (track_freqs_E1[k+1, :] + track_freqs_E2[k+1, :]) / 2
    p_comb = weights_comb / weights_comb.sum() # normalize

    X_initial = np.random.multinomial(n, p_comb)
    
time = np.arange(tmpts+1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# E1
for i in range(L1):
    ax1.plot(time, track_freqs_E1[:, i], alpha=0.1, linewidth=0.5, color='r')
for i in range(L1, L1+L2):
    ax1.plot(time, track_freqs_E1[:, i], alpha=0.1, linewidth=0.5, color='b')
ax1.set_xlabel("Timepoint")
ax1.set_ylabel("Frequency")
ax1.set_title('E1 (blue fitter)')
ax1.set_yscale("log")

# E2
for i in range(L1):
    ax2.plot(time, track_freqs_E2[:, i], alpha=0.1, linewidth=0.5, color='r')
for i in range(L1, L1+L2):
    ax2.plot(time, track_freqs_E2[:, i], alpha=0.1, linewidth=0.5, color='b')
ax2.set_xlabel("Timepoint")
ax2.set_ylabel("Frequency")
ax2.set_title('E2 (red fitter)')
ax2.set_yscale("log")

plt.tight_layout()
plt.show()

#%% Save data:
    
sequencing_depth = 5e6  # 5 million reads per timepoint

read_counts_E1 = np.zeros([tmpts+1, L])
for k in range(tmpts+1):
    reads_E1 = np.random.multinomial(int(sequencing_depth), track_freqs_E1[k,:])
    read_counts_E1[k,:] = reads_E1
    
read_counts_E2 = np.zeros([tmpts+1, L])
for k in range(tmpts+1):
    reads_E2 = np.random.multinomial(int(sequencing_depth), track_freqs_E2[k,:])
    read_counts_E2[k,:] = reads_E2

input_file_E1 = pd.DataFrame(np.transpose(read_counts_E1))
input_file_E1.to_csv('test_input_two_anc_sub_E1.csv', header=None, index=None)
input_file_E2 = pd.DataFrame(np.transpose(read_counts_E2))
input_file_E2.to_csv('test_input_two_anc_sub_E2.csv', header=None, index=None)

delta_t_gen = np.log2(100)  # 6.64 generations per cycle

time_seqs = np.zeros([tmpts+1, 2])
time_seqs[:, 0] = delta_t_gen * np.arange(tmpts+1)  # [0, 6.64, 13.28, ...]
time_seqs[:, 1] = (K/DF) * delta_t_gen             # effective cell number

time_file = pd.DataFrame(time_seqs)
time_file.to_csv('test_time_two_anc_sub.csv',header=None,index=None)

pd.DataFrame(anc_labels).to_csv('test_ancestor_labels_sub.csv', index=False, header=False)


#%% TO CHECK:
    
# First step:
# python main_code/fitmut2_run_two_anc_subdivided_E1E2.py -i1 test/test_input_two_anc_sub_E1.csv -i2 test/test_input_two_anc_sub_E2.csv -t test/test_time_two_anc_sub.csv -al test/test_ancestor_labels_sub.csv -ds1 0.05 -ds2 -0.05 -dt 6.64 -o test_two_anc_sub_results --parallelize


# Second step:
# 
    
    
    
    
    
    
    