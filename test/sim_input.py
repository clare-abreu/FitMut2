#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 16:40:40 2026

@author: clare
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


#%% DEFINE LINEAGES

# 10,000 lineages, 20 timepoints:
L = 50000
num_muts = 500
tmpts = 20
r_anc = 0.4
K = 1e9
DF = 100
mut_fits = np.random.uniform(0.41,0.45,num_muts)
mut_times = random.choices(range(1, tmpts-3), k=num_muts)
pre_fits = r_anc*np.ones(L)
all_fits = r_anc*np.ones(L)
all_fits[0:num_muts] = mut_fits
t_span = [0,24]
delta = 0

# Start with ancestral fitness for all lineages:
all_fits = pre_fits
weights = pre_fits
p = weights / weights.sum() # normalize
n = K/DF
X_initial = np.random.multinomial(n, p)
track_freqs = np.zeros([tmpts,L])

for k in range(tmpts):
    for j in range(len(mut_times)):
        if mut_times[j] == k:
            all_fits[j] = mut_fits[j]
    
    track_freqs[k,:] = X_initial/X_initial.sum()
    
    X_final = solve_ivp(dX_dt_log_multi, t_span, X_initial, args=(all_fits,K,delta,L))
    
    weights = X_final.y[:, -1]
    p = weights / weights.sum() # normalize
    n = K/DF
    X_initial = np.random.multinomial(n, p)
    
time = np.arange(tmpts)

plt.figure(figsize=(6, 4))
for i in range(L):
    plt.plot(time, track_freqs[:, i], alpha=0.1, linewidth=0.5)

plt.xlabel("Timepoint")
plt.ylabel("Frequency")
plt.yscale("log")   # optional but usually useful
plt.tight_layout()
plt.show()

#%% Save data:
    
input_file = pd.DataFrame(np.transpose(track_freqs)*K/DF)

input_file.to_csv('test_input.csv',header=None,index=None)

time_seqs = np.zeros([tmpts,2])
time_seqs[:,0] = np.log(100)/np.log(2)*(np.arange(tmpts)+1)
time_seqs[:,1] = (K/DF)*np.log(100)/np.log(2)

time_file = pd.DataFrame(time_seqs)
time_file.to_csv('test_time.csv',header=None,index=None)

#%% Check FitMut results against ground truth:
    
fitmut_results = pd.read_csv('../test_MutSeq_Result.csv')

plt.scatter(all_fits-0.4, fitmut_results['Fitness'])
plt.plot(xlim := plt.xlim(), xlim, '--k')

plt.show()

#%% TWO ANCESTORS

# 2 x 25,000 lineages, 20 timepoints:
L1 = 25000
L2 = 25000
anc_labels = ['A']*L1+['B']*L2
L = L1+L2
num_muts1 = 500
num_muts2 = 500
tmpts = 20
r_anc1 = 0.4
r_anc2 = 0.42
K = 1e9
DF = 100
mut_fits1 = np.random.uniform(0.405,0.45,num_muts1)
mut_fits2 = np.random.uniform(0.425,0.46,num_muts2)
mut_times1 = random.choices(range(1, tmpts-3), k=num_muts1)
mut_times2 = random.choices(range(1, tmpts-3), k=num_muts2)
pre_fits1 = r_anc1*np.ones(L1)
pre_fits2 = r_anc2*np.ones(L2)
pre_fits = np.ones(L1+L2)
pre_fits[0:L1] = pre_fits1
pre_fits[L1:L1+L2] = pre_fits2
#all_fits1 = r_anc1*np.ones(L1)
#all_fits2 = r_anc2*np.ones(L2)
#all_fits1[0:num_muts1] = mut_fits1
#all_fits2[0:num_muts2] = mut_fits2
#all_fits = np.ones(L1+L2)
#all_fits[0:L1] = all_fits1
#all_fits[L1:L1+L2] = all_fits2
t_span = [0,24]
delta = 0

# Start with ancestral fitness for all lineages:
all_fits = pre_fits
weights = pre_fits
p = weights / weights.sum() # normalize
n = K/DF
X_initial = np.random.multinomial(n, p)
track_freqs = np.zeros([tmpts,L])

for k in range(tmpts):
    for j in range(len(mut_times1)):
        if mut_times1[j] == k:
            all_fits[j] = mut_fits1[j]
    for j in range(len(mut_times2)):
        if mut_times2[j] == k:
            all_fits[L1+j] = mut_fits2[j]
    
    track_freqs[k,:] = X_initial/X_initial.sum()
    
    X_final = solve_ivp(dX_dt_log_multi, t_span, X_initial, args=(all_fits,K,delta))
    
    weights = X_final.y[:, -1]
    p = weights / weights.sum() # normalize
    n = K/DF
    X_initial = np.random.multinomial(n, p)
    
time = np.arange(tmpts)

plt.figure(figsize=(6, 4))
for i in range(L1):
    plt.plot(time, track_freqs[:, i], alpha=0.1, linewidth=0.5, color='r')
for i in range(L1,L1+L2):
    plt.plot(time, track_freqs[:, i], alpha=0.1, linewidth=0.5, color='b')

plt.xlabel("Timepoint")
plt.ylabel("Frequency")
plt.yscale("log")   # optional but usually useful
plt.tight_layout()
plt.show()

#%%

plt.plot(X_final.t,X_final.y.sum(0))
plt.yscale('log')
plt.show()


#%% Save data:
    
sequencing_depth = 5e6  # 5 million reads per timepoint
read_counts = np.zeros([tmpts, L])
for k in range(tmpts):
    reads = np.random.multinomial(int(sequencing_depth), track_freqs[k,:])
    read_counts[k,:] = reads

input_file = pd.DataFrame(np.transpose(read_counts))
input_file.to_csv('test_input_two_anc.csv', header=None, index=None)

delta_t_gen = np.log2(100)  # 6.64 generations per cycle

time_seqs = np.zeros([tmpts, 2])
time_seqs[:, 0] = delta_t_gen * np.arange(tmpts)  # [0, 6.64, 13.28, ...]
time_seqs[:, 1] = (K/DF) * delta_t_gen             # effective cell number

time_file = pd.DataFrame(time_seqs)
time_file.to_csv('test_time_two_anc.csv',header=None,index=None)

pd.DataFrame(anc_labels).to_csv('test_ancestor_labels.csv', index=False, header=False)

# Calculate empirical delta_s from tracked frequencies
# Use early timepoints before mutations appear
early_tmpts = 5  # use first 5 timepoints
freq_A_total = track_freqs[:early_tmpts, :L1].sum(axis=1)
freq_B_total = track_freqs[:early_tmpts, L1:].sum(axis=1)

log_ratio = np.log(freq_B_total / freq_A_total)
generations = delta_t_gen * np.arange(early_tmpts)
delta_s_empirical = np.polyfit(generations, log_ratio, 1)[0]
print(f"Empirical delta_s: {delta_s_empirical:.4f}")

#%% Check FitMut_two_anc results against ground truth:
    
fitmut_results = pd.read_csv('../test_two_anc_MutSeq_Result.csv')

plt.scatter(all_fits[0:L1]-0.4, fitmut_results['Fitness'][0:L1],c='r',alpha=0.05)
plt.scatter(all_fits[L1:L1+L2]-0.42, fitmut_results['Fitness'][L1:L1+L2],c='b')
plt.plot(xlim := plt.xlim(), xlim, '--k')

plt.show()
