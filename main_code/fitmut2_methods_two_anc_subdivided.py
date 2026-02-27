#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy import special
#from scipy.stats import erlang
from scipy.optimize import Bounds
from scipy.optimize import differential_evolution
import math
import itertools
import csv
import time
from multiprocess import Pool, Process
from tqdm import tqdm

# 2/26 update: helper function to calculate delta_s in subdivided environment
def calculate_delta_s_mix(delta_s_E1, delta_s_E2):
    """
    Calculate the effective delta_s for the mix trajectory.
    
    Uses the subdivided fitness formula:
    s_mix = log((exp(s_E1) + exp(s_E2))/2)
    
    With the rule that the less fit ancestor is the reference (s=0) in each environment:
    - If delta_s > 0: B is fitter, so s_A=0, s_B=delta_s
    - If delta_s < 0: A is fitter, so s_A=-delta_s, s_B=0
    
    Parameters:
    -----------
    delta_s_E1 : float
        Fitness difference (s_B - s_A) in environment 1
    delta_s_E2 : float
        Fitness difference (s_B - s_A) in environment 2
        
    Returns:
    --------
    delta_s_mix : float
        Effective fitness difference in the mix
    """
    # Determine absolute fitnesses based on who's reference in each environment
    if delta_s_E1 > 0:
        s_A_E1, s_B_E1 = 0, delta_s_E1
    else:
        s_A_E1, s_B_E1 = -delta_s_E1, 0
    
    if delta_s_E2 > 0:
        s_A_E2, s_B_E2 = 0, delta_s_E2
    else:
        s_A_E2, s_B_E2 = -delta_s_E2, 0
    
    # Calculate mix fitnesses using the subdivided formula
    s_A_mix = np.log((np.exp(s_A_E1) + np.exp(s_A_E2)) / 2)
    s_B_mix = np.log((np.exp(s_B_E1) + np.exp(s_B_E2)) / 2)
    
    # Mix delta_s
    delta_s_mix = s_B_mix - s_A_mix
    
    return delta_s_mix


# fitness inference object
class FitMut_two_anc_sub:
    # 2/26: In this version of the code, you specify A and B populations.
    # B is the HIGHER FITNESS strain.
    # delta_s is the fitness advantage of B compared to A.
    def __init__(self, r_seq,
                       t_list,
                       cell_depth_list,
                       Ub,
                       delta_t, 
                       c,
                       opt_algorithm,
                       max_iter_num,
                       parallelize,
                       save_steps,
                       output_filename,
                       ancestor_labels,
                       delta_s,
                       kappa_value=2.5,
                       # 2/26 update: new parameters for subdivided envs:
                       is_subdivided=False,
                       r_seq_other=None,    # counts from other env
                       other_env_params=None,   # # for step 2 fitness inference refinement
                       other_env_mean_fitness=None):  
        
        # preparing inputs
        self.r_seq = r_seq  # read counts per lineage per timepoint
        self.read_depth_seq = np.sum(self.r_seq, axis=0)    # total reads per timepoint
        self.lineages_num = np.shape(self.r_seq)[0]         # number of lineages
        self.t_list = t_list    # time in generations
        self.seq_num = len(self.t_list)     # number of timepoints
        self.cell_depth_list = cell_depth_list  # effective cell numbers
        
        # Feb 2026: Store ancestor information
        self.ancestor_labels = ancestor_labels  # Array like ['A', 'B', 'A', 'B', ...]
        self.delta_s = delta_s                  # Number like 0.02 (2% advantage for strain B)
        # 2/26 update: Handle signed delta_s and determine reference ancestor
        self.delta_s_signed = delta_s
        self.delta_s = abs(delta_s)
        self.reference_ancestor = 'A' if delta_s >= 0 else 'B'
        
        # 2/26 update: Subdivided environment parameters:
        self.is_subdivided = is_subdivided
        
        if self.is_subdivided:
            # Other environment's read counts (for mixing)
            self.r_seq_other = r_seq_other
            
            # Convert to estimated cells
            ratio_other = np.sum(self.r_seq_other, axis=0) / self.cell_depth_list
            self.n_seq_other = self.r_seq_other / ratio_other
            self.n_seq_other[self.n_seq_other < 1] = 1
            
            # Other environment's inferred parameters (for Step 2 refinement)
            # Format: DataFrame with columns ['s', 'tau', 'p_adaptive']
            self.other_env_params = other_env_params
            
            if self.other_env_params is not None:
                # Extract arrays for quick access
                self.s_other_array = self.other_env_params['Fitness'].values
                self.tau_other_array = self.other_env_params['Establishment_Time'].values
                
                # Add mean fitness from the other env:
                if other_env_mean_fitness is None:
                    raise ValueError("Step 2 requires other_env_mean_fitness")
                self.s_mean_other = other_env_mean_fitness  # Array: (seq_num,)
                
                # Will be calculated after calculate_E() is called
                self.E_other_extend = None
                self.E_other_extend_t_list = None
            else:
                # Step 1: no other environment params yet
                self.s_other_array = None
                self.tau_other_array = None
                self.s_mean_other = None         
                self.E_other_extend = None         
                self.E_other_extend_t_list = None  
        else:
            # Not subdivided - set to None
            self.r_seq_other = None
            self.n_seq_other = None
            self.other_env_params = None
            self.s_other_array = None
            self.tau_other_array = None
            self.s_mean_other = None          
            self.E_other_extend = None         
            self.E_other_extend_t_list = None  
                
        # Feb 2026: Create boolean masks for convenience
        self.is_ancestor_A = (ancestor_labels == 'A')  # True/False array
        self.is_ancestor_B = (ancestor_labels == 'B')  # True/False array
    
        # convert reads to estimated cells:
        self.ratio = self.read_depth_seq/self.cell_depth_list
        self.n_seq = self.r_seq / self.ratio
        # Use kappa constant:
        self.kappa_constant = kappa_value

        # eliminates zeros from data for later convenience -- also have to modify theoretical model
        # in order to not classify neutrals as adaptive
        # It should be possible to modify this ad hoc choice since we can treat the r_theory=0 case
        # of the bessel function separately when calculating log likelihood of a trajectory.
        self.n_seq[self.n_seq < 1] = 1 
        self.r_seq[self.r_seq < 1] = 1

        self.Ub = Ub
        self.delta_t = delta_t
        self.noise_c = c # noise per generation, effective
        
        self.opt_algorithm = opt_algorithm
        self.max_iter_num = max_iter_num
        self.parallelize = parallelize
        #self.parallelize = False

        self.save_steps = save_steps   
        self.output_filename = output_filename


        # set bounds for the optimization
        if self.opt_algorithm == 'differential_evolution':
            self.bounds = Bounds([1e-8, -100], [.5, math.floor(self.t_list[-1] - 1)])
        elif self.opt_algorithm == 'nelder_mead':
            self.bounds = [[1e-8, .5], [-100, math.floor(self.t_list[-1] - 1)]]
        
        
        # define other variables
        self.s_mean_seq_dict = dict() # mean fitness at each iteration
        self.mutant_fraction_dict = dict() # fraction of mutatant cells at each iteration
        
        self.iteration_stop_threhold = 5e-7 # threshold for terminating iterations
        self.threshold_adaptive = 0.3#0.9 # threshold for determining an adaptive lineage
        self.iter_timing_list = [] # running time for each iteration

        # define some variables for convenient vectorization computing
        self.s_stepsize = 0.02
        self.tau_stepsize = 5
        
        self.s_bin = np.arange(0, 0.4, self.s_stepsize)
        self.s_bin[0] = 1e-8
        if len(self.s_bin)%2 == 0:
            self.s_bin = self.s_bin[:-1]

        self.tau_bin = np.arange(-100, self.t_list[-1]-1, self.tau_stepsize)
        if len(self.tau_bin)%2 == 0:
            self.tau_bin = self.tau_bin[:-1]
        
        self.s_coeff = np.array([1] + [4, 2] * int((len(self.s_bin)-3)/2) + [4, 1])
        self.tau_coeff = np.array([1] + [4, 2] * int((len(self.tau_bin)-3)/2) + [4, 1])
             
        # part of joint distribution       
        self.mu_s_mean = 0.1
        self.f_s_tau_joint_log_part = self.get_log_prior_mu(self.s_bin,self.tau_bin)

        # finer grids for direct seach of parameters
        self.s_bin_fine = np.arange(0,self.s_bin[-1],.005)
        self.s_bin_fine[0] = 1e-8
        self.tau_bin_fine = np.arange(self.tau_bin[0],self.tau_bin[-1],2)
        self.f_s_tau_joint_log_part_fine = self.get_log_prior_mu(self.s_bin_fine,self.tau_bin_fine)

    ##########
    def get_log_prior_mu(self,s_array,tau_array):
        """
        Calculate log of the prior (exponential) distribution for mu(s). Returns a 2D array.
        """
        s_len = len(s_array)
        tau_len = len(tau_array)
        mu_s_mean = self.mu_s_mean
        joint_dist1 = np.tile(np.log(self.Ub), (s_len, tau_len))
        joint_dist2 = np.tile(np.log(mu_s_mean), (s_len, tau_len))
        joint_dist3 = np.tile(s_array/mu_s_mean, (tau_len, 1))
        joint_dist4 = np.transpose(joint_dist3, (1,0))
        return joint_dist1 - joint_dist2 - joint_dist4
              
    
    def calculate_kappa(self):
        """
        Calculate kappa value for each timepoint by finding 
        mean and variance of distribution of read number for 
        neutral lineages.
        """
        #self.kappa_seq = np.nan * np.zeros(self.seq_num, dtype=float)
        #self.kappa_seq[0] = 2.5
        
        # Keep kappa at 2.5:
        self.kappa_seq = self.kappa_constant * np.ones(self.seq_num, dtype=float)

        """
        for k in range(self.seq_num-1):
            
            
            r_t1_left, r_t1_right = 20, 40 # neutral lineages with read numbers in [20, 40)
            r_t2_left, r_t2_right = 0, 4*r_t1_right
        
            kappa = np.nan * np.zeros(r_t1_right - r_t1_left, dtype=float)
                        
            for r_t1 in range(r_t1_left, r_t1_right):
                pos = self.r_seq[:,k] == r_t1
                
                if np.sum(pos)>100:
                    pdf_conditional_measure = np.histogram(self.r_seq[pos, k+1],
                                                           bins=np.arange(r_t2_left, r_t2_right+0.001),
                                                           density=True)[0]
            
                    dist_x = np.arange(r_t2_left, r_t2_right)
                    param_mean = np.matmul(dist_x, pdf_conditional_measure)
                    param_variance = np.matmul((dist_x - param_mean)**2, pdf_conditional_measure)
                    
                    kappa[r_t1 - r_t1_left] = param_variance/(2*param_mean)
                
                if np.sum(~np.isnan(kappa)): # if not all values of kappa are nan
                    self.kappa_seq[k+1] = np.nanmean(kappa)
                
                       
        pos_nan = np.isnan(self.kappa_seq)
        if np.sum(pos_nan): # if there is nan type in self.kappa_seq
            self.kappa_seq[pos_nan] = np.nanmean(self.kappa_seq) # define the value as the mean of all non-nan
            
        """
        

    ##########
    def calculate_E(self):
        """
        Pre-calculate a term (capturing the decay in lineage size from the mean fitness)
        to reduce calculations in estimating the number of reads.
        """
        self.t_list_extend = np.concatenate((-np.arange(self.delta_t, 100+self.delta_t, self.delta_t)[::-1], self.t_list))
        seq_num_extend = len(self.t_list_extend)
        #self.s_mean_seq_extend = np.concatenate((1e-8 * np.ones(len(self.t_list_extend) - self.seq_num, dtype=float), self.s_mean_seq))
        self.s_mean_seq_extend = np.concatenate((np.zeros(len(self.t_list_extend) - self.seq_num, dtype=float), self.s_mean_seq))

        self.E_extend = np.ones(seq_num_extend, dtype=float)  
        log_E = 0
        for k in range(1, seq_num_extend):
            #log_E += (self.t_list_extend[k] - self.t_list_extend[k-1]) * self.s_mean_seq_extend[k]
            log_E += (self.t_list_extend[k]-self.t_list_extend[k-1]) * (self.s_mean_seq_extend[k] + self.s_mean_seq_extend[k-1])/2
            self.E_extend[k] = np.exp(-log_E)

        self.E_extend_t_list = np.interp(self.t_list, self.t_list_extend, self.E_extend) # from the very beginning to tk
        
        self.E_t_list = np.zeros(self.seq_num-1, dtype=float) # from tkminus1 to tk
        log_E_t = 0
        for k in range(self.seq_num-1):
            #log_E_t = (self.t_list[k+1]-self.t_list[k]) * (self.s_mean_seq[k+1] + self.s_mean_seq[k])/2  
            # 2/26 update: For subdivided, use mixed mean fitness as starting point
            if self.is_subdivided and hasattr(self, 's_mean_mix_seq'):
                # Integrate from mixed population (after mixing) to this env's endpoint
                log_E_t = (self.t_list[k+1]-self.t_list[k]) * (self.s_mean_seq[k+1] + self.s_mean_mix_seq[k])/2
            else:
                # Original: smooth change within environment
                log_E_t = (self.t_list[k+1]-self.t_list[k]) * (self.s_mean_seq[k+1] + self.s_mean_seq[k])/2
            self.E_t_list[k] =  np.exp(-log_E_t)  

    


    ##########
    # 2/26 update: calculate unmutant in other env in step 2:
    def calculate_other_env_mutant_fraction(self, lineage_idx, k, s_other, tau_other):
        """
        Calculate mutant fraction in the other environment at timepoint k.
        
        This is a simplified trajectory calculation that returns just the mutant
        fraction, not the full trajectory. Used during Step 2 refinement when
        we have inferred (s, tau) from the other environment's Step 1 analysis.
        
        Parameters:
        -----------
        lineage_idx : int
            Index of the lineage being analyzed
        k : int
            Timepoint index
        s_other : float
            Inferred fitness in other environment
        tau_other : float
            Inferred establishment time in other environment
            
        Returns:
        --------
        mutant_frac : float
            Fraction of cells that are mutants (between 0 and 1)
        """
        # If before establishment, no mutants
        if self.t_list[k] < tau_other:
            return 0.0
        
        # Observed total cells in other environment
        n_obs_other = self.n_seq_lineage_other[k]
        
        if n_obs_other < 1:
            return 0.0
        
        # Get mean fitness at establishment time in OTHER environment
        if self.s_mean_other is not None:
            # Step 2: Use actual other environment's mean fitness
            # Extend it like we do for this environment
            s_mean_seq_other_extend = np.concatenate((
                np.zeros(len(self.t_list_extend) - self.seq_num),
                self.s_mean_other
            ))
            s_mean_tau_other = np.interp(tau_other, self.t_list_extend, 
                                           s_mean_seq_other_extend)
        else:
            # Step 1: Shouldn't happen, but fallback to this env's mean fitness
            s_mean_tau_other = np.interp(tau_other, self.t_list_extend, 
                                           self.s_mean_seq_extend)
        
        # Calculate total fitness in other environment
        # Note: The OTHER environment might have different reference ancestor!
        # For now, assume same reference ancestor logic applies
        # (This is a simplification - we could pass reference_ancestor_other if needed)
        if self.lineage_ancestor == self.reference_ancestor:
            total_fitness_other = s_other
        else:
            total_fitness_other = self.delta_s + s_other
        
        # Establishment size
        est_size = self.noise_c / np.maximum(total_fitness_other - s_mean_tau_other, 0.005)
        
        # Get E factor at timepoint k for OTHER environment
        if self.E_other_extend is not None:
            # Use pre-calculated E for other environment
            E_extend_tau_other = np.interp(tau_other, self.t_list_extend, self.E_other_extend)
            E_other_k = self.E_other_extend_t_list[k] / E_extend_tau_other
        else:
            # Fallback: approximate with this environment's E
            E_extend_tau = np.interp(tau_other, self.t_list_extend, self.E_extend)
            E_other_k = self.E_extend_t_list[k] / E_extend_tau
        
        # Calculate mutant cells
        mutant_growth = np.exp(total_fitness_other * (self.t_list[k] - tau_other))
        mutant_n = est_size * mutant_growth * E_other_k
        mutant_n = np.minimum(mutant_n, n_obs_other)
        
        # Return fraction
        mutant_frac = mutant_n / n_obs_other
        
        return mutant_frac

    ##########
    # 2/26 update: calculate E factor of other env in step 2:
    def calculate_E_other(self):
        """
        Calculate E factor for the other environment using its mean fitness.
        
        This is called during Step 2 when we have the other environment's
        mean fitness trajectory from its Step 1 analysis.
        
        Must be called AFTER calculate_E() has been called for this environment
        (so self.t_list_extend is available).
        """
        if self.s_mean_other is None:
            raise ValueError("Cannot calculate E_other without s_mean_other")
        
        # Extend the other environment's mean fitness (same as this env)
        seq_num_extend = len(self.t_list_extend)
        s_mean_seq_other_extend = np.concatenate((
            np.zeros(seq_num_extend - self.seq_num, dtype=float),
            self.s_mean_other
        ))
        
        # Calculate E_extend for other environment
        self.E_other_extend = np.ones(seq_num_extend, dtype=float)
        log_E = 0
        for k in range(1, seq_num_extend):
            # Trapezoidal integration
            log_E += (self.t_list_extend[k] - self.t_list_extend[k-1]) * \
                     (s_mean_seq_other_extend[k] + s_mean_seq_other_extend[k-1]) / 2
            self.E_other_extend[k] = np.exp(-log_E)
        
        # Interpolate to get E at observation timepoints
        self.E_other_extend_t_list = np.interp(self.t_list, self.t_list_extend, 
                                                self.E_other_extend)

    ##########
    def establishment_size_scalar(self, s, tau):
        """
        Calculate establishment size of a mutation with fitness effect s and establishment time tau.
        Inputs: s (scalar)
                tau (scalar)
        Output: established_size (scalar)
        """
        s_mean_tau = np.interp(tau, self.t_list_extend, self.s_mean_seq_extend)
        
        # 2/26 update: Use reference_ancestor logic
        if self.lineage_ancestor == self.reference_ancestor:
            total_fitness = s
        else:
            total_fitness = self.delta_s + s
        
        #established_size = self.noise_c / np.maximum(s - s_mean_tau, 0.005)
        established_size = self.noise_c / np.maximum(total_fitness - s_mean_tau, 0.005)

        return established_size
            

    ##########
    def establishment_size_array(self, s_array, tau_array):
        """
        Calculate establishment size of a mutation with fitness effect s and establishment time tau.
        Inputs: s_array (array, vector)
                tau_array (array, vector) 
        Output: established_size (array, 2D matrix)
        """
        
        # Debug:
        if not hasattr(self, '_debug_delta_s'):
            print(f"DEBUG: self.delta_s = {self.delta_s}, self.reference_ancestor = {getattr(self, 'reference_ancestor', 'N/A')}")
            self._debug_delta_s = True
            
        s_len = len(s_array)
        tau_len = len(tau_array)

        s_matrix = np.transpose(np.tile(s_array, (tau_len, 1)), (1,0))
        
        # 2/26 update: Use reference_ancestor logic
        if self.lineage_ancestor == self.reference_ancestor:
            total_fitness_matrix = s_matrix
        else:
            total_fitness_matrix = s_matrix + self.delta_s

        s_mean_tau = np.tile(np.interp(tau_array, self.t_list_extend, self.s_mean_seq_extend), (s_len, 1)) #(s_len, tau_len)
        
        # Debug:
        if not hasattr(self, '_debug_s_mean_tau'):
            s_idx = np.argmin(np.abs(s_array - 0.10))
            tau_idx = np.argmin(np.abs(tau_array - 130))
            print(f"  s_mean_tau at tau=130: {s_mean_tau[s_idx, tau_idx]:.6f}")
            self._debug_s_mean_tau = True
        
        #established_size = self.noise_c / np.maximum(s_matrix - s_mean_tau, 0.005)
        established_size = self.noise_c / np.maximum(total_fitness_matrix - s_mean_tau, 0.005)

        return established_size


  
    ##########
    def n_theory_scalar(self, s, tau):
        """
        Estimate cell number & mutant cell number all time points for a lineage given s and tau. 
        Inputs: s (scalar)
                tau (scalar)  
        Output: {'cell_number': (array, vector), 
                 'mutant_cell_number': (array, vector)}
        """            
        n_obs = self.n_seq_lineage
        
        n_theory = np.zeros(self.seq_num, dtype=float)
        n_theory[0] = n_obs[0]
        mutant_n_theory = np.zeros(self.seq_num, dtype=float)
        unmutant_n_theory = np.zeros(self.seq_num, dtype=float)
        
        established_size = self.establishment_size_scalar(s, tau)
        E_extend_tau = np.interp(tau, self.t_list_extend, self.E_extend)
        
        for k in range(self.seq_num):
            # For each timepoint, calculate lineage growth, split between mutant/unmutant cells:
            E_tk_minus_tau = self.E_extend_t_list[k]/E_extend_tau
            #mutant1 = np.exp(s * (self.t_list[k] - tau))
            # 2/26 update: Use reference_ancestor logic
            if self.lineage_ancestor == self.reference_ancestor:
                mutant1 = np.exp(s * (self.t_list[k] - tau))
            else:
                mutant1 = np.exp((self.delta_s + s) * (self.t_list[k] - tau))
            mutant2 = established_size*mutant1
            mutant3 = mutant2*E_tk_minus_tau                      
            mutant_n_theory[k] = np.minimum(mutant3, n_obs[k])
            
            unmutant_n_theory[k] = n_obs[k] - mutant_n_theory[k]

            # 2/26 update: Calculate lineage size for next timepoint
            if k > 0:
                E_tk_minus_tkminus1 = self.E_t_list[k-1]
                dt = self.t_list[k] - self.t_list[k-1]
                
                # 2/26 update: For subdivided, calculate mix first
                if self.is_subdivided:
                    # Get other environment's total cells
                    n_other_total = self.n_seq_lineage_other[k-1]
                    
                    # Approximate other environment's mutant/unmutant split
                    if self.s_other_array is not None:
                        # Step 2: Use other environment's inferred parameters
                        mutant_frac_other = self.calculate_other_env_mutant_fraction(
                            self.current_lineage_index, k-1, 
                            self.s_other_array[self.current_lineage_index],
                            self.tau_other_array[self.current_lineage_index]
                        )
                    else:
                        # Step 1: Use this environment's fraction as approximation
                        n_this_total = n_obs[k-1]
                        mutant_frac_other = mutant_n_theory[k-1] / n_this_total if n_this_total > 0 else 0
                    
                    # Calculate other environment's mutant/unmutant cells
                    n_other_mutant = n_other_total * mutant_frac_other
                    n_other_unmutant = n_other_total * (1 - mutant_frac_other)
                    
                    # Calculate mixed mutant/unmutant populations
                    n_mix_mutant = (mutant_n_theory[k-1] + n_other_mutant) / 2
                    n_mix_unmutant = (unmutant_n_theory[k-1] + n_other_unmutant) / 2
                    
                    # Grow from mix in this environment
                    # 2/26 update: Use reference_ancestor logic instead of checking for 'B'
                    if self.lineage_ancestor == self.reference_ancestor:
                        # Reference ancestor (no background advantage)
                        unmutant_growth = 1.0
                        mutant_growth = np.exp(s * dt)
                    else:
                        # Non-reference ancestor (has background advantage delta_s)
                        unmutant_growth = np.exp(self.delta_s * dt)
                        mutant_growth = np.exp((self.delta_s + s) * dt)
                    
                    lineage_size0 = (n_mix_unmutant * unmutant_growth + 
                                    n_mix_mutant * mutant_growth)
                
                else:
                    # Original two-ancestor logic (no mixing)
                    if self.lineage_ancestor == self.reference_ancestor:
                        mutant_growth = np.exp(s * dt)
                        lineage_size0 = (unmutant_n_theory[k-1] +  mutant_n_theory[k-1] * mutant_growth)
                    else:
                        unmutant_growth = np.exp(self.delta_s * dt)
                        mutant_growth = np.exp((self.delta_s + s) * dt)
                        lineage_size0 = (unmutant_n_theory[k-1] * unmutant_growth + 
                        mutant_n_theory[k-1] * mutant_growth)
                    
                    #lineage_size0 = (unmutant_n_theory[k-1] * unmutant_growth + mutant_n_theory[k-1] * mutant_growth)
                
                n_theory[k] = lineage_size0 * E_tk_minus_tkminus1
            
        return {'cell_number': n_theory,'mutant_cell_number': mutant_n_theory}
    


    ##########
    def n_theory_array(self, s_array, tau_array):
        """
        Estimate cell number & mutant cell number all time points for a lineage given s and tau. 
        Inputs: s_array (array, vector)
                tau_array (array, vector) 
        Output: {'cell_number': (array, 3D matrix), 
                 'mutant_cell_number': (array, 3D matrix)}
        """
        s_len = len(s_array)
        tau_len = len(tau_array)

        s_matrix = np.transpose(np.tile(s_array, (tau_len, 1)), (1,0))
        tau_matrix  = np.tile(tau_array, (s_len, 1))
            
        n_obs = np.tile(self.n_seq_lineage, (s_len, tau_len, 1))
        
        n_theory = np.zeros((s_len, tau_len, self.seq_num), dtype=float)
        n_theory[:,:,0] = n_obs[:,:,0]
        mutant_n_theory = np.zeros((s_len, tau_len, self.seq_num), dtype=float)
        unmutant_n_theory = np.zeros((s_len, tau_len, self.seq_num), dtype=float)
            
        established_size = self.establishment_size_array(s_array, tau_array) #(s_len, tau_len)
        
        # Debug
        if not hasattr(self, '_debug_est_size'):
            self._debug_est_size = True
            # Find the grid point closest to s=0.10, tau=130
            s_idx = np.argmin(np.abs(s_array - 0.10))
            tau_idx = np.argmin(np.abs(tau_array - 130))
            print(f"  Est size at s={s_array[s_idx]}, tau={tau_array[tau_idx]}: {established_size[s_idx, tau_idx]:.2f}")
        
        E_extend_tau = np.tile(np.interp(tau_array, self.t_list_extend, self.E_extend), (s_len, 1)) #(s_len, tau_len)
        
        for k in range(self.seq_num):
            E_tk_minus_tau = self.E_extend_t_list[k]/E_extend_tau
            #mutant1 = np.exp(s_matrix * (self.t_list[k] - tau_matrix))
            # 2/26 update: Use reference_ancestor logic
            if self.lineage_ancestor == self.reference_ancestor:
                total_fitness_matrix = s_matrix
            else:
                total_fitness_matrix = s_matrix + self.delta_s
            
            mutant1 = np.exp(total_fitness_matrix * (self.t_list[k] - tau_matrix))
            
            mutant2 = established_size*mutant1
            mutant3 = mutant2*E_tk_minus_tau                     
            mutant_n_theory[:,:,k] = np.minimum(mutant3, n_obs[:,:,k])
            
            unmutant_n_theory[:,:,k] = n_obs[:,:,k] - mutant_n_theory[:,:,k]
                
            # 2/26 update: use unmutant fraction differenty in step 1 vs. 2:
            if k > 0:
                E_tk_minus_tkminus1 = self.E_t_list[k-1]
                dt = self.t_list[k] - self.t_list[k-1]
                
                if self.is_subdivided:
                    # Other environment's total cells (scalar)
                    n_other_total = self.n_seq_lineage_other[k-1]
                    
                    # Determine other environment's mutant/unmutant split
                    if self.s_other_array is not None:
                        # Step 2: Use inferred parameters from other environment
                        mutant_frac_other = self.calculate_other_env_mutant_fraction(
                            self.current_lineage_index, k-1,
                            self.s_other_array[self.current_lineage_index],
                            self.tau_other_array[self.current_lineage_index]
                        )
                        # Broadcast scalar to grid
                        n_other_mutant = n_other_total * mutant_frac_other
                        n_other_unmutant = n_other_total * (1 - mutant_frac_other)
                    else:
                        # Step 1: Use each grid point's own fraction
                        mutant_frac_grid = mutant_n_theory[:, :, k-1] / np.maximum(n_obs[:, :, k-1], 1)
                        # Broadcast scalar × grid
                        n_other_mutant = n_other_total * mutant_frac_grid
                        n_other_unmutant = n_other_total * (1 - mutant_frac_grid)
                    
                    # Calculate mixed populations
                    n_mix_mutant = (mutant_n_theory[:, :, k-1] + n_other_mutant) / 2
                    n_mix_unmutant = (unmutant_n_theory[:, :, k-1] + n_other_unmutant) / 2
                    
                    # Grow from mix using reference ancestor logic
                    if self.lineage_ancestor == self.reference_ancestor:
                        unmutant_growth = 1.0
                        mutant_growth_matrix = np.exp(s_matrix * dt)
                    else:
                        unmutant_growth = np.exp(self.delta_s * dt)
                        total_fitness_matrix = s_matrix + self.delta_s
                        mutant_growth_matrix = np.exp(total_fitness_matrix * dt)
                    
                    lineage_size0 = (n_mix_unmutant * unmutant_growth + 
                                    n_mix_mutant * mutant_growth_matrix)
                
                else:
                    # Original two-ancestor logic (no mixing)
                    if self.lineage_ancestor == self.reference_ancestor:
                        #unmutant_growth = 1.0
                        mutant_growth_matrix = np.exp(s_matrix * dt)
                        lineage_size0 = (unmutant_n_theory[:, :, k-1] + mutant_n_theory[:, :, k-1] * mutant_growth_matrix)
                        
                    else:
                        unmutant_growth = np.exp(self.delta_s * dt)
                        total_fitness_matrix = s_matrix + self.delta_s
                        mutant_growth_matrix = np.exp(total_fitness_matrix * dt)
                        lineage_size0 = (unmutant_n_theory[:, :, k-1] * unmutant_growth + mutant_n_theory[:, :, k-1] * mutant_growth_matrix)
                    
                    #lineage_size0 = (unmutant_n_theory[:, :, k-1] * unmutant_growth + mutant_n_theory[:, :, k-1] * mutant_growth_matrix)
                
                n_theory[:, :, k] = lineage_size0 * E_tk_minus_tkminus1
    
        return {'cell_number': n_theory,'mutant_cell_number': mutant_n_theory}



    ##########
    def loglikelihood_scalar(self, s, tau):
        """
        Calculate log-likelihood value of a lineage given s and tau.
        Inputs: s(scalar)
                tau (scalar) 
        Output: log-likelihood value of all time points (scalar)
        """        
        n_theory = self.n_theory_scalar(s, tau)['cell_number']
        r_theory = n_theory*self.ratio

        # modifies theoretical read number so that one can compare to modified data without zeros
        r_theory[r_theory < 1] = 1
        
        kappa_inverse = 1/self.kappa_seq

        r_obs = self.r_seq_lineage # observed read count
        ive_arg = 2*np.sqrt(r_theory*r_obs)*kappa_inverse
 
        part2 = 1/2 * np.log(r_theory/r_obs)
        part3 = -(r_theory + r_obs)*kappa_inverse
        part4 = np.log(special.ive(1, ive_arg)) + ive_arg

        log_likelihood_seq_lineage = np.log(kappa_inverse) + part2 + part3 + part4
        log_likelihood_lineage = np.sum(log_likelihood_seq_lineage, axis=0)     # sum over all timepoints

        return log_likelihood_lineage
        
    
    ##########
    def loglikelihood_array(self, s_array, tau_array):
        """
        Calculate log-likelihood value of a lineage given s and tau.
        Inputs: s_array (array, vector)
                tau_array (array, vector) 
        Output: log-likelihood value of all time points (array, 2D matrix)
        """
        s_len = len(s_array)
        tau_len = len(tau_array)

        n_theory = self.n_theory_array(s_array, tau_array)['cell_number'] #(s_len, tau_len, seq_num)
        r_theory = n_theory*np.tile(self.ratio, (s_len, tau_len, 1))
        
        # modifies theoretical read number so that one can compare to modified data without zeros
        r_theory[r_theory < 1] = 1
        
        kappa_inverse = np.tile(1/self.kappa_seq, (s_len, tau_len, 1))

        r_obs = np.tile(self.r_seq_lineage, (s_len, tau_len, 1))
        r_obs_inverse = np.tile(1/self.r_seq_lineage, (s_len, tau_len, 1))
        ive_arg = 2*np.sqrt(r_theory*r_obs)*kappa_inverse
 
        part2 = 1/2 * np.log(r_theory*r_obs_inverse)
        part3 = -(r_theory + r_obs)*kappa_inverse
        part4 = np.log(special.ive(1, ive_arg)) + ive_arg

        log_likelihood_seq_lineage = np.log(kappa_inverse) + part2 + part3 + part4
        log_likelihood_lineage = np.sum(log_likelihood_seq_lineage, axis=2)

        return log_likelihood_lineage
    


    ##########
    def posterior_loglikelihood_scalar(self, s, tau):
        """
        Calculate posterior log-likelihood value of a lineage given s and tau.
        Inputs: s (scalar)
                tau (scalar) 
        Output: log-likelihood value of all time poins (scalar)
        """
        mu_s_mean = self.mu_s_mean
        n0 = self.n_seq_lineage[0]
        # weights the prior by lineage size and establishment probability

        # exponential prior
        f_s_tau_joint_log = np.log(self.Ub) - np.log(mu_s_mean) - s/mu_s_mean + np.log(s/self.noise_c * n0)
        # erlang prior
        # f_s_tau_joint_log =  np.log(self.Ub) + np.log(4*s/mu_s_mean**2) - 2*s/mu_s_mean + np.log(s*n0)
        # uniform prior
        # f_s_tau_joint_log =  np.log(self.Ub)  - np.log(2*mu_s_mean) + np.log(s*n0)
        
        return self.loglikelihood_scalar(s, tau) + f_s_tau_joint_log

    
    
    ##########
    def posterior_loglikelihood_array(self, s_array, tau_array,fine=False):
        """
        Calculate posterior log-likelihood value of a lineage given s and tau.
        Calculates the log likelihood on a finer grid if specified
        Inputs: s_array (array, vector)
                tau_array (array, vector)
                fine (boolean)
        Output: log-likelihood value of all time points (array, 2D matrix)
        """
        tau_len = len(tau_array)
        n0 = self.n_seq_lineage[0]

        # weights the prior by lineage size and establishment probability
        joint_dist_T = np.tile(np.log(s_array/self.noise_c  * n0), (tau_len, 1))
        joint_dist = np.transpose(joint_dist_T, (1,0))
        if not fine:
            f_s_tau_joint_log = self.f_s_tau_joint_log_part + joint_dist  # exponential prior distribution
        else:
            f_s_tau_joint_log = self.f_s_tau_joint_log_part_fine + joint_dist

        return self.loglikelihood_array(s_array, tau_array) + f_s_tau_joint_log


    ##########
    def log_ratio_adaptive_integral(self, s_array, tau_array):
        """
        probability of a lineage trajectory, given an array of s and tau (using integral method)
        output is scalar, given by the probability integrated over a grid of s and tau
        Also returns the indices of s and tau in the input arrays which gave the highest probability
        """
        integrand_log = self.posterior_loglikelihood_array(s_array, tau_array)
        
        # Debug: for lineage 0 only
        if not hasattr(self, '_debug_printed_lineage_0'):
            self._debug_printed_lineage_0 = False
        
        if not self._debug_printed_lineage_0:
            max_val = np.max(integrand_log)
            s_idx, tau_idx = np.unravel_index(np.argmax(integrand_log), np.shape(integrand_log))
            print(f"  Lineage 0 Grid max: {max_val:.2f} at s={s_array[s_idx]:.4f}, tau={tau_array[tau_idx]:.1f}")
            self._debug_printed_lineage_0 = True
        
        log_amp_factor = -np.max(integrand_log) + 2
        amp_integrand = np.exp(integrand_log + log_amp_factor)

        s_idx,tau_idx = np.unravel_index(np.argmax(integrand_log),np.shape(integrand_log))
        integral_result = np.dot(np.dot(self.s_coeff, amp_integrand), self.tau_coeff)
        amp_integral = integral_result * self.s_stepsize * self.tau_stepsize / 9
        return np.log(amp_integral) - log_amp_factor,s_idx,tau_idx

    

    ##########
    def posterior_loglikelihood_opt(self, x):
        """
        Calculate posterior log-likelihood value of a lineage given s and tau in optimization
        """
        s, tau = np.maximum(x[0], 1e-8), x[1]
        return -self.posterior_loglikelihood_scalar(s, tau) #minimization only in python

    ##########
    def run_parallel(self, i): 
        """
        i: lineage label
        calculate probability first, then for adaptive lineage output optimized s and tau
        """
        self.r_seq_lineage = self.r_seq[i, :]
        self.n_seq_lineage = self.n_seq[i, :]
        
        # 2/26: Set which ancestor this lineage is from
        self.lineage_ancestor = self.ancestor_labels[i]  # 'A' or 'B'
        
        # 2/26 update: For subdivided, set other environment's data and current index
        if self.is_subdivided:
            self.n_seq_lineage_other = self.n_seq_other[i, :]
            self.current_lineage_index = i
        
        p_ratio_log_adaptive,s_idx,tau_idx = self.log_ratio_adaptive_integral(self.s_bin, self.tau_bin)
        # Calculate prob(neutral):
        # For A lineage: s=0 means fitness = 0
        # For B lineage: s=0 means no NEW mutation, but fitness = delta_s
        #   (n_theory_scalar automatically handles B via self.lineage_ancestor)
        p_ratio_log_neutral = self.loglikelihood_scalar(0, 0)
        
        # Debug: print neutral trajectory for first lineage
        if i == 0:
            neutral_traj = self.n_theory_scalar(0, 0)['cell_number']
            print(f"Lineage 0 neutral trajectory: t=0: {neutral_traj[0]:.1f}, t=5: {neutral_traj[5]:.1f}, t=10: {neutral_traj[10]:.1f}, t=19: {neutral_traj[19]:.1f}")
                
        p_ratio_log = p_ratio_log_adaptive - p_ratio_log_neutral
        if p_ratio_log <= 40:
            p_ratio = np.exp(p_ratio_log)
            p_adaptive = p_ratio /(1 + p_ratio)
        else:
            p_adaptive = 1

        if p_adaptive > self.threshold_adaptive:
            if self.opt_algorithm == 'direct_search':
                # calculate on a finer grid
                log_likelihood_fine = self.posterior_loglikelihood_array(self.s_bin_fine,self.tau_bin_fine,fine=True) 
                s_idx1,tau_idx1 = np.unravel_index(np.argmax(log_likelihood_fine),np.shape(log_likelihood_fine))
                s_opt, tau_opt = self.s_bin_fine[s_idx1], self.tau_bin_fine[tau_idx1]

            elif self.opt_algorithm == 'differential_evolution':
                opt_output = differential_evolution(func = self.posterior_loglikelihood_opt,
                                                    seed = 1,
                                                    bounds = self.bounds,
                                                    x0 = [self.s_bin[s_idx],self.tau_bin[tau_idx]])
                s_opt, tau_opt = opt_output.x[0], opt_output.x[1]

            elif self.opt_algorithm == 'nelder_mead': 
                opt_output =self.nelder_mead(self.posterior_loglikelihood_opt, 
                                                     bounds = self.bounds,
                                                     thresh = 1e-13,
                                                     max_iter = 500,
                                                     x0 = [self.s_bin[s_idx],self.tau_bin[tau_idx]])
                s_opt, tau_opt = opt_output[0], opt_output[1]
            #elif self.opt_algorithm == 'nelder_mead': 
            #    opt_output = minimize(self.posterior_loglikelihood_opt, 
            #                          x0=[self.s_bin[s_idx],self.tau_bin[tau_idx]],
            #                          method='Nelder-Mead',
            #                          bounds=self.bounds, 
            #                          options={'ftol': 1e-8, 'disp': False, 'maxiter': 500})
            #    s_opt, tau_opt = opt_output.x[0], opt_output.x[1]

        else:
            s_opt, tau_opt = 0, 0
            
        # DEBUG:
        if i < 5:  # First 5 lineages
            print(f"Lineage {i} ({self.lineage_ancestor}): p_adaptive={p_adaptive:.4f}, s_opt={s_opt:.4f}, tau_opt={tau_opt:.1f}")

                
        return [p_adaptive, s_opt, tau_opt]

  

                        
    ##########
    def bound_points(self, point, bounds):
        """
        Projects point within bounds, subroutine for nelder_mead
        """
        sol = [min(max(point[0], bounds[0][0]), bounds[0][1]),
               min(max(point[1], bounds[1][0]), bounds[1][1])]
        
        return sol
    

    
    ##########
    def nelder_mead(self, f_opt,bounds=[[-np.inf,np.inf],[-np.inf,np.inf]],thresh=1e-8, max_iter=500,x0=None):
        """
        Manually implements nelder mead algorithm with bounds as specified
        """
        if x0 is None:
            ws = np.array([[0.01,1], [.01,5], [.21,1]])
        else:
            xi,yi = x0
            ws = np.array([[xi,yi],[xi,yi+5],[xi+.05,yi]])
        
        # transformation parameters
        alpha = 1
        beta = 1/2
        gamma = 2
        delta = 1/2
        terminate = False
        
        iter_num=0
        while True:
            iter_num+=1
            f_ws = np.array([f_opt(x) for x in ws])
            sorted_args = np.argsort(f_ws)
            ws = ws[sorted_args] # sort working simplex based on f values
            xl,xs,xh = ws
            fl,fs,fh = f_ws[sorted_args]
            
            f_deviation = np.std(f_ws)
            terminate = f_deviation<thresh or iter_num>max_iter
            if terminate:
                break
            
            centroid = (xl+xs)/2
            
            # reflection
            xr = centroid+alpha*(centroid-xh)
            fr = f_opt(xr)
            if fl<=fr<fs:
                ws[2] = self.bound_points(xr,bounds)
                continue

            # expansion
            if fr<fl:
                xe = centroid+gamma*(xr-centroid)
                fe = f_opt(xe)
                if fe<fr:
                    ws[2] = self.bound_points(xe,bounds)
                    continue
                else:
                    ws[2] = self.bound_points(xr,bounds)
                    continue    

            # contraction
            if fr>=fs:
                if fs<=fr<fh:
                    xc = centroid+beta*(xr-centroid)
                    fc = f_opt(xc)
                    if fc<=fr:
                        ws[2] = self.bound_points(xc,bounds)
                        continue
                else:
                    xc = centroid+beta*(xh-centroid)
                    fc = f_opt(xc)
                    if fc<fh:
                        ws[2] = self.bound_points(xc,bounds)
                        continue
            # shrink
            ws[1] = self.bound_points(xl+delta*(ws[1]-xl),bounds)
            ws[2] = self.bound_points(xl+delta*(ws[2]-xl),bounds)
            
        return np.mean(ws,axis=0)

    
    
    ##########
    def estimation_error_lineage(self, s_opt, tau_opt):
        """
        Estimate estimation error of a lineage for optimization
        """
        d_s, d_tau = 1e-8, 1e-5
    
        f_zero = self.posterior_loglikelihood_opt([s_opt, tau_opt])
        
        f_plus_s = self.posterior_loglikelihood_opt([s_opt + d_s, tau_opt])
        f_minus_s = self.posterior_loglikelihood_opt([s_opt - d_s, tau_opt])
    
        f_plus_tau = self.posterior_loglikelihood_opt([s_opt, tau_opt + d_tau])
        f_minus_tau = self.posterior_loglikelihood_opt([s_opt, tau_opt - d_tau])
    
        f_plus_s_tau = self.posterior_loglikelihood_opt([s_opt + d_s, tau_opt + d_tau])
    
        f_ss = (f_plus_s + f_minus_s - 2*f_zero)/d_s**2
        f_tt = (f_plus_tau + f_minus_tau - 2*f_zero)/d_tau**2
        f_st = (f_plus_s_tau - f_plus_s - f_plus_tau + f_zero)/d_s/d_tau
    
        curvature_matrix = np.array([[f_ss,f_st], [f_st,f_tt]])
        eigs, eigvecs = np.linalg.eig(curvature_matrix)
        v1, v2 = eigvecs[:,0], eigvecs[:,1]
        lambda1, lambda2 = np.abs(eigs[0]), np.abs(eigs[1])
        
        if lambda1==0 or lambda2==0:
            error_s_lineage = np.nan
            error_tau_lineage = np.nan
        else:
            error_s_lineage =  max(np.abs(v1[0]/np.sqrt(lambda1)), np.abs(v2[0]/np.sqrt(lambda2)))
            error_tau_lineage = max(np.abs(v1[1]/np.sqrt(lambda1)), np.abs(v2[1]/np.sqrt(lambda2)))

        return error_s_lineage, error_tau_lineage

    
    ##########
    def estimation_error(self):
        for i in self.idx_adaptive_inferred_index:
            self.r_seq_lineage = self.r_seq[i, :]
            self.n_seq_lineage = self.n_seq[i, :]
                
            s_opt = self.result_s[i]
            tau_opt = self.result_tau[i]
            self.error_s[i], self.error_tau[i] = self.estimation_error_lineage(s_opt, tau_opt)
    

    ##########
    def update_mean_fitness(self, k_iter):
        """
        Updated mean fitness & mutant fraction
        """
        self.mutant_fraction_numerator = np.zeros(self.seq_num, dtype=float)
        self.s_mean_numerator = np.zeros(self.seq_num, dtype=float)
        self.mutant_n_seq_theory = np.zeros(np.shape(self.r_seq), dtype=float)
        
        # 2/26 update: For subdivided, also track mixed mean fitness
        if self.is_subdivided:
            self.s_mean_mix_numerator = np.zeros(self.seq_num, dtype=float)
        
        # Loop over adaptive lineages:
        for i in self.idx_adaptive_inferred_index:
            self.r_seq_lineage = self.r_seq[i, :]
            self.n_seq_lineage = self.n_seq[i, :]
            self.lineage_ancestor = self.ancestor_labels[i]  # 2/26: need this for n_theory_scalar
            
            # 2/26 update: For subdivided, also set other environment data
            if self.is_subdivided:
                self.n_seq_lineage_other = self.n_seq_other[i, :]
                self.current_lineage_index = i
            
            mutant_n = self.n_theory_scalar(self.result_s[i], self.result_tau[i])['mutant_cell_number']
            self.mutant_n_seq_theory[i,:] = mutant_n
            
            # Debug: print for first few adaptive lineages in iteration 2
            if k_iter == 2 and len(self.s_mean_numerator) > 0:
                if i == self.idx_adaptive_inferred_index[0]:  # First adaptive lineage
                    print(f"DEBUG iter 2, lineage {i}: s={self.result_s[i]:.4f}, tau={self.result_tau[i]:.1f}, ancestor={self.lineage_ancestor}")
                    print(f"  mutant_n at t=0: {mutant_n[0]:.1f}, at t=10: {mutant_n[10]:.1f}")
                if i == self.idx_adaptive_inferred_index[1]:  # Second adaptive lineage
                    print(f"DEBUG iter 2, lineage {i}: s={self.result_s[i]:.4f}, tau={self.result_tau[i]:.1f}, ancestor={self.lineage_ancestor}")
                    print(f"  mutant_n at t=0: {mutant_n[0]:.1f}, at t=10: {mutant_n[10]:.1f}")
            
            # 2/26: Contribution to mean fitness depends on ancestor:
            if self.lineage_ancestor == self.reference_ancestor:    # reference strain has lower fitness
                # A mutants have fitness s (relative to A reference)
                fitness_contribution = mutant_n * self.result_s[i]
            else:  # higher-fitness strain
                # non-ref mutants have fitness delta_s + s (background + mutation gain)
                fitness_contribution = mutant_n * (self.delta_s + self.result_s[i])
            
            #self.s_mean_numerator += self.mutant_n_seq_theory[i,:] * self.result_s[i]
            self.s_mean_numerator += fitness_contribution # 2/26: Above line uses self.result_s[i] only, ignoring the ancestor
            self.mutant_fraction_numerator += self.mutant_n_seq_theory[i,:]
            
            # 2/26: For non-ref lineages, also count UNMUTANT cells
            if self.lineage_ancestor != self.reference_ancestor:
                unmutant_n = self.n_seq[i, :] - mutant_n
                unmutant_contribution = unmutant_n * self.delta_s
                self.s_mean_numerator += unmutant_contribution
            
        # 2/26: Add contribution from NEUTRAL non-ref lineages
        for i in range(self.lineages_num):
            # Check if this is a neutral non-ref lineage
            if (self.ancestor_labels[i] != self.reference_ancestor and i not in self.idx_adaptive_inferred_index):
                # All cells in neutral non-ref lineage contribute delta_s
                neutral_contribution = self.n_seq[i, :] * self.delta_s
                self.s_mean_numerator += neutral_contribution
        
        self.s_mean_seq_dict[k_iter] = self.s_mean_numerator/self.cell_depth_list
        self.mutant_fraction_dict[k_iter] = self.mutant_fraction_numerator/self.cell_depth_list
        
        # 2/26 update: For subdivided, calculate mixed mean fitness
        if self.is_subdivided:
            if self.s_mean_other is not None:
                # Step 2: Add actual other environment's contribution
                other_env_contribution = self.s_mean_other * self.cell_depth_list
                self.s_mean_mix_numerator = (self.s_mean_numerator + other_env_contribution) / 2
            else:
                # Step 1: Use only this environment (approximation)
                self.s_mean_mix_numerator = self.s_mean_numerator
            
            # Store mixed mean fitness
            self.s_mean_mix_seq = self.s_mean_mix_numerator / self.cell_depth_list

    
    ##########
    def run_iteration(self):
        """
        run a single interation
        """
        # Calculate proability for each lineage to find adaptive lineages, 
        # Then run optimization for adaptive lineages to find their optimized s & tau for adaptive lineages
        if self.parallelize:
            pool_obj = Pool() # might need to change processes=8
            output0 = pool_obj.map(self.run_parallel, tqdm(range(self.lineages_num)))
            pool_obj.close()
            output = np.array(output0)
            self.result_probability_adaptive = np.array(output[:,0])
            self.result_s = np.array(output[:,1])
            self.result_tau = np.array(output[:,2])

        else:
            self.result_probability_adaptive = np.zeros(self.lineages_num, dtype=float)
            self.result_s = np.zeros(self.lineages_num, dtype=float)
            self.result_tau = np.zeros(self.lineages_num, dtype=float)
            for i in range(self.lineages_num):
                output = self.run_parallel(i)
                self.result_probability_adaptive[i] = output[0]
                self.result_s[i] = output[1]
                self.result_tau[i] = output[2]
        
        self.idx_adaptive_inferred_logical = self.result_probability_adaptive > self.threshold_adaptive
        self.idx_adaptive_inferred_index = np.where(self.idx_adaptive_inferred_logical)[0]
        

        #####
        # number of adaptive lineages
        print(len(self.idx_adaptive_inferred_index))
        #####

        self.error_s = np.zeros(self.lineages_num, dtype=float)
        self.error_tau = np.zeros(self.lineages_num, dtype=float)

    
    #####
    def save_data(self, k_iter, output_label, output_cell_number):
        """
        Save data according to label: if it's saving a step or the final data
        """
        result_output = {'Fitness': self.result_s,
                         'Establishment_Time': self.result_tau,
                         'Error_Fitness': self.error_s,
                         'Error_Establishment_Time': self.error_tau,
                         'Probability_Adaptive': self.result_probability_adaptive,
                         'Mean_Fitness': self.s_mean_seq_dict[k_iter],
                         'Kappa_Value': self.kappa_seq,
                         'Mutant_Cell_Fraction': self.mutant_fraction_dict[k_iter],
                         'Inference_Time': self.iter_timing_list}
        
        to_write = list(itertools.zip_longest(*list(result_output.values())))
        with open(self.output_filename + output_label + '_MutSeq_Result.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(result_output.keys())
            w.writerows(to_write)
        
        to_write = list(itertools.zip_longest(*list(self.s_mean_seq_dict.values())))
        with open(self.output_filename + output_label + '_Mean_fitness_Result.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(self.s_mean_seq_dict.keys())
            w.writerows(to_write)
            
        if output_cell_number == True:
            to_write = pd.DataFrame(self.mutant_n_seq_theory.astype(int))
            to_write.to_csv(self.output_filename + output_label + '_Cell_Number_Mutant_Estimated.csv',
                       index=False, header=False)

            to_write = pd.DataFrame(self.n_seq.astype(int))
            to_write.to_csv(self.output_filename + output_label + '_Cell_Number.csv',
                       index=False, header=False)
   


    #####
    def main(self):
        """
        main function
        """
        start = time.time()
        self.calculate_error = False
        
        self.calculate_kappa()

        for k_iter in range(1, self.max_iter_num+1):
            start_iter = time.time()
            print(f'--- iteration {k_iter} ...')
               
            if k_iter == 1:
                self.s_mean_seq = np.zeros(self.seq_num, dtype=float)
            else:
                self.s_mean_seq = self.s_mean_seq_dict[k_iter-1]
            
        
            self.calculate_E()
            
            # 2/26 update: If Step 2 subdivided, also calculate E for other environment
            if self.is_subdivided and self.s_mean_other is not None:
                self.calculate_E_other()
            
            self.run_iteration()
            self.update_mean_fitness(k_iter)
            
            # Debug: print mean fitness
            if k_iter <= 3:  # Just first 3 iterations
                print(f"DEBUG iteration {k_iter}: mean fitness at t=0: {self.s_mean_seq_dict[k_iter][0]:.6f}, at t=10: {self.s_mean_seq_dict[k_iter][10]:.6f}")

            if self.save_steps == True:
                output_label = '_intermediate_s_' + str(k_iter)
                output_cell_number = False
                self.save_data(k_iter, output_label, output_cell_number)      
                    
            if k_iter > 1:
               stop_check = np.sum((self.s_mean_seq_dict[k_iter] - self.s_mean_seq_dict[k_iter-1])**2)
               print(stop_check)
               if stop_check < self.iteration_stop_threhold:
                   break
                
            end_iter = time.time()
            iter_timing = np.round(end_iter - start_iter, 5)
            self.iter_timing_list.append(iter_timing)
            print(f'    computing time: {iter_timing} seconds', flush=True)
        
        output_label = ''
        output_cell_number = True
        self.estimation_error()
        self.save_data(k_iter, output_label, output_cell_number)
        
        end = time.time()
        inference_timing = np.round(end - start, 5)
        print(f'Total computing time: {inference_timing} seconds',flush=True)
