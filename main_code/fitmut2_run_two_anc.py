#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import fitmut2_methods_two_anc

# try running with command
# python3 ./fitmut2_run.py -i ./simu_0_EvoSimulation_Read_Number.csv -t ./fitmut_input_time_points.csv -o test

###########################
##### PARSE ARGUMENTS #####
###########################

parser = argparse.ArgumentParser(description='Estimate fitness and establishment time of spontaneous adaptive mutations in a competitive pooled growth experiment', 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
parser.add_argument('-i', '--input', type=str, required=True,
                    help='a .csv file: with each column being the read number per barcode at each sequenced time-point')

parser.add_argument('-t', '--t_list', type=str, required=True,
                    help='a .csv file of 2 columns:'
                         '1st column: sequenced time-points evaluated in number of generations, '
                         '2nd column: total effective number of cells of the population for each sequenced time-point.')
           
parser.add_argument('-al', '--ancestor_labels', type=str, required=True,
                    help='a .csv file with one column: ancestor label (A or B) '
                         'for each lineage, in the same order as the input read count file')
         
parser.add_argument('-u', '--mutation_rate', type=float, default=1e-5, 
                    help='total beneficial mutation rate')

parser.add_argument('-dt', '--delta_t', type=float, default=8, 
                    help='number of generations between bottlenecks')

parser.add_argument('-ds', '--delta_s', type=float, default=0, 
                    help='ancestor B fitness minus ancestor A fitness, in units of log frequency change per generation.')

parser.add_argument('-c', '--c', type=float, default=1, 
                    help='half of variance introduced by cell growth and cell transfer')


parser.add_argument('-n', '--maximum_iteration_number', type=int, default=50,
                    help='maximum number of iterations')

parser.add_argument('-a', '--opt_algorithm', type=str, default='direct_search',
                    choices = ['direct_search','differential_evolution', 'nelder_mead'], 
                    help='choose optimization algorithm')

parser.add_argument('-p', '--parallelize', type=str, default='1',
                    help='whether to use multiprocess module to parallelize inference across lineages')

parser.add_argument('-s', '--save_steps', type=str, default='0',
                    help='whether to output files in intermediate step of iterations')
                    
parser.add_argument('-o', '--output_filename', type=str, default='output',
                    help='prefix of output .csv files')

args = parser.parse_args()


##################################################
# read number counts
r_seq = np.array(pd.read_csv(args.input, header=None), dtype=float)

csv_input = pd.read_csv(args.t_list, header=None)
t_list = np.array(csv_input[0][~pd.isnull(csv_input[0])], dtype=float)
cell_depth_list = np.array(csv_input[1][~pd.isnull(csv_input[1])], dtype=float)

# 2/26: Read ancestor labels
ancestor_df = pd.read_csv(args.ancestor_labels, header=None)
ancestor_labels = np.array(ancestor_df[0])

# 2/26: Get delta_s
delta_s = args.delta_s

Ub = args.mutation_rate
delta_t = args.delta_t
c = args.c # per cycle
parallelize = bool(int(args.parallelize))
max_iter_num = args.maximum_iteration_number
opt_algorithm = args.opt_algorithm
output_filename = args.output_filename
save_steps = bool(int(args.save_steps))

my_obj = fitmut2_methods_two_anc.FitMut_two_anc(r_seq = r_seq,
                                   t_list = t_list,
                                   cell_depth_list = cell_depth_list,
                                   ancestor_labels = ancestor_labels,  # 2/26
                                   delta_s = delta_s,                  # 2/26
                                   Ub = Ub,
                                   delta_t = delta_t,
                                   c = c,
                                   opt_algorithm = opt_algorithm,
                                   max_iter_num = max_iter_num,
                                   parallelize = parallelize,
                                   save_steps = save_steps,
                                   output_filename = output_filename)

my_obj.main()

