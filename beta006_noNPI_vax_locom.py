"""
Author: Hali Hambridge
This code was run in parallel on a computing cluster using the following bash command:
    python beta006_noNPI_vax_locom.py $SLURM_ARRAY_TASK_ID
Dependencies: pandas, networkx, numpy, matplotlib, itertools, time
-------------------
Parameter Settings
-------------------
Transmission probability (beta) = 0.006157 per 5 minute exposure, roughly R0 ~ 3.0
Testing Frequencies: every 3, 7, 14, 28 days and symptomatic only
Proportion Vaccinated: 0%, 20%, 40%, 60%, 80%
Probability of External Infection: iid normal(loc = 0.0005, scale = 0.0001),
    roughly 0.3 people infected by outside source each day in a fully susceptible population,
    corresponds to high community transmission scenario in paper
Proportion Mask Wearing: 0%
Proportion Social Distancing: 0%
-------------------
File Outputs
-------------------
beta006_noNPI_vax_locom_detailed_0.csv
beta006_noNPI_vax_locom_detailed_1.csv
beta006_noNPI_vax_locom_detailed_2.csv
beta006_noNPI_vax_locom_detailed_3.csv
beta006_noNPI_vax_locom_detailed_4.csv
"""

import os
import sys
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

from testing_freq import *
from utils import *

# Parse command line arguments
TASKID = int(sys.argv[1])
myseed = TASKID*50000
print('TASKID: ', TASKID)
print('myseed: ', myseed)

sim_name = 'beta006_noNPI_vax_locom'
test_freqs = [0, 3, 7, 14, 28]
time_step = 86400 # one day
nreps = 20

# Read Copenhagen Network Study data
bt = pd.read_csv('bt_data_clean.csv', header = 0, names = ['timestamp','user_a', 'user_b', 'rssi'])

# Construct adjacency matrices
adj_mats, df = construct_adj_mat(bt, time_step = time_step, data_loops = 3, dist_thres = -75)

# Set parameters for simulations
disease_params = dict()
disease_params['asymp'] = 0.3 # 30% remain symptom free for the duration, other pre-symptomatic
disease_params['beta'] = gen_trans_prob(n_nodes = adj_mats.shape[1], univ_val = 0.006157) # Roughly corresponds to R0 = 3
disease_params['sigma_a'] = 1/3 # average incubation period is 3 days
disease_params['sigma_s'] = 1/3
disease_params['gamma_a'] = 1/7 # mild to moderate infectious no longer than 10 days (per CDC)
disease_params['gamma_s'] = 1/12 # severe illness infectious no longer than 20 days after symptom onset
disease_params['n_time_day'] = 1

test_params = dict()
test_params['spec'] = 0.99
test_params['symp_test_delay'] = gen_symp_test_delay(n_nodes = adj_mats.shape[1], univ_delay = 3)
test_params['time_dep'] = True
test_params['time_dep_type'] = 'W'
# % of people seeking testing at each time step, even though not sick -- this is about 3 people per day
test_params['false_symp'] = 0.005

quar_params = dict()
quar_params['quar_delay'] = gen_quar_delay(n_nodes = adj_mats.shape[1], univ_delay = 1)
quar_params['quar_len'] = 10 # 10 day quarantine

# Create the beta scenarios
vax_props = np.linspace(0, 0.8, 5)
beta_scenarios = list(vax_props)

# Create empty df for output
df_out = pd.DataFrame()

# Create empty df for detailed output
det_df_out = pd.DataFrame()

# Loop through each of the beta scenarios
for scenario in beta_scenarios:

    p_vax = scenario

    """
    RUN SIMULATION FOR TESTING SCENARIOS
    """

    # Loop through each of the testing frequencies to consider
    for tf in test_freqs:

        # Set the testing frequency
        test_params['test_freq'] = tf

        # Run simulation with testing and isolation
        for i in range(nreps):

            # Set the parameters that are probabilistic
            rs = np.random.RandomState(myseed)
            disease_params['ext_inf'] = rs.normal(loc = 0.0005, scale = 0.0001, size = 1) # about 0.3 people infected by outside source each day in a fully susceptible population
            while disease_params['ext_inf']<0:
                disease_params['ext_inf'] = rs.normal(loc = 0.0005, scale = 0.0001, size = 1)

            disease_params['init_status'] = gen_init_status(n_nodes = adj_mats.shape[1], asymp = disease_params['asymp'], n_init_inf = 1, n_init_rec = int(adj_mats.shape[1]*p_vax), seed = myseed)

            test_params['nc_schd'] = rs.normal(loc = 0.025, scale = 0.01, size = 1) # Percent non-compliant with scheduled testing
            while test_params['nc_schd']<0:
                test_params['nc_schd'] = rs.normal(loc = 0.025, scale = 0.01, size = 1) # Percent non-compliant with scheduled testing

            test_params['nc_symp'] = rs.normal(loc = 0.25, scale = 0.1, size = 1) # Percent non-compliant with symptomatic testing
            while test_params['nc_symp']<0:
                test_params['nc_symp'] = rs.normal(loc = 0.25, scale = 0.1, size = 1) # Percent non-compliant with symptomatic testing

            quar_params['quar_comp'] = gen_quar_comp(n_nodes = adj_mats.shape[1], seed = myseed)

            # Instantiate the simulation class
            testin = TestFreq(adj_mats, disease_params, test_params, quar_params)

            # Run the simulation
            (ia_nodes_byt, is_nodes_byt, test_pos_schd_byt, test_pos_symp_byt, q_schd_byt, q_symp_byt) = testin.sim_spread_test(seed = myseed)

            # Save detailed results
            tmpdf = pd.DataFrame.from_dict({'rep': np.repeat(i+1, repeats = len(ia_nodes_byt)), 'p_vax': np.repeat(p_vax, repeats = len(ia_nodes_byt)),
                        'tstep': list(range(len(ia_nodes_byt))), 'ext_inf_ct': np.repeat(testin.ext_ict, repeats = len(ia_nodes_byt)),
                        'test_freq': np.repeat(tf, repeats = len(ia_nodes_byt)),
                        'ia_nodes': ia_nodes_byt, 'is_nodes': is_nodes_byt, 'test_pos_schd': test_pos_schd_byt, 'test_pos_symp': test_pos_symp_byt,
                        'q_schd': q_schd_byt, 'q_symp': q_symp_byt})
            det_df_out = det_df_out.append(tmpdf, ignore_index = True)

            # Save aggregate results
            # Flatten the results
            flat_ia = [x for l in ia_nodes_byt for x in l]
            flat_is = [x for l in is_nodes_byt for x in l]
            # Save the results
            tmpdf = pd.DataFrame.from_dict({'rep': [i+1], 'test_freq': [tf], 'p_vax': [p_vax], 'cum_uniq_inf': [len(set(flat_ia + flat_is))]})
            df_out = df_out.append(tmpdf, ignore_index = True)
            # Update the seed
            myseed +=1

# Save out the pandas dataframe results
df_out.to_csv(sim_name + '_' + str(TASKID) + '.csv', index = False)
det_df_out.to_csv(sim_name + '_detailed_' + str(TASKID) + '.csv', index = False)
