# covid-campus

Study of the effectiveness of SARS-CoV-2 repeat testing, social distancing, and mask wearing among a residential college population using the Copenhagen Networks Study data.

## File Overview

* [**testing_freq.py**](https://github.com/onnela-lab/covid-campus/blob/main/testing_freq.py): main class and methods for running simulations examining testing frequency
* [**utils.py**](https://github.com/onnela-lab/covid-campus/blob/main/utils.py): helper functions for constructing weighted adjacency matrices and other inputs to the TestFreq class
* [**CNS_DataProcessing.ipynb**](https://github.com/onnela-lab/covid-campus/blob/main/CNS_DataProcessing.ipynb): code used to clean and explore Copenhagen Network Study data prior to running simulations
* [**Simulation_Examples.ipynb**](https://github.com/onnela-lab/covid-campus/blob/main/Simulation_Examples.ipynb): example code showing how to run simulations

## Usage
Below is a basic example of how to use the files to run testing frequency simulations. For a full working example, see [Simulation_Examples.ipynb](https://github.com/onnela-lab/covid-campus/blob/main/Simulation_Examples.ipynb).

1. Import the necessary packages/files.
```
import pandas as pd
import numpy as np
from testing_freq import *
from utils import *
```
2. Read in the data. Here we use a cleaned version of the Copenhagen Networks Study data. `timestamp` is in seconds starting at 0, `user_a` and `user_b` are integer user IDs, and `rssi` is an integer received signal strength indicator. The cleaned data has been scrubbed of empty scans and non-participating users. For more information on how the cleaned file was generated, see [CNS_DataProcessing.ipynb](https://github.com/onnela-lab/covid-campus/blob/main/CNS_DataProcessing.ipynb).
```
bt = pd.read_csv('bt_data_clean.csv', header = 0, names = ['timestamp','user_a', 'user_b', 'rssi'])
```
3. Construct weighted adjacency matrices for each timestep from the data. `time_step` indicates the number of seconds each time step should represent. `data_loops` indicates how many extra times the data should be looped through to extend the length of the simulation. `dist_thres` indicates the RSSI cutoff for what is considered a close contact. Returns a 3D numpy array with one weighted adjacency matrix per time step. Cell entries indicate the number of interactions between a pair of nodes. `df` is the original dataframe with a `time_step` ID column added.
```
adj_mats, df = construct_adj_mat(bt, time_step = 86400, data_loops = 3, dist_thres = -75)
```
4. Set up simulation parameters. See the [utils.py](https://github.com/onnela-lab/covid-campus/blob/main/utils.py) file for additional options.
```
disease_params = dict()
disease_params['ext_inf'] = 0.002
disease_params['asymp'] = 0.3
disease_params['sigma_a'] = 1/3
disease_params['sigma_s'] = 1/3
disease_params['gamma_a'] = 1/7
disease_params['gamma_s'] = 1/12
disease_params['init_status'] = gen_init_status(n_nodes = adj_mats.shape[1], asymp = disease_params['asymp'], n_init_inf = 1, seed = 1)
disease_params['beta'] = gen_trans_prob_NPI(n_nodes = adj_mats.shape[1], base_beta = 0.006157, p_sd = 0.2, p_fm = 0.2, seed = 1)
disease_params['n_time_day'] = 1

test_params = dict()
test_params['test_freq'] = 7
test_params['spec'] = 0.99
test_params['symp_test_delay'] = gen_symp_test_delay(n_nodes = adj_mats.shape[1], univ_delay = 3)
test_params['nc_schd'] = 0.01 
test_params['nc_symp'] = 0.25
test_params['time_dep'] = True
test_params['time_dep_type'] = 'W'
test_params['false_symp'] = 0.005

quar_params = dict()
quar_params['quar_delay'] = gen_quar_delay(n_nodes = adj_mats.shape[1], univ_delay = 1)
quar_params['quar_comp'] = gen_quar_comp(n_nodes = adj_mats.shape[1], seed = 1)
quar_params['quar_len'] = 10
```
5. Instantiate the simulation class and run a simulation with testing.
```
testin = TestFreq(adj_mats, disease_params, test_params, quar_params)
(ia_nodes_byt, is_nodes_byt, test_pos_schd_byt, test_pos_symp_byt, q_schd_byt, q_symp_byt) = testin.sim_spread_test(seed = 1)
```
