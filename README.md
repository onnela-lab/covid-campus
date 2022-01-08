# COVID Campus

This code simulates the spread of COVID-19, as well as the effect of various mitigation measures, on an empirical contact network of college students collected through the Copenhagen Network Study. This work was published in the _International Journal of Infectious Diseases_. See https://doi.org/10.1016/j.ijid.2021.10.008 for more information.

**Cite the Paper:** Hambridge HL, Kahn R, & Onnela JP (2021). Examining SARS-CoV-2 Interventions in Residential Colleges Using an Empirical Network. _International Journal of Infectious Diseases, 113,_ 325-330.

**Cite the Code:** [![DOI](https://zenodo.org/badge/338376059.svg)](https://zenodo.org/badge/latestdoi/338376059)

## File Overview
### Core Functions
* [**testing_freq.py**](https://github.com/hhambridge/Testing-Frequency/blob/master/Code/testing_freq.py): main class and methods for running simulations examining testing frequency
* [**utils.py**](https://github.com/hhambridge/Testing-Frequency/blob/master/Code/utils.py): helper functions for constructing weighted adjacency matrices, assigning non-pharmaceutical interventions (NPIs), and incorporating NPIs into dyadic transmission probabilities
* [**plotfx.py**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/plotfx.py): some basic plotting functions to show metrics over time and by testing frequency
* [**Simulation_Examples.ipynb**](https://github.com/hhambridge/Testing-Frequency/blob/master/Code/Simulation_Examples.ipynb): example code showing how to run simulations

### Data Processing & Visualization
* [**CNS_DataProcessing.ipynb**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/CNS_DataProcessing.ipynb): loads, cleans, and analyzes Copenhagen Network Study data; removes weak ties, empty scans, and non-participating devices
* [**Network_Plots.ipynb**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/Network_Plots.ipynb): creates network plots for Copenhagen Network Study data with connected components arranged in a grid for each day

### Generate Results
* [**beta003_noNPI_vax_hicom.py**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/beta003_noNPI_vax_hicom.py): runs simulations for R0 ~ 1.5 with high transmission from the community; no mask wearing or social distancing
* [**beta006_noNPI_vax_hicom.py**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/beta006_noNPI_vax_hicom.py): runs simulations for R0 ~ 3.0 with high transmission from the community; no mask wearing or social distancing
* [**beta009_noNPI_vax_hicom.py**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/beta009_noNPI_vax_hicom.py): runs simulations for R0 ~ 4.5 with high transmission from the community; no mask wearing or social distancing
* [**beta003_noNPI_vax_locom.py**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/beta003_noNPI_vax_locom.py): runs simulations for R0 ~ 1.5 with low transmission from the community; no mask wearing or social distancing
* [**beta006_noNPI_vax_locom.py**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/beta006_noNPI_vax_locom.py): runs simulations for R0 ~ 3.0 with low transmission from the community; no mask wearing or social distancing
* [**beta009_noNPI_vax_locom.py**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/beta009_noNPI_vax_locom.py): runs simulations for R0 ~ 4.5 with low transmission from the community; no mask wearing or social distancing
* [**beta006_multiNPI_0vax_hicom.py**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/beta006_multiNPI_0vax_hicom.py): runs simulations for R0 ~ 3.0 with high transmission from the community; no baseline immunity; proportion of the population wearing masks and socially distancing
* [**beta006_multiNPI_20vax_hicom.py**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/beta006_multiNPI_20vax_hicom.py): runs simulations for R0 ~ 3.0 with high transmission from the community; 20% baseline immunity; proportion of the population wearing masks and socially distancing
* [**beta006_multiNPI_40vax_hicom.py**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/beta006_multiNPI_40vax_hicom.py): runs simulations for R0 ~ 3.0 with high transmission from the community; 40% baseline immunity; proportion of the population wearing masks and socially distancing
* [**beta006_multiNPI_0vax_hicom_correlated.py**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/beta006_multiNPI_0vax_hicom_correlated.py): runs simulations for R0 ~ 3.0 with high transmission from the community; no baseline immunity; proportion of the population wearing masks and socially distancing; mask wearing clustered on the proximity network

### Analyze & Visualize Results
* [**Clean_noNPI_Cluster_Results.ipynb**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/Clean_noNPI_Cluster_Results.ipynb): loads, combines, and restructures output from beta003_noNPI_vax_hicom.py, beta006_noNPI_vax_hicom.py, beta009_noNPI_vax_hicom.py, beta003_noNPI_vax_locom.py, beta006_noNPI_vax_locom.py, and beta009_noNPI_vax_locom.py scripts; results used in noNPI_Plots.ipynb
* [**noNPI_Plots.ipynb**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/noNPI_Plots.ipynb): creates plots showing infections, tests, and isolations over time for R0 ~ 1.5, R0 ~ 3.0, and R0 ~ 4.5 for both high and low community transmission settings; generates plots by percent with baseline immunity
* [**NPI_Cluster_Results_hicom.ipynb**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/NPI_Cluster_Results_hicom.ipynb): creates heatmaps comparing average cumulative incidence for various testing frequencies, proportion mask wearing, proportion social distancing under high community transmission scenario
* [**NPI_Cluster_Results_locom.ipynb**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/NPI_Cluster_Results_locom.ipynb): creates heatmaps comparing average cumulative incidence for various testing frequencies, proportion mask wearing, proportion social distancing under low community transmission scenario
* [**NPI_Cluster_Results_Correlated.ipynb**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/NPI_Cluster_Results_Correlated.ipynb): creates heatmaps comparing random assignment of mask wearing with clustered assignment of mask wearing; social distancing was randomly assigned in both scenarios
* [**Relative_Effectiveness.ipynb**](https://github.com/hhambridge/COVID-Campus/blob/master/Code/Relative_Effectiveness.ipynb): analyzes relative effectiveness of various mitigation measures using linear regression model

## Usage
Below is a basic example of how to use the files to run testing frequency simulations. For a full working example, see [Simulation_Examples.ipynb](https://github.com/hhambridge/Testing-Frequency/blob/master/Code/Simulation_Examples.ipynb).

1. Import the necessary packages/files.
```
import pandas as pd
import numpy as np
from testing_freq import *
from utils import *
```
2. Read in the data. Here we use a cleaned version of the Copenhagen Networks Study data. `timestamp` is in seconds starting at 0, `user_a` and `user_b` are integer user IDs, and `rssi` is an integer received signal strength indicator. The cleaned data has been scrubbed of empty scans and non-participating users. For more information on how the cleaned file was generated, see [CNS_DataProcessing.ipynb](https://github.com/hhambridge/Testing-Frequency/blob/master/Code/CNS_DataProcessing.ipynb).
```
bt = pd.read_csv('bt_data_clean.csv', header = 0, names = ['timestamp','user_a', 'user_b', 'rssi'])
```
3. Construct weighted adjacency matrices for each timestep from the data. `time_step` indicates the number of seconds each time step should represent. `data_loops` indicates how many extra times the data should be looped through to extend the length of the simulation. `dist_thres` indicates the RSSI cutoff for what is considered a close contact. Returns a 3D numpy array with one weighted adjacency matrix per time step. Cell entries indicate the number of interactions between a pair of nodes. `df` is the original dataframe with a `time_step` ID column added.
```
adj_mats, df = construct_adj_mat(bt, time_step = 86400, data_loops = 3, dist_thres = -75)
```
4. Set up simulation parameters. See the [utils.py](https://github.com/hhambridge/Testing-Frequency/blob/master/Code/utils.py) file for additional options.
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
