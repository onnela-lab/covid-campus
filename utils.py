import numpy as np
import pandas as pd
import networkx as nx
import itertools

def sigmoid(x):
    """
    Applies the logistic/sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

def gen_init_status(n_nodes, asymp = 0.2, n_init_exp = 0, n_init_inf = 5, n_init_rec = 0, seed = None):
    """
    Randomly assigns an initial status to the nodes in the population
    -------
    Inputs
    -------
    n_nodes: integer
        Number of individuals in the population
    asymp: float
        Percent of infectious who are asymptomatic
    n_init_exp: integer
        Number of initial exposed nodes to randomly assign
    n_init_inf: integer
        Number of initial infections to randomly assign
    n_init_rec: integer
        Number of initial recovered to randomly assign
    seed: integer
        Initializes random processes
    -------
    Outputs
    -------
    init_status: numpy array
        Rows are nodes, columns are S, E, IA, IS, and R respectively; uses
        one-hot encoding to indicate initial node status
    """
    # Set random state
    r = np.random.RandomState(seed)

    # Create list of node_ids
    node_ids = list(range(n_nodes))

    # Initial symptomatic vs. asymptomatic infections
    n_asymp = int(np.round(n_init_inf*asymp))
    n_symp = n_init_inf - n_asymp

    # Select nodes for each status
    r_nodes = list(r.choice(node_ids, n_init_rec))
    ia_nodes = list(r.choice(list(set(node_ids)-set(r_nodes)), n_asymp))
    is_nodes = list(r.choice(list(set(node_ids)-set(r_nodes)-set(ia_nodes)), n_symp))
    e_nodes = list(r.choice(list(set(node_ids)-set(r_nodes)-set(ia_nodes)-set(is_nodes)), n_init_exp))
    s_nodes = list(set(node_ids)-set(r_nodes)-set(ia_nodes)-set(is_nodes))

    # Populate numpy array
    init_status = np.zeros((n_nodes, 5))
    init_status[s_nodes, 0] = 1
    init_status[e_nodes, 1] = 1
    init_status[ia_nodes, 2] = 1
    init_status[is_nodes, 3] = 1
    init_status[r_nodes, 4] = 1

    return init_status

def gen_cluster_NPI(graph, ninit, targetp, spreadp = 0.8, seed = None):
    """
    Generates dictionary of nodes indicating which nodes partake in the NPI
    under consideration; NPI assignment is clustered based on the user-provided
    graph
    -------
    Inputs
    -------
    graph: networkx Graph with edge attribute 'weight'
        Static graph with weighted edges to spread over
    ninit: integer
        Number of initial seed nodes for spreading process
    targetp: float
        Proportion of population that should have NPI after spreading
    spreadp: float
        Probability that a selected node will partake in the NPI
    seed: integer
        Initializes random processes
    -------
    Outputs
    -------
    npi_dict: dictionary
        Keys are node IDs, values are True/False for whether nodes partake in
        the NPI
    """

    # Set random state
    r = np.random.RandomState(seed)

    # Set the target number of nodes
    targetn = np.round(targetp*graph.number_of_nodes())

    # Set NPI attribute and initialize values
    nx.set_node_attributes(graph, False, 'npi')
    seeds = r.choice(list(graph.nodes()), size = ninit)
    nx.set_node_attributes(graph, dict(zip(seeds, np.repeat(True, len(seeds)))), 'npi')

    # Choose multiple nodes at each iteration (max 10 iterations)
    nchoose = int((targetn - len(seeds))//10)

    # Initialize the number of NPI compliers
    n_npi = sum(nx.get_node_attributes(graph, 'npi').values())

    # Spread the NPI across the network
    while n_npi < targetn:
        # Get list of current maskers and their neighbors
        npiers = [x for x,y in graph.nodes(data=True) if y['npi']==True]
        all_neighbors = {n: graph[npier][n]['weight'] for npier in npiers for n in graph.neighbors(npier) if n not in npiers}

        # Choose a neighbor probabilistically using the weights (# proximity events)
        myp = [val/sum(all_neighbors.values()) for val in all_neighbors.values()]
        mychoice = r.choice(list(all_neighbors.keys()), size = min(nchoose, int(targetn-n_npi)), p = myp)

        if r.uniform()<=spreadp:
            nx.set_node_attributes(graph, dict(zip(mychoice, np.repeat(True, len(mychoice)))), 'npi')

        # Update the number who mask post-spreading
        n_npi = sum(nx.get_node_attributes(graph, 'npi').values())

    npi_dict = nx.get_node_attributes(graph, 'npi')

    return npi_dict

def gen_trans_prob(n_nodes, univ_val = None, a = 1, b = 5, seed = None):
    """
    Assigns transmission probability matrix for each possible node pairing; note
    that transmission probability is for each 5 minute interaction; if univ_val
    is specified, all node pairings are given the same transmission probability;
    if univ_val is left unspecified, transmission probabilities are drawn for
    each node from a Beta(a, b) distribution and values are then averaged
    across pairings
    -------
    Inputs
    -------
    n_nodes: integer
        Number of individuals in the population
    univ_val: float
        Universal transmission probability to be used for all node pairings; if
        left as None, delays will be drawn from a distribution for each node
        and then averaged
    a: integer or float
        Alpha parameter for Beta distribution; used to draw node-specific
        transmission probabilities
    b: integer or float
        Beta parameter for Beta distribution; used to draw node-specific
        transmission probabilities
    seed: integer
        Initializes random processes
    -------
    Outputs
    -------
    beta: numpy array
        n_nodes x n_nodes matrix containing transmission probabilities for each
        node pairing
    """
    # Set random state
    r = np.random.RandomState(seed)

    if univ_val is None:
        # Assign each node a transmission probability drawn from a beta distribution
        node_probs = r.beta(a = a, b = b, size = n_nodes)
        # Average the node-specific transmission probabilities to get probability for each node pairing
        combos = np.array(list(itertools.combinations(node_probs, 2)))
        beta = np.zeros((n_nodes, n_nodes))
        xs, ys = np.triu_indices(n = n_nodes, k = 1)
        beta[xs,ys] = np.mean(combos, axis = 1)
        beta[ys,xs] = np.mean(combos, axis = 1)
    else:
        # Assign same value for all pairings
        beta = np.full((n_nodes, n_nodes), fill_value = univ_val)

    return beta

def gen_trans_prob_NPI(n_nodes, base_beta, p_sd = None, p_fm = None, sd_nodes = None, fm_nodes = None, fm_eff = 0.15, fm_eff_std = 0.0684, sd_eff = 0.18, sd_eff_std = 0.0734, seed = None):
    """
    Assigns transmission probability matrix for each possible node pairing
    based on assumed reduction due to mask-wearing and/or social distancing;
    user specified the proportion of nodes who abide by mask wearing or social
    distancing; note that transmission probability is for each 5 minute interaction;
    for both NPIs, user must specify either proportion (for random assignment)
    or dictionary of nodes
    -------
    Inputs
    -------
    n_nodes: integer
        Number of individuals in the population
    base_beta: float
        Baseline beta value for those who take no precautions
    p_sd: float
        Proportion of nodes who social distance but do not wear a face mask;
        if left None, user must specify sd_nodes
    p_fm: float
        Proportion of nodes who wear a face mask but do not social distance;
        if left None, user must specify fm_nodes
    sd_nodes: dictionary
        Keys are node IDs, values are True/False for social distancing;
        if left None, user must specify p_sd
    fm_nodes: dictionary
        Keys are node IDs, values are True/False for mask wearing;
        if left None, user must specify p_fm
    fm_eff: float
        Point estimate for efficacy of face masks
    fm_eff_std: float
        Standard deviation for efficacy of face masks
    sd_eff: float
        Point estimate for efficacy of social distancing
    st_eff_std: float
        Standard deviation for efficacy of face masks
    seed: integer
        Initializes random processes
    -------
    Outputs
    -------
    beta: numpy array
        n_nodes x n_nodes matrix containing transmission probabilities for each
        node pairing
    """
    # Set random state
    r = np.random.RandomState(seed)

    node_ids = list(range(n_nodes))

    # Randomly assign nodes to socially distance
    if sd_nodes is None:
        r.shuffle(node_ids)
        sd = node_ids[0:int(n_nodes*p_sd)]
    # Or use specified nodes
    else:
        sd = [x for x,y in sd_nodes.items() if y==True]

    # Randomly assign nodes to wear face masks
    if fm_nodes is None:
        r.shuffle(node_ids)
        fm = node_ids[0:int(n_nodes*p_fm)]
    # Or use specified nodes
    else:
        fm = [x for x,y in fm_nodes.items() if y==True]

    # Initiate numpy array
    beta = np.zeros((n_nodes, n_nodes))

    # Update values based on NPI
    # 2 face masks + social distancing
    fm2 = list(itertools.product(fm, fm))
    fm2sd = [(x,y) for (x,y) in fm2 if (x in sd) or (y in sd)]
    fm2sd_xx = [x for (x,y) in fm2sd]
    fm2sd_yy = [y for (x,y) in fm2sd]
    fm_draws1 = r.normal(loc = fm_eff, scale = fm_eff_std, size = len(fm2sd))
    fm_draws2 = r.normal(loc = fm_eff, scale = fm_eff_std, size = len(fm2sd))
    sd_draws = r.normal(loc = sd_eff, scale = sd_eff_std, size = len(fm2sd))
    beta[fm2sd_xx, fm2sd_yy] = fm_draws1*fm_draws2*sd_draws*base_beta
    beta[fm2sd_yy, fm2sd_xx] = fm_draws1*fm_draws2*sd_draws*base_beta

    # 1 face mask + social distancing
    fm1 = list(itertools.product(fm, set(node_ids)-set(fm)))
    fm1sd = [(x,y) for (x,y) in fm1 if (x in sd) or (y in sd)]
    fm1sd_xx = [x for (x,y) in fm1sd]
    fm1sd_yy = [y for (x,y) in fm1sd]
    fm_draws = r.normal(loc = fm_eff, scale = fm_eff_std, size = len(fm1sd))
    sd_draws = r.normal(loc = sd_eff, scale = sd_eff_std, size = len(fm1sd))
    beta[fm1sd_xx, fm1sd_yy] = fm_draws*sd_draws*base_beta
    beta[fm1sd_yy, fm1sd_xx] = fm_draws*sd_draws*base_beta

    # 0 face masks + social distancing
    fm0sd = list(itertools.product(set(sd)-set(fm), set(node_ids)-set(fm)))
    fm0sd_xx = [x for (x,y) in fm0sd]
    fm0sd_yy = [y for (x,y) in fm0sd]
    sd_draws = r.normal(loc = sd_eff, scale = sd_eff_std, size = len(fm0sd))
    beta[fm0sd_xx, fm0sd_yy] = sd_draws*base_beta
    beta[fm0sd_yy, fm0sd_xx] = sd_draws*base_beta

    # 2 face masks, NO social distancing
    fm2nosd = [(x,y) for (x,y) in fm2 if (x not in sd) and (y not in sd)]
    fm2nosd_xx = [x for (x,y) in fm2nosd]
    fm2nosd_yy = [y for (x,y) in fm2nosd]
    fm_draws1 = r.normal(loc = fm_eff, scale = fm_eff_std, size = len(fm2nosd))
    fm_draws2 = r.normal(loc = fm_eff, scale = fm_eff_std, size = len(fm2nosd))
    beta[fm2nosd_xx, fm2nosd_yy] = fm_draws1*fm_draws2*base_beta
    beta[fm2nosd_yy, fm2nosd_xx] = fm_draws1*fm_draws2*base_beta

    # 1 face mask, NO social distancing
    fm1nosd = [(x,y) for (x,y) in fm1 if (x not in sd) and (y not in sd)]
    fm1nosd_xx = [x for (x,y) in fm1nosd]
    fm1nosd_yy = [y for (x,y) in fm1nosd]
    fm_draws = r.normal(loc = fm_eff, scale = fm_eff_std, size = len(fm1nosd))
    beta[fm1nosd_xx, fm1nosd_yy] = fm_draws*base_beta
    beta[fm1nosd_yy, fm1nosd_xx] = fm_draws*base_beta

    # 0 face masks, NO social distancing
    fm0nosd = list(itertools.product(set(node_ids)-set(sd)-set(fm), set(node_ids)-set(sd)-set(fm)))
    fm0nosd_xx = [x for (x,y) in fm0nosd]
    fm0nosd_yy = [y for (x,y) in fm0nosd]
    beta[fm0nosd_xx, fm0nosd_yy] = base_beta
    beta[fm0nosd_yy, fm0nosd_xx] = base_beta

    return beta

def gen_symp_test_delay(n_nodes, norm_mean = 1, norm_std = 0.5, univ_delay = None, seed = None):
    """
    Assigns delay from becoming symptomatic infectious to presenting for
    symptomatic testing for each node in the population; if node never becomes
    symptomatic infectious, this delay is not used
    -------
    Inputs
    -------
    n_nodes: integer
        Number of individuals in the population
    norm_mean: float
        Mean of the normal distribution to draw node-specific delays from
        (only used if univ_delay = None)
    norm_std: float
        Standard deviation of the normal distribution to draw node-specific
        delays from (only used if univ_delay = None)
    univ_delay: integer
        Delay to use for all nodes in number of time steps; if left as None,
        delays will be drawn from a normal distribution and rounded to the
        nearest integer
    seed: integer
        Initializes random processes
    -------
    Outputs
    -------
    symp_test_delay: dictionary
        Delay from developing symptoms to seeking testing (values) for each node (keys)
    """
    # Set random state
    r = np.random.RandomState(seed)

    if univ_delay is not None:
        # Use a constant value for all nodes
        delays = np.repeat(univ_delay, n_nodes)
    else:
        # Draw from a normal distribution and round the results to integers
        delays = np.round(r.normal(norm_mean, norm_std, size = n_nodes))

    # Construct the dictionary
    symp_test_delay = dict(zip(list(range(n_nodes)), delays))

    return symp_test_delay

def gen_quar_delay(n_nodes, norm_mean = 1, norm_std = 0.5, univ_delay = None, seed = None):
    """
    Assigns delay from testing to quarantine for each node in the population; if
    node never tests positive, this delay is not used
    -------
    Inputs
    -------
    n_nodes: integer
        Number of individuals in the population
    norm_mean: float
        Mean of the normal distribution to draw node-specific delays from
        (only used if univ_delay = None)
    norm_std: float
        Standard deviation of the normal distribution to draw node-specific
        delays from (only used if univ_delay = None)
    univ_delay: integer
        Delay to use for all nodes in number of time steps; if left as None,
        delays will be drawn from a normal distribution and rounded to the
        nearest integer
    seed: integer
        Initializes random processes
    -------
    Outputs
    -------
    quar_delay: dictionary
        Delay from testing to quarantine (values) for each node (keys)
    """
    # Set random state
    r = np.random.RandomState(seed)

    if univ_delay is not None:
        # Use a constant value for all nodes
        delays = np.repeat(univ_delay, n_nodes)
    else:
        # Draw from a normal distribution and round the results to integers
        delays = np.round(r.normal(norm_mean, norm_std, size = n_nodes))

    # Construct the dictionary
    quar_delay = dict(zip(list(range(n_nodes)), delays))

    return quar_delay

def gen_quar_comp(n_nodes, alpha = 5, beta = 0.5, seed = None):
    """
    Assigns compliance probabilities to each node in the population
    -------
    Inputs
    -------
    n_nodes: integer
        Number of individuals in the population
    alpha: float
        Alpha parameter for Beta distribution
    beta: float
        Beta parameter for Beta distribution
    seed: integer
        Initializes random processes
    -------
    Outputs
    -------
    quar_comp: dictionary
        Probability of quarantine compliance (values) for each node (keys)
    """
    # Set random state
    r = np.random.RandomState(seed)
    # Draw probabilities from a beta distribution
    comp_probs = r.beta(alpha, beta, size = n_nodes)
    # Construct the dictionary
    quar_comp = dict(zip(list(range(n_nodes)), comp_probs))

    return quar_comp

def construct_adj_mat(df, time_step = 86400, data_loops = 0, dist_thres = -75):
    """
    Constructs weighted adjacency matrices where entries are the number of
    five minute intervals where an interaction (Bluetooth ping) occurred
    -------
    Inputs
    -------
    df: pandas dataframe
        Contains timestamp, user IDs, and RSSI for each interaction
    time_step: integer
        Number of seconds for each time step; daily matrices are 86400;
        6 hour window is 21600
    data_loops: integer
        Number of times to cycle through data in addition to the original copy
    dist_thres: integer
        Cut-off point to use for RSSI; closer to zero indicates stronger signal;
        any RSSI below this value will be considered too distant an interaction
        to cause disease spread
    -------
    Outputs
    -------
    adj_mats: 3D numpy array
        Contains one weighted adjacency matrix for each time step; cells
        indicate the number of 5-minute time bins where an interaction occurred
        during that time window
    df: pandas dataframe
        Original dataframe with 'time_step' ID column added
    """
    # Extract total number of nodes
    num_nodes = len(set(df.user_a.unique().tolist() + df.user_b.unique().tolist()))
    ids = np.array(range(num_nodes))

    # Create the time steps and make it a column in the dataframe
    times = list(range(0, max(df.timestamp) + time_step, time_step))
    idx = pd.IntervalIndex.from_breaks(times, closed = 'left')
    df['time_step'] = idx.get_indexer(df['timestamp'].values)

    # Create a dataframe to output
    df_out = df.copy()

    # Number of time_steps in the original data
    n_times = len(idx)

    # Intialize list of adjacency matrices
    adj_mats = np.zeros((n_times*(1 + data_loops), num_nodes, num_nodes))

    # Loop through the dataframe and populate the adjacency matrices
    for index, row in df.iterrows():
        if row.rssi>=dist_thres:
            adj_mats[row.time_step, row.user_a, row.user_b] +=1
            adj_mats[row.time_step, row.user_b, row.user_a] +=1

    if data_loops!=0:
        for i in range(data_loops):
            # Copy the adjacency matrices for the rest of the time steps
            adj_mats[(n_times*(i+1)):(n_times*(i+2)),:,:] = adj_mats[:n_times,:,:]
            # Copy the original dataframe
            df_copy = df.copy()
            df_copy['time_step'] += n_times*(i+1)
            # Append the new version to the output copy
            df_out = df_out.append(df_copy, ignore_index = True)

    return adj_mats, df_out
