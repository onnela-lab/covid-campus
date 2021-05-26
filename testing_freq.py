import numpy as np
import networkx as nx
from utils import *

class TestFreq:
    """
    A class that implements testing simulations on a real contact network.
    ----------
    Attributes
    ----------
    adj_mats: list of np arrays
        Contains adjacency matrices, one for each time step
    disease_params: dictionary
        Contains disease spread parameters
    test_params: dictionary
        Contains testing parameters
    quar_params: dictionary
        Contains quarantine parameters
    ----------
    Methods
    ----------
    __init__()
        Defines the setup for the simulations
    spread()
        Propagates the infection from the set of infected nodes to their susceptible neighbors
    recover()
        Moves infected individuals into a recovered state probabilistically
    time_dep_sens()
        Returns the test sensitivity based on the age of the infection
    scheduled_test()
        Simulates scheduled testing of the entire population at the current time point
    symp_test()
        Simulates testing of individuals who present with symptoms (both infectious and non-infectious)
    quarantine()
        Simulates quarantine of those identified as positive through testing;
        uses an all or nothing compliance
    sim_spread()
        Simulates the unchecked spread of the epidemic (no testing or isolation)
    sim_spread_test()
        Simulates the spread of the epidemic with testing and isolation implemented
    """

    def __init__(self, adj_mats, disease_params, test_params, quar_params):
        """
        This method defines the setup for the simulations.
        """

        self.adj_mats = adj_mats # list of weighted adjacency matrices, one per time step
        self.n_nodes = adj_mats[0,:,:].shape[0] # total number of nodes in population

        # Set up disease spread parameters
        self.ext_inf = disease_params['ext_inf'] # probability of external infection per node at each time step
        self.asymp = disease_params['asymp'] # percent asymptomatic
        self.init_status = disease_params['init_status'] # initial statuses for all nodes as numpy array
        self.beta = disease_params['beta'] # numpy array with transmission probabilities for each node pairing per 5-min interaction
        self.sigma_a = disease_params['sigma_a'] # probability of E to I transition for asymptomatic individuals per node per time step
        self.sigma_s = disease_params['sigma_s'] # probability of E to I transition for symptomatic individuals per node per time step
        self.gamma_a = disease_params['gamma_a'] # probability of I to R transition for asymptomatic individuals per node per time step
        self.gamma_s = disease_params['gamma_s'] # probability of I to R transition for symptomatic individuals per node per time step
        self.n_time_day = disease_params['n_time_day'] # number of time steps per day (e.g. 6 hour time steps = 4 steps per day)

        # Set up testing parameters
        self.symp_test_delay = test_params['symp_test_delay'] # dictionary of delays between experiencing symptoms and presenting for symptomatic testing (time steps)
        self.test_freq = test_params['test_freq'] # testing frequency (time steps)
        self.nc_schd = test_params['nc_schd'] # percent non-compliant with scheduled testing
        self.nc_symp = test_params['nc_symp'] # percent of symptomatic non-compliant with symptomatic testing
        self.false_symp = test_params['false_symp'] # probability of non-infectious presenting for symptomatic testing at each time step
        self.time_dep = test_params['time_dep'] # indicator of whether sensitivity should be time dependent
        self.spec = test_params['spec'] # specificity
        if self.time_dep is False:
            self.sens_i = test_params['sens_i'] # sensitivity for infectious (if constant)
            self.sens_e = test_params['sens_e'] # sensitivity for exposed (if constant)
        if self.time_dep is True:
            self.time_dep_type = test_params['time_dep_type'] # indicator of which version of time dependence to use: 'K' indicates Kucirka and 'W' indicates Wikramaratna

        # Set up quarantine/isolation parameters
        self.quar_delay = quar_params['quar_delay'] # dictionary of delays between testing positive and quarantining (time steps)
        self.quar_comp = quar_params['quar_comp'] # dictionary of node-specific compliance probabilities
        self.quar_len = quar_params['quar_len'] # number of time steps individuals are asked to quarantine

        # Set up counter for community transmitted infections
        self.ext_ict = 0

    def spread(self, adj_mat, s_nodes, e_nodes, ia_nodes, is_nodes, q_nodes, r):
        """
        Carries out the susceptible to infectious spreading process for 1 time step;
        probability of exposure depends on number of 5-min interactions
        -------
        Inputs
        -------
        adj_mat: 2D numpy array
            Adjacency matrix for current time step
        s_nodes: list
            Susceptible node IDs at the current time step
        e_nodes: list
            Exposed node IDs at the current time step
        ia_nodes: list
            Infectious asymptomatic node IDs at the current time step
        is_nodes: list
            Infectious symptomatic node IDs at the current time step
        q_nodes: list
            Node IDs of those in quarantine during the current time step (any disease status)
        r: numpy RandomState
            Initializes random processes
        ----------------
        Other Parameters
        ----------------
        beta: numpy array
            Probability of S to E transition per node per 5-min interaction;
            each entry is a node-pair specific transmission probabilitiy
        sigma_a: float
            Probability of E to I transition for asymptomatic individuals per node per time step
        sigma_s: float
            Probability of E to I transition for symptomtic individuals per node per time step
        asymp: float
            Probability of an infectious individual being asymptomatic
        ext_inf: float
            Probability of a susceptible node being infected by external source per time step
        -------
        Outputs
        -------
        new_asymp: set
            Node IDs for new asymptomatic infections
        new_symp: set
            Node IDs for new symptomatic infections
        """

        # Create sets for newly exposed, asymptomatic, and symptomatic nodes
        new_exposed = set()
        new_asymp = set()
        new_symp = set()

        # Infect susceptibles with probability beta via infectious interaction
        for node in ia_nodes + is_nodes:
            # Only infect if the node is not in the quarantine list at this time step
            if node not in q_nodes:
                # Identify contacts of node
                contacts = np.where(adj_mat[node,]!=0)[0]

                if len(contacts)>0:
                    # Extract only susceptible contacts who aren't in quarantine
                    s_contacts = contacts[(np.isin(contacts, s_nodes)) & (~np.isin(contacts, q_nodes))]
                    for contact in s_contacts:
                        # Draw uniform for each 5-min interaction and infect if any below threshold
                        if (r.uniform(size = int(adj_mat[node, contact])) <= self.beta[node, contact]).any():
                            new_exposed.add(contact)

        # Infect susceptibles via external source
        for node in s_nodes:
            if r.uniform() <= self.ext_inf:
                new_exposed.add(node)
                self.ext_ict +=1

        # For each exposed node, transition to infectious (symp or asymp) probabilistically
        for node in e_nodes:
            # Asymptomatic
            if r.uniform() <= self.asymp:
                if r.uniform() <= self.sigma_a:
                    new_asymp.add(node)
                    ia_nodes.append(node)
            # Symptomatic
            else:
                if r.uniform() <= self.sigma_s:
                    new_symp.add(node)
                    is_nodes.append(node)
        # Update status lists
        for node in new_asymp:
            e_nodes.remove(node)
        for node in new_symp:
            e_nodes.remove(node)
        for node in new_exposed:
            e_nodes.append(node)
            s_nodes.remove(node)

        return(new_asymp, new_symp)

    def recover(self, ia_nodes, is_nodes, r_nodes, r):
        """
        Carries out the infectious to recovered spreading process for 1 time step
        -------
        Inputs
        -------
        ia_nodes: list
            Infectious asymptomatic node IDs at the current time step
        is_nodes: list
            Infectious symptomatic node IDs at the current time step
        r_nodes: list
            Recovered node IDs at the current time step
        r: numpy RandomState
            Initializes random processes
        ----------------
        Other Parameters
        ----------------
        gamma_a: float
            Probability of I to R transition for asymptomatic individuals per time step
        gamma_s: float
            Probability of I to R transition for symptomatic individuals per time step
        -------
        Outputs
        -------
        new_recoveries: list
            Node IDs for new recoveries
        """

        # Create list of newly recovered nodes
        new_recoveries = list()

        # Recover asymptomatic nodes probabilistically
        for node in ia_nodes:
            if r.uniform() <= self.gamma_a:
                new_recoveries.append(node)
                r_nodes.append(node)
                ia_nodes.remove(node)
        # Recover symptomatic nodes probabilistically
        for node in is_nodes:
            if r.uniform() <=self.gamma_s:
                new_recoveries.append(node)
                r_nodes.append(node)
                is_nodes.remove(node)

        return new_recoveries

    def time_dep_sens(self, exp_age, inf_age):
        """
        Returns the test sensitivity based on age of infection
        -------
        Inputs
        -------
        exp_age: integer
            Number of time steps since exposure; 1 indicates current time step
        inf_age: integer
            Number of time steps since node became infectious;
            1 indicates current time step, 0 indicates never infectious
        ----------------
        Other Parameters
        ----------------
        n_time_day: integer
            Number of time steps in a day
        -------
        Outputs
        -------
        sens: float
            Sensitivity for that infection age
        """
        # Convert exposure/infection age to days
        days_exp = exp_age/self.n_time_day
        days_inf = inf_age/self.n_time_day

        # Uses Chang et al's approximation (2020, Health Care Management Science)
        # of Kucirka et al's results (2020, Annals of Internal Medicine)
        if self.time_dep_type=='K':
            # Combine exposed + infectious time since model is for days since exposure
            days_since_exp = days_exp + days_inf - 1/self.n_time_day
            # Compute sensitivity
            if days_since_exp == 0:
                sens = 0
            elif days_since_exp <=21:
                sens = sigmoid(-29.966+37.713*np.log(days_since_exp)-14.452*np.power(np.log(days_since_exp), 2)+1.721*np.power(np.log(days_since_exp), 3))
            else:
                sens = sigmoid(6.878-2.436*np.log(days_since_exp))
            return sens

        # Uses nasal swab results from Wikramaratna et al (2020 Euro Surveillance)
        else:
            # From Wikramaratna supplemental materials with two entries prepended
            # First two entries are mirror of third and fourth entries and are designed
            # to account for time before symptom onset
            sens = [0.945897051, 0.956035396, 0.964345916, 0.956035396, 0.945897051,
            0.933584589, 0.918713483, 0.900871215, 0.879635065, 0.854598785,
            0.8254113590000001, 0.791827558, 0.753761722, 0.711341394, 0.664950473,
            0.615247128, 0.563146104, 0.509754778, 0.456274387, 0.40390623000000003,
            0.35374840399999996, 0.30670959900000005, 0.26345573099999997,
            0.22439299899999998, 0.18968316500000004, 0.159281275, 0.132984471,
            0.11048262799999997, 0.091403491, 0.07535069699999997, 0.061931225000000034,
            0.05077403700000005, 0.041540205, 0.033927078000000055]
            sens_dict = dict(zip(range(len(sens)), sens))

            if days_inf!=0:
                if np.round(days_inf-1) in sens_dict.keys():
                    return sens_dict[np.round(days_inf-1)]
                else:
                    # Covers the edge case where someone is infectious for a really long time
                    return 0
            else:
                # No probability of detection when exposed under this model
                return 0

    def scheduled_test(self, test_nodes, e_nodes_byt, ia_nodes_byt, is_nodes_byt, r):
        """
        Simulates scheduled testing of the specified nodes at the current time step
        -------
        Inputs
        -------
        test_nodes: list
            IDs of nodes to be tested at this time step
        e_nodes_byt: list of lists
            Exposed node IDs at all time steps
        ia_nodes_byt: list of lists
            Infectious asymptomatic node IDs at all time steps
        is_nodes_byt: list of lists
            Infectious symptomatic node IDs at all time steps
        r: numpy RandomState
            Initializes random processes
        ----------------
        Other Parameters
        ----------------
        nc_schd: float
            Percent non-compliant with scheduled testing
        time_dep: Boolean
            Indicator of whether test sensitivity should be time dependent
        sens_i: float
            Test sensitivity for infectious (only used if time_dep = False)
        sens_e: float
            Test sensitivity for exposed (only used if time_dep = False)
        spec: float
            Test specificity
        -------
        Outputs
        -------
        test_pos: list
            Node IDs who tested positive
        test_neg: list
            Node IDs who tested negative
        test_nc: list
            Node IDs who refused to be tested
        """

        # Initialize empty lists
        test_pos = []
        test_neg = []
        test_nc = []

        # Loop through all the nodes earmarked to be tested
        for node in test_nodes:
            # Non-compliant nodes who refuse scheduled testing
            if r.uniform() < self.nc_schd:
                test_nc.append(node)
            # Compliant nodes
            else:
                # Exposed or infected nodes
                if node in e_nodes_byt[-1] + ia_nodes_byt[-1] + is_nodes_byt[-1]:

                    # Test sensitivity is constant
                    if self.time_dep == False:
                        # Exposed node
                        if node in e_nodes_byt[-1]:
                            if r.uniform() < self.sens_e:
                                # True positive
                                test_pos.append(node)
                            else:
                                # False negative
                                test_neg.append(node)
                        # Infectious node
                        else:
                            if r.uniform() < self.sens_i:
                                # True positive
                                test_pos.append(node)
                            else:
                                # False negative
                                test_neg.append(node)

                    # Test sensitivity depends on time since exposure
                    else:
                        # Determine time steps node has been exposed
                        node_e_loc = [node in lst for lst in e_nodes_byt]
                        if sum(node_e_loc)==0:
                            # If node isn't in any exposed, it's an initial infection/seed node
                            # Set time exposed to 3 days
                            time_exp = 3*self.n_time_day
                        else:
                            time_exp = np.max(np.where(node_e_loc))-np.min(np.where(node_e_loc))+1
                        # Determine time steps node has been infectious
                        if node in ia_nodes_byt[-1]:
                            node_i_loc = [node in lst for lst in ia_nodes_byt]
                            time_inf = np.max(np.where(node_i_loc))-np.min(np.where(node_i_loc))+1
                        elif node in is_nodes_byt[-1]:
                            node_i_loc = [node in lst for lst in is_nodes_byt]
                            time_inf = np.max(np.where(node_i_loc))-np.min(np.where(node_i_loc))+1
                        else:
                            # Node not infectious yet
                            time_inf = 0

                        if r.uniform() < self.time_dep_sens(time_exp, time_inf):
                            # True positive
                            test_pos.append(node)
                        else:
                            # False negative
                            test_neg.append(node)

                # Uninfected/unexposed nodes
                else:
                    if r.uniform() < self.spec:
                        # True negative
                        test_neg.append(node)
                    else:
                        # False positive
                        test_pos.append(node)

        return test_pos, test_neg, test_nc

    def symp_test(self, e_nodes_byt, is_nodes_byt, s_nodes, r_nodes, r):
        """
        Simulates testing of individuals who present with symptoms
        -------
        Inputs
        -------
        e_nodes_byt: list of lists
            Exposed node IDs at all time steps
        is_nodes_byt: list of lists
            Infectious symptomatic node IDs at all time steps
        s_nodes: list
            Susceptible node IDs at the current time step
        r_nodes: list
            Recovered node IDs at the current time step
        r: numpy RandomState
            Initializes random processes
        ----------------
        Other Parameters
        ----------------
        symp_test_delay: dictionary
            Values are number of time steps between experiencing symptoms and
            presenting for symptomatic testing; node IDs are keys
        nc_symp: float
            Percent of infectious symptomatic individuals who do not present for
            symptomatic testing
        time_dep: Boolean
            Indicator of whether test sensitivity should be time dependent
        sens_i: float
            Test sensitivity for infectious (only used if time_dep = False)
        sens_e: float
            Test sensitivity for exposed (only used if time_dep = False)
        spec: float
            Test specificity
        -------
        Outputs
        -------
        test_pos: list
            Node IDs who tested positive
        test_neg: list
            Node IDs who tested negative
        test_nc: list
            Node IDs of infectious symptomatic individuals who did not present
            for symptomatic testing
        """

        # Initialize empty lists
        test_pos = []
        test_neg = []
        test_nc = []

        # Create list of symptomatic infectious nodes to be tested
        # using time first infectious and node-specific delays
        test_nodes = []
        for node in is_nodes_byt[-1]:
            # Generate time steps where node is symptomatic infectious
            locs = [i for i, loc in enumerate(is_nodes_byt) if node in loc]
            first_loc = min(locs)
            last_loc = max(locs)
            # If symptom delay just ended for this node, test it
            if (last_loc - first_loc) == self.symp_test_delay[node]:
                test_nodes.append(node)

        # Test infectious nodes that present with symptoms
        for node in test_nodes:
            if r.uniform() < self.nc_symp:
                # Node that doesn't get tested despite symptoms
                test_nc.append(node)
            else:
                # Test sensitivity is constant
                if self.time_dep == False:
                    if r.uniform() < self.sens_i:
                        # True positive
                        test_pos.append(node)
                    else:
                        # False negative
                        test_neg.append(node)
                # Test sensitivity depends on time since infection
                else:
                    # Determine time steps node has been exposed
                    node_e_loc = [node in lst for lst in e_nodes_byt]
                    if sum(node_e_loc)==0:
                        # If node isn't in any exposed, it's an initial infection/seed node
                        # Set time exposed to 3 days
                        time_exp = 3*self.n_time_day
                    else:
                        time_exp = np.max(np.where(node_e_loc))-np.min(np.where(node_e_loc))+1
                    # Determine time steps node has been infectious
                    node_i_loc = [node in lst for lst in is_nodes_byt]
                    time_inf = np.max(np.where(node_i_loc))-np.min(np.where(node_i_loc))+1

                    if r.uniform() < self.time_dep_sens(time_exp, time_inf):
                        # True positive
                        test_pos.append(node)
                    else:
                        # False negative
                        test_neg.append(node)

        # Test non-infectious nodes that erroneously present with symptoms
        non_inf_symp = s_nodes + e_nodes_byt[-1] + r_nodes
        to_test = r.choice(non_inf_symp, int(np.round(self.false_symp*len(non_inf_symp))))
        for node in to_test:
            # Exposed node
            if node in e_nodes_byt[-1]:
                # Test sensitivity is constant
                if self.time_dep == False:
                    if r.uniform() < self.sens_e:
                        # True positive
                        test_pos.append(node)
                    else:
                        # False negative
                        test_neg.append(node)
                # Test sensitivity depends on time since infection
                else:
                    # Determine time steps node has been exposed
                    node_e_loc = [node in lst for lst in e_nodes_byt]
                    # Number of time steps node has been exposed/infectious
                    time_exp = np.max(np.where(node_e_loc)) - np.min(np.where(node_e_loc))+1
                    time_inf = 0

                    if r.uniform() < self.time_dep_sens(time_exp, time_inf):
                        # True positive
                        test_pos.append(node)
                    else:
                        # False negative
                        test_neg.append(node)
            # Unexposed node
            else:
                if r.uniform() < self.spec:
                    # True negative
                    test_neg.append(node)
                else:
                    # False positive
                    test_pos.append(node)

        return test_pos, test_neg, test_nc

    def quarantine(self, test_pos_byt, s_nodes, e_nodes, ia_nodes, is_nodes, r_nodes, r):
        """
        Simulates quarantine of those identified as positive through testing;
        each node has an average probability of compliance for each time step
        -------
        Inputs
        -------
        test_pos_byt: list of lists
            IDs of nodes who tested positive for each time step
        s_nodes: list
            Susceptible node IDs at the current time step
        e_nodes: list
            Exposed node IDs at the current time step
        ia_nodes: list
            Infectious asymptomatic node IDs at the current time step
        is_nodes: list
            Infectious symptomatic node IDs at the current time step
        r_nodes: list
            Recovered node IDs at the current time step
        r: numpy RandomState
            Initializes random processes
        ----------------
        Other Parameters
        ----------------
        quar_delay: dictionary
            Dictionary of node-specific quartine delays with node IDs as keys
            and delays as values
        quar_comp: dictionary
            Dictionary of node-specific compliance probabilities with node IDs
            as keys and compliance probabilities as values
        quar_len: integer
            Number of time steps students are instructed to quarantine after
            testing positive
        -------
        Outputs
        -------
        q_nodes: list
            Node IDs of those who quarantine/isolate at this time step
        """

        # Nodes that have ever tested positive
        ever_pos = set([val for sublist in test_pos_byt for val in sublist])
        current_t = len(test_pos_byt) - 1

        # Create list of quarantined nodes
        q_nodes = []

        for node in ever_pos:
            # Get time step node tested positive
            test_t = max([i for i, loc in enumerate(test_pos_byt) if node in loc])
            min_t = test_t + self.quar_delay[node]
            max_t = test_t + self.quar_delay[node] + self.quar_len

            # If within these bounds, node is eligible for quarantine
            if (current_t>=min_t) and (current_t<=max_t):
                # Check if node complies based on their probability of compliance
                if r.uniform() < self.quar_comp[node]:
                    # If compliant, add to quarantine list for this time step
                    q_nodes.append(node)

        return q_nodes

    def sim_spread(self, seed = None):
        """
        Simulates the spreading of the epidemic
        -------
        Inputs
        -------
        seed: integer
            Initializes random processes
        ----------------
        Other Parameters
        ----------------
        init_status: numpy array
            Rows are nodes, columns are S, E, IA, IS, and R respectively; uses
            one-hot encoding to indicate initial node status
        adj_mats: list of np arrays
            Contains adjacency matrices, one for each time step
        -------
        Outputs
        -------
        s_nodes_byt: list of lists
            Node IDs for susceptible nodes for all time steps
        e_nodes_byt: list of lists
            Node IDs for exposed nodes for all time steps
        ia_nodes_byt: list of lists
            Node IDs for infectious asymptomatic nodes for all time steps
        is_nodes_byt: list of lists
            Node IDs for infectious symptomatic nodes for all time steps
        r_nodes_byt: list of lists
            Node IDs for recovered nodes for all time steps
        """
        # Set random state
        r = np.random.RandomState(seed)

        # Initialize the overall node lists
        s_nodes = np.where(self.init_status[:,0]==1)[0].tolist()
        e_nodes = np.where(self.init_status[:,1]==1)[0].tolist()
        ia_nodes = np.where(self.init_status[:,2]==1)[0].tolist()
        is_nodes = np.where(self.init_status[:,3]==1)[0].tolist()
        r_nodes = np.where(self.init_status[:,4]==1)[0].tolist()

        # This list will always be empty since we're not simulating quarantine, but is required for the spread function
        q_nodes = []

        # Initialize the node lists for each time step
        s_nodes_byt = [s_nodes.copy()]
        e_nodes_byt = [e_nodes.copy()]
        ia_nodes_byt = [ia_nodes.copy()]
        is_nodes_byt = [is_nodes.copy()]
        r_nodes_byt = [r_nodes.copy()]

        for tstep in range(self.adj_mats.shape[0]):
            new_recoveries = self.recover(ia_nodes, is_nodes, r_nodes, r)
            r_nodes_byt.append(list(r_nodes))

            (new_asymp, new_symp) = self.spread(self.adj_mats[tstep,:,:], s_nodes, e_nodes, ia_nodes, is_nodes, q_nodes, r)
            e_nodes_byt.append(list(e_nodes))

            s_nodes_byt.append(list(s_nodes))
            ia_nodes_byt.append(list(ia_nodes))
            is_nodes_byt.append(list(is_nodes))

        return(s_nodes_byt, e_nodes_byt, ia_nodes_byt, is_nodes_byt, r_nodes_byt)

    def sim_spread_test(self, seed = None):
        """
        Simulates the spreading of the epidemic
        -------
        Inputs
        -------
        seed: integer
            Initializes random processes
        ----------------
        Other Parameters
        ----------------
        n_nodes: integer
            Number of individuals in the population
        init_status: numpy array
            Rows are nodes, columns are S, E, IA, IS, and R respectively; uses
            one-hot encoding to indicate initial node status
        adj_mats: list of np arrays
            Contains adjacency matrices, one for each time step
        symp_test_delay: dictionary
            Values are number of time steps between experiencing symptoms and
            presenting for symptomatic testing; node IDs are keys
        test_freq: integer
            Testing frequency in number of time steps
        -------
        Outputs
        -------
        ia_nodes_byt: list of lists
            Node IDs for infectious asymptomatic nodes for all time steps
        is_nodes_byt: list of lists
            Node IDs for infectious symptomatic nodes for all time steps
        test_pos_schd_byt: list of lists
            Node IDs for nodes that tested positive via scheduled testing for all time steps
        test_pos_symp_byt: list of lists
            Node IDs for nodes that tested positive via symptomatic testing for all time steps
        q_schd_byt: list of lists
            Node IDs for nodes that quarantined after scheduled testing for all time steps
        q_symp_byt: list of lists
            Node IDs for nodes that quarantined after symptomatic testing for all time steps
        """
        # Set random state
        r = np.random.RandomState(seed)

        # Initialize the overall node lists
        s_nodes = np.where(self.init_status[:,0]==1)[0].tolist()
        e_nodes = np.where(self.init_status[:,1]==1)[0].tolist()
        ia_nodes = np.where(self.init_status[:,2]==1)[0].tolist()
        is_nodes = np.where(self.init_status[:,3]==1)[0].tolist()
        r_nodes = np.where(self.init_status[:,4]==1)[0].tolist()

        # Initialize the node lists for each time step
        s_nodes_byt = [s_nodes.copy()]
        e_nodes_byt = [e_nodes.copy()]
        ia_nodes_byt = [ia_nodes.copy()]
        is_nodes_byt = [is_nodes.copy()]
        r_nodes_byt = [r_nodes.copy()]

        # Initialize lists to keep track of positive tests and quarantines over time
        test_pos_schd_byt = [[]]
        test_pos_symp_byt = [[]]
        q_schd_byt = [[]]
        q_symp_byt = [[]]

        # Create list of node_ids
        node_ids = list(range(self.n_nodes))

        # Divide up the nodes randomly into bins for scheduled testing
        r.shuffle(node_ids)
        if self.test_freq!=0:
            test_lists = [node_ids[i:i + int(np.ceil(len(node_ids)/self.test_freq))] for i in range(0, len(node_ids), int(np.ceil(len(node_ids)/self.test_freq)))]

        # Loop through the time steps
        for tstep in range(self.adj_mats.shape[0]):

            # Initialize the quarantine list for this time step
            q_nodes = []

            # Perform symptomatic testing if there are symptomatic nodes available from delay time step back
            test_pos_symp, test_neg_symp, test_nc_symp = self.symp_test(e_nodes_byt, is_nodes_byt, s_nodes, r_nodes, r)
            test_pos_symp_byt.append(test_pos_symp)

            # Perform scheduled testing before propagating the disease
            if self.test_freq!=0:
                test_pos_schd, test_neg_schd, test_nc_schd = self.scheduled_test(test_lists[tstep % self.test_freq], e_nodes_byt, ia_nodes_byt, is_nodes_byt, r)
                test_pos_schd_byt.append(test_pos_schd)
            else:
                test_pos_schd_byt.append([])

            # Quarantine individuals who tested positive (with delay)
            # Quarantine symptomatic cases that tested positive
            q_nodes_symp = self.quarantine(test_pos_symp_byt, s_nodes, e_nodes, ia_nodes, is_nodes, r_nodes, r)
            q_symp_byt.append(q_nodes_symp)
            q_nodes.extend(q_nodes_symp)

            # Quarantine cases that tested positive through scheduled testing
            q_nodes_schd = self.quarantine(test_pos_schd_byt, s_nodes, e_nodes, ia_nodes, is_nodes, r_nodes, r)
            q_schd_byt.append(q_nodes_schd)
            q_nodes.extend(q_nodes_schd)

            # Propagate the disease forward
            new_recoveries = self.recover(ia_nodes, is_nodes, r_nodes, r)
            r_nodes_byt.append(list(r_nodes))
            (new_asymp, new_symp) = self.spread(self.adj_mats[tstep,:,:], s_nodes, e_nodes, ia_nodes, is_nodes, q_nodes, r)
            e_nodes_byt.append(list(e_nodes))
            s_nodes_byt.append(list(s_nodes))
            ia_nodes_byt.append(list(ia_nodes))
            is_nodes_byt.append(list(is_nodes))

        return(ia_nodes_byt, is_nodes_byt, test_pos_schd_byt, test_pos_symp_byt, q_schd_byt, q_symp_byt)
