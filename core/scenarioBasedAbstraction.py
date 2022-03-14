#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 ______________________________________
|                                      |
|  SCENARIO-BASED ABSTRACTION PROGRAM  |
|______________________________________|

Implementation of the method proposed in the paper:

  Thom Badings, Alessandro Abate, David Parker, Nils Jansen, Hasan Poonawala & 
  Marielle Stoelinga (2021). Sampling-based Robust Control of Autonomous 
  Systems with Non-Gaussian Noise. AAAI 2022.

Originally coded by:        Thom S. Badings
Contact e-mail address:     thom.badings@ru.nl>
______________________________________________________________________________
"""

import numpy as np              # Import Numpy for computations
import itertools                # Import to crate iterators
import csv                      # Import to create/load CSV files
import sys                      # Allows to terminate the code at some point
import os                       # Import OS to allow creationg of folders
import random                   # Import to use random variables
import pandas as pd             # Import Pandas to store data in frames

from .define_model import find_connected_components
from .define_partition import definePartitions, defStateLabelSet
from .compute_probabilities import computeScenarioBounds_sparse, \
    computeScenarioBounds_error
from .commons import tic, ticDiff, tocDiff, table, printWarning
from .compute_actions import defEnabledActions, defEnabledActions_UA, \
    def_all_BRS
from .create_iMDP import mdp
from .postprocessing.createPlots import partition_plot

'''
------------------------------------------------------------------------------
Main filter-based abstraction object definition
------------------------------------------------------------------------------
'''

class Abstraction(object):
    '''
    Main abstraction object    
    '''
    
    def __init__(self):
        '''
        Initialize the scenario-based abstraction object.

        Returns
        -------
        None.

        '''
        
        print('Model loaded:')
        print(' -- Dimension of the state:',self.model.n)
        print(' -- Dimension of the control input:',self.model.p)
        
        # Number of simulation time steps
        self.N = int(self.model.setup['endTime'] / self.model.setup['lump'])
        
        if hasattr(self.model, 'A_set'):
            self.flags['parametric_A'] = True
        else:
            self.flags['parametric_A'] = False
        
        if self.model.p < self.model.n:
            print(' --- Model is not fully actuated, so set flag')
            self.flags['underactuated'] = True
        else:
            self.flags['underactuated'] = False
        
        self.time['0_init'] = tocDiff(False)
        print('Abstraction object initialized - time:',self.time['0_init'])
    
    def _defAllCorners(self):
        '''
        Returns the vertices of every region in the partition (as nested list)

        Returns
        -------
        list
            Nested list of all vertices of all regions.

        '''
            
        # Create all combinations of n bits, to reflect combinations of all 
        # lower/upper bounds of partitions
        bitCombinations = list(itertools.product([0, 1], 
                               repeat=self.model.n))
        bitRelation     = ['low','upp']
        
        # Calculate all corner points of every partition. Every partition has 
        # an upper and lower bounnd in every state (dimension). Hence, the 
        # every partition has 2^n corners, with n the number of states.
        allOriginPointsNested = [[[
            self.partition['R'][bitRelation[bit]][i][bitIndex] 
                for bitIndex,bit in enumerate(bitList)
            ] for bitList in bitCombinations 
            ] for i in range(self.partition['nr_regions'])
            ]
        
        return np.array(allOriginPointsNested)
    
    def _create_manual_targets(self):
        '''
        Create target points, based on the vertices given

        Returns
        -------
        target : dict
            Dictionary with all data about the target points.

        '''
        
        print(' -- Compute manual target points; no. per dim:',self.model.setup['targets']['number'])
    
        # Create target points (similar to a partition of the state space)
        d = definePartitions(self.model.n,
                self.model.setup['targets']['number'],
                self.model.setup['targets']['width'],
                self.model.setup['targets']['origin'],
                onlyCenter = True)
        
        return d
        
    def define_states(self):
        ''' 
        Define the discrete state space partition and target points
        
        Returns
        -------
        None.
        '''
        
        # Create partition
        print('\nComputing partition of the state space...')
        
        # Define partitioning of state-space
        self.partition = dict()
        
        # Determine origin of partition
        self.model.setup['partition']['number'] = np.array(self.model.setup['partition']['number'])
        self.model.setup['partition']['width'] = np.array([-1,1]) @ self.model.setup['partition']['boundary'].T / self.model.setup['partition']['number']
        self.model.setup['partition']['origin'] = 0.5 * np.ones(2) @ self.model.setup['partition']['boundary'].T
        
        self.partition['R'] = definePartitions(self.model.n,
                            self.model.setup['partition']['number'],
                            self.model.setup['partition']['width'],
                            self.model.setup['partition']['origin'],
                            onlyCenter = False)
        
        self.partition['nr_regions'] = len(self.partition['R']['center'])
        
        # Determine goal regions
        self.partition['goal'], self.partition['goal_slices'], self.partition['goal_idx'] = defStateLabelSet(
            allCenters = self.partition['R']['c_tuple'], 
            sets = self.model.setup['specification']['goal'],
            partition = self.model.setup['partition'])
        
        # Determine critical regions
        self.partition['critical'], self.partition['critical_slices'], self.partition['critical_idx'] = defStateLabelSet(
            allCenters = self.partition['R']['c_tuple'], 
            sets = self.model.setup['specification']['critical'],
            partition = self.model.setup['partition'])
        
        print(' -- Number of regions:',self.partition['nr_regions'])
        print(' -- Number of goal regions:',len(self.partition['goal']))
        print(' -- Number of critical regions:',len(self.partition['critical']))

        self.time['1_partition'] = tocDiff(False)
        print('Discretized states defined - time:',self.time['1_partition'])
        
        self.partition['allCorners']     = self._defAllCorners()
        self.partition['allCornersFlat'] = np.concatenate(self.partition['allCorners'])

    def define_actions(self):
        ''' 
        Determine which actions are actually enabled.
        
        Returns
        -------
        None.
        '''
        
        print('\nDefining target points...')
        self.actions = {}
        
        # Create the target point for every action (= every state)
        if self.model.setup['targets']['number'] == 'auto':
            # Set default target points to the center of every region
            self.model.setup['targets']['number'] = self.model.setup['partition']['number']
            self.model.setup['targets']['width'] = self.model.setup['partition']['width']
            self.model.setup['targets']['origin'] = self.model.setup['partition']['origin']
            
            self.actions['T'] = {'center': self.partition['R']['center'],
                                 'idx': self.partition['R']['idx'] }
            
        else:
            self.model.setup['targets']['number'] = np.array(self.model.setup['targets']['number'])
            self.model.setup['targets']['width'] = np.array([-1,1]) @ self.model.setup['targets']['boundary'].T / self.model.setup['targets']['number']
            self.model.setup['targets']['origin'] = 0.5 * np.ones(2) @ self.model.setup['targets']['boundary'].T
            
            self.actions['T'] = self._create_manual_targets()
        
        # Add additional target points if this is requested
        if 'extra_targets' in self.model.setup:
            self.actions['T']['center'] = np.vstack(( self.actions['T']['center'],
                                                   self.model.setup['extra_targets'] ))
        
        self.actions['nr_actions'] = len(self.actions['T']['center'])
        
        print('\nComputing all backward reachable sets...')
        
        self.actions['backreach'] = def_all_BRS(self.model, self.actions['T']['center'])
        
        print('\nComputing set of enabled actions...')
        
        ### 1 ###
        # If underactuated...
        if self.flags['underactuated']:
            print('- For underactuated system')
            
            # Find the connected components of the system
            dim_n, dim_p = find_connected_components(self.model.A, self.model.B,
                                                     self.model.n, self.model.p)
            
            print(' -- Number of actions (target points):', self.actions['nr_actions'])

            A = [None for i in range(len(dim_n))]
            A_inv = [None for i in range(len(dim_n))]
            CE = [None for i in range(len(dim_n))]

            for i,(dn,dp) in enumerate(zip(dim_n, dim_p)):
            
                print(' --- In dimensions of state', dn,'and control', dp)    
            
                A[i], A_inv[i], CE[i] = \
                    defEnabledActions_UA(self.flags, self.partition, self.actions, self.model, dn, dp)
                    
                    
            self.CE = CE
            
            ### Compositional model building
                    
            # Initialize variables
            self.actions['enabled'] = [set() for i in range(self.partition['nr_regions'])]
            self.actions['enabled_inv'] = [set() for i in range(self.actions['nr_actions'])]
            self.actions['control_error'] = {}
            
            ## Merge together enabled actions (per state)
            keys_prod = itertools.product(*[A[i].keys() for i in range(len(dim_n))])
            vals_prod = itertools.product(*[A[i].values() for i in range(len(dim_n))])
            
            # Zipping over the product of the keys/values of the dictionaries
            for keys,vals in zip(keys_prod,vals_prod):
                
                # Add tuples to get the compositional state
                sum_key = np.sum(keys, axis=0)
                i = self.partition['R']['idx'][tuple(sum_key)]
                
                # Skip if i is a critical state
                if i in self.partition['critical']:
                    continue
                
                v_list = list(itertools.product(*vals))
                v_sum  = np.sum(v_list, axis=1)
                enabled = set()
                for v in v_sum:
                    enabled.add( self.actions['T']['idx'][tuple(v)] )
                
                self.actions['enabled'][i] = enabled
              
            ## Merge together successor states (per action)
            keys_prod = itertools.product(*[A_inv[i].keys() for i in range(len(dim_n))])
            vals_inv = itertools.product(*[A_inv[i].values() for i in range(len(dim_n))])
            vals_CE = itertools.product(*[CE[i].values() for i in range(len(dim_n))])
            
            # Precompute matrix to put together control error
            mats = [None] * len(dim_n)
            for h,dim in enumerate(dim_n):
                mats[h] = np.zeros((self.model.n, len(dim)))
                for j,i in enumerate(dim):
                    mats[h][i,j] = 1
            
            nr_A = 0
            
            # Zipping over the product of the keys/values of the dictionaries
            for keys, valsA, valsB in zip(keys_prod, vals_inv, vals_CE):
                
                # Add tuples to get the compositional state
                sum_key = np.sum(keys, axis=0)
                a = self.actions['T']['idx'][tuple(sum_key)]
                
                v_list = list(itertools.product(*valsA))
                v_sum  = np.sum(v_list, axis=1)
                enabled_inv = set()
                for v in v_sum:
                    
                    state = self.partition['R']['idx'][tuple(v)]
                    
                    # Skip if v is a critical state
                    if state in self.partition['critical']:
                        continue
                    
                    enabled_inv.add( state )
                
                self.actions['enabled_inv'][a] = enabled_inv
                
                if len(enabled_inv) > 0:
                    nr_A += 1
                
                # Compute control error
                self.actions['control_error'][a] = {
                    'pos': np.sum([mats[z] @ valsB[z]['pos'] for z in range(len(valsB))], axis=0),
                    'neg': np.sum([mats[z] @ valsB[z]['neg'] for z in range(len(valsB))], axis=0)
                    }
        
            self.CE = CE
        
        ### 3 ###
        else:
            nr_A, self.actions['enabled'], \
             self.actions['enabled_inv'], _ = defEnabledActions(self.setup, self.partition, self.actions, self.model)
              
        a = np.round(self.actions['nr_actions'] / 2).astype(int)
        partition_plot((0,1), (), self.setup, self.model, self.partition,
                       np.array([]), self.actions['backreach'][a])
                
        print(nr_A,'actions enabled')
        if nr_A == 0:
            printWarning('No actions enabled at all, so terminate')
            sys.exit()
    
        self.time['2_enabledActions'] = tocDiff(False)
        print('Enabled actions define - time:',self.time['2_enabledActions'])
        
        # assert False
        
    def build_iMDP(self, problem_type='reachavoid'):
        '''
        Build the (i)MDP and create all respective PRISM files.

        Returns
        -------
        model_size : dict
            Dictionary describing the number of states, choices, and 
            transitions.

        '''
        
        # Initialize MDP object
        self.mdp = mdp(self.setup, self.N, self.partition, self.actions)
        
        # Create PRISM file (explicit way)
        model_size, self.mdp.prism_file, self.mdp.spec_file, \
        self.mdp.specification = \
            self.mdp.writePRISM_explicit(self.actions, self.partition, self.trans, problem_type, 
                                         self.setup.mdp['mode'])   

        self.time['4_MDPcreated'] = tocDiff(False)
        print('MDP created - time:',self.time['4_MDPcreated'])
        
        return model_size
            
    def solve_iMDP(self):
        '''
        Solve the (i)MDP usign PRISM

        Returns
        -------
        None.

        '''
            
        # Solve the MDP in PRISM (which is called via the terminal)
        policy_file, vector_file = self._solveMDPviaPRISM()
        
        # Load PRISM results back into Python
        self.loadPRISMresults(policy_file, vector_file)
            
        self.time['5_MDPsolved'] = tocDiff(False)
        print('MDP solved in',self.time['5_MDPsolved'])
        
    def _solveMDPviaPRISM(self):
        '''
        Call PRISM to solve (i)MDP while executing the Python codes.

        Returns
        -------
        policy_file : str
            Name of the file in which the optimal policy is stored.
        vector_file : str
            Name of the file in which the optimal rewards are stored.

        '''
        
        import subprocess

        prism_folder = self.setup.mdp['prism_folder'] 
        
        print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        
        print('Starting PRISM...')
        
        spec = self.mdp.specification
        mode = self.setup.mdp['mode']
        java_memory = self.setup.mdp['prism_java_memory']
        
        print(' -- Running PRISM with specification for mode',
              mode.upper()+'...')
    
        file_prefix = self.setup.directories['outputFcase'] + "PRISM_" + mode
        policy_file = file_prefix + '_policy.csv'
        vector_file = file_prefix + '_vector.csv'
    
        options = ' -ex -exportadv "'+policy_file+'"'+ \
                  ' -exportvector "'+vector_file+'"'
    
        # Switch between PRISM command for explicit model vs. default model
        if self.setup.mdp['prism_model_writer'] == 'explicit':
    
            print(' --- Execute PRISM command for EXPLICIT model description')        
    
            model_file      = '"'+self.mdp.prism_file+'"'             
        
            # Explicit model
            command = prism_folder+"bin/prism -javamaxmem "+ \
                str(java_memory)+"g -importmodel "+model_file+" -pf '"+ \
                spec+"' "+options
        else:
            
            print(' --- Execute PRISM command for DEFAULT model description')
            
            model_file      = '"'+self.mdp.prism_file+'"'
            
            # Default model
            command = prism_folder+"bin/prism -javamaxmem "+ \
                str(java_memory)+"g "+model_file+" -pf '"+spec+"' "+options    
        
        subprocess.Popen(command, shell=True).wait()    
        
        return policy_file, vector_file
        
    def loadPRISMresults(self, policy_file, vector_file):
        '''
        Load results from existing PRISM output files.

        Parameters
        ----------
        policy_file : str
            Name of the file to load the optimal policy from.
        vector_file : str
            Name of the file to load the optimal policy from.

        Returns
        -------
        None.

        '''
        
        self.results = dict()
        
        # Read policy CSV file
        policy_all = pd.read_csv(policy_file, header=None).iloc[:, 3:].\
            fillna(-1).to_numpy()
            
        # Flip policy upside down (PRISM generates last time step at top!)
        policy_all = np.flipud(policy_all)
        
        self.results['optimal_policy'] = np.zeros(np.shape(policy_all), dtype=int)
        
        rewards_k0 = pd.read_csv(vector_file, header=None).iloc[3:].to_numpy()
        self.results['optimal_reward'] = rewards_k0.flatten()
        
        for i,row in enumerate(policy_all):    
            for j,value in enumerate(row):
                
                # If value is not -1 (means no action defined)
                if value != -1:
                    # Split string
                    value_split = value.split('_')
                    # Store action ID
                    self.results['optimal_policy'][i,j] = int(value_split[1])
                else:
                    # If no policy is known, set to -1
                    self.results['optimal_policy'][i,j] = int(value)
        
    def generate_probability_plots(self):
        '''
        Generate (optimal reachability probability) plots
        '''
        
        print('\nGenerate plots')
        
        if self.partition['nr_regions'] <= 1000:
        
            from .postprocessing.createPlots import createProbabilityPlots
        
            if not hasattr(self, 'mc'):
                self.mc = None    
        
            if self.setup.plotting['probabilityPlots']:
                createProbabilityPlots(self.setup, self.N, self.model,
                                       self.results, self.partition, self.mc)
                    
        else:
            printWarning("Omit probability plots (nr. of regions too large)")
        
    def generate_UAV_plots(self, case_id, writer, exporter, itersToSim = 1000):
        
        # The code below plots the trajectories for the UAV benchmark
        if self.model.name in ['UAV', 'shuttle']:
            
            from core.postprocessing.createPlots import UAVplots
        
            # Create trajectory plot
            performance_df = UAVplots(self, case_id, writer, itersToSim)
                
            exporter.add_to_df(performance_df, 'performance')
            
    def generate_heatmap(self):
            
            from core.postprocessing.createPlots import reachabilityHeatMap
            
            # Create heat map
            reachabilityHeatMap(self)
                
        
###############################

class scenarioBasedAbstraction(Abstraction):
    def __init__(self, setup, model):
        '''
        Initialize scenario-based abstraction (ScAb) object

        Parameters
        ----------
        setup : dict
            Setup dictionary.
        model : dict
            Base model for which to create the abstraction.

        Returns
        -------
        None.

        '''
        
        # Copy setup to internal variable
        self.setup = setup
        self.model = model
        
        # Start timer
        tic()
        ticDiff()
        self.time = dict()
        self.flags = {}
        
        Abstraction.__init__(self)
        
    def _loadScenarioTable(self, tableFile, k):
        '''
        Load tabulated bounds on the transition probabilities (computed using
        the scenario approach).

        Parameters
        ----------
        tableFile : str
            File from which to load the table.

        Returns
        -------
        memory : dict
            Dictionary containing all loaded probability bounds / intervals.

        '''
        
        if not os.path.isfile(tableFile):
            sys.exit('ERROR: the following table file does not exist:'+
                     str(tableFile))
        
        with open(tableFile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            
            # Skip header row
            next(reader)
            
            memory = np.full((k+1, 2), fill_value = -1, dtype=float)
                
            for i,row in enumerate(reader):
                
                strSplit = row[0].split(',')
                
                value = [float(i) for i in strSplit[-2:]]
                memory[int(strSplit[0])] = value
                    
        return memory
        
    def _computeProbabilityBounds(self, tab, k):
        '''
        Compute transition probability intervals (bounds)

        Parameters
        ----------
        tab : dict
            Table dictionary.
        k : int
            Discrete time step.

        Returns
        -------
        prob : dict
            Dictionary containing the computed transition probabilities.

        '''
        
        prob = dict()
        printEvery = min(100, max(1, int(self.actions['nr_actions']/10)))

        if self.setup.scenarios['gaussian'] is True:
            # Compute Gaussian noise samples
            samples_zero_mean = np.random.multivariate_normal(
                            np.zeros(self.model.n), self.model.noise['w_cov'], 
                            size=(self.actions['nr_actions'],
                                  self.setup.scenarios['samples']))

        # For every action (i.e. target point)
        for a in range(self.actions['nr_actions']):
            
            # Check if action a is available in any state at all
            if len(self.actions['enabled_inv'][a]) > 0:
                    
                prob[a] = dict()
                mu = self.actions['T']['center'][a]
                
                if self.setup.scenarios['gaussian'] is True:
                    samples = mu + samples_zero_mean[a]                                        
                else:
                    # Determine non-Gaussian noise samples (relative from 
                    # target point)
                    samples = mu + np.array(
                        random.choices(self.model.noise['samples'], 
                        k=self.setup.scenarios['samples']) )
                    
                if self.flags['underactuated']:
                    
                    exclude = exclude_samples(samples, 
                                      self.model.setup['partition']['width'])
                    
                    prob[a] = computeScenarioBounds_error(self.setup, 
                          self.model.setup['partition'], 
                          self.partition, self.trans, samples, self.actions['control_error'][a], exclude)
                    
                    # Print normal row in table
                    if a % printEvery == 0:
                        nr_transitions = len(prob[a]['successor_idxs'])
                        tab.print_row([k, a, 
                           'Probabilities computed (transitions: '+
                           str(nr_transitions)+')'])
                        
                    from .compute_probabilities import plot_transition
                    
                    a_plot = self.actions['T']['c_tuple'][(21,38.05)]
                    if a == a_plot:
                        
                        plot_transition(samples, self.actions['control_error'][a], 
                            (0,1), (), self.setup, self.model, self.partition,
                            np.array([]), self.actions['backreach'][a])
                
                else:
                    
                    prob[a] = computeScenarioBounds_sparse(self.setup, 
                          self.model.setup['partition'], 
                          self.partition, self.trans, samples)
                
                    # Print normal row in table
                    if a % printEvery == 0:
                        nr_transitions = len(prob[a]['successor_idxs'])
                        tab.print_row([k, a, 
                           'Probabilities computed (transitions: '+
                           str(nr_transitions)+')'])
                
        return prob
    
    def define_probabilities(self):
        '''
        Define the transition probabilities of the finite-state abstraction 
        (perform for every iteration of the iterative scheme).

        Returns
        -------
        None.

        '''
           
        # Column widths for tabular prints
        col_width = [8,8,8,46]
        tab = table(col_width)
        
        self.trans = {'prob': {}}
                
        print(' -- Loading scenario approach table...')
        
        tableFile = self.setup.directories['base'] + '/input/SaD_probabilityTable_N='+ \
                        str(self.setup.scenarios['samples'])+'_beta='+ \
                        str(self.setup.scenarios['confidence'])+'.csv'
        
        # Load scenario approach table
        self.trans['memory'] = self._loadScenarioTable(tableFile = tableFile,
                                       k = self.setup.scenarios['samples'])
        
        # Retreive type of horizon
        k_range = [0]
        
        print('Computing transition probabilities...')
        
        self.trans['prob'] = dict()
        
        # For every time step in the horizon
        for k in k_range:
            
            # Print header row
            tab.print_row(['K','ACTION','STATUS'], head=True)    
            
            self.trans['prob'][k] = \
                self._computeProbabilityBounds(tab, k)
            
        # Delete iterable variables
        del k
        
        self.time['3_probabilities'] = tocDiff(False)
        print('Transition probabilities calculated - time:',
              self.time['3_probabilities'])
        
def exclude_samples(samples, width):
    
    N,n = samples.shape
    
    S = np.reshape(samples, (N,n,1))
    diff = S - S.T
    width_tile = np.tile(width, (N,1)).T
    boolean = np.any(diff > width_tile, axis=1) | np.any(diff < -width_tile, axis=1)
    
    mp = map(np.nonzero, boolean)
    exclude = [set(m[0]) for m in mp]
    
    return exclude
    
    # exclude = [None for n in range(self.setup.scenarios['samples'])]
    
    # for n in range(self.setup.scenarios['samples']):
    #     nonzero = np.nonzero(
    #                 np.any(samples - samples[n] >  width, axis=1) ^
    #                 np.any(samples - samples[n] < -width, axis=1)
    #             )[0]
        
    #     distance = np.linalg.norm(samples[nonzero] - samples[n], axis=1)
    #     sort = np.argsort(distance)
        
    #     exclude[n] = set(nonzero[sort])