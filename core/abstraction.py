#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Implementation of the method proposed in the paper:
 "Probabilities Are Not Enough: Formal Controller Synthesis for Stochastic 
  Dynamical Models with Epistemic Uncertainty"

Originally coded by:        <anonymized>
Contact e-mail address:     <anonymized>
______________________________________________________________________________
"""

import numpy as np              # Import Numpy for computations
import itertools                # Import to crate iterators
import sys                      # Allows to terminate the code at some point
import pandas as pd             # Import Pandas to store data in frames
import sys                      # Allows to terminate the code at some point
import os                       # Import OS to allow creationg of folders
import csv                      # Import to create/load CSV files
import random                   # Import to use random variables
import subprocess               # Import to call prism via terminal command

from .define_model import find_connected_components
from .define_partition import define_partition, define_spec_region, \
    partition_plot, state2region
from .commons import tocDiff, printWarning
from .create_iMDP import mdp
from .action_classes import action

'''
------------------------------------------------------------------------------
Main abstraction object definition
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
        
        print('Size of the continuous model:')
        print(' -- Dimension of the state:',self.model.n)
        print(' -- Dimension of the control input:',self.model.p)
        
        # Number of simulation time steps
        self.N = int(self.spec.end_time / self.model.lump)
        
        # Determine if model has parametric uncertainty
        if hasattr(self.model, 'A_set'):
            print(' --- Model has parametric uncertainty, so set flag')
            self.flags['parametric'] = True
        else:
            print(' --- Model does not have parametric uncertainty')
            self.flags['parametric'] = False
        
        # Determine if model is underactuated
        if self.model.p < self.model.n:
            print(' --- Model is not fully actuated, so set flag')
            self.flags['underactuated'] = True
        else:
            print(' --- Model is fully actuated')
            self.flags['underactuated'] = False
        
        self.time['0_init'] = tocDiff(False)
        print('Abstraction object initialized - time:',self.time['0_init'])
    
    

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
        self.spec.partition['number'] = np.array(self.spec.partition['number'])
        self.spec.partition['width'] = np.array([-1,1]) @ \
            self.spec.partition['boundary'].T / self.spec.partition['number']
        self.spec.partition['origin'] = 0.5 * np.ones(2) @ \
            self.spec.partition['boundary'].T
        
        print(' -- Width of regions in each dimension:',self.spec.partition['width'])

        self.partition['R'] = define_partition(self.model.n,
                            self.spec.partition['number'],
                            self.spec.partition['width'],
                            self.spec.partition['origin'])
        
        self.partition['nr_regions'] = len(self.partition['R']['center'])
        
        print(' -- Number of regions:',self.partition['nr_regions'])

        # Determine goal regions
        self.partition['goal'], self.partition['goal_slices'], \
            self.partition['goal_idx'] = define_spec_region(
                allCenters = self.partition['R']['c_tuple'], 
                sets = self.spec.goal,
                partition = self.spec.partition,
                borderOutside = True)
        
        print(' -- Number of goal regions:',len(self.partition['goal']))

        # Determine critical regions
        self.partition['critical'], self.partition['critical_slices'], \
            self.partition['critical_idx'] = define_spec_region(
                allCenters = self.partition['R']['c_tuple'], 
                sets = self.spec.critical,
                partition = self.spec.partition,
                borderOutside = True)
        
        print(' -- Number of critical regions:',len(self.partition['critical']))

        self.time['1_partition'] = tocDiff(False)
        print('Discretized states defined - time:',self.time['1_partition'])

        # Create all combinations of n bits, to reflect combinations of all 
        # lower/upper bounds of partitions
        bitCombinations = list(itertools.product([0, 1], 
                               repeat=self.model.n))
        bitRelation     = ['low','upp']
        
        # Calculate all corner points of every partition. Every partition has 
        # an upper and lower bounnd in every state (dimension). Hence, the 
        # every partition has 2^n corners, with n the number of states.
        self.partition['allCorners'] = np.array( [[[
            self.partition['R'][bitRelation[bit]][i][bitIndex] 
                for bitIndex,bit in enumerate(bitList)
            ] for bitList in bitCombinations 
            ] for i in range(self.partition['nr_regions'])
            ] )



    def define_target_points(self):
        
        self.actions = {'obj': {},
                        'backreach_obj': {},
                        'tup2idx': {},
                        'extra_act': []} 

        print('\nDefining backward reachable sets...')
        
        # Define backward reachable sets
        self.define_backreachsets()
            
        backreach_obj = self.actions['backreach_obj']['default']
                
        print('\nDefining target points...')
        
        # Create the target point for every action (= every state)
        if type(self.spec.targets['number']) == str:
            # Set default target points to the center of every region
            
            for center, (tup,idx) in zip(self.partition['R']['center'],
                                        self.partition['R']['idx'].items()):
                
                self.actions['obj'][idx] = action(idx, self.model, center, tup, 
                                                  backreach_obj)
                self.actions['tup2idx'][tup] = idx
            
        else:  
            
            print(' -- Compute manual target points; no. per dim:',self.spec.targets['number'])
        
            ranges = map(np.linspace, self.spec.targets['boundary'][:,0],
                         self.spec.targets['boundary'][:,1], self.spec.targets['number'])
            
            tuples = map(np.arange, np.zeros(self.model.n),
                         self.spec.targets['number'])
            
            for idx,(center,tup) in enumerate(zip(itertools.product(*ranges),
                                                itertools.product(*tuples))):
                
                self.actions['obj'][idx] = action(idx, self.model, 
                                                  np.array(center), tup, 
                                                  backreach_obj)
                self.actions['tup2idx'][tup] = idx
        
        nr_default_act = len(self.actions['obj'])
        
        # Add additional target points if this is requested
        if 'extra' in self.spec.targets:
            
            backreach_obj = self.actions['backreach_obj']['extra']
            
            for i,center in enumerate(self.spec.targets['extra']):    
                
                self.actions['obj'][nr_default_act+i] = action(nr_default_act+i, self.model, center, -i, backreach_obj)
                self.actions['extra_act'] += [self.actions['obj'][nr_default_act+i]]
        
        self.actions['nr_actions'] = len(self.actions['obj'])



    def define_enabled_actions(self):
        ''' 
        Determine which actions are actually enabled.
        
        Returns
        -------
        None.
        '''
            
        print('\nComputing set of enabled actions...')
        
        # Find the connected components of the system
        dim_n, dim_p = find_connected_components(self.model.A, self.model.B,
                                                 self.model.n, self.model.p)
        
        print(' -- Number of actions (target points):', self.actions['nr_actions'])

        enabled = [None for i in range(len(dim_n))]
        enabled_inv = [None for i in range(len(dim_n))]
        error = [None for i in range(len(dim_n))]

        for i,(dn,dp) in enumerate(zip(dim_n, dim_p)):
        
            print(' --- In dimensions of state', dn,'and control', dp)    
        
            enabled[i], enabled_inv[i], error[i] = self.get_enabled_actions(self.model, self.spec, dn, dp)

        print('\nCompose enabled actions in independent dimensions...')

        nr_act = self._composeEnabledActions(dim_n, enabled, 
                                             enabled_inv, error)

        self.set_extra_actions()
            
        ### PLOT ###
        if self.args.partition_plot:
            
            if 'partition_plot_action' in self.model.setup:
                a = self.model.setup['partition_plot_action']
            else:
                a = np.round(self.actions['nr_actions'] / 2).astype(int)
                
            partition_plot((0,1), (), self, cut_value=np.array([]), act=self.actions['obj'][a] )
            for a in range(0,self.actions['nr_actions'],100):
                print('Create plot of partition with backward reachable set...')
                
                partition_plot((0,1), (), self, cut_value=np.array([]), act=self.actions['obj'][a] )
                
        print(nr_act,'actions enabled')
        if nr_act == 0:
            printWarning('No actions enabled at all, so terminate')
            sys.exit()

        s_init = state2region(self.args.x_init, self.spec.partition, self.partition['R']['c_tuple'])[0]
        print('In initial state '+str(s_init)+', the following actions are enabled:')
        print([self.actions['obj'][a].center for a in self.actions['enabled'][s_init]])

        self.time['2_enabledActions'] = tocDiff(False)
        print('Enabled actions define - time:',self.time['2_enabledActions'])
        
        
        
    def _composeEnabledActions(self, dim_n, enabled_sub, enabled_sub_inv, 
                               control_error_sub):
        
        if None in control_error_sub:
            no_error = True
        else:
            no_error = False

        # Initialize variables
        self.actions['enabled'] = [set() for i in range(self.partition['nr_regions'])]
            
        ## Merge together successor states (per action)
        enabled_inv_keys = itertools.product(*[enabled_sub_inv[i].keys() 
                                               for i in range(len(dim_n))])
        enabled_inv_vals = itertools.product(*[enabled_sub_inv[i].values() 
                                                for i in range(len(dim_n))])
        
        # Precompute matrix to put together control error
        mats = [None] * len(dim_n)
        for h,dim in enumerate(dim_n):
            mats[h] = np.zeros((self.model.n, len(dim)))
            for j,i in enumerate(dim):
                mats[h][i,j] = 1
        
        nr_act = 0
        
        # Zipping over the product of the keys/values of the dictionaries
        for keys, vals_enab in zip(enabled_inv_keys, enabled_inv_vals):

            # Check if we have to save an error term as well
            if not no_error:
                vals_error = [control_error_sub[i][key] for i,key in enumerate(keys)]
            
            # Add tuples to get the compositional state
            act_idx = self.actions['tup2idx'][tuple(np.sum(keys, axis=0))]
            act_obj = self.actions['obj'][act_idx]
            
            s_elems = list(itertools.product(*vals_enab))

            # If action not enabled in one of the subcomponenets, then skip
            if len(s_elems) == 0:
                continue
                
            s_enabledin  = np.sum(s_elems, axis=1)
            
            for s in s_enabledin:
                
                state = self.partition['R']['idx'][tuple(s)]
                
                # Skip if v is a critical state
                if state in self.partition['critical']:
                    continue
                
                act_obj.enabled_in.add( state )
                self.actions['enabled'][state].add( act_idx )
            
            # Check if action is enabled in any state
            if len(act_obj.enabled_in) > 0:
                nr_act += 1
            
            if not no_error:
                # Compute control error for this action
                if hasattr(self.model, 'Q_uncertain'):
                    # Also account for the uncertain disturbances
                    act_obj.error = {
                        'pos': np.sum([mats[z] @ vals_error[z]['pos'] for z in 
                                    range(len(dim_n))], axis=0) + self.model.Q_uncertain['max'],
                        'neg': np.sum([mats[z] @ vals_error[z]['neg'] for z in 
                                    range(len(dim_n))], axis=0) + self.model.Q_uncertain['min']
                        }
                
                else:
                    # No uncertain disturbance to account for
                    act_obj.error = {
                        'pos': np.sum([mats[z] @ vals_error[z]['pos'] for z in 
                                    range(len(dim_n))], axis=0),
                        'neg': np.sum([mats[z] @ vals_error[z]['neg'] for z in 
                                    range(len(dim_n))], axis=0)
                        }
            
        return nr_act
        


    def noise_sampler(self):

        if self.args.nongaussian_noise:

            # Use non-Gaussian noise samples (loaded from external file)
            noise_samples = np.array(
                        random.choices(self.model.noise['samples'], 
                        k=self.args.noise_samples) )

        else:

            # Compute Gaussian noise samples
            noise_samples = np.random.multivariate_normal(
                            np.zeros(self.model.n), self.model.noise['w_cov'], 
                            size=self.args.noise_samples)

        return noise_samples

        
        
    def build_iMDP(self):
        '''
        Build the (i)MDP and create all respective PRISM files.

        Returns
        -------
        model_size : dict
            Dictionary describing the number of states, choices, and 
            transitions.

        '''
        
        problem_type = self.spec.problem_type
        
        # Initialize MDP object
        self.mdp = mdp(self.setup, self.N, self.partition, self.actions)
        
        # Create PRISM file (explicit way)
        model_size, self.mdp.prism_file, self.mdp.spec_file, \
        self.mdp.specification = \
            self.mdp.writePRISM_explicit(self.actions, self.partition, 
                                 self.trans, problem_type, self.args.mdp_mode)   

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

        prism_folder = self.args.prism_folder
        
        print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        
        print('Starting PRISM...')
        
        spec = self.mdp.specification
        mode = self.args.mdp_mode
        
        print(' -- Running PRISM with specification for mode',
              mode.upper()+'...')
    
        file_prefix = self.setup.directories['outputFcase'] + "PRISM_" + mode
        policy_file = file_prefix + '_policy.csv'
        vector_file = file_prefix + '_vector.csv'
    
        options = ' -ex -exportadv "'+policy_file+'"'+ \
                  ' -exportvector "'+vector_file+'"'
    
        print(' --- Execute PRISM command for EXPLICIT model description')        

        model_file      = '"'+self.mdp.prism_file+'"'             
    
        # Explicit model
        command = prism_folder+"bin/prism -javamaxmem "+ \
            str(self.args.prism_java_memory)+"g -importmodel "+model_file+" -pf '"+ \
            spec+"' "+options
        
        subprocess.Popen(command, shell=True).wait()    
        
        # Load PRISM results back into Python
        self.loadPRISMresults(policy_file, vector_file)
            
        self.time['5_MDPsolved'] = tocDiff(False)
        print('MDP solved in',self.time['5_MDPsolved'])
        
        
    
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
        
        # Convert avoid probability to the safety probability
        if self.spec.problem_type == 'avoid':
            self.results['optimal_reward'] = 1 - self.results['optimal_reward']
        
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
            
        
###############################


def loadScenarioTable(self, tableFile, k):
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