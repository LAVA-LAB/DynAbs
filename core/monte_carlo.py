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
import pandas as pd             # Import Pandas to store data in frames
import random                   # Import to use random variables
from progressbar import progressbar # Import to create progress bars

from .define_partition import computeRegionCenters
from .commons import tocDiff, table
from .cvx_opt import Controller

class MonteCarloSim():
    '''
    Class to run Monte Carlo simulations under a derived controller
    '''
    
    def __init__(self, Ab, iterations=100, writer=False, random_initial_state=False, **kwargs):
        '''
        Initialization function

        Parameters
        ----------
        Ab : abstraction instance
            Full object of the abstraction being plotted for
        iterations : int, optional
            Number of Monte Carlo iterations. The default is 100.

        Returns
        -------
        None.

        '''
        
        print(' -- Starting Monte Carlo simulations...')

        if Ab.flags['underactuated']:
            self.controller = Controller(Ab.model)

        self.results = dict()
        self.traces = dict()
        
        # Copy necessary data from abstraction object
        self.flags = Ab.flags
        self.model = Ab.model
        self.partition = Ab.partition
        self.actions = Ab.actions
        self.policy = Ab.results['optimal_policy']
        self.args = Ab.args
        self.horizon = Ab.N
        self.spec = Ab.spec

        self.random_initial_state = random_initial_state

        # Column widths for tabular prints
        if self.args.verbose:
            col_width = [8,6,6,46]
            self.tab = table(col_width)

        if 'init_states' in kwargs:
            self.init_states = kwargs['init_states']
            print('-- Manual initial states:', self.init_states)
        else:               
            self.init_states = np.arange(Ab.partition['nr_regions'])
            
        self.results['reachability_probability'] = {}
            
        self.iterations = iterations
        
        #####
        
        # Only precompute noise samples if in Gaussian mode
        if not self.args.nongaussian_noise:
            self.defineDisturbances()

        # Run Monte Carlo simulations
        self.run()

        if writer:
            # Store Monte Carlo results as dataframe
            MCsims_df = pd.Series( 
                self.results['reachability_probability'], 
                index=self.init_states )
                
            # Write Monte Carlo results to Excel
            MCsims_df.to_excel(writer, sheet_name='Empirical reach.')
        
        #####
        
        del self.flags
        del self.model
        del self.partition
        del self.actions
        del self.policy
        del self.horizon
        del self.spec
        if not self.args.nongaussian_noise:
            del self.noise
        del self.args
        

    def defineDisturbances(self):
        
        if self.args.verbose:
            print(' -- Computing required random variables...')
        
        # Gaussian noise mode
        self.noise = np.random.multivariate_normal(
                np.zeros(self.model.n), self.model.noise['w_cov'],
                (len(self.init_states), self.iterations, self.horizon ))
            
    
    def run(self):
        
        if not self.args.verbose:
            iterator = progressbar(self.init_states, 
                                   redirect_stdout=True)
        else:
            iterator = self.init_states

        for s_abs, s_init in enumerate(iterator):
            
            if self.args.verbose:
                self.tab.print_row(['STATE','ITER','K','STATUS'], head=True)
            
            # Create dictionaries for results related to partition i
            self.results[s_init] = {'goalReached': 
                                    np.full(self.iterations, False, dtype=bool)}
            self.traces[s_init]  = {}
            
            # For each of the monte carlo iterations
            for m in range(self.iterations):
                
                self.traces[s_init][m], self.results[s_init]['goalReached'][m] = \
                    self._runIteration(s_abs, s_init, m)
                    
            self.results['reachability_probability'][s_init] = \
                np.mean( self.results[s_init]['goalReached'] )
    
    def _runIteration(self, s_abs, s_init, m):
        
        # Initialize variables at start of iteration
        success = False
        trace = {'k': [], 'x': [], 'action': []}
        k = 0
        
        if self.args.verbose:
            self.tab.print_row([s_init, m, k, 'Start Monte Carlo iteration'])
        
        # Initialize the current simulation
        x           = np.zeros((self.horizon + 1, self.model.n))
        x_target    = np.zeros((self.horizon + 1, self.model.n))
        x_region    = np.zeros(self.horizon + 1).astype(int)
        u           = np.zeros((self.horizon, self.model.p))
        action      = np.zeros(self.horizon).astype(int)
        
        # Determine nitial state
        if self.random_initial_state:
            x[0] = np.random.uniform(
                    low  = self.partition['R']['low'][s_init],
                    high = self.partition['R']['upp'][s_init])
        else:
            x[0] = self.partition['R']['center'][s_init]

        # Add current state, belief, etc. to trace
        trace['k'] += [0]
        trace['x'] += [x[0]]
        
        ######

        # For each time step in the finite time horizon
        while k <= self.horizon:
            
            # Determine to which region the state belongs
            region_center = computeRegionCenters(x[k], 
                    self.spec.partition).flatten()

            if tuple(region_center) in self.partition['R']['c_tuple']:
                # Save that state is currently in region ii
                x_region[k] = self.partition['R']['c_tuple'][tuple(region_center)]
                
            else:
                # Absorbing region reached
                x_region[k] = -1

                if self.args.verbose:
                    self.tab.print_row([s_init, m, k, 'Absorbing state reached, so abort'], sort="Warning")
                return trace, success

            # If current region is the goal state ... 
            if x_region[k] in self.partition['goal']:
                # Then abort the current iteration, as we have achieved the goal
                success = True
                if self.args.verbose:
                    self.tab.print_row([s_init, m, k, 'Goal state reached, x ='+str(np.round(x[k],2))], sort="Success")
                return trace, success
                
            # If current region is in critical states...
            elif x_region[k] in self.partition['critical']:
                # Then abort current iteration
                if self.args.verbose:
                    self.tab.print_row([s_init, m, k, 'Critical state reached, so abort'], sort="Warning")
                return trace, success

            # Check if we can still perform another action within the horizon
            elif k >= self.horizon:
                return trace, success

            # Retreive the action from the policy
            action[k] = self.policy[k, x_region[k]]

            if action[k] == -1:
                if self.args.verbose:
                    self.tab.print_row([s_init, m, k, 'No policy known, so abort'], sort="Warning")
                return trace, success

            ###
            
            # If loop was not aborted, we have a valid action            
            if self.args.verbose:
                self.tab.print_row([s_init, m, k, 'In state: '+str(x_region[k])+' ('+str(x[k])+'), take action: '+str(action[k])])
        
            # Set target state equal to the center of the target set
            x_target[k+1] = self.actions['obj'][action[k]].center

            # Reconstruct the control input required to achieve this target point
            # Note that we do not constrain the control input; we already know that a suitable control exists!
            if self.flags['underactuated']:
                success, _, u[k] = self.controller.solve(x_target[k+1], x[k], 
                    self.actions['obj'][action[k]].backreach_obj.target_set_size)
                
                if not success:
                    print('>> Failed to compute control input <<')
                    assert False
                
                # Implement the control into the physical (unobservable) system
                x_hat = self.model.A_true @ x[k] + self.model.B_true @ u[k] + self.model.Q_flat
            else:
                x_nom = x[k]

                u[k] = np.array(self.model.B_pinv @ ( x_target[k+1] - self.model.A @ x_nom.flatten() - self.model.Q_flat ))
            
                # Implement the control into the physical (unobservable) system
                x_hat = self.model.A @ x[k] + self.model.B @ u[k] + self.model.Q_flat

            # Use Gaussian process noise
            if self.args.nongaussian_noise:
                # Use generated non-Gaussian samples                                    
                noise = random.choice(self.model.noise['samples'])
                x[k+1] = x_hat + noise

            else:
                # Use Gaussian noise samples
                x[k+1] = x_hat + self.noise[s_abs, m, k]
               
            if any(self.model.uMin > u[k]) or any(self.model.uMax < u[k]):
                self.tab.print_row([s_init, m, k, 'Control input '+str(u[k])+' outside limits'], sort="Warning")
    
            # Add current state, belief, etc. to trace
            trace['k'] += [k+1]
            trace['x'] += [x[k+1]]
            trace['action'] += [action[k]]
            
            # Increase iterator variable by one
            k += 1
                        
        ######
        
        return trace, success