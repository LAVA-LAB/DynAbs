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
        init_states : list, optional
            List of initial states to start simulations from. Default is False.

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
    






def monte_carlo(Ab, iterations='auto', init_states='auto', 
               printDetails=False, writer=False,
               random_initial_state=False):
    '''
    Perform Monte Carlo simulations to validate the obtained results

    Parameters
    ----------
    iterations : str or int, optional
        Number of Monte Carlo iterations. The default is 'auto'.
    init_states : str or list, optional
        Initial states to start simulations from. The default is 'auto'.

    Returns
    -------
    None.

    '''
    
    if Ab.flags['underactuated']:
        controller = Controller(Ab.model)
    
    tocDiff(False)
    if Ab.args.verbose:
        print(' -- Starting Monte Carlo simulations...')
    
    mc = {'goal_reached': {}, 'traces': {}, 'action_traces': {}}
    
    if iterations != 'auto':
        Ab.setup.montecarlo['iterations'] = int(iterations)
        
    if init_states != 'auto':
        Ab.setup.montecarlo['init_states'] = list(init_states)
    else:
        Ab.setup.montecarlo['init_states'] = False
        
    mc['reachability'] = \
        np.zeros(Ab.partition['nr_regions'])
    
    # Column widths for tabular prints
    if Ab.args.verbose:
        col_width = [6,8,6,6,46]
        tab = table(col_width)

        print(' -- Computing required Gaussian random variables...')
    
    if Ab.setup.montecarlo['init_states'] == False:
        init_state_idxs = np.arange(Ab.partition['nr_regions'])
        
    else:
        init_state_idxs = Ab.setup.montecarlo['init_states']
    
    # The gaussian random variables are precomputed to speed up the code
    w_array = np.random.multivariate_normal(
        np.zeros(Ab.model.n), 
        Ab.model.noise['w_cov'],
       ( len(init_state_idxs), 
         Ab.setup.montecarlo['iterations'], Ab.N ))

    # For each of the monte carlo iterations
    if len(init_state_idxs) > 1:
        s_loop = progressbar(
                init_state_idxs, 
                redirect_stdout=True)
    else:
        s_loop = init_state_idxs

    # For each initial state
    for i_abs, i in enumerate(s_loop):
        
        # Check if we should perform Monte Carlo sims for this state
        if Ab.setup.montecarlo['init_states'] is not False and \
                    i not in Ab.setup.montecarlo['init_states']:
            print('Warning: this should not happen!')
            # If current state is not in the list of initial states,
            # continue to the next iteration
            continue
        # Otherwise, continue with the Monte Carlo simulation
        
        if Ab.args.verbose:
            tab.print_row(['STATE','ITER','K','STATUS'], 
                          head=True)
        
        # Create dictionaries for results related to partition i
        mc['goal_reached'][i] = np.full(
            Ab.setup.montecarlo['iterations'], False, dtype=bool)
        mc['traces'][i]  = dict()    
        mc['action_traces'][i] = dict()
        
        # For each of the monte carlo iterations
        if Ab.args.verbose or len(init_state_idxs) > 1:
            loop = range(Ab.setup.montecarlo['iterations'])
        else:
            loop = progressbar(
                    range(Ab.setup.montecarlo['iterations']), 
                    redirect_stdout=True)
            
        for m in loop:
            
            # Set initial time
            k = 0
            
            mc['traces'][i][m] = []
            mc['action_traces'][i][m] = []
            
            # Retreive the initial action time-grouping to be chosen
            # (given by the optimal policy to the MDP)
            action = Ab.results['optimal_policy'][k, i]
            
            if i in Ab.partition['goal']:
                # If initial state is already the goal state, succes
                # Then, abort the iteration, as we reached the goal
                mc['goal_reached'][i][m] = True
                
                if printDetails:
                    tab.print_row([i, m, k, 
                       'Initial state is goal state'], sort="Success")
            
            elif action == -1:
                # If action=-1, no policy known, and reachability is 0
                if Ab.args.verbose:
                    tab.print_row([i, m, k, 
                       'No initial policy known, so abort'], 
                       sort="Warning")
            
            else:
                if Ab.args.verbose:
                    tab.print_row([i, m, k, 
                       'Start Monte Carlo iteration'])
                                    
                # Initialize the current simulation
                x           = np.zeros((Ab.N+1, Ab.model.n))
                x_goal      = np.zeros((Ab.N+1, Ab.model.n))
                x_region    = np.zeros(Ab.N).astype(int)
                u           = np.zeros((Ab.N, Ab.model.p))
                
                act         = np.zeros(Ab.N).astype(int)
                
                # True state model dynamics
                if random_initial_state:
                    x[k] = np.random.uniform(
                            low  = Ab.partition['R']['low'][i],
                            high = Ab.partition['R']['upp'][i])
                else:
                    x[k] = Ab.partition['R']['center'][i]
                
                # Add current state to trace
                mc['traces'][i][m] += [x[k]]
                
                # For each time step in the finite time horizon
                while k < Ab.N:
                    
                    if Ab.args.verbose:
                        tab.print_row([i, m, k, 'New time step'])
                    
                    # Compute all centers of regions associated with points
                    center_coord = computeRegionCenters(x[k], 
                        Ab.spec.partition).flatten()
                    
                    if tuple(center_coord) in Ab.partition['R']['c_tuple']:
                        # Save that state is currently in region ii
                        x_region[k] = Ab.partition['R']['c_tuple'][tuple(center_coord)]
                        
                        # Retreive the action from the policy
                        act[k] = Ab.results['optimal_policy'][k, x_region[k]]
                    else:
                        x_region[k] = -1
                    
                    # If current region is the goal state ... 
                    if x_region[k] in Ab.partition['goal']:
                        # Then abort the current iteration, as we have achieved the goal
                        mc['goal_reached'][i][m] = True
                        
                        if Ab.args.verbose:
                            tab.print_row([i, m, k, 'Goal state reached'], sort="Success")
                        break
                    # If current region is in critical states...
                    elif x_region[k] in Ab.partition['critical']:
                        # Then abort current iteration
                        if Ab.args.verbose:
                            tab.print_row([i, m, k, 'Critical state reached, so abort'], sort="Warning")
                        break
                    elif x_region[k] == -1:
                        if Ab.args.verbose:
                            tab.print_row([i, m, k, 'Absorbing state reached, so abort'], sort="Warning")
                        break
                    elif act[k] == -1:
                        if Ab.args.verbose:
                            tab.print_row([i, m, k, 'No policy known, so abort'], sort="Warning")
                        break
                    
                    # If loop was not aborted, we have a valid action
                    if Ab.args.verbose:
                        tab.print_row([i, m, k, 'In state: '+str(x_region[k])+' ('+str(x[k])+'), take action: '+str(act[k])])
                
                    # Move predicted mean to the future belief to the target point of the next state
                    x_goal[k+1] = Ab.actions['obj'][act[k]].center

                    # Reconstruct the control input required to achieve this target point
                    # Note that we do not constrain the control input; we already know that a suitable control exists!
                    if Ab.flags['underactuated']:
                        success, x_hat, u[k] = controller.solve(x_goal[k+1], x[k], 
                            Ab.actions['obj'][act[k]].backreach_obj.target_set_size)
                        
                        if not success:
                            print('>> Failed to compute control input <<')
                            assert False
                        
                        # Implement the control into the physical (unobservable) system
                        x_plus = Ab.model.A_true @ x[k] + Ab.model.B_true @ u[k] + Ab.model.Q_flat
                    else:
                        x_nom = x[k]
                    
                        u[k] = np.array(Ab.model.B_pinv @ ( x_goal[k+1] - Ab.model.A @ x_nom.flatten() - Ab.model.Q_flat ))
                    
                        # Implement the control into the physical (unobservable) system
                        x_plus = Ab.model.A @ x[k] + Ab.model.B @ u[k] + Ab.model.Q_flat
                    
                    # Use Gaussian process noise
                    x[k+1] = x_plus + w_array[i_abs, m, k]
                        
                    # Add current state to trace
                    mc['traces'][i][m] += [x[k+1]]
                    
                    # Increase iterator variable by one
                    k += 1
                
                mc['action_traces'][i][m] = act
            
        # Set of monte carlo iterations completed for specific initial state
        
        # Calculate the overall reachability probability for initial state i
        mc['reachability'][i] = \
            np.sum(mc['goal_reached'][i]) / Ab.setup.montecarlo['iterations']
                
    Ab.time['6_MonteCarlo'] = tocDiff(False)
    print('Monte Carlo simulations finished:',Ab.time['6_MonteCarlo'])
    
    if writer:
    
        # Store Monte Carlo results as dataframe
        MCsims_df = pd.Series( 
            mc['reachability'], index=range(Ab.partition['nr_regions']) )
            
        # Write Monte Carlo results to Excel
        MCsims_df.to_excel(writer, sheet_name='Empirical reach.')
    
    return mc