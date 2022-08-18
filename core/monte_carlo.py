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
from progressbar import progressbar # Import to create progress bars

from .define_partition import computeRegionCenters
from .commons import tocDiff, table
from .cvx_opt import Controller

def monte_carlo(ScAb, iterations='auto', init_states='auto', 
               init_times='auto', printDetails=False, writer=False,
               random_initial_state=False):
    '''
    Perform Monte Carlo simulations to validate the obtained results

    Parameters
    ----------
    iterations : str or int, optional
        Number of Monte Carlo iterations. The default is 'auto'.
    init_states : str or list, optional
        Initial states to start simulations from. The default is 'auto'.
    init_times : str or list, optional
        Initial time steps to start sims. from. The default is 'auto'.

    Returns
    -------
    None.

    '''
    
    if ScAb.flags['underactuated']:
        controller = Controller(ScAb.model)
    
    tocDiff(False)
    if ScAb.args.verbose:
        print(' -- Starting Monte Carlo simulations...')
    
    mc = {'goal_reached': {}, 'traces': {}, 'action_traces': {}}
    
    if iterations != 'auto':
        ScAb.setup.montecarlo['iterations'] = int(iterations)
        
    if init_states != 'auto':
        ScAb.setup.montecarlo['init_states'] = list(init_states)
    else:
        ScAb.setup.montecarlo['init_states'] = False
        
    mc['reachability'] = \
        np.zeros(ScAb.partition['nr_regions'])
    
    # Column widths for tabular prints
    if ScAb.args.verbose:
        col_width = [6,8,6,6,46]
        tab = table(col_width)

        print(' -- Computing required Gaussian random variables...')
    
    if ScAb.setup.montecarlo['init_states'] == False:
        init_state_idxs = np.arange(ScAb.partition['nr_regions'])
        
    else:
        init_state_idxs = ScAb.setup.montecarlo['init_states']
    
    # The gaussian random variables are precomputed to speed up the code
    w_array = np.random.multivariate_normal(
        np.zeros(ScAb.model.n), 
        ScAb.model.noise['w_cov'],
       ( len(init_state_idxs), 
         ScAb.setup.montecarlo['iterations'], ScAb.N ))

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
        if ScAb.setup.montecarlo['init_states'] is not False and \
                    i not in ScAb.setup.montecarlo['init_states']:
            print('Warning: this should not happen!')
            # If current state is not in the list of initial states,
            # continue to the next iteration
            continue
        # Otherwise, continue with the Monte Carlo simulation
        
        if ScAb.args.verbose:
            tab.print_row(['STATE','ITER','K','STATUS'], 
                          head=True)
        
        # Create dictionaries for results related to partition i
        mc['goal_reached'][i] = np.full(
            ScAb.setup.montecarlo['iterations'], False, dtype=bool)
        mc['traces'][i]  = dict()    
        mc['action_traces'][i] = dict()
        
        # For each of the monte carlo iterations
        if ScAb.args.verbose or len(init_state_idxs) > 1:
            loop = range(ScAb.setup.montecarlo['iterations'])
        else:
            loop = progressbar(
                    range(ScAb.setup.montecarlo['iterations']), 
                    redirect_stdout=True)
            
        for m in loop:
            
            # Set initial time
            k = 0
            
            mc['traces'][i][m] = []
            mc['action_traces'][i][m] = []
            
            # Retreive the initial action time-grouping to be chosen
            # (given by the optimal policy to the MDP)
            action = ScAb.results['optimal_policy'][k, i]
            
            if i in ScAb.partition['goal']:
                # If initial state is already the goal state, succes
                # Then, abort the iteration, as we reached the goal
                mc['goal_reached'][i][m] = True
                
                if printDetails:
                    tab.print_row([i, m, k, 
                       'Initial state is goal state'], sort="Success")
            
            elif action == -1:
                # If action=-1, no policy known, and reachability is 0
                if ScAb.args.verbose:
                    tab.print_row([i, m, k, 
                       'No initial policy known, so abort'], 
                       sort="Warning")
            
            else:
                if ScAb.args.verbose:
                    tab.print_row([i, m, k, 
                       'Start Monte Carlo iteration'])
                                    
                # Initialize the current simulation
                x           = np.zeros((ScAb.N+1, ScAb.model.n))
                x_goal      = np.zeros((ScAb.N+1, ScAb.model.n))
                x_region    = np.zeros(ScAb.N).astype(int)
                u           = np.zeros((ScAb.N, ScAb.model.p))
                
                act         = np.zeros(ScAb.N).astype(int)
                
                # True state model dynamics
                if random_initial_state:
                    x[k] = np.random.uniform(
                            low  = ScAb.partition['R']['low'][i],
                            high = ScAb.partition['R']['upp'][i])
                else:
                    x[k] = ScAb.partition['R']['center'][i]
                
                # Add current state to trace
                mc['traces'][i][m] += [x[k]]
                
                # For each time step in the finite time horizon
                while k < ScAb.N:
                    
                    if ScAb.args.verbose:
                        tab.print_row([i, m, k, 'New time step'])
                    
                    # Compute all centers of regions associated with points
                    center_coord = computeRegionCenters(x[k], 
                        ScAb.spec.partition).flatten()
                    
                    if tuple(center_coord) in ScAb.partition['R']['c_tuple']:
                        # Save that state is currently in region ii
                        x_region[k] = ScAb.partition['R']['c_tuple'][tuple(center_coord)]
                        
                        # Retreive the action from the policy
                        act[k] = ScAb.results['optimal_policy'][k, x_region[k]]
                    else:
                        x_region[k] = -1
                    
                    # If current region is the goal state ... 
                    if x_region[k] in ScAb.partition['goal']:
                        # Then abort the current iteration, as we have achieved the goal
                        mc['goal_reached'][i][m] = True
                        
                        if ScAb.args.verbose:
                            tab.print_row([i, m, k, 'Goal state reached'], sort="Success")
                        break
                    # If current region is in critical states...
                    elif x_region[k] in ScAb.partition['critical']:
                        # Then abort current iteration
                        if ScAb.args.verbose:
                            tab.print_row([i, m, k, 'Critical state reached, so abort'], sort="Warning")
                        break
                    elif x_region[k] == -1:
                        if ScAb.args.verbose:
                            tab.print_row([i, m, k, 'Absorbing state reached, so abort'], sort="Warning")
                        break
                    elif act[k] == -1:
                        if ScAb.args.verbose:
                            tab.print_row([i, m, k, 'No policy known, so abort'], sort="Warning")
                        break
                    
                    # If loop was not aborted, we have a valid action
                    if ScAb.args.verbose:
                        tab.print_row([i, m, k, 'In state: '+str(x_region[k])+' ('+str(x[k])+'), take action: '+str(act[k])])
                
                    # Move predicted mean to the future belief to the target point of the next state
                    x_goal[k+1] = ScAb.actions['obj'][act[k]].center

                    # Reconstruct the control input required to achieve this target point
                    # Note that we do not constrain the control input; we already know that a suitable control exists!
                    if ScAb.flags['underactuated']:
                        success, x_hat, u[k] = controller.solve(x_goal[k+1], x[k], 
                            ScAb.actions['obj'][act[k]].backreach_obj.max_control_error)
                        
                        if not success:
                            print('>> Failed to compute control input <<')
                            assert False
                        
                        # Implement the control into the physical (unobservable) system
                        x_plus = ScAb.model.A_true @ x[k] + ScAb.model.B_true @ u[k] + ScAb.model.Q_flat
                    else:
                        x_nom = x[k]
                    
                        u[k] = np.array(ScAb.model.B_pinv @ ( x_goal[k+1] - ScAb.model.A @ x_nom.flatten() - ScAb.model.Q_flat ))
                    
                        # Implement the control into the physical (unobservable) system
                        x_plus = ScAb.model.A @ x[k] + ScAb.model.B @ u[k] + ScAb.model.Q_flat
                    
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
            np.sum(mc['goal_reached'][i]) / ScAb.setup.montecarlo['iterations']
                
    ScAb.time['6_MonteCarlo'] = tocDiff(False)
    print('Monte Carlo simulations finished:',ScAb.time['6_MonteCarlo'])
    
    if writer:
    
        # Store Monte Carlo results as dataframe
        MCsims_df = pd.Series( 
            mc['reachability'], index=range(ScAb.partition['nr_regions']) )
            
        # Write Monte Carlo results to Excel
        MCsims_df.to_excel(writer, sheet_name='Empirical reach.')
    
    return mc