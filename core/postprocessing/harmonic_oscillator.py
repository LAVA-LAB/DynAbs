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
import pandas as pd             # Import Pandas to store data in frames
import matplotlib.pyplot as plt # Import Pyplot to generate plots

# Load main classes and methods
from matplotlib import cm
from scipy.spatial import ConvexHull
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

from ..commons import printWarning, mat_to_vec, cm2inch
from core.monte_carlo import monte_carlo

def oscillator_heatmap(ScAb, title = 'auto'):
    '''
    Create heat map for the reachability probability from any initial state.

    Parameters
    ----------
    ScAb : abstraction instance
        Full object of the abstraction being plotted for

    Returns
    -------
    None.

    '''
    
    import seaborn as sns
    from ..define_partition import definePartitions

    x_nr = ScAb.spec.partition['number'][0]
    y_nr = ScAb.spec.partition['number'][1]
        
    cut_centers = definePartitions(ScAb.model.n, [x_nr, y_nr], 
           ScAb.spec.partition['width'], 
           ScAb.spec.partition['origin'], onlyCenter=True)['center']
                          
    cut_values = np.zeros((x_nr, y_nr))
    cut_coords = np.zeros((x_nr, y_nr, ScAb.model.n))
    
    cut_idxs = [ScAb.partition['R']['c_tuple'][tuple(c)] for c in cut_centers 
                                   if tuple(c) in ScAb.partition['R']['c_tuple']]              
    
    for i,(idx,center) in enumerate(zip(cut_idxs, cut_centers)):
        
        j = i % y_nr
        k = i // y_nr
        
        difference = ScAb.mc['reachability'][idx] - ScAb.results['optimal_reward'][idx]
        
        if difference >= 0:
            # Guarantees safe (model checking >= empirical)
            cut_values[k,j] = 1
        else:
            # Guarantees unsafe (model checking < empirical)
            cut_values[k,j] = 0
        cut_coords[k,j,:] = center
    
    plot_dataframe = pd.DataFrame( cut_values, index=cut_coords[:,0,0], 
                           columns=cut_coords[0,:,1] )
    
    # Compute the fraction of states for which the empirical reachability
    # guarantees (i.e. simulated performance) are lower than the guarantees
    # obtained from the iMDP model checking
    average_value = plot_dataframe.mean().mean()
    
    fig = plt.figure(figsize=cm2inch(9, 8))
    ax = sns.heatmap(plot_dataframe.T, cmap="jet", #YlGnBu
             vmin=0, vmax=1)
    ax.figure.axes[-1].yaxis.label.set_size(20)
    ax.invert_yaxis()
    
    ax.set_xlabel('Var 1', fontsize=15)
    ax.set_ylabel('Var 2', fontsize=15)
    if title == 'auto':
        ax.set_title("N = "+str(ScAb.setup.sampling['samples']), fontsize=20)
    else:
        ax.set_title(str(title), fontsize=20)
    
    # Set tight layout
    fig.tight_layout()

    # Save figure
    filename = ScAb.setup.directories['outputFcase']+'safeset_N=' + \
                str(ScAb.setup.sampling['samples'])
    for form in ScAb.setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    plt.show()
    
    return average_value

def oscillator_traces(ScAb, traces, action_traces, plot_trace_ids=None,
              line=False, stateLabels=False, title = 'auto'):
    '''
    Create 2D trajectory plots for the harmonic oscillator benchmark

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    model : dict
        Main dictionary of the LTI system model.
    partition : dict
        Dictionay containing all information of the partitioning.
    traces : list
        Nested list containing the trajectories (traces) to plot for
    line : Boolean, optional
        If true, also plot line that connects points of traces. 
        The default is False.
    stateLabels : Boolean, optional
        If true, plot IDs of the regions as well. The default is False.

    Returns
    -------
    None.

    '''
    
    from scipy.interpolate import interp1d
    
    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))
    
    plt.xlabel('$x$', labelpad=0)
    plt.ylabel('$y$', labelpad=0)

    width = np.array(ScAb.spec.partition['width'])
    domainMax = width * np.array(ScAb.spec.partition['number']) / 2
    
    min_xy = ScAb.spec.partition['origin'] - domainMax
    max_xy = ScAb.spec.partition['origin'] + domainMax
    
    major_ticks_x = np.arange(min_xy[0]+1, max_xy[0]+1, 4*width[0])
    major_ticks_y = np.arange(min_xy[1]+1, max_xy[1]+1, 4*width[1])
    
    minor_ticks_x = np.arange(min_xy[0], max_xy[0]+1, width[0])
    minor_ticks_y = np.arange(min_xy[1], max_xy[1]+1, width[1])
    
    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(minor_ticks_y, minor=True)
    
    for axi in (ax.xaxis, ax.yaxis):
        for tic in axi.get_minor_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
    
    plt.grid(which='minor', color='#CCCCCC', linewidth=0.3)
    
    # Goal x-y limits
    ax.set_xlim(min_xy[0] - 2, max_xy[0] + 2)
    ax.set_ylim(min_xy[1] - 2, max_xy[1] + 2)
    
    
    if title == 'auto':
        ax.set_title("N = "+str(ScAb.setup.sampling['samples']),fontsize=10)
    else:
        ax.set_title(str(title),fontsize=10)
    
    # Draw goal states
    for goal in ScAb.partition['goal']:
        
        goal_lower = ScAb.partition['R']['low'][goal, [0, 1]]
        goalState = Rectangle(goal_lower, width=width[0], 
                              height=width[1], color="green", 
                              alpha=0.3, linewidth=None)
        ax.add_patch(goalState)
    
    # Draw critical states
    for crit in ScAb.partition['critical']:
        
        critStateLow = ScAb.partition['R']['low'][crit, [0, 1]]
        criticalState = Rectangle(critStateLow, width=width[0], 
                                  height=width[1], color="red", 
                                  alpha=0.3, linewidth=None)
        ax.add_patch(criticalState)

    with plt.rc_context({"font.size": 5}):        
        # Draw every X-th label
        if stateLabels:
            skip = 1
            for i in range(0, ScAb.partition['nr_regions'], skip):
                                        
                ax.text(ScAb.partition['R']['center'][i,0], 
                        ScAb.partition['R']['center'][i,1], i, \
                        verticalalignment='center', 
                        horizontalalignment='center' ) 
        
    # Add traces
    for i,trace in traces.items():
        
        if not plot_trace_ids is None and i not in plot_trace_ids:
            continue
        
        if len(trace) < 2:
            printWarning('Warning: trace '+str(i)+
                         ' has length of '+str(len(trace)))
            continue
        
        # Convert nested list to 2D array
        trace_array = np.array(trace)
        
        # Extract x,y coordinates of trace
        x = trace_array[:, 0]
        y = trace_array[:, 1]
        points = np.array([x,y]).T
        
        # Plot precise points
        plt.plot(*points.T, 'o', markersize=1, color="black");
        
        action_centers = np.array([ScAb.actions['obj'][a].center 
                                   for a in action_traces[i][:len(trace)-1]] )
        action_errors  = np.array([ScAb.actions['obj'][a].error 
                                   for a in action_traces[i][:len(trace)-1]] )           
        plt.plot(*action_centers.T, 'o', markersize=1, color="red");
        
        for center, error in zip(action_centers, action_errors):
            
            low = center + error['neg']
            diff = error['pos'] - error['neg']
            
            rect = patches.Rectangle(low, diff[0], diff[1], 
                                     linewidth=0.5, edgecolor='red', 
                                     linestyle='dashed', facecolor='none')
            
            # Add the patch to the Axes
            ax.add_patch(rect)
        
        if line:
        
            # Linear length along the line:
            distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, 
                                                  axis=1 )) )
            distance = np.insert(distance, 0, 0)/distance[-1]
            
            # Interpolation for different methods:
            alpha = np.linspace(0, 1, 75)
            
            interpolator =  interp1d(distance, points, kind='quadratic', 
                                     axis=0)
            interpolated_points = interpolator(alpha)
            
            # Plot trace
            plt.plot(*interpolated_points.T, '-', color="blue", linewidth=1);
    
    # Set tight layout
    fig.tight_layout()
    
    # Save figure
    filename = ScAb.setup.directories['outputFcase']+'drone_trajectory'
    for form in ScAb.setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    plt.show()
    
class oscillator_experiment(object):
    
    def __init__(self, f_min, f_max, f_step, monte_carlo_iterations):
        '''
        Initialize object for collecting and exporting the fraction of safe
        controllers (as reported in the paper)

        Returns
        -------
        None.

        '''

        self.fraction_safe = []        

        self.f_min = f_min
        self.f_max = f_max
        self.f_step = f_step
        
        self.monte_carlo_iterations = monte_carlo_iterations
        
    def add_iteration(self, ScAb, case_id):
        '''
        Perform Monte Carlo simulations and add the resulting fraction of
        states for which the resulting controller is safe to the list        

        Parameters
        ----------
        ScAb : Abstraction object
        case_id : Index (int) of the case being run

        Returns
        -------
        None.

        '''
        
        ScAb.setup.set_monte_carlo(iterations=self.monte_carlo_iterations)

        f_list = np.arange(self.f_min, self.f_max+0.01, self.f_step)
        frac_sr = pd.Series(index=np.round(f_list, 3), dtype=float, name=case_id)

        for f in f_list:
            f = np.round(f, 3)
            spring = np.round(ScAb.model.spring_nom * (1-f) + ScAb.model.spring_max * f, 3)
            mass   = np.round(ScAb.model.mass_nom * (1-f) + ScAb.model.mass_min * f, 3)

            ScAb.model.set_true_model(mass=mass, spring=spring)
            ScAb.mc = monte_carlo(ScAb, random_initial_state=True)
            
            # reachabilityHeatMap(ScAb, montecarlo = True, title='Monte Carlo, mass='+str(mass)+'; spring='+str(spring))
            frac_sr[f] = oscillator_heatmap(ScAb, title='Monte Carlo, mass='+str(mass)+'; spring='+str(spring))

        self.fraction_safe += [frac_sr]
        
    def export(self, ScAb):
        
        fraction_safe_df = pd.concat(self.fraction_safe, axis=1)
        fraction_safe_df.index.name = 'x'
        df = fraction_safe_df.T.describe().T

        fraction_safe_df.to_csv(ScAb.setup.directories['outputF']+
                                'fraction_safe_parametric='+
                                str(ScAb.flags['parametric'])+'.csv', sep=';',
                                encoding='utf-8-sig')

        df.to_csv(ScAb.setup.directories['outputF']+
                                'fraction_safe_parametric='+
                                str(ScAb.flags['parametric'])+'_stats.csv', sep=';',
                                encoding='utf-8-sig')
        
    def plot_trace(self, ScAb, spring, state_center, number_to_plot=1):
        '''
        Plot a simulated trace

        Parameters
        ----------
        ScAb : Abstraction object
        spring : Actual/true spring coefficient
        state_center : Tuple of center of the state to plot
        number_to_plot : Number of traces to plot

        Returns
        -------
        None.

        '''
        
        ScAb.setup.set_monte_carlo(iterations=self.monte_carlo_iterations)

        df = pd.DataFrame(columns=['guaranteed', 'simulated', 'ratio'], dtype=float)
        df.index.name = 'mass'

        # eval_state = ScAb.partition['R']['c_tuple'][(-9.5,0.5)]
        eval_state = ScAb.partition['R']['c_tuple'][state_center]

        f_list = np.arange(self.f_min, self.f_max+0.01, self.f_step)

        for f in f_list:
            f = np.round(f, 3)
            spring = np.round(ScAb.model.spring_nom * (1-f) + ScAb.model.spring_max * f, 3)
            mass   = np.round(ScAb.model.mass_nom * (1-f) + ScAb.model.mass_min * f, 3)

            mass = np.round(mass, 3)
            spring = np.round(spring, 3)

            ScAb.model.set_true_model(mass=mass, spring=spring)
            ScAb.mc = monte_carlo(ScAb, random_initial_state=True, init_states = [eval_state])
            
            df.loc[mass, 'guaranteed'] = np.round(ScAb.results['optimal_reward'][eval_state], 4)
            df.loc[mass, 'simulated']  = np.round(ScAb.mc['reachability'][eval_state], 4)
            df.loc[mass, 'ratio']  = np.round(ScAb.mc['reachability'][eval_state] / ScAb.results['optimal_reward'][eval_state], 4)
            
            oscillator_traces(ScAb, ScAb.mc['traces'][eval_state], ScAb.mc['action_traces'][eval_state], plot_trace_ids=[0,1,2,3,4,5,6,7,8,9], title='Traces for mass='+str(mass)+'; spring='+str(spring))
          
        csv_file = ScAb.setup.directories['outputF']+'gauranteed_vs_simulated.csv'
        df.to_csv(csv_file, sep=',')