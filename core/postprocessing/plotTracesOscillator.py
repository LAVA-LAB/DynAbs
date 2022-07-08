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

def oscillator_traces(ScAb, traces, action_traces, plot_trace_ids=None,
              line=False, stateLabels=False):
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
    ax.set_xlim(min_xy[0], max_xy[0])
    ax.set_ylim(min_xy[1], max_xy[1])
    
    ax.set_title("N = "+str(ScAb.setup.scenarios['samples']),fontsize=10)
    
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