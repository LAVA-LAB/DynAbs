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
import matplotlib.pyplot as plt # Import Pyplot to generate plots
import seaborn as sns
from matplotlib import cm

from core.commons import printWarning, mat_to_vec, cm2inch
from core.define_partition import define_partition

def set_axes_equal(ax: plt.Axes):
    """
    Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)
    


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])



def reachability_plot(setup, results, mc=None):
    '''
    Create the result plots for the partitionaction instance.

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    results : dict
        Dictionary containing all results from solving the MDP.
    mc : dict
        Dictionary containing all data relevant to the Monte Carlo simulations.

    Returns
    -------
    None.

    '''
    
    if mc != None:
        fig = plt.figure(figsize=cm2inch(14, 7))
    else:
        fig = plt.figure(figsize=cm2inch(8, 6))
    ax = plt.gca()
    
    # Plot probability reachabilities
    color = next(ax._get_lines.prop_cycler)['color']
    
    plt.plot(results['optimal_reward'], label='k=0', linewidth=1, color=color)
    if mc != None and not setup.montecarlo['init_states']:
        plt.plot(mc['reachability'], label='Monte carlo (k=0)', \
                 linewidth=1, color=color, linestyle='dashed')
                
    # Styling plot
    plt.xlabel('States')
    plt.ylabel('Reachability probability')
    plt.legend(loc="upper left")
    
    # Set tight layout
    fig.tight_layout()
                
    # Save figure
    filename = setup.directories['outputFcase']+'reachability_probability'
    for form in setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
    
    plt.show()
    


def heatmap_3D_view(model, setup, spec, region_centers, results):

    ######################
    # Determine dimension of model
    m = spec.partition['number']
    
    if model.n > 2:
        printWarning('Nr of dimensions > 2, so 3D reachability plot omitted')
    
    else:
        # Plot 3D probability results
        plot3D      = dict()
        
        plot3D['x'] = np.reshape(region_centers[:,0], (m[0],m[1]))
        plot3D['y'] = np.reshape(region_centers[:,1], (m[0],m[1]))

        # Create figure
        fig = plt.figure(figsize=cm2inch(8,5.33))
        ax  = plt.axes(projection='3d')

        # Determine matrix of probability values
        Z   = np.reshape(results['optimal_reward'], (m[0],m[1]))
        
        # Plot the surface
        surf = ax.plot_surface(plot3D['x'], plot3D['y'], Z, 
                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        # Customize the z axis
        ax.set_zlim(0,1)
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # Set title and axis format
        ax.title.set_text('Reachability probability at time k = 0')
        x_row = mat_to_vec( plot3D['x'] )
        y_col = mat_to_vec( plot3D['y'] )
        n_ticks = 5
        plt.xticks(np.arange(min(x_row), max(x_row)+1, \
                             (max(x_row) - min(x_row))/n_ticks ))
        plt.yticks(np.arange(min(y_col), max(y_col)+1, \
                             (max(y_col) - min(y_col))/n_ticks ))
        plt.tick_params(pad=-3)
            
        plt.xlabel('x_1', labelpad=-6)
        plt.ylabel('x_2', labelpad=-6)
        
        # Set tight layout
        fig.tight_layout()
        
        # Save figure
        filename = setup.directories['outputFcase']+\
            '3d_reachability_k=0'
        
        for form in setup.plotting['exportFormats']:
            plt.savefig(filename+'.'+str(form), format=form, 
                        bbox_inches='tight')
        
        plt.show()
        
        
    
def heatmap_2D(args, model, setup, c_tuple, spec, values, title = 'auto'):
    '''
    Create heat map for the reachability probability from any initial state.

    Parameters
    ----------
    Ab : abstraction instance
        Full object of the abstraction being plotted for

    Returns
    -------
    None.

    '''

    if model.n == 2:
        
        x_nr = spec.partition['number'][0]
        y_nr = spec.partition['number'][1]
        
        cut_centers = define_partition(model.n, [x_nr, y_nr], 
               spec.partition['width'], 
               spec.partition['origin'])['center']

    if model.name == 'building_2room':
    
        x_nr = spec.partition['number'][0]
        y_nr = spec.partition['number'][1]
        
        cut_centers = define_partition(model.n, [x_nr, y_nr, 1, 1], 
               spec.partition['width'], 
               spec.partition['origin'])['center']
        
    if model.name == 'anaesthesia_delivery':
    
        x_nr = spec.partition['number'][0]
        y_nr = spec.partition['number'][1]
        
        orig = np.concatenate((spec.partition['origin'][0:2],
                              [9.25]))
        
        cut_centers = define_partition(model.n, [x_nr, y_nr, 1], 
               spec.partition['width'], 
               orig)['center']
        
    elif model.n == 4:
        
        x_nr = spec.partition['number'][0]
        y_nr = spec.partition['number'][2]
        
        cut_centers = define_partition(model.n, [x_nr, 1, y_nr, 1], 
               spec.partition['width'], 
               spec.partition['origin'])['center']
                          
    cut_values = np.zeros((x_nr, y_nr))
    cut_coords = np.zeros((x_nr, y_nr, model.n))
    
    cut_idxs = [c_tuple[tuple(c)] for c in cut_centers 
                if tuple(c) in c_tuple]              
    
    for i,(idx,center) in enumerate(zip(cut_idxs, cut_centers)):
        
        j = i % y_nr
        k = i // y_nr
        
        cut_values[k,j] = values[idx] 
        cut_coords[k,j,:] = center
    
    cut_df = pd.DataFrame( cut_values, index=cut_coords[:,0,0], 
                           columns=cut_coords[0,:,1] )
    
    fig = plt.figure(figsize=cm2inch(9, 8))
    ax = sns.heatmap(cut_df.T, cmap="jet", #YlGnBu
             vmin=0, vmax=1)
    ax.figure.axes[-1].yaxis.label.set_size(20)
    ax.invert_yaxis()
    
    ax.set_xlabel('Var 1', fontsize=15)
    ax.set_ylabel('Var 2', fontsize=15)
    if title == 'auto':
        ax.set_title("N = "+str(args.noise_samples), fontsize=20)
    else:
        ax.set_title(str(title), fontsize=20)
    
    # Set tight layout
    fig.tight_layout()

    # Save figure
    filename = setup.directories['outputFcase']+'2D_Heatmap_N=' + \
                str(args.noise_samples)
    for form in setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    plt.show()