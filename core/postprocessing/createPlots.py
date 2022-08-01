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

def draw_hull(points, color, linewidth=0.1):

    # Plot hull of the vertices        
    try: 
        hull = ConvexHull(points)
        # plt.plot(points[hull.vertices,0], 
        #          points[hull.vertices,1], str(color)+'-', lw=linewidth)
        # plt.plot([points[hull.vertices[0],0], 
        #           points[hull.vertices[-1],0]],
        #          [points[hull.vertices[0],1], 
        #           points[hull.vertices[-1],1]], str(color)+'-', lw=linewidth)
        
        # Get the indices of the hull points.
        hull_indices = hull.vertices
        
        # These are the actual points.
        hull_pts = points[hull_indices, :]
        
        plt.fill(hull_pts[:,0], hull_pts[:,1], fill=False, edgecolor=color, lw=linewidth)
        
        print('Convex hull plotted')
        
    except:
        plt.plot(points[:,0], points[:,1],
                 color=color, lw=linewidth)
        
        print('Line plotted')

def partition_plot(i_show, i_hide, ScAb, cut_value, act=None, stateLabels=False):
    '''

    Returns
    -------
    None.

    '''
    
    is1, is2 = i_show
    
    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))
    
    plt.xlabel('Var $1$', labelpad=0)
    plt.ylabel('Var $2$', labelpad=0)

    width = ScAb.spec.partition['width']
    number = ScAb.spec.partition['number']
    
    min_xy = ScAb.spec.partition['boundary'][:,0]
    max_xy = ScAb.spec.partition['boundary'][:,1]
    
    # Compute where to show ticks on the axis
    ticks_x = np.round(np.linspace(min_xy[is1], max_xy[is1], number[is1]+1), 4)
    ticks_y = np.round(np.linspace(min_xy[is2], max_xy[is2], number[is2]+1), 4)
    
    show_every = np.round(number / 5)
    ticks_x_labels = [v if i % show_every[is1] == 0 else '' for i,v in enumerate(ticks_x)]
    ticks_y_labels = [v if i % show_every[is2] == 0 else '' for i,v in enumerate(ticks_y)]
    
    # Set ticks and tick labels
    ax.set_xticks(ticks_x)
    ax.set_yticks(ticks_y)
    ax.set_xticklabels(ticks_x_labels)
    ax.set_yticklabels(ticks_y_labels)
    
    # x-axis
    for i, tic in enumerate(ax.xaxis.get_major_ticks()):
        if ticks_x_labels[i] == '':
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
    
    # y-axis
    for i, tic in enumerate(ax.yaxis.get_major_ticks()):
        if ticks_y_labels[i] == '':
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
    
    # Show gridding of the state space
    plt.grid(which='major', color='#CCCCCC', linewidth=0.3)
    
    # Goal x-y limits
    ax.set_xlim(1.5*min_xy[is1], 1.5*max_xy[is1])
    ax.set_ylim(1.5*min_xy[is2], 1.5*max_xy[is2])
    
    # ax.set_xlim(19.1, 22.5)
    # ax.set_ylim(37.48, 37.54)
    
    ax.set_title("Partition plot", fontsize=10)
    
    # Draw goal states
    for goal in ScAb.partition['goal']:
        
        if all(ScAb.partition['R']['center'][goal, list(i_hide)] == cut_value):
        
            goal_lower = ScAb.partition['R']['low'][goal, [is1, is2]]
            goalState = patches.Rectangle(goal_lower, width=width[is1], 
                                  height=width[is2], color="green", 
                                  alpha=0.3, linewidth=None)
            ax.add_patch(goalState)
    
    # Draw critical states
    for crit in ScAb.partition['critical']:
        
        if all(ScAb.partition['R']['center'][crit, list(i_hide)] == cut_value):
        
            critStateLow = ScAb.partition['R']['low'][crit, [is1, is2]]
            criticalState = patches.Rectangle(critStateLow, width=width[is1], 
                                      height=width[is2], color="red", 
                                      alpha=0.3, linewidth=None)
            ax.add_patch(criticalState)
    
    with plt.rc_context({"font.size": 5}):        
        # Draw every X-th label
        if stateLabels:
            skip = 1
            for i in range(0, ScAb.partition['nr_regions'], skip):
                
                if all(ScAb.partition['R']['center'][i, list(i_hide)] == cut_value):
                                
                    ax.text(ScAb.partition['R']['center'][i,is1], 
                            ScAb.partition['R']['center'][i,is2], i, \
                            verticalalignment='center', 
                            horizontalalignment='center' ) 
    
    if not act is None:
        
        plt.scatter(act.center[is1], act.center[is2], c='red', s=20)
        
        print(' - Print backward reachable set of action', act.idx)
        draw_hull(act.backreach, color='red')
        
        if hasattr(act, 'backreach_infl'):
            draw_hull(act.backreach_infl, color='blue')
        
        for s in act.enabled_in:
            center = ScAb.partition['R']['center'][s]
            plt.scatter(center[is1], center[is2], c='blue', s=8)
        
        # for s in actions['enabled_inv'],
    
    # Set tight layout
    fig.tight_layout()
    
    # Save figure
    filename = ScAb.setup.directories['outputF']+'partition_plot'
    for form in ScAb.setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    plt.show()

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

def createProbabilityPlots(setup, N, model, spec, results, partition, mc=None):
    '''
    Create the result plots for the partitionaction instance.

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    plot : dict
        Dictionary containing info about plot settings
    N : int
        Finite time horizon.
    model : dict
        Main dictionary of the LTI system model.
    results : dict
        Dictionary containing all results from solving the MDP.
    partition : dict
        Dictionay containing all information of the partitioning
    mc : dict
        Dictionary containing all data relevant to the Monte Carlo simulations.

    Returns
    -------
    None.

    '''
    
    # Plot 2D probability plot over time
    if N/2 != round(N/2):
        printWarning('WARNING: '+str(N/2)+' is no valid integer index')
        printWarning('Print results for time index k='+
                     str(int(np.floor(N/2)-1))+' instead')
    
    if setup.montecarlo['enabled']:
        fig = plt.figure(figsize=cm2inch(14, 7))
    else:
        fig = plt.figure(figsize=cm2inch(8, 6))
    ax = plt.gca()
    
    # Plot probability reachabilities
    color = next(ax._get_lines.prop_cycler)['color']
    
    plt.plot(results['optimal_reward'], label='k=0', linewidth=1, color=color)
    if setup.montecarlo['enabled'] and not setup.montecarlo['init_states']:
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
    
    ######################
    # Determine dimension of model
    m = spec.partition['number']
    
    # Plot 3D probability plot for selected time steps
    if model.n > 2:
        printWarning('Nr of dimensions > 2, so 3D reachability plot omitted')
    
    else:
        # Plot 3D probability results
        plot3D      = dict()
        
        # Retreive X and Y values of the centers of the regions
        plot3D['x'] = np.zeros(partition['nr_regions'])
        plot3D['y'] = np.zeros(partition['nr_regions'])
        
        for i in range(partition['nr_regions']):
            plot3D['x'][i], plot3D['y'][i] = partition['R']['center'][i]
        
        plot3D['x'] = np.reshape(plot3D['x'], (m[0],m[1]))
        plot3D['y'] = np.reshape(plot3D['y'], (m[0],m[1]))

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
        
def UAV_3D_plotLayout(ScAb):
    '''
    Create a plot that shows the layout of the UAV problem (without results)
    
    Parameters
    ----------
    ScAb : abstraction instance
        Full object of the abstraction being plotted for
        
    Returns
    -------
    None.
        
    '''
    
    cut_value = np.zeros(3)
    for i,d in enumerate(range(1, ScAb.model.n, 2)):
        if ScAb.spec.partition['number'][d]/2 != round( 
                ScAb.spec.partition['number'][d]/2 ):
            cut_value[i] = 0
        else:
            cut_value[i] = ScAb.spec.partition['width'][d] / 2    
    
    UAVplot3d_visvis( ScAb.setup, ScAb.model, ScAb.spec, ScAb.partition, 
                      traces=[], cut_value=cut_value ) 

def UAVplots(ScAb, case_id, writer=None, itersToSim=10000, itersToPlot=1):
    '''
    Create the trajectory plots for the UAV benchmarks

    Parameters
    ----------
    ScAb : abstraction instance
        Full object of the abstraction being plotted for
    case_id : int
        Index for the current abstraction iteration
    writer : XlsxWriter
        Writer object to write results to Excel
    itersToPlot : Int
        Number of traces/trajectories to plot

    Returns
    -------
    performance_df : Pandas DataFrame
        DataFrame containing the empirical performance results

    '''
    
    from core.define_partition import computeRegionCenters
    from core.commons import setStateBlock
    
    # Determine desired state IDs
    if ScAb.model.name == 'UAV':
        if ScAb.model.modelDim == 2:
            x_init = ScAb.spec.x0
            
            cut_value = np.zeros(2)
            for i,d in enumerate(range(1, ScAb.model.n, 2)):
                if ScAb.spec.partition['number'][d]/2 != \
                  round( ScAb.spec.partition['number'][d]/2 ):
                      
                    cut_value[i] = 0
                else:
                    cut_value[i] = ScAb.spec.partition[
                                    'width'][d] / 2                
            
        elif ScAb.model.modelDim == 3:
            x_init = ScAb.spec.x0
            
            cut_value = np.zeros(3)
            for i,d in enumerate(range(1, ScAb.model.n, 2)):
                if ScAb.spec.partition['number'][d]/2 != \
                  round( ScAb.spec.partition['number'][d]/2 ):
                      
                    cut_value[i] = 0
                else:
                    cut_value[i] = ScAb.spec.partition[
                                    'width'][d] / 2         
                    
    elif ScAb.model.name == 'shuttle':
        x_init = setStateBlock(ScAb.spec.partition, 
                               a=[-.75], b=[-.85], c=[0], d=[0])
        
        cut_value = np.array([0.005, 0.005])
            
    # Compute all centers of regions associated with points
    x_init_centers = computeRegionCenters(np.array(x_init), 
                                          ScAb.spec.partition)
    
    # Filter to only keep unique centers
    x_init_unique = np.unique(x_init_centers, axis=0)
    
    state_idxs = [ScAb.partition['R']['c_tuple'][tuple(c)] for c in x_init_unique 
                                   if tuple(c) in ScAb.partition['R']['c_tuple']]
    
    print(' -- Perform simulations for initial states:',state_idxs)
    
    ScAb.setup.montecarlo['init_states'] = state_idxs
    ScAb.setup.montecarlo['iterations'] = itersToSim
    ScAb.mc = monte_carlo(ScAb)
    
    PRISM_reach = ScAb.results['optimal_reward'][state_idxs]
    empirical_reach = ScAb.mc['reachability'][state_idxs]
    
    print('Probabilistic reachability (PRISM): ',PRISM_reach)
    print('Empirical reachability (Monte Carlo):',empirical_reach)
    
    if writer:
        # Create performance dataframe
        performance_df = pd.DataFrame( {
                'PRISM reachability': PRISM_reach.flatten(),
                'Empirical reachability': empirical_reach.flatten() 
            }, index=[case_id] )
        
        # Write to Excel
        performance_df.to_excel(writer, sheet_name='Performance')
    
    traces = []
    for i in state_idxs:
        for j in range(itersToPlot):
            traces += [ScAb.mc['traces'][i][j]]
            
    alltraces = []
    for i in state_idxs:
        for j in range(ScAb.setup.montecarlo['iterations']):
            alltraces += [ScAb.mc['traces'][i][j]]
    
    if ScAb.model.modelDim == 2:
        if ScAb.model.name == 'UAV':
            i_show = (0,2)
            i_hide = (1,3)
            
            UAVplot2D( i_show, i_hide, ScAb.setup, ScAb.model, ScAb.spec, 
                       ScAb.partition, traces, cut_value )
            
        elif ScAb.model.name == 'shuttle':
            i_show = (2,3)
            i_hide = (0,1)
        
            UAVplot2D( i_show, i_hide, ScAb.setup, ScAb.model, ScAb.spec,
                       ScAb.partition, traces, cut_value )
    
            i_show = (0,1)
            i_hide = (2,3)
        
            UAVplot2D( i_show, i_hide, ScAb.setup, ScAb.model, ScAb.spec,
                       ScAb.partition, traces, cut_value )
    
    elif ScAb.model.modelDim == 3:
        if ScAb.setup.main['iterations'] == 1 or \
          ScAb.setup.plotting['3D_UAV']:
        
            # Only plot trajectory plot in non-iterative mode (because it 
            # pauses the script)
            UAVplot3d_visvis( ScAb.setup, ScAb.model, ScAb.spec, ScAb.partition, 
                              traces, cut_value ) 
    
    traces_df = pd.DataFrame(alltraces)
    traces_df.to_pickle(ScAb.setup.directories['outputFcase']+'traces.pickle')
    
    return performance_df
    
def UAVplot2D(i_show, i_hide, setup, model, spec, partition, traces, cut_value, 
              line=False, stateLabels=False):
    '''
    Create 2D trajectory plots for the 2D UAV benchmark

    Parameters
    ----------
    i_show : list of ints.
        List of indices of the state vector to show in plot.
    i_hide : list of ints.
        List of indices of the state vector to hide in plot.
    setup : dict
        Setup dictionary.
    model : dict
        Main dictionary of the LTI system model.
    partition : dict
        Dictionay containing all information of the partitioning.
    traces : list
        Nested list containing the trajectories (traces) to plot for
    cut_value : array
        Values to create the cross-section for
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
    
    is1, is2 = i_show
    ih1, ih2 = i_hide
    
    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))
    
    plt.xlabel('$x$', labelpad=0)
    plt.ylabel('$y$', labelpad=0)

    width = np.array(spec.partition['width'])
    domainMax = width * np.array(spec.partition['number']) / 2
    
    min_xy = spec.partition['origin'] - domainMax
    max_xy = spec.partition['origin'] + domainMax
    
    major_ticks_x = np.arange(min_xy[is1]+1, max_xy[is1]+1, 4*width[is1])
    major_ticks_y = np.arange(min_xy[is2]+1, max_xy[is2]+1, 4*width[is2])
    
    minor_ticks_x = np.arange(min_xy[is1], max_xy[is1]+1, width[is1])
    minor_ticks_y = np.arange(min_xy[is2], max_xy[is2]+1, width[is2])
    
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
    ax.set_xlim(min_xy[is1], max_xy[is1])
    ax.set_ylim(min_xy[is2], max_xy[is2])
    
    ax.set_title("N = "+str(setup.sampling['samples']),fontsize=10)
    
    # Draw goal states
    for goal in partition['goal']:
        
        if all(partition['R']['center'][goal, [ih1,ih2]] == cut_value):
        
            goal_lower = partition['R']['low'][goal, [is1, is2]]
            goalState = Rectangle(goal_lower, width=width[is1], 
                                  height=width[is2], color="green", 
                                  alpha=0.3, linewidth=None)
            ax.add_patch(goalState)
    
    # Draw critical states
    for crit in partition['critical']:
        
        if all(partition['R']['center'][crit, [ih1,ih2]] == cut_value):
        
            critStateLow = partition['R']['low'][crit, [is1, is2]]
            criticalState = Rectangle(critStateLow, width=width[is1], 
                                      height=width[is2], color="red", 
                                      alpha=0.3, linewidth=None)
            ax.add_patch(criticalState)
    
    with plt.rc_context({"font.size": 5}):        
        # Draw every X-th label
        if stateLabels:
            skip = 1
            for i in range(0, partition['nr_regions'], skip):
                
                if all(partition['R']['center'][i, [ih1,ih2]] == cut_value):
                                
                    ax.text(partition['R']['center'][i,is1], 
                            partition['R']['center'][i,is2], i, \
                            verticalalignment='center', 
                            horizontalalignment='center' ) 
            
    # Add traces
    for i,trace in enumerate(traces):
        
        if len(trace) < 3:
            printWarning('Warning: trace '+str(i)+
                         ' has length of '+str(len(trace)))
            continue
        
        # Convert nested list to 2D array
        trace_array = np.array(trace)
        
        # Extract x,y coordinates of trace
        x = trace_array[:, is1]
        y = trace_array[:, is2]
        points = np.array([x,y]).T
        
        # Plot precise points
        plt.plot(*points.T, 'o', markersize=1, color="black");
        
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
    filename = setup.directories['outputFcase']+'drone_trajectory'
    for form in setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    plt.show()
    
def UAVplot3d_visvis(setup, model, spec, partition, traces, cut_value):
    '''
    Create 3D trajectory plots for the 3D UAV benchmark

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
    cut_value : array
        Values to create the cross-section for

    Returns
    -------
    None.

    '''
    
    from scipy.interpolate import interp1d
    import visvis as vv
    
    fig = vv.figure()
    f = vv.clf()
    a = vv.cla()
    fig = vv.gcf()
    ax = vv.gca()
    
    ix = 0
    iy = 2
    iz = 4
    
    regionWidth_xyz = np.array([spec.partition['width'][0], 
                                spec.partition['width'][2], 
                                spec.partition['width'][4]])    
    
    # Draw goal states
    for goal in partition['goal']:
        
        goalCenter = partition['R']['center'][goal]
        if all(goalCenter[[1,3,5]] == cut_value):
            
            goal = vv.solidBox(tuple(goalCenter), 
                               scaling=tuple(regionWidth_xyz))
            goal.faceColor = (0,1,0,0.8)
            
    # Draw critical states
    for crit in partition['critical']:
        
        critCenter = partition['R']['center'][crit]
        if all(critCenter[[1,3,5]] == cut_value):
        
            critical = vv.solidBox(tuple(critCenter), 
                                   scaling=tuple(regionWidth_xyz))
            critical.faceColor = (1,0,0,0.8)
    
    # Add traces
    for i,trace in enumerate(traces):
        
        if len(trace) < 3:
            printWarning('Warning: trace '+str(i)+' has length of '+
                         str(len(trace)))
            continue
        
        # Convert nested list to 2D array
        trace_array = np.array(trace)
        
        # Extract x,y coordinates of trace
        x = trace_array[:, ix]
        y = trace_array[:, iy]
        z = trace_array[:, iz]
        points = np.array([x,y,z]).T
        
        # Plot precise points
        vv.plot(x,y,z, lw=0, mc='b', ms='.')
        
        # Linear length along the line:
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, 
                                              axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
        
        # Interpolation for different methods:
        alpha = np.linspace(0, 1, 75)
        
        interpolator =  interp1d(distance, points, kind='quadratic', axis=0)
        interpolated_points = interpolator(alpha)
        
        xp = interpolated_points[:,0]
        yp = interpolated_points[:,1]
        zp = interpolated_points[:,2]
        
        # Plot trace
        vv.plot(xp,yp,zp, lw=1, lc='b')
    
    ax.axis.xLabel = 'X'
    ax.axis.yLabel = 'Y'
    ax.axis.zLabel = 'Z'
    
    app = vv.use()
    
    f.relativeFontSize = 1.6
    # ax.position.Correct(dh=-5)
    
    vv.axis('tight', axes=ax)
    
    fig.position.w = 700
    fig.position.h = 600
    
    im = vv.getframe(vv.gcf())
    
    ax.SetView({'zoom':0.042, 'elevation':25, 'azimuth':-35})
    
    if 'outputFcase' in setup.directories:
    
        filename = setup.directories['outputFcase'] + \
                    'UAV_paths_screenshot.png'
        
    else:
        
        filename = setup.directories['outputF'] + 'UAV_paths_screenshot.png'
    
    vv.screenshot(filename, sf=3, bg='w', ob=vv.gcf())
    app.Run()
    
def load_traces_manual(ScAb, pathLow, pathHigh, idxLow=0, idxHigh=0):
    '''
    Function to plot 3D UAV benhmark with two distinct cases (as in paper)

    Parameters
    ----------
    ScAb : abstraction instance
        Full object of the abstraction being plotted for
    pathLow : string
        Path to first pickle file describing traces (low noise case).
    pathHigh : string
        Path to second pickle file describing traces (high noise case).
    idxLow : int, optional
        Index of trace for low noise case to be plotted. The default is 0.
    idxHigh : int, optional
        Index of trace for high noise case to be plotted. The default is 0.

    Returns
    -------
    None.

    '''
    
    tracesLow  = pd.read_pickle(pathLow)
    tracesHigh = pd.read_pickle(pathHigh)
    
    traceLow  = list(tracesLow.loc[idxLow].dropna())
    traceHigh  = list(tracesHigh.loc[idxHigh].dropna())
    
    cut_value = np.zeros(3)
    for i,d in enumerate(range(1, ScAb.model.n, 2)):
        if ScAb.spec.partition['number'][d]/2 != round( 
                ScAb.spec.partition['number'][d]/2 ):
            cut_value[i] = 0
        else:
            cut_value[i] = ScAb.spec.partition['width'][d] / 2
    
    UAVplot3d_visvis_manual( ScAb.setup, ScAb.model, ScAb.spec, ScAb.partition, 
                             cut_value, traceLow, traceHigh) 
    
def UAVplot3d_visvis_manual(setup, model, spec, partition, cut_value, traceLow, 
                            traceHigh):
    '''
    Create 3D trajectory plots for the 3D UAV benchmark

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
    cut_value : array
        Values to create the cross-section for

    Returns
    -------
    None.

    '''
    
    from scipy.interpolate import interp1d
    import visvis as vv
    
    fig = vv.figure()
    f = vv.clf()
    a = vv.cla()
    fig = vv.gcf()
    ax = vv.gca()
    
    ix = 0
    iy = 2
    iz = 4
    
    regionWidth_xyz = np.array([spec.partition['width'][0], 
                                spec.partition['width'][2], 
                                spec.partition['width'][4]])    
    
    # Draw goal states
    for goal in partition['goal']:
        
        goalCenter = partition['R']['center'][goal]
        if all(goalCenter[[1,3,5]] == cut_value):
            
            goal = vv.solidBox(tuple(goalCenter), 
                               scaling=tuple(regionWidth_xyz))
            goal.faceColor = (0,0.8,0,0.8)
            
    # Draw critical states
    for crit in partition['critical']:
        
        critCenter = partition['R']['center'][crit]
        if all(critCenter[[1,3,5]] == cut_value):
        
            critical = vv.solidBox(tuple(critCenter), 
                                   scaling=tuple(regionWidth_xyz))
            critical.faceColor = (0.8,0,0,0.8)
    
    # Add traces
    for i,trace in enumerate([traceLow]):
        
        if len(trace) < 3:
            printWarning('Warning: trace '+str(i)+' has length of '+
                         str(len(trace)))
            continue
        
        # Convert nested list to 2D array
        trace_array = np.array(trace)
        
        print(trace_array)
        
        # Extract x,y coordinates of trace
        x = trace_array[:, ix]
        y = trace_array[:, iy]
        z = trace_array[:, iz]
        points = np.array([x,y,z]).T
        
        # Plot precise points
        vv.plot(x,y,z, lw=0, mc=(102/255, 178/255, 255/255), ms='.', mw=12)
        
        # Linear length along the line:
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, 
                                              axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
        
        # Interpolation for different methods:
        alpha = np.linspace(0, 1, 75)
        
        interpolator =  interp1d(distance, points, kind='quadratic', axis=0)
        interpolated_points = interpolator(alpha)
        
        xp = interpolated_points[:,0]
        yp = interpolated_points[:,1]
        zp = interpolated_points[:,2]
        
        # Plot trace
        vv.plot(xp,yp,zp, lw=5, lc=(102/255, 178/255, 255/255))
        
    # Add traces
    for i,trace in enumerate([traceHigh]):
        
        if len(trace) < 3:
            printWarning('Warning: trace '+str(i)+' has length of '+
                         str(len(trace)))
            continue
        
        # Convert nested list to 2D array
        trace_array = np.array(trace)
        
        # Extract x,y coordinates of trace
        x = trace_array[:, ix]
        y = trace_array[:, iy]
        z = trace_array[:, iz]
        points = np.array([x,y,z]).T
        
        # Plot precise points
        vv.plot(x,y,z, lw=0, mc=(204/255, 153/255, 255/255), ms='x', mw=12)
        
        # Linear length along the line:
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, 
                                              axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
        
        # Interpolation for different methods:
        alpha = np.linspace(0, 1, 75)
        
        interpolator =  interp1d(distance, points, kind='quadratic', axis=0)
        interpolated_points = interpolator(alpha)
        
        xp = interpolated_points[:,0]
        yp = interpolated_points[:,1]
        zp = interpolated_points[:,2]
        
        # Plot trace
        vv.plot(xp,yp,zp, lw=5, lc=(204/255, 153/255, 255/255))
    
    ax.axis.xLabel = 'X'
    ax.axis.yLabel = 'Y'
    ax.axis.zLabel = 'Z'
    
    app = vv.use()
    
    f.relativeFontSize = 1.6
    # ax.position.Correct(dh=-5)
    vv.axis('tight', axes=ax)
    
    fig.position.w = 1000
    fig.position.h = 750
    
    im = vv.getframe(vv.gcf())
    
    #ax.SetView({'zoom':0.03, 'elevation':55, 'azimuth':-20})
    ax.SetView({'zoom':0.03, 'elevation':70, 'azimuth':20})
    
    ax.legend = "Low noise", "Low noise", "High noise", "High noise"
    
    if 'outputFcase' in setup.directories:
    
        filename = setup.directories['outputFcase']+'UAV_paths_screenshot.png'
        
    else:
        
        filename = setup.directories['outputF'] + 'UAV_paths_screenshot.png'
    
    vv.screenshot(filename, sf=3, bg='w', ob=vv.gcf())
    app.Run()
    
def reachabilityHeatMap(ScAb, montecarlo = False, title = 'auto'):
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

    if ScAb.model.n == 2:
        
        x_nr = ScAb.spec.partition['number'][0]
        y_nr = ScAb.spec.partition['number'][1]
        
        cut_centers = definePartitions(ScAb.model.n, [x_nr, y_nr], 
               ScAb.spec.partition['width'], 
               ScAb.spec.partition['origin'], onlyCenter=True)['center']

    if ScAb.model.name == 'building_2room':
    
        x_nr = ScAb.spec.partition['number'][0]
        y_nr = ScAb.spec.partition['number'][1]
        
        cut_centers = definePartitions(ScAb.model.n, [x_nr, y_nr, 1, 1], 
               ScAb.spec.partition['width'], 
               ScAb.spec.partition['origin'], onlyCenter=True)['center']
        
    if ScAb.model.name == 'anaesthesia_delivery':
    
        x_nr = ScAb.spec.partition['number'][0]
        y_nr = ScAb.spec.partition['number'][1]
        
        orig = np.concatenate((ScAb.spec.partition['origin'][0:2],
                              [9.25]))
        
        cut_centers = definePartitions(ScAb.model.n, [x_nr, y_nr, 1], 
               ScAb.spec.partition['width'], 
               orig, onlyCenter=True)['center']
        
        print(cut_centers)
        
    elif ScAb.model.n == 4:
        
        x_nr = ScAb.spec.partition['number'][0]
        y_nr = ScAb.spec.partition['number'][2]
        
        cut_centers = definePartitions(ScAb.model.n, [x_nr, 1, y_nr, 1], 
               ScAb.spec.partition['width'], 
               ScAb.spec.partition['origin'], onlyCenter=True)['center']
                          
    cut_values = np.zeros((x_nr, y_nr))
    cut_coords = np.zeros((x_nr, y_nr, ScAb.model.n))
    
    cut_idxs = [ScAb.partition['R']['c_tuple'][tuple(c)] for c in cut_centers 
                                   if tuple(c) in ScAb.partition['R']['c_tuple']]              
    
    for i,(idx,center) in enumerate(zip(cut_idxs, cut_centers)):
        
        j = i % y_nr
        k = i // y_nr
        
        if montecarlo:
            cut_values[k,j] = ScAb.mc['reachability'][idx] 
        else:
            cut_values[k,j] = ScAb.results['optimal_reward'][idx]
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