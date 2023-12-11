#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np              # Import Numpy for computations
import itertools                # Import to crate iterators
import matplotlib.pyplot as plt # Import Pyplot to generate plots
import matplotlib.patches as patches

from scipy.spatial import ConvexHull
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

from .commons import cm2inch

def computeRegionCenters(points, partition):
    '''
    Function to compute to which region (center) a list of points belong

    Parameters
    ----------
    points : 2D Numpy array
        Array, with every row being a point to determine the center point for.
    partition : dict
        Dictionary of the partition.

    Returns
    -------
    2D Numpy array
        Array, with every row being the center coordinate of that row of the
        input array.

    '''
    
    # Check if 'points' is a vector or matrix
    if len(np.shape(points)) == 1:
        points = np.reshape(points, (1,len(points)))
    
    # Retreive partition parameters
    region_width = np.array(partition['width'])
    region_nrPerDim = partition['number']
    dim = len(region_width)
    
    # Boolean list per dimension if it has a region with the origin as center
    originCentered = [True if nr % 2 != 0 else False for nr in region_nrPerDim]

    # Initialize centers array
    centers = np.zeros(np.shape(points)) 
    
    # Shift the points to account for a non-zero origin
    originShift = np.array(partition['origin'])
    pointsShift = points - originShift
    
    for q in range(dim):
        # Compute the center coordinates of every shifted point
        if originCentered[q]:
            
            centers[:,q] = ((pointsShift[:,q]+0.5*region_width[q]) //
                             region_width[q]) * region_width[q]
        
        else:
            
            centers[:,q] = (pointsShift[:,q] // region_width[q]) * \
                            region_width[q] + 0.5*region_width[q]
    
    # Add the origin again to obtain the absolute center coordinates
    return np.round(centers + originShift, decimals=5)



def state2region(state, partition, c_tuple):

    region_centers = computeRegionCenters(state, partition)

    try:
        region_idx = [c_tuple[tuple(c)] for c in region_centers]
        return region_idx
    except:
        print('ERROR: state',state,'does not belong to any region')
        return False



def computeRegionIdx(points, partition, borderOutside=False):
    '''
    Function to compute the indices of the regions that a list of points belong

    Parameters
    ----------
    points : 2D Numpy array
        Array, with every row being a point to determine the center point for.
    partition : dict
        Dictionary of the partition.

    Returns
    -------
    2D Numpy array
        Array, with every row being the indices.

    '''
    
    # Shift the points to account for a non-zero origin
    pointsZero = points - partition['origin'] + \
                    partition['width']*partition['number']/2

    indices = (pointsZero // partition['width']).astype(int)
    
    # Reduce index by one if it is exactly on the border
    indices -= ((pointsZero % partition['width'] == 0).T * borderOutside).T
    
    indices_nonneg = np.minimum(np.maximum(0, indices), 
                                np.array(partition['number'])-1).astype(int)
    
    return indices, indices_nonneg



def define_partition(dim, nrPerDim, regionWidth, origin):
    '''
    Define the partitions object `partitions` based on given settings.

    Parameters
    ----------
    dim : int
        Dimension of the state (of the LTI system).
    nrPerDim : list
        List of integers, where each value is the number of regions in that 
        dimension.
    regionWidth : list
        Width of the regions in every dimension.
    origin : list
        Coordinates of the origin of the continuous state space. 

    Returns
    -------
    partition : dict
        Dictionary containing the info regarding partisions.

    '''
    
    regionWidth = np.array(regionWidth)
    origin      = np.array(origin)
    
    elemVector   = dict()
    for i in range(dim):
        
        elemVector[i] = np.linspace(-(nrPerDim[i]-1)/2, 
                                     (nrPerDim[i]-1)/2,
                                     int(nrPerDim[i]))
        
    widthArrays = [[x*regionWidth[i] for x in elemVector[i]] 
                                              for i in range(dim)]
    
    idxArrays = [range(len(arr)) for arr in widthArrays]
    
    
    nr_regions = np.prod(nrPerDim)
    partition = {'center': np.zeros((nr_regions, dim), dtype=float), 
                 'idx': {},
                 'c_tuple': {}}
    
    partition['low'] = np.zeros((nr_regions, dim), dtype=float)
    partition['upp'] = np.zeros((nr_regions, dim), dtype=float)
    
    for i,(pos,idx) in enumerate(zip(itertools.product(*widthArrays),
                                     itertools.product(*idxArrays))):
        
        center = np.array(pos) + origin
        
        dec = 5
        
        partition['center'][i] = np.round(center, decimals=dec)
        partition['c_tuple'][tuple(np.round(center, decimals=dec))] = i
        partition['low'][i] = np.round(center - regionWidth/2, 
                                        decimals=dec)
        partition['upp'][i] = np.round(center + regionWidth/2, 
                                        decimals=dec)
        partition['idx'][tuple(idx)] = i
    
    return partition



def define_spec_region(allCenters, sets, partition, borderOutside=False):
    '''
    Return the indices of regions associated with the unique centers.

    Parameters
    ----------
    allCenters : List
        List of the center coordinates for all regions.
    partition : Dict
        Partition dictionary.
    subset : List
        List of points to return the unique centers for.

    Returns
    -------
    list
        List of unique center points.

    '''
    
    delta = 1e-5
    
    if sets is None:
        return [], [], set()
    
    else:
    
        points = [None] * len(sets)
        slices = {'min': [None] * len(sets), 'max': [None] * len(sets)}
        index_tuples = set()
        
        # Convert regions to all individual points (centers of regions)
        for i,set_boundary in enumerate(sets):
        
            set_boundary = np.array([
                            S if type(S) != str else partition['boundary'][j] 
                            for j,S in enumerate(set_boundary) ])
            
            # Increase by small margin to avoid issues on region boundaries
            set_boundary = np.hstack((set_boundary[:,[0]] + delta, 
                                      set_boundary[:,[1]] - delta))
            
            vertices = np.array(list(itertools.product(*set_boundary)))
            
            _, indices_nonneg = computeRegionIdx(vertices, partition, borderOutside)
            
            slices['min'][i] = indices_nonneg.min(axis=0)
            slices['max'][i] = indices_nonneg.max(axis=0)
            
            # Create slices
            index_tuples.update(set(itertools.product(*map(range, slices['min'][i], slices['max'][i]+1))))
            
            # Define iterator
            if borderOutside:
                M = map(np.arange, set_boundary[:,0]+delta, set_boundary[:,1]-delta, partition['width']/2)
            else:
                M = map(np.arange, set_boundary[:,0], set_boundary[:,1], partition['width']/2)
            
            # Retreive list of points and convert to array
            points[i] = np.array(list(itertools.product(*M)))
            
        points = np.concatenate(points)
        
        # Compute all centers of regions associated with points
        centers = computeRegionCenters(points, partition)
        
        # Filter to only keep unique centers
        centers_unique = np.unique(centers, axis=0)
        
        states = [allCenters[tuple(c)] for c in centers_unique 
                           if tuple(c) in allCenters]
        
        if len(states) != len(index_tuples):
            print('ERROR: lengths of goal and goal_idx lists are not the same.')
            assert False
        
        # Return the ID's of regions associated with the unique centers            
        return states, slices, index_tuples
        


def partition_plot(i_show, i_hide, Ab, cut_value, act=None, stateLabels=False):
    '''
    Create partition plot
    '''
    
    is1, is2 = i_show
    
    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))
    
    plt.xlabel('Var $1$', labelpad=0)
    plt.ylabel('Var $2$', labelpad=0)

    width = Ab.spec.partition['width']
    number = Ab.spec.partition['number']
    origin = Ab.spec.partition['origin']
    
    min_xy = Ab.spec.partition['boundary'][:,0]
    max_xy = Ab.spec.partition['boundary'][:,1]
    
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
    min_xy_scaled = 1.5 * (min_xy - origin) + origin
    max_xy_scaled = 1.5 * (max_xy - origin) + origin
    
    ax.set_xlim(min_xy_scaled[is1], max_xy_scaled[is1])
    ax.set_ylim(min_xy_scaled[is2], max_xy_scaled[is2])
    
    ax.set_title("Partition plot", fontsize=10)
    
    # Draw goal states
    for goal in Ab.partition['goal']:
        
        if all(Ab.partition['R']['center'][goal, list(i_hide)] == cut_value):
        
            goal_lower = Ab.partition['R']['low'][goal, [is1, is2]]
            goalState = patches.Rectangle(goal_lower, width=width[is1], 
                                  height=width[is2], color="green", 
                                  alpha=0.3, linewidth=None)
            ax.add_patch(goalState)
    
    # Draw critical states
    for crit in Ab.partition['critical']:
        
        if all(Ab.partition['R']['center'][crit, list(i_hide)] == cut_value):
        
            critStateLow = Ab.partition['R']['low'][crit, [is1, is2]]
            criticalState = patches.Rectangle(critStateLow, width=width[is1], 
                                      height=width[is2], color="red", 
                                      alpha=0.3, linewidth=None)
            ax.add_patch(criticalState)
    
    with plt.rc_context({"font.size": 5}):        
        # Draw every X-th label
        if stateLabels:
            skip = 1
            for i in range(0, Ab.partition['nr_regions'], skip):
                
                if all(Ab.partition['R']['center'][i, list(i_hide)] == cut_value):
                                
                    ax.text(Ab.partition['R']['center'][i,is1], 
                            Ab.partition['R']['center'][i,is2], i, \
                            verticalalignment='center', 
                            horizontalalignment='center' ) 
    
    if not act is None:
        
        plt.scatter(act.center[is1], act.center[is2], c='red', s=20)
        
        print(' - Print backward reachable set of action', act.idx)
        draw_hull(act.backreach, color='red')
        
        if hasattr(act, 'backreach_infl'):
            draw_hull(act.backreach_infl, color='blue')
        
        for s in act.enabled_in:
            center = Ab.partition['R']['center'][s]
            plt.scatter(center[is1], center[is2], c='blue', s=8)
    
    # Set tight layout
    fig.tight_layout()
    
    # Save figure
    filename = Ab.setup.directories['outputF']+'partition_plot'
    for form in Ab.setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    plt.show(block = False)



def draw_hull(points, color, linewidth=0.1):

    # Plot hull of the vertices        
    try: 
        hull = ConvexHull(points)
        
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