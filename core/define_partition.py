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
    originShift = np.array(partition['origin'] )
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
        