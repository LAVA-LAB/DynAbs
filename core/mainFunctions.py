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

import numpy as np
import itertools
from .commons import floor_decimal
from scipy.spatial import Delaunay

def in_hull(p, hull):
    '''
    Test if points in `p` are in `hull`.

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    '''
    
    if not isinstance(hull,Delaunay):
        print(' -- Creating hull...')
        hull = Delaunay(hull, qhull_options='QJ')

    boolArray = hull.find_simplex(p) >= 0

    return boolArray

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
    region_nrPerDim = partition['nrPerDim']
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

def computeScenarioBounds_sparse(setup, partition, abstr, trans, samples):
    '''
    Compute the transition probability intervals

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    partition : dict
        Dictionary of the partition.
    abstr : dict
        Dictionay containing all information of the finite-state abstraction.
    trans : dict
        Dictionary with all data for the transition probabilities
    samples : 2D Numpy array
        Numpy array, with every row being a sample of the process noise.

    Returns
    -------
    returnDict : dict
        Dictionary with the computed (intervals of) transition probabilities.

    '''
    
    # Number of decision variables always equal to one
    d = 1
    Nsamples = setup.scenarios['samples']
    beta = setup.scenarios['confidence']
    
    # Initialize counts array
    counts = dict()
    
    centers = computeRegionCenters(samples, partition)
    
    for s in range(Nsamples):
        
        key = tuple(centers[s])
        
        if key in abstr['allCenters']:
            idx = abstr['allCenters'][ key ]
            if idx in counts:
                counts[idx] += 1
            else:
                counts[idx] = 1
    
    # Count number of samples not in any region (i.e. in absorbing state)
    k = int( Nsamples - sum(counts.values()) )
    
    deadlock_low = 1 - trans['memory'][k][1]
    if k > Nsamples:
        deadlock_upp = 1
    else:
        deadlock_upp = 1 - trans['memory'][k][0]

    # Initialize vectors for probability bounds
    probability_low = np.zeros(len(counts))
    probability_upp = np.zeros(len(counts))
    probability_approx = np.zeros(len(counts))
    
    interval_idxs = np.zeros(len(counts), dtype=int)
    approx_idxs = np.zeros(len(counts), dtype=int)

    # Enumerate over all the non-zero bins
    for i, (region,count) in enumerate(counts.items()):
        
        k = Nsamples - count
        
        if k > Nsamples:
            probability_low[i] = 0                
        else:
            probability_low[i] = trans['memory'][k][0]
        probability_upp[i] = trans['memory'][k][1]
        
        interval_idxs[i] = int(region)
        
        # Point estimate transition probability (count / total)
        probability_approx[i] = count / Nsamples
        approx_idxs[i] = int(region)
    
    nr_decimals = 5
    
    #### PROBABILITY INTERVALS
    probs_lb = floor_decimal(probability_low, nr_decimals)
    probs_ub = floor_decimal(probability_upp, nr_decimals)
    
    # Create interval strings (only entries for prob > 0)
    interval_strings = ["["+
                      str(floor_decimal(max(1e-4, lb),5))+","+
                      str(floor_decimal(min(1,    ub),5))+"]"
                      for (lb, ub) in zip(probs_lb, probs_ub)]
    
    # Compute deadlock probability intervals
    deadlock_lb = floor_decimal(deadlock_low, nr_decimals)
    deadlock_ub = floor_decimal(deadlock_upp, nr_decimals)
    
    deadlock_string = '['+ \
                       str(floor_decimal(max(1e-4, deadlock_lb),5))+','+ \
                       str(floor_decimal(min(1,    deadlock_ub),5))+']'
    
    #### POINT ESTIMATE PROBABILITIES
    probability_approx = np.round(probability_approx, nr_decimals)
    
    # Create approximate prob. strings (only entries for prob > 0)
    approx_strings = [str(p) for p in probability_approx]
    
    # Compute approximate deadlock transition probabilities
    deadlock_approx = np.round(1-sum(probability_approx), nr_decimals)
    
    returnDict = {
        'interval_strings': interval_strings,
        'interval_idxs': interval_idxs,
        'approx_strings': approx_strings,
        'approx_idxs': approx_idxs,
        'deadlock_interval_string': deadlock_string,
        'deadlock_approx': deadlock_approx,
    }
    
    return returnDict

def definePartitions(dim, nrPerDim, regionWidth, origin, onlyCenter=False):
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
    onlyCenter : Boolean, default=False
        If True, only the center of the regions is computed. 
        If False, the full partition (e.g. vertices) is computed        

    Returns
    -------
    partitions : dict
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
    
    if onlyCenter:        
        partitions = np.array(list(itertools.product(*widthArrays))) + origin
        
    else:
        partitions = dict()
        
        for i,elements in enumerate(itertools.product(*widthArrays)):
            partitions[i] = dict()
            
            center = np.array(elements) + origin
            
            dec = 5
            
            partitions[i]['center'] = np.round(center, decimals=dec)
            partitions[i]['low'] = np.round(center - regionWidth/2, 
                                            decimals=dec)
            partitions[i]['upp'] = np.round(center + regionWidth/2, 
                                            decimals=dec)
    
    return partitions

def makeModelFullyActuated(model, manualDimension='auto', observer=False):
    '''
    Given a model in `model`, render it fully actuated.

    Parameters
    ----------
    model : dict
        Main dictionary of the LTI system model.
    manualDimension : int or str, optional
        Desired dimension of the state of the model The default is 'auto'.
    observer : Boolean, default=False
        If True, it is assumed that the system is not directly observable, so
        an observer is created.

    Returns
    -------
    model : dict
        Main dictionary of the LTI system model, which is now fully actuated.

    '''
    
    if manualDimension == 'auto':
        # Determine dimension for actuation transformation
        dim    = int( np.size(model.A,1) / np.size(model.B,1) )
    else:
        # Group a manual number of time steps
        dim    = int( manualDimension )
    
    # Determine fully actuated system matrices and parameters
    A_hat  = np.linalg.matrix_power(model.A, (dim))
    B_hat  = np.concatenate([ np.linalg.matrix_power(model.A, (dim-i)) \
                                      @ model.B for i in range(1,dim+1) ], 1)
    
    Q_hat  = sum([ np.linalg.matrix_power(model.A, (dim-i)) @ model.Q
                       for i in range(1,dim+1) ])
    
    w_sigma_hat  = sum([ np.array( np.linalg.matrix_power(model.A, (dim-i) )
                                  @ model.noise['w_cov'] @
                                  np.linalg.matrix_power(model.A.T, (dim-i) )
                                ) for i in range(1,dim+1) ])
    
    # Overwrite original system matrices
    model.A               = A_hat
    model.B               = B_hat
    model.Q               = Q_hat
    model.Q_flat          = Q_hat.flatten()
    
    model.noise['w_cov']  = w_sigma_hat
    
    # Redefine sampling time of model
    model.tau             *= (dim+1)
    
    return model