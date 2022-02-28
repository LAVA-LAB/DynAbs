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
from .define_partition import computeRegionCenters, computeRegionIdx

def computeScenarioBounds_sparse(setup, partition_setup, partition, trans, samples):
    '''
    Compute the transition probability intervals

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    partition_setup : dict
        Dictionary of the partition.
    partition : dict
        Dictionay containing all information of the partitioning.
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
    
    centers = computeRegionCenters(samples, partition_setup)
    
    for s in range(Nsamples):
        
        key = tuple(centers[s])
        
        if key in partition['R']['c_tuple']:
            idx = partition['R']['c_tuple'][ key ]
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
    
    successor_idxs = np.zeros(len(counts), dtype=int)

    # Enumerate over all the non-zero bins
    for i, (region,count) in enumerate(counts.items()):
        
        k = Nsamples - count
        
        if k > Nsamples:
            probability_low[i] = 0                
        else:
            probability_low[i] = trans['memory'][k][0]
        probability_upp[i] = trans['memory'][k][1]
        
        successor_idxs[i] = int(region)
        
        # Point estimate transition probability (count / total)
        probability_approx[i] = count / Nsamples
    
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
        'successor_idxs': successor_idxs,
        'approx_strings': approx_strings,
        'deadlock_interval_string': deadlock_string,
        'deadlock_approx': deadlock_approx,
    }
    
    return returnDict

def computeScenarioBounds_uncertain(setup, partition_setup, partition, trans, samples,
                                    brs, model):
    '''
    Compute the transition probability intervals

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    partition_setup : dict
        Dictionary of the partition.
    partition : dict
        Dictionay containing all information of the partitioning.
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
    
    offset = np.concatenate([((A - model.A) @ brs.T).T for A in model.A_set])
    offset_min = np.min(offset, axis=0)
    offset_max = np.max(offset, axis=0)
    
    # print('OFFSET:',offset_min,offset_max)
    
    # Initialize counts array
    counts_low = np.zeros(partition_setup['nrPerDim'])
    counts_upp = np.zeros(partition_setup['nrPerDim'])
    
    width       = partition_setup['width']
    nr_per_dim  = partition_setup['nrPerDim']
    origin      = partition_setup['origin']
    
    imin, iMin = computeRegionIdx(samples + offset_min, width, nr_per_dim, origin)
    imax, iMax = computeRegionIdx(samples + offset_max, width, nr_per_dim, origin,
                                  borderOutside=[True]*len(samples))
    
    counts_absorb_low = 0
    counts_absorb_upp = 0
    
    for s in range(Nsamples):
        
        # If all indices are outside the partition, then it is certain that 
        # this sample is outside the partition
        if any([imax[s][d] < 0 or imin[s][d] > partition_setup['nrPerDim'][d] for d in range(model.n)]):
            # print('Fully outside; imin:',imin[s],'imax:',imax[s])
            counts_absorb_low += 1
            counts_absorb_upp += 1
            
        # If not fully outside, we now that at least some samples are within 
        # the partition
        else:
            
            # Create slice and add one to the upper bound
            slices = tuple(map(slice, iMin[s], iMax[s]+1))
            counts_upp[slices] += 1
            
            # If we precisely know where the sample is, add one to the lower
            # bound of the count as well
            if all(imin[s] == imax[s]):
                counts_low[slices] += 1
                
            # If the sample is partially outside the partition, then also add
            # to the upper bound count of the absorbing region
            elif any([imin[s][d] < 0 or imax[s][d] > partition_setup['nrPerDim'][d] for d in range(model.n)]):
                
                # print('Partially outside; imin:',imin[s],'imax:',imax[s])
                counts_absorb_upp += 1
            
    # Number of samples not in any region (i.e. in absorbing state)
    deadlock_low = 1 - trans['memory'][counts_absorb_low][1]
    if counts_absorb_upp > Nsamples:
        deadlock_upp = 1
    else:
        deadlock_upp = 1 - trans['memory'][counts_absorb_upp][0]

    counts = {}

    # Convert from numpy array of counts to a sparse dictionary
    for i, (idx, cu) in enumerate(np.ndenumerate(counts_upp)):
        
        if cu > 0:
            state = partition['R']['idx'][idx]
            counts[state] = (counts_low[idx], cu)

    # Initialize vectors for probability bounds
    probability_low = np.zeros(len(counts))
    probability_upp = np.zeros(len(counts))
    probability_approx = np.zeros(len(counts))
    
    successor_idxs = np.zeros(len(counts), dtype=int)

    # Enumerate over all the non-zero bins
    for i, (region,count) in enumerate(counts.items()):
        
        k_upp = Nsamples - count[0]
        k_low = Nsamples - count[1]
        
        if k_upp > Nsamples:
            probability_low[i] = 0                
        else:
            probability_low[i] = trans['memory'][k_upp][0]
        probability_upp[i] = trans['memory'][k_low][1]
        
        successor_idxs[i] = int(region)
        
        # Point estimate transition probability (count / total)
        probability_approx[i] = np.mean(count) / Nsamples
    
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
        'successor_idxs': successor_idxs,
        'approx_strings': approx_strings,
        'deadlock_interval_string': deadlock_string,
        'deadlock_approx': deadlock_approx,
    }
    
    return returnDict

def computeScenarioBounds_error(setup, partition_setup, partition, trans, samples, error, exclude):
    '''
    Compute the transition probability intervals

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    partition_setup : dict
        Dictionary of the partition.
    partition : dict
        Dictionay containing all information of the partitioning.
    trans : dict
        Dictionary with all data for the transition probabilities
    samples : 2D Numpy array
        Numpy array, with every row being a sample of the process noise.

    Returns
    -------
    returnDict : dict
        Dictionary with the computed (intervals of) transition probabilities.

    '''
    
    # tocDiff(False)
    
    # Number of decision variables always equal to one
    Nsamples = setup.scenarios['samples']
    Nrange   = np.arange(Nsamples)
    
    # Initialize counts array
    counts_low = np.zeros(partition_setup['nrPerDim'])
    counts_upp = np.zeros(partition_setup['nrPerDim'])
    
    i_excl = {}

    imin, iMin = computeRegionIdx(samples + error['neg'], partition_setup)
    imax, iMax = computeRegionIdx(samples + error['pos'], partition_setup,
                                  borderOutside=[True]*len(samples))
    
    counts_absorb_low = 0
    counts_absorb_upp = 0
    
    nrPerDim = np.array(partition_setup['nrPerDim'])
    
    # Compute number of samples fully outside of partition.
    # If all indices are outside the partition, then it is certain that 
    # this sample is outside the partition
    fully_out = (imax < 0).any(axis=1) + (imin > nrPerDim - 1).any(axis=1)
    counts_absorb_low = fully_out.sum()
    
    # Determine samples that are partially outside
    partially_out = (imin < 0).any(axis=1) + (imax > nrPerDim - 1).any(axis=1)
    counts_absorb_upp = partially_out.sum()
    
    # If we precisely know where the sample is, and it is not outside the
    # partition, then add one to both lower and upper bound count
    in_single_region = (imin == imax).all(axis=1) * np.bitwise_not(fully_out)
    
    for c, key in zip(Nrange[in_single_region], imin[in_single_region]):
        counts_low[tuple(key)] += 1
        counts_upp[tuple(key)] += 1
            
    # For the remaining samples, only increment the upper bound count
    for c,(i,j) in enumerate(zip(iMin, iMax)):
        if not in_single_region[c] and not fully_out[c]:
            counts_upp[ tuple(map(slice, i, j+1)) ] += 1
            
            for key in itertools.product(*map(np.arange, i, j+1)):
                
                # If first occurence of this region, this just add it
                if key not in i_excl:
                    i_excl[key] = {c}
                    
                # If not first occurence of region
                else:
                    # Check if there is a conflicting sample in there, then 
                    # skip this region (ONE TIME!)
                    union = i_excl[key] & exclude[c]
                    
                    if len(union) == 0:
                        i_excl[key].add(c)
                        
                    else:
                        counts_upp[key] -= 1
                        i_excl[key].pop()

    # Convert from numpy array of counts to a sparse array
    counts = np.array([[partition['R']['idx'][idx], counts_low[idx], cu] 
                       for idx, cu in np.ndenumerate(counts_upp) if cu > 0], 
                      dtype = int)

    print(counts)

    # Number of samples not in any region (i.e. in absorbing state)
    deadlock_low = 1 - trans['memory'][counts_absorb_low][1]
    if counts_absorb_upp > Nsamples:
        deadlock_upp = 1
    else:
        deadlock_upp = 1 - trans['memory'][counts_absorb_upp][0]

    if len(counts) > 0:
        k_upp = np.minimum(Nsamples - counts[:, 1], Nsamples)
        k_low = Nsamples - counts[:, 2]
        
        probability_low     = trans['memory'][k_upp, 0]
        probability_upp     = trans['memory'][k_low, 1]
        probability_approx  = counts[:, 1:3].mean(axis=1) / Nsamples
        successor_idxs = counts[:,0]
        
    else:

        probability_low     = np.array([])
        probability_upp     = np.array([])
        probability_approx  = np.array([])
        successor_idxs = np.array([])
    
    nr_decimals = 5
    
    #### PROBABILITY INTERVALS
    probs_lb = np.maximum(1e-4, floor_decimal(probability_low, nr_decimals))
    probs_ub = np.minimum(1,    floor_decimal(probability_upp, nr_decimals))
    
    # Create interval strings (only entries for prob > 0)
    interval_strings = ["["+
                      str(lb)+","+
                      str(ub)+"]"
                      for (lb, ub) in zip(probs_lb, probs_ub)]
    
    # Compute deadlock probability intervals
    deadlock_lb = np.maximum(1e-4, floor_decimal(deadlock_low, nr_decimals))
    deadlock_ub = np.minimum(1,    floor_decimal(deadlock_upp, nr_decimals))
    
    deadlock_string = '['+ \
                       str(deadlock_lb)+','+ \
                       str(deadlock_ub)+']'
    
    #### POINT ESTIMATE PROBABILITIES
    probability_approx = np.round(probability_approx, nr_decimals)
    
    # Create approximate prob. strings (only entries for prob > 0)
    approx_strings = [str(p) for p in probability_approx]
    
    # Compute approximate deadlock transition probabilities
    deadlock_approx = np.round(1-sum(probability_approx), nr_decimals)
    
    returnDict = {
        'interval_strings': interval_strings,
        'successor_idxs': successor_idxs,
        'approx_strings': approx_strings,
        'deadlock_interval_string': deadlock_string,
        'deadlock_approx': deadlock_approx,
    }
    
    return returnDict



import matplotlib.pyplot as plt # Import Pyplot to generate plots

# Load main classes and methods
from scipy.spatial import ConvexHull
import matplotlib.patches as patches

from .commons import cm2inch

def plot_transition(samples, error, i_show, i_hide, setup, model, partition, 
                    cut_value, backreach=False, stateLabels=False):
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
    
    is1, is2 = i_show
    
    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))
    
    plt.xlabel('$x$', labelpad=0)
    plt.ylabel('$y$', labelpad=0)

    width = np.array(model.setup['partition']['width'])
    domainMax = width * np.array(model.setup['partition']['nrPerDim']) / 2
    
    min_xy = model.setup['partition']['origin'] - domainMax
    max_xy = model.setup['partition']['origin'] + domainMax
    
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
    
    ax.set_title("N = "+str(setup.scenarios['samples']),fontsize=10)
    
    # Draw goal states
    for goal in partition['goal']:
        
        if all(partition['R']['center'][goal, list(i_hide)] == cut_value):
        
            goal_lower = partition['R']['low'][goal, [is1, is2]]
            goalState = patches.Rectangle(goal_lower, width=width[is1], 
                                  height=width[is2], color="green", 
                                  alpha=0.3, linewidth=None)
            ax.add_patch(goalState)
    
    # Draw critical states
    for crit in partition['critical']:
        
        if all(partition['R']['center'][crit, list(i_hide)] == cut_value):
        
            critStateLow = partition['R']['low'][crit, [is1, is2]]
            criticalState = patches.Rectangle(critStateLow, width=width[is1], 
                                      height=width[is2], color="red", 
                                      alpha=0.3, linewidth=None)
            ax.add_patch(criticalState)
    
    with plt.rc_context({"font.size": 5}):        
        # Draw every X-th label
        if stateLabels:
            skip = 1
            for i in range(0, partition['nr_regions'], skip):
                
                if all(partition['R']['center'][i, list(i_hide)] == cut_value):
                                
                    ax.text(partition['R']['center'][i,is1], 
                            partition['R']['center'][i,is2], i, \
                            verticalalignment='center', 
                            horizontalalignment='center' ) 
    
    for sample in samples:
        
        sample_min = sample + error['neg']
        diff       = error['pos'] - error['neg']
        
        rect = patches.Rectangle(sample_min, diff[0], diff[1], 
                                 linewidth=0.5, edgecolor='red', 
                                 linestyle='dashed', facecolor='none')
        
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    if type(backreach) == np.ndarray:
        
        plt.plot(backreach[:,0], backreach[:,1],
                 color='blue')
    
    # Set tight layout
    fig.tight_layout()
    
    # Save figure
    filename = setup.directories['outputFcase']+'drone_trajectory'
    for form in setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    plt.show()