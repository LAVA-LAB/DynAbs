#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from operator import itemgetter
import collections

from .commons import cm2inch, floor_decimal, tocDiff
from .define_partition import computeRegionIdx, computeRegionCenters, draw_hull

def compute_intervals_error(args, partition_setup, partition, trans, 
                                clusters, error, exclude=False, verbose=False):
    '''
    Compute the transition probability intervals

    Parameters
    ----------
    args : obj
        Object with arguments passed to the program
    partition_setup : dict
        Dictionary of the partition.
    partition : dict
        Dictionay containing all information of the partitioning.
    trans : dict
        Dictionary with all data for the transition probabilities
    clusters : dict
        Dictionary of cluster information
    error : dict
        Control/epistemic error dictionary
    
    Returns
    -------
    returnDict : dict
        Dictionary with the computed (intervals of) transition probabilities.

    '''
    
    # tocDiff(False)
    if len(exclude) > 0:
        check_exclude = True
    else:
        check_exclude = False
    
    Nsamples = args.noise_samples
    
    # Initialize counts array
    counts_low = np.zeros(partition_setup['number'])
    counts_upp = np.zeros(partition_setup['number'])
    
    i_excl = {}

    # If hoeffding inequality is used to obtain upper bound probabilities,
    # then we don't modify the uncertainty boxes
    imin, iMin = computeRegionIdx(clusters['lb'] + error['neg'], partition_setup)
    imax, iMax = computeRegionIdx(clusters['ub'] + error['pos'], partition_setup,
                                  borderOutside=[True]*len(clusters['value']))
    
    epsilon = np.sqrt( 1/(2*Nsamples) * np.log(
                2/args.confidence) )
    
    counts_absorb_low = 0
    counts_absorb_upp = 0
    
    counts_goal_low = 0
    counts_goal_upp = 0
    
    counts_critical_low = 0
    counts_critical_upp = 0
    
    nrPerDim = np.array(partition_setup['number'])
    
    ###
    
    # Compute number of samples fully outside of partition.
    # If all indices are outside the partition, then it is certain that 
    # this sample is outside the partition
    fully_out = (imax < 0).any(axis=1) + (imin > nrPerDim - 1).any(axis=1)
    counts_absorb_low = clusters['value'][fully_out].sum()
    
    # Determine samples that are partially outside
    partially_out = (imin < 0).any(axis=1) + (imax > nrPerDim - 1).any(axis=1)
    
    if verbose:
        print('Partially out sum:', partially_out.sum())
    
    counts_absorb_upp = clusters['value'][partially_out].sum().astype(int)
    
    # If we precisely know where the sample is, and it is not outside the
    # partition, then add one to both lower and upper bound count
    in_single_region = (imin == imax).all(axis=1) * np.bitwise_not(fully_out)
    
    for key,val in zip (imin[in_single_region], 
                        clusters['value'][in_single_region]):
        key = tuple(key)
        if key in partition['goal_idx']:
            counts_goal_low += val
            counts_goal_upp += val
        
        elif key in partition['critical_idx']:
            counts_critical_low += val
            counts_critical_upp += val
        
        else:
            counts_low[key] += val
            counts_upp[key] += val
            
    keep = ~in_single_region & ~fully_out
    iMin_rem = iMin[keep]
    iMax_rem = iMax[keep]
    c_rem    = np.arange( len(clusters['value']) )[keep]
            
    ###
    
    # For the remaining samples, only increment the upper bound count
    for x,c in enumerate(c_rem):        
        counts_upp[ tuple(map(slice, iMin_rem[x], iMax_rem[x]+1)) ] += clusters['value'][c]
        
        '''
        if check_exclude:
          for key in itertools.product(*map(np.arange, iMin_rem[x], iMax_rem[x]+1)):
            
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
        '''
                
        index_tuples = set(itertools.product(*map(range, iMin_rem[x], iMax_rem[x]+1)))
        
        # Check if all are goal states
        if index_tuples.issubset( partition['goal_idx'] ) and not partially_out[x]:
            counts_goal_low += clusters['value'][c]
            counts_goal_upp += clusters['value'][c]
            
        # Check if all are critical states
        elif index_tuples.issubset( partition['critical_idx'] ) and not partially_out[x]:
            counts_critical_low += clusters['value'][c]
            counts_critical_upp += clusters['value'][c]
            
        # Otherwise, check if part of them are goal/critical states
        else:
            if not index_tuples.isdisjoint( partition['goal_idx'] ):
                counts_goal_upp += clusters['value'][c]
                
            if not index_tuples.isdisjoint( partition['critical_idx'] ):
                counts_critical_upp += clusters['value'][c]
    
    ###
    
    counts_nonzero = [[partition['R']['idx'][idx], counts_low[idx], cu] 
                        for idx, cu in np.ndenumerate(counts_upp) if cu > 0 
                        and idx not in partition['goal_idx'] 
                        and idx not in partition['critical_idx']]
    
    counts_header = []
    if len(partition['critical_idx']) > 0:
        counts_header += [[-2, counts_critical_low, counts_critical_upp]]
    if len(partition['goal_idx']) > 0:
        counts_header += [[-1, counts_goal_low, counts_goal_upp]]
    
    counts = np.array(counts_header + counts_nonzero, dtype = int)
    
    # Number of samples not in any region (i.e. in absorbing state)
    deadlock_low = np.maximum(0, counts_absorb_low / Nsamples - epsilon)    
    deadlock_upp = 1 - trans['memory'][counts_absorb_upp][0]

    if len(counts) > 0:
        discard_upp = np.minimum(Nsamples - counts[:, 1], Nsamples)
        
        # Compute lower bound probability
        probability_low     = trans['memory'][discard_upp, 0]
        
        # Compute upper bound probability either with hoeffding's bound
        probability_upp     = np.minimum(1, counts[:, 2] / Nsamples + epsilon)
        
        probability_approx  = counts[:, 1:3].mean(axis=1) / Nsamples
        successor_idxs = counts[:,0]
        
    else:
        probability_low     = np.array([])
        probability_upp     = np.array([])
        probability_approx  = np.array([])
        successor_idxs = np.array([])
    
    nr_decimals = 5
    Pmin = 1e-4
    
    #### PROBABILITY INTERVALS
    probs_lb = np.maximum(Pmin, floor_decimal(probability_low, nr_decimals))
    probs_ub = np.minimum(1,    floor_decimal(probability_upp, nr_decimals))
    
    # Create interval strings (only entries for prob > 0)
    interval_strings = ["["+
                      str(lb)+","+
                      str(ub)+"]"
                      for (lb, ub) in zip(probs_lb, probs_ub)]
    
    # Compute deadlock, goal, and critical probability intervals
    deadlock_lb = np.maximum(Pmin, floor_decimal(deadlock_low, nr_decimals))
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



def compute_intervals_default(args, partition_setup, partition, trans, samples, 
                              successor_indices, regions_list = False, nr_decimals = 5,):
    '''
    Compute the transition probability intervals

    Parameters
    ----------
    args : obj
        Object with arguments passed to the program
    partition_setup : dict
        Dictionary of the partition.
    partition : dict
        Dictionay containing all information of the finite-state abstraction.
    trans : dict
        Dictionary with all data for the transition probabilities
    samples : 2D Numpy array
        Numpy array, with every row being a sample of the process noise.
    successor_indices : list
        List of successor state indices (used for improved synthesis scheme)

    Returns
    -------
    returnDict : dict
        Dictionary with the computed (intervals of) transition probabilities.

    '''
    

    Nsamples = args.noise_samples

    if not regions_list:
        # Only keep those samples that are within the partitioned portion of the state space
        samples = samples[np.all(samples >= partition_setup['boundary'][:,0], axis=1) * 
                        np.all(samples <= partition_setup['boundary'][:,1], axis=1)]

        # Relative value to lower boundary
        samples_rel = samples - partition_setup['boundary'][:,0]

        # Determine the indices of the respective samples
        index = (samples_rel // partition_setup['width']).astype(int)
        index_tuples = map(tuple, index)

        regions_list = list(itemgetter(*index_tuples)(partition['R']['idx']))

    #### CONVERT FROM REGION COUNT TO VALUE PARTITION COUNT
    states_list = successor_indices[regions_list]
    count_dict = collections.Counter(states_list)

    # Determine probability intervals
    successor_idxs = np.fromiter(count_dict.keys(), dtype=int)
    counts_value = np.fromiter(count_dict.values(), dtype=int)

    if args.improved_synthesis and all(successor_idxs == 0):
        ignore = True
    else:
        ignore = False

    #### PROBABILITY INTERVALS
    probs_lb = floor_decimal(trans['memory'][Nsamples - counts_value, 0], nr_decimals)
    probs_ub = floor_decimal(trans['memory'][Nsamples - counts_value, 1], nr_decimals)
    
    # Create interval strings (only entries for prob > 0)
    interval_strings = ["["+
                      str(floor_decimal(max(1e-4, lb),5))+","+
                      str(floor_decimal(min(1,    ub),5))+"]"
                      for (lb, ub) in zip(probs_lb, probs_ub)]
    
    # Count number of samples not in any region (i.e. in absorbing state)
    k_deadlock = int( Nsamples - sum(counts_value) )

    # Compute deadlock probability intervals
    deadlock_lb = floor_decimal(1 - trans['memory'][k_deadlock][1], nr_decimals)
    deadlock_ub = floor_decimal(1 - trans['memory'][k_deadlock][0], nr_decimals)
    
    deadlock_string = '['+ \
                       str(floor_decimal(max(1e-4, deadlock_lb),5))+','+ \
                       str(floor_decimal(min(1,    deadlock_ub),5))+']'
    
    #### POINT ESTIMATE PROBABILITIES
    probability_approx = np.round(counts_value / Nsamples, nr_decimals)
    
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
    
    return returnDict, regions_list, ignore



def transition_plot(samples, error, i_show, i_hide, args, setup, model, spec, partition, 
                    cut_value, backreach=False, backreach_inflated=False,
                    stateLabels=False):
    '''
    Create transition plot
    '''
    
    is1, is2 = i_show
    
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
    
    ax.set_title("N = "+str(args.noise_samples),fontsize=10)
    
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
        
    if type(backreach_inflated) == np.ndarray:
        
        draw_hull(backreach_inflated, color='blue')
            
    # Set tight layout
    fig.tight_layout()
    
    # Save figure
    filename = setup.directories['outputFcase']+'transition_plot'
    for form in setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    plt.show()