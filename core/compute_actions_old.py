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
from progressbar import progressbar # Import to create progress bars
import itertools                # Import to crate iterators
from scipy.spatial import Delaunay # Import to create convex hulls

import cvxpy as cp

from .define_partition import computeRegionIdx
from .commons import in_hull, overapprox_box
from .postprocessing.createPlots import createPartitionPlot
from .cvx_opt import abstraction_error

def defInvArea(model):
    '''
    Compute the predecessor set (without the shift due to the target
    point). This acccounts to computing, for all u_k, the set
    A^-1 (B u_k - q_k)

    Parameters
    ----------
    model : dict

    Returns
    -------
    x_inv_area : 2D Numpy array
        Predecessor set (every row is a vertex).

    '''
    
    # Determine the set of extremal control inputs
    u = [[model.uMin[i], model.uMax[i]] for i in 
          range(model.p)]
    
    # Determine the inverse image of the extreme control inputs
    x_inv_area = np.zeros((2**model.p, model.n))
    for i,elem in enumerate(itertools.product(*u)):
        list_elem = list(elem)
        
        # Calculate inverse image of the current extreme control input
        x_inv_area[i,:] = model.A_inv @ \
            (model.B @ np.array(list_elem).T + 
             model.Q_flat)  

    return x_inv_area

def defInvHull(x_inv_area):
    '''
    Define the convex hull object of the predecessor set        

    Parameters
    ----------
    x_inv_area : 2D Numpy array
        Predecessor set (every row is a vertex).

    Returns
    -------
    x_inv_hull : hull
        Convex hull object.

    '''
    
    x_inv_hull = Delaunay(x_inv_area, qhull_options='QJ')
    
    return x_inv_hull

def defBasisVectors(model, verbose=False):
    '''
    Compute the basis vectors of the predecessor set, computed from the
    average control inputs to the maximum in every dimension of the
    control space.
    Note that the drift does not play a role here.  

    Parameters
    ----------
    model : dict

    Returns
    -------
    basis_vectors : 2D Numpy array
        Numpy array of basis vectors (every row is a vector).

    '''
    
    u_avg = np.array(model.uMax + model.uMin)/2    

    # Compute basis vectors (with average of all controls as origin)
    u = np.tile(u_avg, (model.n,1)) + np.diag(model.uMax - u_avg)
    
    origin = model.A_inv @ (model.B @ np.array(u_avg).T)   
            
    basis_vectors = np.zeros((model.n, model.n))
    
    for i,elem in enumerate(u):
        
        # Calculate inverse image of the current extreme control input
        point = model.A_inv @ (model.B @ elem.T)    
        
        basis_vectors[i,:] = point - origin
    
        if verbose:
            print(' ---- Length of basis',i,':',
              np.linalg.norm(basis_vectors[i,:]))
    
    return basis_vectors

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [int(x) for x in seq if not (x in seen or seen_add(x))]

def def_backward_reach(model):
    '''
    Compute the backward reachable set for the given model (assuming target 
    point is zero)

    Parameters
    ----------
    model : dictionary
        Dictionary of the current linear dynamical model.

    Returns
    -------
    backreach : 2d array
        Backward reachable set, with every row being a point.

    '''
    
    # Compute matrix of all possible control inputs
    control_mat = [[model.uMin[i], model.uMax[i]] for i in 
          range(model.p)]
    
    u_all = np.array(list(itertools.product(*control_mat)))
    
    # Compute backward reachable set (inverse dynamics under all extreme
    # control inputs)
    # The transpose needed to do computation for all control inputs at once
    inner = -model.B @ u_all.T - model.Q
    backreach = (model.A_inv @ inner).T
        
    return backreach

def defEnabledActions_UA(partition, model, prop, verbose=False):
    
    # Compute the backward reachable set (not accounting for target point yet)
    G_zero = def_backward_reach(model)
    
    action_range = f7(np.concatenate(( partition['goal'],
                       np.arange(partition['nr_actions']) )))
    
    # Number of vertices in backward reachable set
    v = len(G_zero)
    
    # Define optimization problem (with region as a parameter)
    region_min = cp.Parameter(model.n)
    region_max = cp.Parameter(model.n)
    G_curr = cp.Parameter(G_zero.shape)
    
    x = cp.Variable(model.n)
    alpha = cp.Variable(v, nonneg=True)
    
    # Point x is a convex combination of the backward reachable set vertices,
    # and is within the specified region
    constraints = [sum(alpha) == 1,
                   x == cp.sum([G_curr[i] * alpha[i] for i in range(v)]),
                   x >= region_min,
                   x <= region_max]
    
    obj = cp.Minimize(0)
    prob = cp.Problem(obj, constraints)
    
    actions         = [set() for i in range(partition['nr_regions'])]
    actions_inv     = [set() for i in range(partition['nr_actions'])]
    control_error   = [{'pos': np.zeros(model.n), 'neg': np.zeros(model.n)} 
                       for i in range(partition['nr_actions'])]
    
    actions_enabled = np.full(partition['nr_actions'], False)
    
    backreach = {}
    
    # Initialize object to compute abstraction errors
    abstr_error = abstraction_error(model, no_verts = 2**model.n)
    
    # For every action
    for a in progressbar(action_range, redirect_stdout=True):
        
        backreach[a] = G_zero + model.A_inv @ partition['targets'][a]
        
        # Shift the backward reachable set by the target point
        G_curr.value = backreach[a]
        
        # Overapproximate the backward reachable set
        G_curr_box = overapprox_box(backreach[a])
        
        # Determine set of regions that potentially intersect with G
        _, idx_edge = computeRegionIdx(G_curr_box, prop.partition,
                                       borderOutside=[False,True])
        
        # Transpose because we want the min/max combinations per dimension
        # (not per point)
        idx_mat = [np.arange(low, high+1) for low,high in idx_edge.T]
        
        if verbose:
            print('Evaluate action',a,'(try',
              len(list(itertools.product(*idx_mat))),'potential predecessors)')
        
        for tup in itertools.product(*idx_mat):
            
            s = partition['R']['idx'][tup]
            
            # Skip if this is a critical state
            if s in partition['critical']:
                continue
            
            region_min.value = partition['R']['low'][s]
            region_max.value = partition['R']['upp'][s]
            
            prob.solve(warm_start=True, solver='ECOS')
            
            if prob.status != "infeasible":
                # If problem not infeasible, then action enabled in this state
                # But we do another step to limit the abstraction error
                
                # Compute abstraction error
                flag, p, err_pos, err_neg = abstr_error.solve(partition['allCorners'][s], 
                                                         backreach[a])
                
                if not flag:
                    
                    actions[s].add(a)
                    actions_inv[a].add(s)
                    actions_enabled[a] = True
                    
                    control_error[a]['pos'] = np.maximum(err_pos, control_error[a]['pos'])
                    control_error[a]['neg'] = np.minimum(err_neg, control_error[a]['neg'])
                    
                # else:
                #     print('Do not enable action',a,'in state',s,'because the error is too large:',err_pos,err_neg)
                
    return sum(actions_enabled), actions, actions_inv, control_error, backreach

def defEnabledActions(setup, partition, model, A_idx=None, verbose=False):
    '''
    Define dictionaries to sture points in the preimage of a state, and
    the corresponding polytope points.

    Parameters
    ----------
    model : dict
    
    Returns
    -------
    total_actions_enabled : int
        Total number of enabled actions.
    enabled_actions : list
        Nested list of actions enabled in every region.
    enabledActions_inv : list
        Nested list of inverse actions enabled in every region.
        
    '''
    
    from .commons import angle_between
    
    # Compute inverse reachability area
    x_inv_area = defInvArea(model)
    
    total_actions_enabled = 0
    
    enabled_polypoints = dict()
    
    enabled_in_states = [[] for i in range(partition['nr_actions'])]
    enabled_actions   = [set() for i in range(partition['nr_regions'])]
    
    nr_corners = 2**model.n
    
    printEvery = min(100, max(1, int(partition['nr_actions']/10)))
    
    # Check if dimension of control area equals that if the state vector
    dimEqual = model.p == model.n
    
    if dimEqual:
        
        print(' -- Computing inverse basis vectors...')
        # Use preferred method: map back the skewed image to squares
        
        basis_vectors = defBasisVectors(model, verbose=verbose)   
        
        if verbose:
            for i,v1 in enumerate(basis_vectors):
                for j,v2 in enumerate(basis_vectors):
                    if i != j:
                        print(' ---- Angle between control',i,'and',j,':',
                              angle_between(v1,v2) / np.pi * 180)
        
        parralelo2cube = np.linalg.inv( basis_vectors )
        
        x_inv_area_normalized = x_inv_area @ parralelo2cube
        
        predSet_originShift = -np.average(x_inv_area_normalized, axis=0)
        if verbose:
            print('Transformation:',parralelo2cube)            
            print('Normal inverse area:',x_inv_area)
            
            print('Normalized hypercube:',x_inv_area_normalized)
            
            print('Off origin:',predSet_originShift)
        
            print('Shifted normalized hypercube:',x_inv_area @ parralelo2cube
                    + predSet_originShift)
            
        allRegionVertices = partition['allCornersFlat'] @ parralelo2cube \
                - predSet_originShift
        
    else:
        
        print(' -- Creating inverse hull...')
        
        # Use standard method: check if points are in (skewed) hull
        x_inv_hull = defInvHull(x_inv_area)
    
        if verbose:
            print('Normal inverse area:',x_inv_area)
    
        allRegionVertices = partition['allCornersFlat'] 
    
    action_range = f7(np.concatenate(( partition['goal'],
                       np.arange(partition['nr_actions']) )))
    
    backreach = np.zeros((len(action_range), x_inv_area.shape[0],
                          x_inv_area.shape[1]))
    
    # For every action
    for action_id in progressbar(action_range, redirect_stdout=True):
        
        targetPoint = partition['targets'][action_id]
        
        if dimEqual:
        
            # Shift the origin points (instead of the target point)
            A_inv_d = model.A_inv @ np.array(targetPoint)
            
            # Python implementation
            allVerticesNormalized = (A_inv_d @ parralelo2cube) - \
                                     allRegionVertices
                            
            # Reshape the whole matrix that we obtain
            poly_reshape = np.reshape( allVerticesNormalized,
                            (partition['nr_regions'], 
                             nr_corners*model.n))
            
            # Enabled actions are ones that have all corner points within
            # the origin-centered hypercube with unit length
            enabled_in = np.maximum(np.max(poly_reshape, axis=1), 
                                    -np.min(poly_reshape, axis=1)) <= 1.0
            
        else:
            
            # Shift the origin points (instead of the target point)
            A_inv_d = model.A_inv @ np.array(targetPoint)
        
            # Subtract the shift from all corner points
            allVertices = A_inv_d - allRegionVertices
        
            # Check which points are in the convex hull
            polypoints_vec = in_hull(allVertices, x_inv_hull)
        
            # Map enabled vertices of the partitions to actual partitions
            enabled_polypoints[action_id] = np.reshape(  polypoints_vec, 
                          (partition['nr_regions'], nr_corners))
        
            # Polypoints contains the True/False results for all vertices
            # of every partition. An action is enabled in a state if it 
            # is enabled in all its vertices
            enabled_in = np.all(enabled_polypoints[action_id] == True, 
                                axis = 1)
        
        backreach[action_id] = A_inv_d - x_inv_area
        
        # Shift the inverse hull to account for the specific target point
        if setup.plotting['partitionPlot'] and \
            action_id == 88: #int(partition['nr_regions']/2):

            if verbose:
                print('x_inv_area:',x_inv_area)
                print('origin shift:',A_inv_d)       
                print('targetPoint:',targetPoint,' - drift:',
                      model.Q_flat)
            
                # Partition plot for the goal state, also showing pre-image
                print('Create partition plot...')
            
            createPartitionPlot((0,1), (2,3), partition['goal'], 
                setup, model, partition, 
                partition['allCorners'], backreach[action_id], prefix=A_idx)
        
        # Retreive the ID's of all states in which the action is enabled
        enabled_in_states[action_id] = np.nonzero(enabled_in)[0]
        
        # Remove critical states from the list of enabled actions
        enabled_in_states[action_id] = np.setdiff1d(
            enabled_in_states[action_id], partition['critical'])
        
        if action_id % printEvery == 0:
            if action_id in partition['goal']:
                if verbose:
                    print(' -- GOAL action',str(action_id),'enabled in',
                          str(len(enabled_in_states[action_id])),
                          'states - target point:',
                          str(targetPoint))
                
            else:
                if verbose:
                    print(' -- Action',str(action_id),'enabled in',
                          str(len(enabled_in_states[action_id])),
                          'states - target point:',
                          str(targetPoint))
        
        if len(enabled_in_states[action_id]) > 0:
            total_actions_enabled += 1
        
        for origin in enabled_in_states[action_id]:
            enabled_actions[origin].add(action_id)
            
    enabledActions_inv = [set(enabled_in_states[i])
                          for i in range(partition['nr_actions'])]
    
    return total_actions_enabled, enabled_actions, enabledActions_inv, \
            backreach