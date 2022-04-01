# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 15:30:09 2022

@author: Thom Badings
"""

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
from copy import deepcopy

import cvxpy as cp

from .define_partition import computeRegionIdx
from .commons import in_hull, overapprox_box
from .cvx_opt import abstraction_error, LP_vertices_contained

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

def partial_model(flags, model, dim_n, dim_p):
    
    #TODO: improve this function.
    
    n_start = min(dim_n)
    n_end   = max(dim_n)+1
    p_start = min(dim_p)
    p_end   = max(dim_p)+1
    
    model.A         = model.A[n_start:n_end, n_start:n_end]
    model.A_inv     = model.A_inv[n_start:n_end, n_start:n_end]
    model.B         = model.B[n_start:n_end, p_start:p_end]
    model.Q         = model.Q[n_start:n_end, :]
    model.n         = len(dim_n)
    model.p         = len(dim_p)
    
    if flags['parametric_A']:
        model.A_set     = [A[n_start:n_end, n_start:n_end] for A in model.A_set]
    
    model.setup['partition']['number'] = model.setup['partition']['number'][dim_n]
    model.setup['partition']['width'] = model.setup['partition']['width'][dim_n]
    model.setup['partition']['origin'] = model.setup['partition']['origin'][dim_n]
    
    model.uMin = model.uMin[dim_p]
    model.uMax = model.uMax[dim_p]
    
    if 'max_control_error' in model.setup:
        model.setup['max_control_error'] = model.setup['max_control_error'][dim_n]

    return model

def def_all_BRS(model, targets, control_error_bound):
    
    G_zero = def_backward_reach(model)
    G_infl_zero = compute_BRS_inflated(model, G_zero, control_error_bound)
    backreach = {}
    backreach_inflated = {}

    for a in range(len(targets)):
        shift = model.A_inv @ targets[a]
        backreach[a] = G_zero + shift
        backreach_inflated[a] = G_infl_zero + shift
        
    return backreach, backreach_inflated

def rotate_BRS(vector):
    '''
    Compute the rotation vector to rotate a vector back to x-axis aligned

    Parameters
    ----------
    vector : np.array
        Input vector.

    Returns
    -------
    matrix : np.array
        Rotation matrix.

    '''
    
    n = len(vector)
    
    rotate_to = np.zeros(n)
    rotate_to[0] = 1
    
    ROT = cp.Variable((n,n))
    constraints = [ROT @ vector == rotate_to]
    obj = cp.Minimize(cp.norm(ROT))
    prob = cp.Problem(obj, constraints)
    
    prob.solve()
    
    matrix = ROT.value
    
    return matrix

def project_to_line(BRS, points):
    '''
    Project given points to the line spanned by a 2-point backward reach set

    Parameters
    ----------
    BRS : np.array
        Backward reachable set.
    points : np.array
        List of points to project.

    Returns
    -------
    None.

    '''
    
    base = BRS[0]
    line = BRS[1]-base
    
    constant = line / np.dot(line, line)
    
    proj_points = np.array([np.dot(line, point-base) * constant for point in points]) + base
    
    return proj_points

def defEnabledActions_UA(flags, partition, actions, model, dim_n=False, dim_p=False, verbose=False):
    
    MODE = 'rotate'
    
    full_n = model.n
    nrPerDim = [model.setup['targets']['number'][i] if i in dim_n else 1 for i in range(full_n)]
    action_range = list(itertools.product(*map(range, [0]*full_n, nrPerDim)))
    
    # Compute the backward reachable set (not accounting for target point yet)    
    if dim_n is False or dim_p is False:
        dim_n = np.array(model.n)
        dim_p = np.array(model.p)
        
        compositional = False
        
    else:
        model = partial_model(flags, deepcopy(model), dim_n, dim_p)
    
        compositional = True
    
    G_zero = def_backward_reach(model)
    
    ROTMAT = rotate_BRS( G_zero[1]-G_zero[0] )
    
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
    
    # Initialize object to compute abstraction errors
    abstr_error = abstraction_error(model, no_verts = 2**model.n)
    
    enabled = {}
    enabled_inv = {}   
    control_error = {}
    
    # For every action
    for a_tup in progressbar(action_range, redirect_stdout=True):
        
        idx = actions['T']['idx'][a_tup]
        
        # Get backward reachable set
        BRS = np.unique(actions['backreach'][idx][:, dim_n], axis=0)
        
        # Overapproximate the backward reachable set
        G_curr_box = overapprox_box(BRS)
        
        # Determine set of regions that potentially intersect with G
        _, idx_edgeV = computeRegionIdx(G_curr_box, model.setup['partition'],
                                       borderOutside=[False,True])
        
        idx_edge = np.zeros((full_n, 2))
        idx_edge[dim_n] += idx_edgeV.T
        
        # Transpose because we want the min/max combinations per dimension
        # (not per point)
        idx_mat = [np.arange(low, high+1).astype(int) for low,high in idx_edge]
        
        if verbose:
            print('Evaluate action',a_tup,'(try',
              len(list(itertools.product(*idx_mat))),'potential predecessors)')
        
        # Initialize control error 
        nr_idx_mat = np.product(list(map(len, idx_mat)))
        control_error[a_tup] = {'neg': np.zeros((nr_idx_mat, model.n)),
                                'pos': np.zeros((nr_idx_mat, model.n)) }
        
        # Set current backward reachable set as parameter
        G_curr.value = BRS
        
        for si, s_tup in enumerate(itertools.product(*idx_mat)):
            
            s = partition['R']['idx'][s_tup]
            
            # Skip if this is a critical state
            if s in partition['critical'] and not compositional:
                continue
            
            region_min.value = partition['R']['low'][s][dim_n]
            region_max.value = partition['R']['upp'][s][dim_n]
            
            prob.solve(warm_start=True, solver='ECOS')
            
            # print('a_tup',a_tup,'s_tup',s_tup,'feasible?',prob.status)
            
            if prob.status != "infeasible":
                # If problem not infeasible, then action enabled in this state
                # But we do another step to limit the abstraction error
                
                vertices = np.unique(partition['allCorners'][s][:, dim_n], axis=0)
                
                if MODE == 'optimize':
                    # Compute abstraction error
                    flag, project_vec, projected_points = abstr_error.solve(vertices, BRS)
                    
                    # Skip this action if flag is set or if not all projections 
                    # are along the same vector
                    if flag or not np.isclose(project_vec, project_vec[0], 
                                    rtol=1e-1, atol=1e-2).all():
                        continue
                    
                else:
                    shift_vertices = vertices - BRS[0]
                    norm_vertices = (ROTMAT @ shift_vertices.T).T
                    vertices_x = norm_vertices[:,0]
                    
                    if np.any(vertices_x > 1) or np.any(vertices_x < 0):                        
                        continue
                    
                    # Project points
                    projected_points = project_to_line(BRS, vertices)
                    
                    if a_tup == (5,5) and s_tup == (5,5):
                        print('---')
                        print('s:', s,' s_tup:', s_tup)
                        print('vertices:', vertices)
                        print('norm_vertices:', norm_vertices)
                        print('BRS:', BRS)
                        print('projected points:', projected_points)
                        print('---')
                #####
                # Compute control error
                projected_diff = vertices - projected_points
                c_error = (model.A @ (projected_diff).T).T
                
                control_error[a_tup]['neg'][si] = c_error.min(axis=0)
                control_error[a_tup]['pos'][si] = c_error.max(axis=0)
                
                if a_tup == (5,5) and s_tup == (5,5):
                    print('difference:', projected_diff)
                    print('control error:', c_error)
                
                # Check if we also have to account for the epistemic error
                if flags['parametric_A']:
                
                    e_error_neg = np.zeros((len(model.A_set), model.n))
                    e_error_pos = np.zeros((len(model.A_set), model.n))
                
                    # Compute epistemic error
                    for i,A_vertex in enumerate(model.A_set):
                        e_error = ((A_vertex - model.A) @ vertices.T).T
                        e_error_neg[i] = e_error.min(axis=0)
                        e_error_pos[i] = e_error.max(axis=0)
                        
                    control_error[a_tup]['neg'][si] += e_error_neg.min(axis=0)
                    control_error[a_tup]['pos'][si] += e_error_pos.max(axis=0)
                
                if s_tup in enabled:
                    enabled[s_tup].add(a_tup)
                else:
                    enabled[s_tup] = {a_tup}
                    
                if a_tup in enabled_inv:
                    enabled_inv[a_tup].add(s_tup)
                else:
                    enabled_inv[a_tup] = {s_tup}
                        
        control_error[a_tup] = {'neg': control_error[a_tup]['neg'].min(axis=0),
                                'pos': control_error[a_tup]['pos'].max(axis=0)}
            
    return enabled, enabled_inv, control_error

def find_backward(A_hat, error, alpha, G):

    x = cp.Variable(len(A_hat))
    y = cp.Variable(len(A_hat))
    
    constraints = [ A_hat @ (x - y) == error,
                    y == sum([a * g for a,g in zip(alpha, G)])
                    ]

    obj = cp.Minimize(1)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver='ECOS')
    
    return prob, x.value, y.value

def compute_BRS_inflated(model, G, control_error_bound):
    
    BRS_inflated = []
    alphas = np.eye(len(G))
    
    for err in itertools.product(*control_error_bound):
        for alpha in alphas:
            prob,x,_ = find_backward(model.A, np.array(err), alpha, G)
            
            BRS_inflated += [x]
    
    return np.array(BRS_inflated)

def compute_epistemic_error(model, vertices):
    
    e_error_neg = np.zeros((len(model.A_set), model.n))
    e_error_pos = np.zeros((len(model.A_set), model.n))

    # Compute epistemic error
    for i,A_vertex in enumerate(model.A_set):
        e_error = ((A_vertex - model.A) @ vertices.T).T
        e_error_neg[i] = e_error.min(axis=0)
        e_error_pos[i] = e_error.max(axis=0)
        
    return e_error_neg.min(axis=0), e_error_pos.max(axis=0)

def defEnabledActions_UA_V2(flags, partition, actions, model, dim_n=False, dim_p=False, verbose=False):
    
    full_n = model.n
    nrPerDim = [model.setup['targets']['number'][i] if i in dim_n else 1 for i in range(full_n)]
    action_range = list(itertools.product(*map(range, [0]*full_n, nrPerDim)))
    
    # Compute the backward reachable set (not accounting for target point yet)    
    if dim_n is False or dim_p is False:
        dim_n = np.arange(model.n)
        dim_p = np.arange(model.p)
        
        compositional = False
        
    else:
        model = partial_model(flags, deepcopy(model), dim_n, dim_p)
    
        compositional = True
    
    enabled = {}
    enabled_inv = {}   
    control_error = {}
    
    control_error_neg = model.setup['max_control_error'][:,0]
    control_error_pos = model.setup['max_control_error'][:,1]
    
    # Create LP object
    Q = LP_vertices_contained(model, np.unique(actions['backreach_inflated'][0][:, dim_n], axis=0))
    
    # For every action
    for a_tup in progressbar(action_range, redirect_stdout=True):
        
        idx = actions['T']['idx'][a_tup]
        a_center = actions['T']['center'][idx]
        
        # Retrieve inflated backward reachable set
        BRS = np.unique(actions['backreach_inflated'][idx][:, dim_n], axis=0)
        
        # Set current backward reachable set as parameter
        Q.set_backreach(BRS)
        
        # Overapproximate the backward reachable set
        BRS_overapprox_box = overapprox_box(BRS)
        
        # Determine set of regions that potentially intersect with G
        _, idx_edgeV = computeRegionIdx(BRS_overapprox_box, 
                                        model.setup['partition'],
                                        borderOutside=[False,True])
        
        # Transpose because we want the min/max combinations per dimension
        # (not per point)
        idx_edge = np.zeros((full_n, 2))
        idx_edge[dim_n] += idx_edgeV.T
        
        idx_mat = [np.arange(low, high+1).astype(int) for low,high in idx_edge]
        
        if verbose:
            print('Evaluate action',a_tup,'(try',
              len(list(itertools.product(*idx_mat))),'potential predecessors)')
        
        # Initialize control error 
        nr_idx_mat = np.product(list(map(len, idx_mat)))
        epist_error_neg = np.zeros((nr_idx_mat, model.n))
        epist_error_pos = np.zeros((nr_idx_mat, model.n))
        
        for si, s_tup in enumerate(itertools.product(*idx_mat)):
            
            # Retrieve current state index
            s = partition['R']['idx'][s_tup]
            
            # Skip if this is a critical state
            if s in partition['critical'] and not compositional:
                continue
            
            unique_verts = np.unique(partition['allCorners'][s][:, dim_n], axis=0)
            
            # If the problem is feasible, then action is enabled in this state
            if Q.solve(unique_verts):
                
                # Check if we also have to account for the epistemic error
                if flags['parametric_A']:                    
                    epist_error_neg[si], epist_error_pos[si] = \
                        compute_epistemic_error(model, unique_verts)    
                
                if 'max_action_distance' in model.setup:
                    s_center = partition['R']['center'][s]
                    
                    if any(a_center - s_center > model.setup['max_action_distance']) or \
                       any(a_center - s_center < -model.setup['max_action_distance']):
                           print('Disable action from', s_tup, s_center, 'to', a_tup, a_center)
                           
                           continue
                
                # Enable the current action in the current state
                if s_tup in enabled:
                    enabled[s_tup].add(a_tup)
                else:
                    enabled[s_tup] = {a_tup}
                    
                if a_tup in enabled_inv:
                    enabled_inv[a_tup].add(s_tup)
                else:
                    enabled_inv[a_tup] = {s_tup}
                    
        control_error[a_tup] = {'neg': control_error_neg + epist_error_neg.min(axis=0),
                                'pos': control_error_pos + epist_error_pos.max(axis=0)}
            
    return enabled, enabled_inv, control_error

def defEnabledActions(setup, partition, actions, model, A_idx=None, verbose=True):
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
    
    enabled_in_states = [[] for i in range(actions['nr_actions'])]
    enabled_actions   = [set() for i in range(partition['nr_regions'])]
    
    nr_corners = 2**model.n
    
    printEvery = min(100, max(1, int(actions['nr_actions']/10)))
    
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
    
    # action_range = f7(np.concatenate(( partition['goal'],
    #                    np.arange(actions['nr_actions']) )))
    action_range = np.arange(actions['nr_actions'])
    
    backreach = np.zeros((len(action_range), x_inv_area.shape[0],
                          x_inv_area.shape[1]))
    
    if verbose:
        RANGE = action_range
    else:
        RANGE = progressbar(action_range, redirect_stdout=True)
    
    # For every action
    for action_id in RANGE:
        
        targetPoint = actions['T']['center'][action_id]
        
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
            action_id == np.round(np.mean(partition['goal'])):

            # Partition plot for the goal state, also showing pre-image
            print('Create partition plot for action '+str(action_id)+'...')             

            if verbose:
                print('x_inv_area:',x_inv_area)
                print('origin shift:',A_inv_d)       
                print('targetPoint:',targetPoint,' - drift:',
                      model.Q_flat)
        
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
                          for i in range(actions['nr_actions'])]
    
    return total_actions_enabled, enabled_actions, enabledActions_inv, \
            backreach