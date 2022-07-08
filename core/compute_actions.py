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
from .commons import in_hull, overapprox_box, tocDiff
from .cvx_opt import LP_vertices_contained

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


def partial_model(flags, model, spec, dim_n, dim_p):
    
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
    
    if flags['parametric']:
        model.A_set     = [A[n_start:n_end, n_start:n_end] for A in model.A_set]
        model.B_set     = [B[n_start:n_end, p_start:p_end] for B in model.B_set]
    
    spec.partition['number'] = spec.partition['number'][dim_n]
    spec.partition['width'] = spec.partition['width'][dim_n]
    spec.partition['origin'] = spec.partition['origin'][dim_n]
    
    model.uMin = model.uMin[dim_p]
    model.uMax = model.uMax[dim_p]

    return model, spec


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

def rotate_2D_vector(vector):
    '''
    Compute the rotation matrix to rotate a vector back to x-axis aligned

    Parameters
    ----------
    vector : np.array
        Input vector.

    Returns
    -------
    matrix : np.array
        Rotation matrix.

    '''
    
    unit_vector = vector / np.linalg.norm(vector)
    
    n = len(vector)
    rotate_to = np.zeros(n)
    rotate_to[0] = 1
    
    dot_product = np.dot(unit_vector, rotate_to)
    angle = np.arccos(dot_product)
    
    matrix = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    
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


def find_backward_inflated(A_hat, error, alpha, G):

    x = cp.Variable(len(A_hat))
    y = cp.Variable(len(A_hat))
    
    constraints = [ A_hat @ (x - y) == error,
                    y == sum([a * g for a,g in zip(alpha, G)])
                    ]

    obj = cp.Minimize(1)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver='ECOS')
    
    return prob, x.value, y.value


class epistemic_error(object):
    
    def __init__(self, model):
        '''
        Initialize method to compute the epistemic error

        Parameters
        ----------
        model : Model object

        Returns
        -------
        None.

        '''
        
        # Compute matrix of all possible control inputs
        control_mat = [[model.uMin[i], model.uMax[i]] for i in 
                             range(model.p)]
        
        rows = len(model.A_set) * 2**model.p
        self.plus = np.zeros((rows, model.n))
        self.mult = [[]]*rows
        i=0
        
        for A_vertex, B_vertex in zip(model.A_set, model.B_set):
            for u in itertools.product(*control_mat):
                
                self.mult[i] = A_vertex - model.A
                self.plus[i,:] = (B_vertex - model.B) @ u
                i+=1
                
        
    def compute(self, vertices):
        '''
        Compute the epistemic error for a given list of vertices

        Parameters
        ----------
        vertices : 2D array, with every row being a n-dimensional vertex.

        Returns
        -------
        minimum/maximum epistemic error.

        '''
        
        e_error = np.vstack([ (m @ vertices.T).T + p
                              for m,p in zip(self.mult, self.plus) ])

        return e_error.min(axis=0), e_error.max(axis=0)


def compute_epistemic_error(model, vertices):

    # Compute matrix of all possible control inputs
    control_mat = [[model.uMin[i], model.uMax[i]] for i in 
          range(model.p)]
    
    e_error_neg = np.zeros((len(model.A_set) * 2**model.p, model.n))
    e_error_pos = np.zeros((len(model.A_set) * 2**model.p, model.n))
    
    i = 0

    # Compute epistemic error
    for A_vertex, B_vertex in zip(model.A_set, model.B_set):
        for u in itertools.product(*control_mat):
        
            e_error = ((A_vertex - model.A) @ vertices.T).T + (B_vertex - model.B) @ u
            
            e_error_neg[i] = e_error.min(axis=0)
            e_error_pos[i] = e_error.max(axis=0)
            i += 1
        
    return e_error_neg.min(axis=0), e_error_pos.max(axis=0)


def enabledActionsImprecise(setup, flags, partition, actions, model, spec, 
                            dim_n=False, dim_p=False, verbose=False):
    
    # Compute the backward reachable set (not accounting for target point yet)    
    if dim_n is False or dim_p is False or len(dim_n) == model.n:
        dim_n = np.arange(model.n)
        dim_p = np.arange(model.p)
        compositional = False
        
    else:
        dim_excl = np.array([i for i in range(model.n) if i not in dim_n])
        
        model, spec = partial_model(flags, deepcopy(model), deepcopy(spec), dim_n, dim_p)
        compositional = True
        
    
    enabled = {}
    enabled_inv = {}   
    error = {}
    
    # Create LP object
    BRS_0 = actions['backreach_obj']['default'].verts_infl
    LP = LP_vertices_contained(model, 
                               np.unique(BRS_0[:, dim_n], axis=0).shape, 
                               solver=setup.cvx['solver'])
    
    if flags['parametric']:
        epist = epistemic_error(model)
    else:
        epist = None
    
    # If the (noninflated) backward reachable set is a line, compute the rot.
    # matrix. Used to make computation of enabled actions faster, by reducing
    # the number of potentially active actions.
    if len(actions['backreach_obj']['default'].verts) == 2 and len(dim_n) == 2:
        verts = actions['backreach_obj']['default'].verts[:, dim_n]
        brs_0_shift = verts - verts[0]
        BRS_0_shift = BRS_0[:, dim_n] - verts[0]
        
        ROT = rotate_2D_vector(brs_0_shift[1] - brs_0_shift[0])
        distances = (ROT @ (BRS_0_shift[:, dim_n]).T).T
        max_distance_to_brs = np.max(np.abs(distances[:, 1:]))
        
    else:
        ROT = None
    
    # For every action
    for i,a_tup in enumerate(actions['tup2idx']): #progressbar(enumerate(actions['tup2idx']), redirect_stdout=True):
        
        # If we are in compositional mode, only check this action if all 
        # excluded dimensions are zero        
        if compositional and any(np.array(a_tup)[dim_excl] != 0):
            continue
        
        # Get reference to current action object
        a_idx = actions['tup2idx'][a_tup]
        act   = actions['obj'][a_idx]
        
        # Get backward reachable set
        BRS = np.unique(act.backreach_infl[:, dim_n], axis=0)
        brs_basis = act.backreach[0, dim_n]
        
        # Set current backward reachable set as parameter
        LP.set_backreach(BRS)
        
        # Overapproximate the backward reachable set
        BRS_overapprox_box = overapprox_box(BRS)
        
        if 'max_action_distance' in spec.error:
            BRS_overapprox_box = np.maximum(np.minimum(BRS_overapprox_box - act.center[dim_n], 
                                    spec.error['max_action_distance']), 
                                   -spec.error['max_action_distance']) + act.center[dim_n]
        
        # Determine set of regions that potentially intersect with G
        _, idx_edgeV = computeRegionIdx(BRS_overapprox_box, 
                                        spec.partition,
                                        borderOutside=[False,True])
        
        # Transpose because we want the min/max combinations per dimension
        # (not per point)
        idx_edge = np.zeros((model.n, 2))
        idx_edge[dim_n] = idx_edgeV.T
        
        idx_mat = [np.arange(low, high+1).astype(int) for low,high in idx_edge]
        
        s_min_list = []
        
        # Try each potential predecessor state
        for si, s_tup in enumerate(itertools.product(*idx_mat)):
            tocDiff(False)
            
            # Retrieve current state index
            s_min = partition['R']['idx'][s_tup]
            
            # Skip if this is a critical state
            if s_min in partition['critical'] and not compositional:
                continue
            
            unique_verts = np.unique(partition['allCorners'][s_min][:, dim_n], axis=0)
            
            # if not ROT is None:
            if not ROT is None:
                rotated_region = (ROT @ (unique_verts - brs_basis).T).T
                
                distance = max(np.abs(rotated_region[:,1:]))
                if distance > max_distance_to_brs:
                    continue
            
            # If the problem is feasible, then action is enabled in this state
            if LP.solve(unique_verts):
                
                # Add state to the list of enabled states
                s_min_list += [s_min]
                
                # Enable the current action in the current state
                if s_tup in enabled:
                    enabled[s_tup].add(a_tup)
                else:
                    enabled[s_tup] = {a_tup}
                    
                if a_tup in enabled_inv:
                    enabled_inv[a_tup].add(s_tup)
                else:
                    enabled_inv[a_tup] = {s_tup}
                
        # Retrieve control error negative/positive
        control_error = np.zeros((model.n, 2))
        control_error[dim_n] = act.backreach_obj.max_control_error[dim_n, :]
        
        # If a parametric model is used
        if not epist is None and len(s_min_list) > 0:
            
            # Retrieve list of unique vertices of predecessor states
            s_vertices = partition['allCorners'][s_min_list][:, :, dim_n]
            s_vertices_unique = np.unique(np.vstack(s_vertices), axis=0)
        
            # Compute the epistemic error
            epist_error_neg, epist_error_pos = epist.compute(s_vertices_unique)
            
            # Store the control error for this action
            error[a_tup] = {'neg': control_error[:,0] + epist_error_neg,
                            'pos': control_error[:,1] + epist_error_pos}
            
        elif len(s_min_list) > 0:
            
            # Store the control error for this action
            error[a_tup] = {'neg': control_error[:,0],
                            'pos': control_error[:,1]}        
            
        if i % 1 == 0:
            print('Action to',act.center,'enabled in ',len(s_min_list),'states')
            
    return enabled, enabled_inv, error


def enabledActions(setup, partition, actions, model, A_idx=None, verbose=True):
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