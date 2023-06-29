#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import cvxpy as cp

from models.stabilize import compute_stabilized_control_vertices

class action(object):
    '''
    Action object
    ''' 
    
    def __init__(self, idx, model, center, idx_tuple, backreach_obj, shift):
        
        '''
        Initialize action and compute the (inflated) backward reachable set 
        
        Parameters
        ----------
        model : Model object
        '''
        
        self.idx                = idx
        self.center             = center
        self.tuple              = idx_tuple
        self.backreach_obj      = backreach_obj
        
        self.error              = None
        self.enabled_in         = set()
        
        self.backreach = self.backreach_obj.verts + shift
        
        if not backreach_obj.target_set_size is None:
            self.backreach_infl = self.backreach_obj.verts_infl + shift
        
        
        
class backreachset(object):
    '''
    Backward reachable set
    '''
    
    def __init__(self, name, target_point = 0, target_set_size = None):
        
        self.name = name
        self.target_point = target_point
        self.target_set_size = target_set_size
        
    def compute_default_set(self, model, verbose=False):
        '''
        Compute the default (inflated) backward reachable set for a target
        point at the origin (zero).
        
        Parameters
        ----------
        model : Model object

        '''

        if hasattr(model, 'K'):

            u_vertices = compute_stabilized_control_vertices(model, self.target_point)

        else:

            # Compute matrix of all possible control inputs
            control_mat = [[model.uMin[i], model.uMax[i]] for i in range(model.p)]
        
            u_vertices = np.array(list(itertools.product(*control_mat)))
        
        if len(u_vertices) == 0:
            print('- WARNING: backward reachable set for target point {} is empty'.format(self.target_point.flatten()))

            # Backward reachable set is empty
            self.verts = np.array([])

        else:
            # Backward reachable set is non empty
            self.verts = def_backward_reach(model, 
                                            np.reshape(self.target_point, (model.n, 1)), 
                                            u_vertices)

            if not self.target_set_size is None:
                
                alphas = np.eye(len(self.verts))
                
                BRS_inflated = []
                
                # Compute the control error
                for err in itertools.product(*self.target_set_size):
                    for alpha in alphas:
                        prob,x,_ = find_backward_inflated(model.A, np.array(err), 
                                                        alpha, self.verts)
                        
                        BRS_inflated += [x]
                
                self.verts_infl = np.array(BRS_inflated)

        if verbose:
            print('target_point: ', self.target_point)
            print('u_vertices: ', u_vertices)
            print('backward reachable set:', self.verts)


def def_backward_reach(model, target_point, u_vertices):
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
    
    # Compute backward reachable set (inverse dynamics under all extreme
    # control inputs)
    # The transpose needed to do computation for all control inputs at once    
    backreach = (model.A_inv @ (target_point - model.B @ u_vertices.T - model.Q)).T
    
    return backreach



def find_backward_inflated(A_hat, error, alpha, G):
    '''
    Find the inflated backward reachable set
    '''

    x = cp.Variable(len(A_hat))
    y = cp.Variable(len(A_hat))
    
    constraints = [ A_hat @ (x - y) == error,
                    y == sum([a * g for a,g in zip(alpha, G)])
                    ]

    obj = cp.Minimize(1)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver='ECOS')
    
    return prob, x.value, y.value



def partial_model(flags, model, spec, dim_n, dim_p):
    '''
    Define partial model for compositional definition of actions
    '''
    
    model.A         = model.A[dim_n][:, dim_n]
    model.A_inv     = model.A_inv[dim_n][:, dim_n]
    model.B         = model.B[dim_n][:, dim_p]
    model.Q         = model.Q[dim_n]
    model.n         = len(dim_n)
    model.p         = len(dim_p)
    
    if flags['parametric']:
        model.A_set     = [A[dim_n][:, dim_n] for A in model.A_set]
        model.B_set     = [B[dim_n][:, dim_p] for B in model.B_set]
    
    spec.partition['number'] = spec.partition['number'][dim_n]
    spec.partition['width'] = spec.partition['width'][dim_n]
    spec.partition['origin'] = spec.partition['origin'][dim_n]
    
    model.uMin = model.uMin[dim_p]
    model.uMax = model.uMax[dim_p]

    return model, spec



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