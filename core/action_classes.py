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

import itertools
import numpy as np
import cvxpy as cp

class action(object):
    '''
    Action object
    ''' 
    
    def __init__(self, idx, model, center, idx_tuple, backreach_obj):
        
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
          
        shift = model.A_inv @ self.center
        self.backreach = self.backreach_obj.verts + shift
        
        if not backreach_obj.target_set_size is None:
            self.backreach_infl = self.backreach_obj.verts_infl + shift
        
        
        
class backreachset(object):
    '''
    Backward reachable set
    '''
    
    def __init__(self, name, target_set_size=None):
        
        self.name = name
        self.target_set_size = target_set_size
        
    def compute_default_set(self, model):
        '''
        Compute the default (inflated) backward reachable set for a target
        point at the origin (zero).
        
        Parameters
        ----------
        model : Model object

        '''
        
        self.verts = def_backward_reach(model)
        
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