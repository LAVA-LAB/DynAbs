#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np              # Import Numpy for computations
from scipy.sparse.csgraph import connected_components

def define_model(model_raw, spec, stabilize_lqr=False):
    '''
    Define model within abstraction object for given value of lump
    '''
    
    spec.partition['boundary'] = np.array(spec.partition['boundary']).astype(float)
    spec.partition['number'] = np.array(spec.partition['number']).astype(int)
    
    if type(spec.targets['boundary']) != str:        
        spec.targets['boundary'] = np.array(spec.targets['boundary']).astype(float)
    if type(spec.targets['number']) != str:
        spec.targets['number'] = np.array(spec.targets['number']).astype(int)
    
    # Control limitations
    model_raw.uMin   = np.array( spec.control['uMin'] ).astype(float)
    model_raw.uMax   = np.array( spec.control['uMax'] ).astype(float)
    
    lump = model_raw.lump
    
    if lump == 0:
        model = make_fully_actuated(model_raw, 
                   manualDimension = 'auto')
    else:
        model = make_fully_actuated(model_raw, 
                   manualDimension = lump)
        

    #####
    # Stabilize system
    if stabilize_lqr:
        from models.stabilize import lqr
        
        Q = np.eye(model.n)
        R = 2*np.eye(model.n)

        model.K = lqr(model.A, model.B, Q, R)
        model.A = (model.A - model.B @ model.K)
    #####


    # Determine inverse A matrix
    model.A_inv  = np.linalg.inv(model.A)
    
    # Determine pseudo-inverse B matrix
    model.B_pinv = np.linalg.pinv(model.B)
    
    # Retreive system dimensions
    model.p      = np.size(model.B,1)   # Nr of inputs
    
    uAvg = (model.uMin + model.uMax) / 2
    
    if np.linalg.matrix_rank(np.eye(model.n) - model.A) == model.n:
        model.equilibrium = np.linalg.inv(np.eye(model.n) - model.A) @ \
            (model.B @ uAvg + model.Q_flat)

    return model, spec

def find_connected_components(A, B, n, p):
    
    no_comp, array = connected_components(A, directed=False)
    
    print(' - State space can be decomposed in {} blocks'.format(no_comp))
    print(' - Decomposition vector:', array)

    dim_n = [[] for i in range(no_comp)]
    dim_p = [[] for i in range(no_comp)]
    
    # Iterate over the components
    for c in range(no_comp):
        # Retrieve state variables belonging to this component
        states = np.arange(n)[array == c]
        
        dim_n[c] = states
        
        # Retrieve control variables belonging to this component
        inputs = np.arange(p)[np.any(B[states] != 0, axis=0)]
        dim_p[c] = inputs

    # Merge components if controls are not disjoint
    dim_p, dim_n = merge_sublists(dim_p, dim_n)
    no_comp = len(dim_n)

    if sum([len(dim_p[c]) for c in range(no_comp)]) != p:
        print('WARNING: control inputs not decoupled between components')
        print('Resolve by merging multiple blocks')
    
    return dim_n, dim_p

def merge_sublists(sublists, sublists_synch):
    '''
    Merge list of lists until they are all disjoint
    '''

    # Check if there are any further intersections among the merged sublists
    while True:
        merged = False
        
        # Iterate over the merged sublists
        for i in range(len(sublists)):

            # Iterate over the remaining merged sublists
            for j in range(len(sublists)):
                if i == j:
                    continue

                # If there is an intersection between the two merged sublists
                if any(state in sublists[j] for state in sublists[i]):
                    print(' - Merge state components {} and {} as controls are not disjoint'.format(i,j))

                    # Merge the sublists
                    sublists[i] = np.concatenate((sublists[i], sublists[j]))
                    sublists_synch[i] = np.concatenate((sublists_synch[i], sublists_synch[j]))

                    # Remove the ones that were merged
                    sublists.pop(j)
                    sublists_synch.pop(j)
                    merged = True

                    break

            if merged:
                break
        
        # If no further intersections are found, exit the loop
        if not merged:
            break

    sublists = [np.unique(list) for list in sublists]
    sublists_synch = [np.unique(list) for list in sublists_synch]
    
    print('--------')

    return sublists, sublists_synch

def make_fully_actuated(model, manualDimension='auto'):
    '''
    Given a model in `model`, render it fully actuated.

    Parameters
    ----------
    model : dict
        Main dictionary of the LTI system model.
    manualDimension : int or str, optional
        Desired dimension of the state of the model The default is 'auto'.

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
    
    model.orig = {
        'A': model.A,
        'B': model.B,
        'Q': model.Q,
        'Q_flat': model.Q.flatten()
    }

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
    model.tau             *= dim
    
    model.uBarMin = 0.2*np.repeat(model.uMin, dim)
    model.uBarMax = 0.2*np.repeat(model.uMax, dim)

    model.uMin = np.repeat(model.uMin, dim)
    model.uMax = np.repeat(model.uMax, dim)
    
    return model