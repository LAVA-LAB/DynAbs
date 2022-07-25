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

import numpy as np              # Import Numpy for computations
from scipy.sparse.csgraph import connected_components

def define_model(setup, model_raw, spec):
    '''
    Define model within abstraction object for given value of lump

    Parameters
    ----------
    lump : int
        Value of lump to create model for.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

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
    
    # Create noise samples (for 3D UAV benchmark)
    if model_raw.name in ['UAV'] and model_raw.modelDim == 3:
        setup.setOptions(category='scenarios', gaussian=False)
        model_raw.setTurbulenceNoise(setup.directories['base'],
                                 setup.scenarios['samples'])
    
    if lump == 0:
        model = makeModelFullyActuated(model_raw, 
                   manualDimension = 'auto', observer=False)
    else:
        model = makeModelFullyActuated(model_raw, 
                   manualDimension = lump, observer=False)
        
    # Determine inverse A matrix
    model.A_inv  = np.linalg.inv(model.A)
    
    # Determine pseudo-inverse B matrix
    model.B_pinv = np.linalg.pinv(model.B)
    
    # Retreive system dimensions
    model.p      = np.size(model.B,1)   # Nr of inputs
    
    # If noise samples are used, recompute them
    if setup.scenarios['gaussian'] is False:
        
        f = model_raw.setup['noiseMultiplier']
        
        model.noise['samples'] = f * np.vstack(
            (2*model.noise['samples'][:,0],
             0.2*model.noise['samples'][:,0],
             2*model.noise['samples'][:,1],
             0.2*model.noise['samples'][:,1],
             2*model.noise['samples'][:,2],
             0.2*model.noise['samples'][:,2])).T
    
    uAvg = (model.uMin + model.uMax) / 2
    
    if np.linalg.matrix_rank(np.eye(model.n) - model.A) == model.n:
        model.equilibrium = np.linalg.inv(np.eye(model.n) - model.A) @ \
            (model.B @ uAvg + model.Q_flat)
    
    return {'model': model, 'spec': spec}

def find_connected_components(A, B, n, p):
    
    no_comp, array = connected_components(A, directed=False)
    
    dim_n = [[] for i in range(no_comp)]
    dim_p = [[] for i in range(no_comp)]
    
    for c in range(no_comp):
        states = np.arange(n)[array == c]
        
        dim_n[c] = states
        
        inputs = np.arange(p)[np.any(B[states] != 0, axis=0)]
        dim_p[c] = inputs
        
    if sum([len(dim_p[c]) for c in range(no_comp)]) != p:
        print('WARNING: control inputs not decoupled between components')
    
    return dim_n, dim_p

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
    model.tau             *= dim
    
    model.uMin = np.repeat(model.uMin, dim)
    model.uMax = np.repeat(model.uMax, dim)
    
    return model