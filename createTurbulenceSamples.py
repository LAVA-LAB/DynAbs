#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 ______________________________________
|                                      |
|  SCENARIO-BASED ABSTRACTION PROGRAM  |
|______________________________________|

This script belongs to the following paper:
    
  Thom Badings, Alessandro Abate, David Parker, Nils Jansen, Hasan Poonawala & 
  Marielle Stoelinga (2021). Sampling-based Robust Control of Autonomous 
  Systems with Non-Gaussian Noise. AAAI 2022.

Originally coded by:        Thom S. Badings
Contact e-mail address:     thom.badings@ru.nl

Specifically, this script generates samples of the process noise caused by
turbulence, acting on a UAV. This noise is non-Gaussian, and can, therefore,
not be approximates as a Gaussian distribution.
______________________________________________________________________________
"""

import numpy as np              # Import Numpy for computations

def createTurbulenceSamples(N, length):
    '''
    Create samples of the effect of wind turbulence on the state of the UAV.

    Parameters
    ----------
    N : int
        Number of samples to create.
    length : int
        Length of the trajectories used to create the samples.

    Returns
    -------
    samples : 2D Numpy array
        Array of samples (every row is a sample).

    '''

    from core.UAV.dryden import DrydenGustModel
            
    # V_a = speed in 
    turb = DrydenGustModel(dt=1, b=5, h=20, V_a = 25, intensity="moderate")
    
    iters = N
    sample_length = length
    
    samples = np.zeros((iters,3))
    
    for i in range(iters):
        
        if i % 100 == 0:
            print(' -- Create turbulence noise sample:',i)
        
        turb.reset()
        turb.simulate(sample_length)
        timeseries = turb.vel_lin
        
        samples[i,:] = timeseries[:,-1]
          
    return samples
    
#############################
# Create turbulence samples #
#############################
    
# Main settings
N = 100000         # Number of samples
length = 1000   # Length of trajectories used to compute samples

# Create samples
samples = createTurbulenceSamples(N=N, length=length) / 5

# Store samples
store_folder = "input/"
store_file   = "TurbulenceNoise_N="+str(N)+".csv"

# Save file
np.savetxt(store_folder+store_file, samples, delimiter=",")