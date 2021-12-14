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
from ..commons import nchoosek

def discretizeGearsMethod(A_cont, B_cont, W_cont, tau):
    '''
    Perform discretization based on Gears method

    Parameters
    ----------
    A_cont : 2D numpy array
        A matrix of the continuous-time dynamical system.
    B_cont : 2D numpy array
        B matrix of the continuous-time dynamical system.
    W_cont : 2D numpy array
        Deterministic disturbance of the continuous-time dynamical system.
    tau : float
        Discretization time.

    Returns
    -------
    A : 2D numpy array
        A matrix of the discretized system.
    B : 2D numpy array
        B matrix of the discretized system.
    W : 2D numpy array
        Deterministic disturbance of the discretized system.

    '''
    
    # Dimension of the state
    n = len(A_cont)
    
    # Discretize model with respect to time
    alpha, beta0, alphaSum  = gears_order(s=1)
    
    A_bar = np.linalg.inv( np.eye(n) - tau*beta0*A_cont )
    O_bar = tau * beta0 * A_bar
    
    A = A_bar * alphaSum
    B = O_bar @ B_cont
    W = O_bar @ W_cont
    
    return A,B,W

def gears_order(s):
    '''
    Function to calculate gears discretization parameters, 
    based on the order `s` (input)
    
    ''' 
    
    beta0 = 0                   # Define beta0
    for i in range(1,s+1):
        beta0 = beta0 + 1/i     # Add to sum of beta0
    beta0 = beta0**(-1)
    
    alpha = np.zeros(s)
    alphaT = np.zeros(s)
    for i in range(1,s+1):
        alpha[i-1] = (-1)**(i+1)*beta0
        alphaT[i-1] = 0
        for j in range(i,s+1):
            alphaT[i-1] = alphaT[i-1] + j**(-1)*nchoosek(j,i)
        alpha[i-1] = alpha[i-1]*alphaT[i-1]
        
    alphaSum = 0
    for j in range(s):
        alphaSum += alpha[j]
        
    return alpha, beta0, alphaSum