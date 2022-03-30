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

Specifically, this script generates the tables of probability intervals, used
for computing the transition probabilities of our sampling-based abstractions.
______________________________________________________________________________
"""

import numpy as np
import pandas as pd

from scipy.stats import beta as betaF

def createUniformSamples(N, low=-1, upp=1):
    
    if N > 1:
        rands = low + (upp-low)*np.random.rand(N)
    else:
        rands = low + (upp-low)*np.random.rand()
    
    return rands

def computeBetaPPF(N, k, d, beta):
    
    epsilon = betaF.ppf(beta, k+d, N-(d+k)+1)
    
    return epsilon

def computeBetaCDF(N, k, d, epsilon):
    
    cum_prob = betaF.cdf(epsilon, k+d, N-(d+k)+1)
    
    return cum_prob

def validate_eps(trials, N, beta, d, krange, eps_low, eps_upp):
    
    correct_eps_sum_low     = np.zeros(len(krange))
    correct_eps_sum_upp     = np.zeros(len(krange))
    correct_eps_sum_both    = np.zeros(len(krange))
    correct_all             = 0
    
    for tr in range(trials):
    
        if tr % 1000 == 0:
            print('Trial number',tr)
        
        fac = 1e-6
        
        width = trial_SAD(N, beta, d, krange, fac)
    
        # Validate computed epsilon
        V_prob = np.zeros(len(width))
        for i,w in enumerate(width):
            V_prob[i] = 1 - w  
    
        # Check if bounds are correct
        correct_eps_sum_low += V_prob > eps_low
        correct_eps_sum_upp += V_prob < eps_upp
        correct_eps_sum_both += (V_prob > eps_low) * (V_prob < eps_upp)
        
        correct_all += all(V_prob > eps_low) and all(V_prob < eps_upp)
    
        beta_empirical_low      = correct_eps_sum_low / trials
        beta_empirical_upp      = correct_eps_sum_upp / trials
        beta_empirical_both     = correct_eps_sum_both / trials
        
        beta_overall            = correct_all / trials
        
    print('Beta empirical low:',    beta_empirical_low)
    print('Beta empirical upp:',    beta_empirical_upp)
    print('Beta empirical both:',   beta_empirical_both)
    
    print('Overall confidence level:', beta_overall,'(expected was: '+str(1-beta)+')')

def trial_SAD(N, beta, d, krange, fac):
      
    # Create interval for N samples
    samples = createUniformSamples(N, 0,1)
    
    # Create interval for N samples
    samples_sort = np.sort(samples)
    
    # For every value of samples to discard
    width = np.array([ np.max(samples_sort[:int(N-k)]) for i,k in enumerate(krange)])
    
    return width

def tabulate(N, eps_low, eps_upp, kstep, krange):
    
    P_low = np.zeros(N+1)
    P_upp = np.zeros(N+1)
    
    for k in range(0,N+1):
        
        # If k > N-kstep, we need to discard all samples to get a lower bound
        # probability, so the result is zero. The upper bound probability is
        # then given by removing exactly N-kstep samples.
        if k > np.max(krange):
            P_low[k] = 0
            P_upp[k] = 1 - eps_low[-1]
        
        else:  
            
            # Determine the index of the upper bound violation probability to
            # use for computing the lower bound probability.
            id_in = (k-1) // kstep + 1
            
            # Lower bound probability is 1 minus the upper bound violation
            P_low[k] = 1 - eps_upp[id_in]
            
            # If no samples are discarded, even for computing the lower bound
            # probability, we can only conclude an upper bound of one
            if k == 0:
                P_upp[k] = 1
            else:
                
                # Else, the upper bound probability is given by the lower 
                # bound probability, for discarding "kstep samples less" than
                # for the lower bound probability
                id_out = id_in - 1
                P_upp[k] = 1 - eps_low[id_out]
                
    return P_low, P_upp

#####

# Experiment settings
beta_list = [0.01/(1000**2)]     # Confidence level
d = 1                       # Nr. of decision variables (always 1 for us)

# Number of trials to validate obtained guarantees
trials = 0

# List of kstep sizes (batches in which to discard)
kstep_list_all = np.array([1])

# List of number of samples
N_list = np.array([3200])

P_low = {}
P_upp = {}

EXPORT = True

for beta in beta_list:

    print('\nStart for overall confidence level beta=',beta)    

    P_low[beta] = {}
    P_upp[beta] = {}    

    for n,N in enumerate(N_list):
    
        P_low[beta][N] = {}
        P_upp[beta][N] = {}    
    
        print('\nStart for sample size N='+str(N))    
    
        # Remove valuesthat are above the number of samples
        kstep_list = np.array(kstep_list_all)[ np.array(kstep_list_all) < N]
    
        for q, kstep in enumerate(kstep_list):
            
            krange = np.arange(0, N, kstep)
            
            eps_low                 = np.zeros(len(krange))
            eps_upp                 = np.zeros(len(krange))
            
            beta_bar = beta / (2*len(krange))
            
            # Compute violation level (epsilon)
            for i,k in enumerate(krange):
                # Compute violation levels for a specific level of k (nr of
                # discarded constraints)
                eps_low[i] = computeBetaPPF(N, k, d, beta_bar)
                eps_upp[i] = computeBetaPPF(N, k, d, 1 - beta_bar)
            
            P_low[beta][N][kstep], P_upp[beta][N][kstep] = tabulate(N, eps_low, eps_upp, kstep, krange)
            
            if trials > 0:
                validate_eps(trials, N, beta, d, krange, eps_low, eps_upp)
        
        filename = 'input/SaD_probabilityTable_N='+str(N)+'_beta='+str(beta)+'.csv'
        
        if EXPORT:
            df = pd.DataFrame(np.vstack((P_low[beta][N][1], P_upp[beta][N][1])).T,
                              index=np.arange(N+1),
                              columns=['P_low','P_upp']
                              )
            df.index.name = 'N_out'
            
            df.to_csv(filename, sep=',')
            
            print('Exported table to *.csv file')