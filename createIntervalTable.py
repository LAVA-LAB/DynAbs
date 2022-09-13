#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script belongs to the paper with the title:
    
 "Probabilities Are Not Enough: Formal Controller Synthesis for Stochastic 
  Dynamical Models with Epistemic Uncertainty"

Originally coded by:        <anonymized>
Contact e-mail address:     <anonymized>

Specifically, this script generates the tables of probability intervals, used
for computing the transition probabilities of our sampling-based abstractions.
______________________________________________________________________________
"""

# %run "~/documents/sample-abstract/createIntervalTable.py"

import numpy as np
import pandas as pd

import os
from pathlib import Path

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

def create_table(N, beta, kstep, trials, export=False):

    d = 1
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
                
        # Sanity check to see if the upper bound is actually decreasing with
        # the number of discarded constraints
        if k > 0:
            if P_upp[k] > P_upp[k-1]:
                print('-- Fix issue in P_upp['+str(k)+']')
                P_upp[k] = P_upp[k-1]
                
    # Due to numerical issues, P_low for N_out=0 can be incorrect. Check if 
    # this is the case, and change accordingly.
    if P_low[0] < P_low[1]:
        print('-- Fix numerical error in P_low[0]')
        P_low[0] = 1 - P_upp[N]
                
    if trials > 0:
        validate_eps(trials, N, beta, d, krange, eps_low, eps_upp)

    if export:
        filename = 'input/SaD_probabilityTable_N='+str(N)+'_beta='+str(beta)+'.csv'

        cwd = os.path.dirname(os.path.abspath(__file__))
        root_dir = Path(cwd)

        filepath = Path(root_dir, filename)
        
        print(P_low)
        print(P_upp)

        df = pd.DataFrame(np.column_stack((P_low, P_upp)),
                        index=np.arange(N+1),
                        columns=['P_low','P_upp']
                        )
        df.index.name = 'N_out'
        
        df.to_csv(filepath, sep=',')
        
        print('exported table to *.csv file')

    return P_low, P_upp

#####

if __name__ == '__main__':

    cwd = os.path.dirname(os.path.abspath(__file__))
    root_dir = Path(cwd)
    print('Root directory:', root_dir)
    
    #####
    
    # Experiment settings
    beta_list = [1e-3, 1e-6, 1e-9]     # Confidence level
    d = 1                  # Nr. of decision variables (always 1 for us)
    
    # Number of trials to validate obtained guarantees
    trials = 0
    
    # List of kstep sizes (batches in which to discard)
    kstep_list_all = np.array([1])
    
    # List of number of samples
    N_list = np.array([25,50,100,200,400,800,1600,3200,6400,12800])
    
    P_low = {}
    P_upp = {}
    
    export = False
    
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
                
                P_low[beta][N][kstep], P_upp[beta][N][kstep] = \
                    create_table(N, beta, kstep, trials, export)
                
    # %%
    
    # Create plot for probability intervals over increasing values of N
    print('\n Create plot for intervals over N, for multiple values of beta...')
    
    probabilities_lb    = dict()
    probabilities_ub    = dict()
    average_lb          = dict()
    average_ub          = dict()
    std_lb              = dict()
    std_ub              = dict()
    
    hoeff_lb            = dict()
    hoeff_ub            = dict()
    hoeff_avg_lb        = dict()
    hoeff_avg_ub        = dict()
    hoeff_std_lb        = dict()
    hoeff_std_ub        = dict()
    
    # Create figure as in Appendix of paper
    reps = 100000
    true_prob = 0.1
    
    kstep = 1
    
    for beta in beta_list:
        
        print('For beta:', beta)
        
        probabilities_lb[beta]  = {}
        probabilities_ub[beta]  = {}
        average_lb[beta]        = np.zeros(len(N_list))
        average_ub[beta]        = np.zeros(len(N_list))
        std_lb[beta]            = np.zeros(len(N_list))
        std_ub[beta]            = np.zeros(len(N_list))
        
        hoeff_lb[beta]          = {}
        hoeff_ub[beta]          = {}
        hoeff_avg_lb[beta]      = np.zeros(len(N_list))
        hoeff_avg_ub[beta]      = np.zeros(len(N_list))
        hoeff_std_lb[beta]      = np.zeros(len(N_list))
        hoeff_std_ub[beta]      = np.zeros(len(N_list))
        
        for n,N in enumerate(N_list):
            
            print('For N:', N)
            
            probabilities_lb[beta][N] = np.zeros(reps)
            probabilities_ub[beta][N] = np.zeros(reps)
            
            hoeff_lb[beta][N] = np.zeros(reps)
            hoeff_ub[beta][N] = np.zeros(reps)
            
            hoeff_eps = np.sqrt( np.log(beta / 2) / (-2 * N) )
            
            for i in range(reps):
            
                # Create samples
                samples = np.random.rand(N)
                
                # Compute N_j^out
                N_out = np.sum(samples > true_prob)
                N_in  = np.sum(samples <= true_prob)
                
                hoeff_lb[beta][N][i] = np.maximum(N_in/N - hoeff_eps, 0)
                hoeff_ub[beta][N][i] = np.minimum(N_in/N + hoeff_eps, 1)
                
                # Look-up probability interval
                probabilities_lb[beta][N][i] = P_low[beta][N][kstep][N_out]
                probabilities_ub[beta][N][i] = P_upp[beta][N][kstep][N_out]
    
            average_lb[beta][n] = np.mean(probabilities_lb[beta][N])
            average_ub[beta][n] = np.mean(probabilities_ub[beta][N])
            
            std_lb[beta][n] = np.std(probabilities_lb[beta][N])
            std_ub[beta][n] = np.std(probabilities_ub[beta][N])
            
            hoeff_avg_lb[beta][n] = np.mean(hoeff_lb[beta][N])
            hoeff_avg_ub[beta][n] = np.mean(hoeff_ub[beta][N])
            
            hoeff_std_lb[beta][n] = np.std(hoeff_lb[beta][N])
            hoeff_std_ub[beta][n] = np.std(hoeff_ub[beta][N])
            
    # %%
    
    # Create plot for intervals over N, for multiple confidence levels
    
    import matplotlib.pyplot as plt
    
    def cm2inch(*tupl):
        '''
        Convert centimeters to inches
        '''
        
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)
    
    # Plot font family and size
    plt.rc('font', family='serif')
    SMALL_SIZE = 7
    MEDIUM_SIZE = 9
    BIGGER_SIZE = 9
    
    fig, ax1 = plt.subplots(1, 1, figsize=cm2inch(14,9))  #(9,6))
    
    x = list(N_list)
    
    lw = 1.4
    
    b0 = beta_list[0]
    b1 = beta_list[2]
    
    ax1.errorbar(N_list, average_lb[b0], yerr=std_lb[b0], capsize=2, linewidth=lw, label=r'Scenario $\beta = '+str(b0)+'$', color='grey')
    ax1.errorbar(N_list, average_ub[b0], yerr=std_ub[b0], capsize=2, linewidth=lw, color='grey')
    
    ax1.errorbar(N_list, hoeff_avg_lb[b0], yerr=hoeff_std_lb[b0], capsize=2, linewidth=lw, label=r'Hoeffding $\beta = '+str(b0)+'$', color='blue')
    ax1.errorbar(N_list, hoeff_avg_ub[b0], yerr=hoeff_std_ub[b0], capsize=2, linewidth=lw, color='blue')
    
    ax1.errorbar(N_list, average_lb[b1], yerr=std_lb[b1], capsize=2, linewidth=lw, label=r'Scenario $\beta = '+str(b1)+'$', color='k')
    ax1.errorbar(N_list, average_ub[b1], yerr=std_ub[b1], capsize=2, linewidth=lw, color='k')
    
    ax1.errorbar(N_list, hoeff_avg_lb[b1], yerr=hoeff_std_lb[b1], capsize=2, linewidth=lw, label=r'Hoeffding $\beta = '+str(b1)+'$', color='green')
    ax1.errorbar(N_list, hoeff_avg_ub[b1], yerr=hoeff_std_ub[b1], capsize=2, linewidth=lw, color='green')
    
    ax1.plot(N_list, np.repeat(true_prob, len(N_list)), 'r--', linewidth=lw, label='True prob.')
    # ax1.set_ylim(-0.1, 0.7)
    ax1.set_xscale('log')
    
    ax1.legend(loc='upper right')
        
    ax1.set_xlabel('Number of samples (N)')
    ax1.set_ylabel('Probability')
    
    ax1.grid(which='both', axis='y', linestyle='dotted')
    
    # General styling
    fig.tight_layout()
    
    plt.show()
    plt.pause(0.001)
    
    filename = 'probabilityBoundsOverSampleSize'
    fig.savefig(filename+'.pdf', bbox_inches='tight')
    fig.savefig(filename+'.png', bbox_inches='tight')
    
    ###
    
    # export plot data
    DIC = {
           str(b0)+'_sc_low': average_lb[b0],
           str(b0)+'_sc_low_e': std_lb[b0],
           #
           str(b1)+'_sc_low': average_lb[b1],
           str(b1)+'_sc_low_e': std_lb[b1],
           #
           str(b0)+'_sc_upp': average_ub[b0],
           str(b0)+'_sc_upp_e': std_ub[b0],
           #
           str(b1)+'_sc_upp': average_ub[b1],
           str(b1)+'_sc_upp_e': std_ub[b1],
           ###
           str(b0)+'_hf_low': hoeff_avg_lb[b0],
           str(b0)+'_hf_low_e': hoeff_std_lb[b0],
           #
           str(b1)+'_hf_low': hoeff_avg_lb[b1],
           str(b1)+'_hf_low_e': hoeff_std_lb[b1],
           #
           str(b0)+'_hf_upp': hoeff_avg_ub[b0],
           str(b0)+'_hf_upp_e': hoeff_std_ub[b0],
           #
           str(b1)+'_hf_upp': hoeff_avg_ub[b1],
           str(b1)+'_hf_upp_e': hoeff_std_ub[b1],
           }
    
    DF = pd.DataFrame(DIC, index=N_list)
    DF.index.name = 'N'
    
    DF.to_csv(Path(root_dir, 'nr_samples_vs_intervals.csv'), sep=',')
    
    # %%
    
    # Create plot for intervals over the true probability
    print('\n Create plot for intervals over increasing true probability...')
    
    b0 = beta_list[2]
    # b1 = beta_list[2]
    
    N0 = 800
    k_step = 1
    
    hoeff_eps_b0 = np.sqrt( 1/(2*N0) * np.log(2/b0) )
    varyN_hf_lb_b0 = np.maximum(0, [(N0-n_in) / N0 - hoeff_eps_b0 for n_in in range(0, N0+1) ])
    varyN_hf_ub_b0 = np.minimum(1, [(N0-n_in) / N0 + hoeff_eps_b0 for n_in in range(0, N0+1) ])
    
    # hoeff_eps_b1 = np.sqrt( 1/(2*N0) * np.log(2/b1) )
    # varyN_hf_lb_b1 = np.maximum(0, [(N0-n_in) / N0 - hoeff_eps_b1 for n_in in range(0, N0+1) ])
    # varyN_hf_ub_b1 = np.minimum(1, [(N0-n_in) / N0 + hoeff_eps_b1 for n_in in range(0, N0+1) ])
    
    fig, ax2 = plt.subplots(1, 1, figsize=cm2inch(14,9))
    
    f_range = np.arange(0, N0+1) / N0
    
    ax2.plot(f_range, varyN_hf_lb_b0, linewidth=lw, label=r'Hoeffding $\beta = '+str(b0)+'$', color='blue')
    ax2.plot(f_range, varyN_hf_ub_b0, linewidth=lw, color='blue')
    
    ax2.plot(f_range, P_low[b0][N0][k_step], linewidth=lw, label=r'Scenario $\beta = '+str(b0)+'$', color='grey')
    ax2.plot(f_range, P_upp[b0][N0][k_step], linewidth=lw, color='grey')
    
    # ax2.plot(f_range, varyN_hf_lb_b1, linewidth=lw, label=r'Hoeffding $\beta = '+str(b1)+'$', color='green')
    # ax2.plot(f_range, varyN_hf_ub_b1, linewidth=lw, color='green')
    
    # ax2.plot(f_range, P_low[b1][N0][k_step], linewidth=lw, label=r'Scenario $\beta = '+str(b1)+'$', color='k')
    # ax2.plot(f_range, P_upp[b1][N0][k_step], linewidth=lw, color='k')
    
    
    ax2.plot(f_range, 1 - f_range, 'r--', linewidth=lw, label=r'$N^{out} / N$')
    
    ax2.legend(loc='upper right')
        
    ax2.set_xlabel('Fraction of samples outside region')
    ax2.set_ylabel('Probability')
    
    ax2.grid(which='both', axis='y', linestyle='dotted')
    
    # General styling
    fig.tight_layout()
    
    plt.show()
    plt.pause(0.001)
    
    filename = 'boundsOverViolationFraction'
    fig.savefig(filename+'.pdf', bbox_inches='tight')
    fig.savefig(filename+'.png', bbox_inches='tight')
    
    # export plot data
    DIC = {
           str(b0)+'_sc_low': P_low[b0][N0][k_step],
           # str(b1)+'_sc_low': P_low[b1][N0][k_step],
           #
           str(b0)+'_sc_upp': P_upp[b0][N0][k_step],
           # str(b1)+'_sc_upp': P_upp[b1][N0][k_step],
           ###
            str(b0)+'_hf_low': varyN_hf_lb_b0,
           # str(b1)+'_hf_low': varyN_hf_lb_b1,
           # #
            str(b0)+'_hf_upp': varyN_hf_ub_b0,
           # str(b1)+'_hf_upp': varyN_hf_ub_b1,
           }
    
    DF = pd.DataFrame(DIC, index=np.arange(0, N0+1))
    DF.index.name = 'k'
    
    DF.to_csv(Path(root_dir, 'fraction_vs_intervals.csv'), sep=',')