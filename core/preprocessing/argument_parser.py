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

import argparse
from ast import literal_eval

def parse_arguments():
    """
    Function to parse arguments provided

    Parameters
    ----------
    :manualModel: Override model as provided as argument in the command
    :nobisim: Override bisimulatoin option as provided as argument in the command

    Returns
    -------
    :args: Dictionary with all arguments

    """
    
    parser = argparse.ArgumentParser(description="Sampling-Based Abstraction Method",
                                     prefix_chars='--')
    # Scenario problem main arguments
    
    parser.add_argument('--noise_samples', type=int, action="store", dest='noise_samples', 
                        default=20000, help="Number of noise samples to use")
    
    parser.add_argument('--confidence', type=float, action="store", dest='confidence', 
                        default=1e-8, help="Confidence level on individual transitions")
    
    parser.add_argument('--sample_clustering', type=float, action="store", dest='sample_clustering', 
                        default=1e-2, help="Distance at which to cluster noise samples")
    
    parser.add_argument('--prism_java_memory', type=int, action="store", dest='prism_java_memory', 
                        default=1, help="Max. memory usage by JAVA / PRISM")
    
    parser.add_argument('--iterations', type=int, action="store", dest='iterations', 
                        default=1, help="Number of repetitions of computing iMDP probability intervals")
    
    # Argument for model to load
    parser.add_argument('--model', type=str, action="store", dest='model', 
                        default=False, help="Model to load", required=True)
    
    # Number of Monte Carlo simulation iterations
    parser.add_argument('--monte_carlo_iter', type=int, action="store", dest='monte_carlo_iter', 
                        default=0, help="Number of Monte Carlo simulations to perform")
    
    parser.add_argument('--partition_plot', dest='partition_plot', action='store_true',
                        help="If enabled, create plot of state space partition")
    parser.set_defaults(partition_plot=False)
    
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="If enabled, provide more detailed outputs of script")
    parser.set_defaults(verbose=False)
    
    ####
    #### Drone model arguments ####
    parser.add_argument('--drone_spring', dest='drone_spring', action='store_true',
                        help="Enable spring coefficient in drone benchmark")
    parser.set_defaults(drone_spring=False)
    
    parser.add_argument('--drone_par_uncertainty', dest='drone_par_uncertainty', action='store_true',
                        help="Enable parameter uncertainty in drone benchmark")
    parser.set_defaults(drone_par_uncertainty=False)
    
    parser.add_argument('--drone_mc_step', type=int, action="store", dest='drone_mc_step', 
                        default=0.2, help="Steps (factor) at which to increase parameter deviation from nominal value")
    
    parser.add_argument('--drone_mc_iter', type=int, action="store", dest='drone_mc_iter', 
                        default=100, help="Monte Carlo simulations to evaluate controller safety")
    
    ####
    #### Building temperature model arguments ####
    parser.add_argument('--bld_partition', type=str, action="store", dest='bld_partition', 
                        default='[25,35]', help="Size of the state space partition")
    
    parser.add_argument('--bld_target_size', type=str, action="store", dest='bld_target_size', 
                        default='[[-.1, .1], [-.3, .3]]', help="Size of the target sets used")
    
    parser.add_argument('--bld_par_uncertainty', dest='bld_par_uncertainty', action='store_true',
                        help="Enable parameter uncertainty in temperature control benchmark")
    parser.set_defaults(bld_par_uncertainty=False)
    
    ####
    ####
    #### Anaesthesia delivery model arguments ####
    parser.add_argument('--drug_partition', type=str, action="store", dest='drug_partition', 
                        default='[20,20,20]', help="Size of the state space partition")
    
    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args, unknown = parser.parse_known_args()
    
    args.mdp_mode = 'interval'
    
    if len(unknown) > 0:
        print('\nWarning: There are unknown arguments:\n', unknown,'\n')
    
    args.bld_target_size = literal_eval(args.bld_target_size)
    
    try:
        args.bld_partition = [int(args.bld_partition)]
    except:
        args.bld_partition = list(literal_eval(args.bld_partition))    
    
    try:
        args.drug_partition = [int(args.drug_partition)]
    except:
        args.drug_partition = list(literal_eval(args.drug_partition))


    return args