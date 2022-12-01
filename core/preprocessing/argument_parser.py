#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from ast import literal_eval

def parse_arguments():
    """
    Function to parse arguments provided

    Returns
    -------
    :args: Dictionary with all arguments

    """
    
    parser = argparse.ArgumentParser(description="Sampling-Based Abstraction Method",
                                     prefix_chars='--')

    ### Scenario problem main arguments
    parser.add_argument('--noise_samples', type=int, action="store", dest='noise_samples', 
                        default=20000, help="Number of noise samples to use")
    
    parser.add_argument('--confidence', type=float, action="store", dest='confidence', 
                        default=1e-8, help="Confidence level on individual transitions")
    
    parser.add_argument('--sample_clustering', type=float, action="store", dest='sample_clustering', 
                        default=1e-2, help="Distance at which to cluster noise samples")
    
    parser.add_argument('--iterations', type=int, action="store", dest='iterations', 
                        default=1, help="Number of repetitions of computing iMDP probability intervals")

    parser.add_argument('--nongaussian_noise', dest='nongaussian_noise', action='store_true',
                        help="If enabled, non-Gaussian noise samples (if available) are used")
    parser.set_defaults(nongaussian_noise=False)

    ### Memory allocation
    # Prism java memory
    parser.add_argument('--prism_java_memory', type=int, action="store", dest='prism_java_memory', 
                        default=1, help="Max. memory usage by JAVA / PRISM")
    
    # Enable/disable improved policy synthesis scheme
    parser.add_argument('--improved_synthesis', dest='improved_synthesis', action='store_true',
                        help="If enabled, the improved policy synthesis scheme is enabled")
    parser.set_defaults(improved_synthesis=False)

    ### Abstraction options
    # File from which to load model
    parser.add_argument('--model_file', type=str, action="store", dest='model_file', 
                        default=False, help="File to load model from", required=True)
    
    # Argument for model to load
    parser.add_argument('--model', type=str, action="store", dest='model', 
                        default=False, help="Model to load", required=True)

    # Type of abstraction to create
    parser.add_argument('--abstraction_type', type=str, action="store", dest='abstraction_type', 
                        default='default', help="Type of abstraction to generate (can be 'default' or 'epistemic')", required=False)
    
    # Number of Monte Carlo simulation iterations
    parser.add_argument('--monte_carlo_iter', type=int, action="store", dest='monte_carlo_iter', 
                        default=0, help="Number of Monte Carlo simulations to perform")

    # Initial state for Monte Carlo simulations
    parser.add_argument('--x_init', type=str, action="store", dest='x_init', 
                        default='[]', help="Initial state for Monte Carlo simulations")
    
    ### Plotting options
    parser.add_argument('--partition_plot', dest='partition_plot', action='store_true',
                        help="If enabled, create plot of state space partition")
    parser.set_defaults(partition_plot=False)

    parser.add_argument('--plot', dest='plot', action='store_true',
                        help="If enabled, plots are created after finishing the programme")
    parser.set_defaults(plot=False)
    
    ### Verbose switch
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="If enabled, provide more detailed outputs of script")
    parser.set_defaults(verbose=False)

    ####
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

    ####
    ####
    
    #### UAV model arguments ####
    parser.add_argument('--UAV_dim', type=int, action="store", dest='UAV_dim', 
                        default=2, help="Dimension of the UAV model to run")

    parser.add_argument('--noise_factor', type=float, action="store", dest='noise_factor', 
                        default=1, help="Multiplication factor for the process noise (covariance)")
    
    ####
    ####

    parser.add_argument('--mdp_mode', type=str, action="store", dest='mdp_mode', 
                        default='interval', help="Is either `estimate` (MDP) or `interval` (iMDP; default option)")

    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args, unknown = parser.parse_known_args()
    
    if len(unknown) > 0:
        print('\nWarning: There are unknown arguments:\n', unknown,'\n')
    
    args.bld_target_size = literal_eval(args.bld_target_size)
    
    args.x_init = np.array(list(literal_eval(args.x_init)), dtype=float)    

    try:
        args.bld_partition = [int(args.bld_partition)]
    except:
        args.bld_partition = list(literal_eval(args.bld_partition))    
    
    try:
        args.drug_partition = [int(args.drug_partition)]
    except:
        args.drug_partition = list(literal_eval(args.drug_partition))


    return args