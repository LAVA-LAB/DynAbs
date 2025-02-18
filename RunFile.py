#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

DynAbs: A Python tool for abstraction-based controller synthesis for stochastic dynamical systems
For more information about the papers forming the foundations of this tool, see the ReadMe or the git repository below.

Originally coded by:        Thom Badings
Contact e-mail address:     thombadings@gmail.com
Git repository:             https://github.com/LAVA-LAB/DynAbs
Latest version:             February 18, 2025
______________________________________________________________________________
"""

# Load general packages
from datetime import datetime  # Import Datetime to get current date/time
import pandas as pd  # Import Pandas to store data in frames
import numpy as np  # Import Numpy for computations
import os
import sys
import matplotlib as mpl
from inspect import getmembers, isclass  # To get list of all available models
import importlib
import pathlib
import json
import random
from pathlib import Path
from fnmatch import fnmatch

# Load main classes and methods
from core.abstraction_default import abstraction_default
from core.abstraction_parameter import abstraction_parameter
from core.monte_carlo import MonteCarloSim
from core.pick2learn import P2L, bound_risk
from core.commons import createDirectory
from core.export import result_exporter, pickle_results
from core.define_partition import state2region

from core.preprocessing.master_classes import settings
from core.preprocessing.argument_parser import parse_arguments

from plotting.harmonic_oscillator import oscillator_experiment
from plotting.anaesthesia_delivery import heatmap_3D

mpl.rcParams['figure.dpi'] = 300

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------

args = parse_arguments()
args.base_dir = os.path.dirname(os.path.abspath(__file__))
print('Base directory:', args.base_dir)

# Set random seeds
np.random.seed(args.seed)
random.seed(args.seed)

if not pathlib.Path(args.prism_executable).is_file():
    raise Exception(f"Could not find the prism executable. Please check if the following path to the executable is correct: {str(args.prism_executable)}")

print('Run using arguments:')
for key, val in vars(args).items():
    print(' - `' + str(key) + '`: ' + str(val))

# Abort if both the improved synthesis scheme and unbounded property is set
if args.improved_synthesis and args.timebound == np.inf:
    sys.exit("Cannot run script with both improved synthesis scheme and unbounded property")

# Abort if both the improved synthesis scheme and monte carlo simulations is set
if args.monte_carlo_iter and args.timebound == np.inf:
    sys.exit("Cannot run script with both Monte Carlo simulations and unbounded property")

# -----------------------------------------------------------------------------
# Load model and set specification
# -----------------------------------------------------------------------------

# Define model
print('Run `' + args.model + '` model from set `' + args.model_file + '`...')

# Retreive a list of all available models
models = importlib.import_module("models." + args.model_file, package=None)
modelClasses = np.array(getmembers(models, isclass))
names = modelClasses[:, 0]
methods = modelClasses[:, 1]

# Check if the input model name is also present in the list
if sum(names == args.model) > 0:
    # Get method corresponding to this model
    method = methods[names == args.model][0]
    model = method(args)
    spec = model.set_spec()
else:
    sys.exit("There does not exist a model with name `" + str(args.model) + "`.")

# -----------------------------------------------------------------------------
# Load default settings and overwrite by options provided by the user
# -----------------------------------------------------------------------------

# Create settings object
setup = settings(application=model.name, base_dir=args.base_dir)

print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n' +
      'PROGRAM STARTED AT \n' + setup.time['datetime'] +
      '\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

# Create the main object for the current instance
if args.abstraction_type == 'default':
    method = abstraction_default
elif args.abstraction_type == 'parameter':
    method = abstraction_parameter
else:
    sys.exit('ERROR: Abstraction type `' + args.abstraction_type + '` not valid')
Ab = method(args, setup, model, spec)

# Define states of abstraction
Ab.define_states()

# Initialize results dictionaries
Ab.initialize_results()

# -----------------------------------------------------------------------------
# Define actions (only required once outside iterative scheme)
# -----------------------------------------------------------------------------

# Create directories
createDirectory(Ab.setup.directories['outputF'])

# Create target points associated with actions
Ab.define_target_points()

# Determine enabled state-action paris
Ab.define_enabled_actions()

# Define object for improved synthesis scheme, if this scheme is enabled
from core.improved_synthesis import improved_synthesis

if args.improved_synthesis:
    Ab.blref = improved_synthesis(100, Ab.partition['goal'],
                                  Ab.partition['nr_regions'], Ab.N)

# -----------------------------------------------------------------------------
# Code below is repeated every iteration of the iterative scheme
# -----------------------------------------------------------------------------

# Shortcut to the number of samples
N = args.noise_samples

# Initialize case ID at zero
case_id = 0

# Create empty DataFrames to store iterative results
exporter = result_exporter()

if Ab.model.name == 'drone':
    harm_osc = oscillator_experiment(f_min=0, f_max=2.01,
                                     f_step=args.drone_mc_step,
                                     monte_carlo_iterations=args.drone_mc_iter)
else:
    harm_osc = False

# For every iteration... (or once, if iterations are disabled)
for case_id in range(0, Ab.args.iterations):

    # Set directories for this iteration
    folder = Ab.setup.create_output_directory(N, case_id)

    # Export the results of the current iteration
    exporter.set_folder(folder)
    writer = exporter.create_writer(Ab, N)
    exporter.export_args(Ab.args)

    done = False

    #####
    # Pre-sample from noise distribution
    if Ab.args.nongaussian_noise:
        # Use non-Gaussian noise samples (loaded from external file)
        noise_idxs = np.random.randint(len(Ab.model.noise['samples']), size=(Ab.args.noise_samples, Ab.N))
        noise_samples = Ab.model.noise['samples'][noise_idxs]
    else:
        # Compute Gaussian noise samples
        noise_samples = np.random.multivariate_normal(
            np.zeros(Ab.model.n), Ab.model.noise['w_cov'],
            size=(Ab.args.noise_samples, Ab.N))

    if args.P2L:
        # Initialize pick2learn meta algorithm
        pick2learn = P2L(Ab, noise_samples)
    else:
        # If Pick2Learn is not enabled, still create the object but directly set all samples to be used
        pick2learn = P2L(Ab, noise_samples, enabled=False)

    #####

    while not done:
        if args.improved_synthesis:
            print('\nBLOCK REFINEMENT - TIME STEP k =', Ab.blref.k)
            case_string = str(case_id) + 'k=' + str(Ab.blref.k)
            print(' -- Number of value states used:', Ab.blref.num_lb_used, '\n')
        else:
            case_string = str(case_id)

        # Calculate transition probabilities
        Ab.define_probabilities(pick2learn.T[:, 0, :])  # Only pass noise samples for time k=0

        # Build and solve interval MDP
        model_size = Ab.build_iMDP()
        Ab.solve_iMDP()

        # If an initial state is given, print the optimal reach-avoid prob. in that state
        if len(args.x_init) == Ab.model.n:
            s_init = state2region(args.x_init, Ab.spec.partition, Ab.partition['R']['c_tuple'])
            print(f" > Optimal satisfaction probability computed PRISM: {Ab.results['optimal_reward'][s_init[0]]:.6f} (in state {s_init[0]})")

        # Store run times of current iterations        
        time_df = pd.DataFrame(data=Ab.time, index=[case_string])
        time_df.to_excel(writer, sheet_name='Run time')

        if pick2learn.enabled:
            # Update the pick2learn datasets
            done = pick2learn.update(Ab)

        else:
            if args.improved_synthesis:
                # Store policy for current time step into overall policy
                Ab.blref.append_policy(policy=Ab.results['optimal_policy'])

                # Check if we are done yet
                blref_done = Ab.blref.decrease_time()

                # If not done yet, prepare for next block refinement iteration
                if not blref_done:
                    Ab.blref.set_values(Ab.results['optimal_reward'])
                else:
                    # Load general policy (from block refinement scheme) back into general abstraction object
                    Ab.results['optimal_policy'] = Ab.blref.general_policy

            if not args.improved_synthesis or blref_done:
                done = True

        # Export results
        exporter.add_results(Ab, Ab.blref.general_policy if args.improved_synthesis else Ab.results['optimal_policy'],
                             model_size, case_string)
        exporter.add_to_df(pd.DataFrame(data=N, index=[case_string], columns=['N']),
                           'general')
        exporter.add_to_df(time_df, 'run_times')
        exporter.add_to_df(pd.DataFrame(data=model_size, index=[case_string]),
                           'model_size')

    # Export JSON results file
    expDic = {
        'seed': int(args.seed),
        'abstraction': str('IMDP' if args.mdp_mode == 'interval' else 'MDP'),
        'N': int(args.noise_samples),
        'num_montecarlo': int(Ab.args.monte_carlo_iter),
        'beta_per_interval)': float(np.round(args.confidence, 8)),
        'clopper_pearson': args.clopper_pearson,
    }

    if args.P2L:
        generalization_bound, empirical_bound = pick2learn.compute_bound(Ab)
        expDic['P2L_iter'] = int(pick2learn.iteration)
        expDic['P2L_delta'] = float(np.round(1 - args.P2L_delta, 8))
        expDic['P2L_bound'] = float(np.round(generalization_bound, 8))
        expDic['P2L_montecarlo'] = float(np.round(empirical_bound, 8)),
    else:
        if len(args.x_init) == Ab.model.n:
            # Retrieve bound on satisfaction probability from optimal values
            s_init = state2region(args.x_init, Ab.spec.partition, Ab.partition['R']['c_tuple'])
            generalization_bound = Ab.results['optimal_reward'][s_init[0]]

            # Perform monte carlo simulations
            Ab.mc = MonteCarloSim(Ab, iterations=Ab.args.monte_carlo_iter, writer=writer, init_states=s_init)
            empirical_bound = Ab.mc.results['reachability_probability'][s_init[0]]
            print(f" > Empirical reach-avoid probability (over {Ab.args.monte_carlo_iter} simulations): {empirical_bound:.6f}\n")

            expDic['no-P2L_bound'] = float(np.round(generalization_bound, 8))
            expDic['no-P2L_montecarlo'] = float(np.round(empirical_bound, 8))
        else:
            Ab.mc = MonteCarloSim(Ab, iterations=Ab.args.monte_carlo_iter, writer=writer)

            # If no initial state is specified, there is not a single value we can report
            expDic['no-P2L_bound'] = -1
            expDic['no-P2L_montecarlo'] = -1

    # Export to JSON
    filepath = Path(Ab.setup.directories['outputFcase'], 'out.json')
    with open(filepath, "w") as outfile:
        json.dump(expDic, outfile)

    # Export objects into data file
    pickle_results(Ab)

    if harm_osc:
        print(' -- Monte Carlo simulations to determine controller safety...')
        harm_osc.add_iteration(Ab, case_id)

    writer.close()

# Save overall data in Excel (for all iterations combined)   
exporter.save_to_excel(Path(Ab.setup.directories['outputF'], Ab.setup.time['datetime'] + '_iterative_results.xlsx'))

# -----------------------------------------------------------------------------
# Create plots
# -----------------------------------------------------------------------------

# Plots for AAAI 2023 paper
if harm_osc:
    print(' -- Export results for longitudinal drone dynamics as in paper...')

    harm_osc.export(Ab)

    harm_osc.plot_trace(Ab, spring=Ab.model.spring_nom,
                        state_center=(-6.5, -2.5))

if Ab.model.name == 'anaesthesia_delivery':
    centers = Ab.partition['R']['center']
    values = Ab.results['optimal_reward']

    heatmap_3D(Ab.setup, centers, values)

# Plots for JAIR paper / AAAI 2022
if args.plot:
    from RunPlots import plot

    plot(path=Path(Ab.setup.directories['outputF'], 'data_dump.p'))

# Finally, clean up files if enabled
if args.clean_prism_model:
    extensions = ['*.tra', '*.lab', '*.sta', '*.pctl']
    for path, subdirs, files in os.walk(Ab.setup.directories['outputFcase']):
        for name in files:
            if any([fnmatch(name, extension) for extension in extensions]):
                print(f' - Delete file: {name}')
                os.remove(os.path.join(path, name))

print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
print('APPLICATION FINISHED AT', datetime.now().strftime("%m-%d-%Y %H-%M-%S"))
