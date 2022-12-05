#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Implementation of the method proposed in the paper:
 "Robust Control for Dynamical Systems with Non-Gaussian Noise via
  Formal Abstractions"

Originally coded by:        Thom Badings
Contact e-mail address:     thom.badings@ru.nl
Git repository: https://gitlab.science.ru.nl/tbadings/sample-abstract
______________________________________________________________________________
"""

# %run "~/documents/sample-abstract/RunFile.py"

# Load general packages
from datetime import datetime   # Import Datetime to get current date/time
import pandas as pd             # Import Pandas to store data in frames
import numpy as np              # Import Numpy for computations
import os
import sys
import matplotlib as mpl
from inspect import getmembers, isclass # To get list of all available models
import importlib

# Load main classes and methods
from core.abstraction_default import abstraction_default
from core.abstraction_epistemic import abstraction_epistemic
from core.monte_carlo import MonteCarloSim
from core.commons import createDirectory
from core.export import result_exporter, pickle_results
from core.define_partition import state2region

from core.preprocessing.master_classes import settings
from core.preprocessing.argument_parser import parse_arguments

from plotting.harmonic_oscillator import oscillator_experiment
from plotting.anaesthesia_delivery import heatmap_3D

np.random.seed(1)
mpl.rcParams['figure.dpi'] = 300

#-----------------------------------------------------------------------------
# Parse arguments
#-----------------------------------------------------------------------------

args = parse_arguments()
args.base_dir = os.path.dirname(os.path.abspath(__file__))
print('Base directory:', args.base_dir)

print('Run using arguments:')
for key,val in vars(args).items():
    print(' - `'+str(key)+'`: '+str(val))

with open(os.path.join(args.base_dir, 'path_to_prism.txt')) as f:
    args.prism_folder = str(f.readlines()[0])
    print('-- Path to PRISM is:', args.prism_folder)

# Abort if both the improved synthesis scheme and unbounded property is set
if args.improved_synthesis and args.timebound == np.inf:
    sys.exit("Cannot run script with both improved synthesis scheme and unbounded property")

# Abort if both the improved synthesis scheme and monte carlo simulations is set
if args.monte_carlo_iter and args.timebound == np.inf:
    sys.exit("Cannot run script with both Monte Carlo simulations and unbounded property")

#-----------------------------------------------------------------------------
# Load model and set specification
#-----------------------------------------------------------------------------

# Define model
print('Run `'+args.model+'` model from set `'+args.model_file+'`...')

# Retreive a list of all available models
models = importlib.import_module("models."+args.model_file, package=None)
modelClasses = np.array(getmembers(models, isclass))
names   = modelClasses[:,0]
methods = modelClasses[:,1]

# Check if the input model name is also present in the list
if sum(names == args.model) > 0:
    # Get method corresponding to this model
    method = methods[names == args.model][0]
    model = method(args)
    spec = model.set_spec()
else:
    sys.exit("There does not exist a model with name `"+str(args.model)+"`.")

#-----------------------------------------------------------------------------
# Load default settings and overwrite by options provided by the user
#-----------------------------------------------------------------------------

# Create settings object
setup = settings(application=model.name, base_dir = args.base_dir)

print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n' +
      'PROGRAM STARTED AT \n'+setup.time['datetime'] +
      '\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

# Create the main object for the current instance
if args.abstraction_type == 'default':
    method = abstraction_default
elif args.abstraction_type == 'epistemic':
    method = abstraction_epistemic
else:
    sys.exit('ERROR: Abstraction type `'+args.abstraction_type+'` not valid')
Ab = method(args, setup, model, spec)

# Define states of abstraction
Ab.define_states()

# Initialize results dictionaries
Ab.initialize_results()

#-----------------------------------------------------------------------------
# Define actions (only required once outside iterative scheme)
#-----------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------
# Code below is repeated every iteration of the iterative scheme
#-----------------------------------------------------------------------------

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
    Ab.setup.prepare_iteration(N, case_id)

    # Export the results of the current iteration
    writer = exporter.create_writer(Ab, N)

    done = False

    while not done:
        if args.improved_synthesis:
            print('\nBLOCK REFINEMENT - TIME STEP k =',Ab.blref.k)
            case_string = str(case_id) + 'k=' + str(Ab.blref.k)
            print('-- Number of value states used:',Ab.blref.num_lb_used,'\n')
        else:
            case_string = str(case_id)

        # Calculate transition probabilities
        Ab.define_probabilities()
        
        # Build and solve interval MDP
        model_size = Ab.build_iMDP()
        Ab.solve_iMDP()

        # Store run times of current iterations        
        time_df = pd.DataFrame( data=Ab.time, index=[case_string] )
        time_df.to_excel(writer, sheet_name='Run time')

        exporter.add_results(Ab, model_size, case_string)

        exporter.add_to_df(pd.DataFrame(data=N, index=[case_string], columns=['N']), 
                            'general')
        exporter.add_to_df(time_df, 'run_times')
        exporter.add_to_df(pd.DataFrame(data=model_size, index=[case_string]), 
                            'model_size')

        # Check if we are done yet
        if not args.improved_synthesis or Ab.blref.decrease_time():
            done = True

            if Ab.args.monte_carlo_iter > 0:
                if len(args.x_init) == Ab.model.n:
                    s_init = state2region(args.x_init, Ab.spec.partition, Ab.partition['R']['c_tuple'])

                    Ab.mc = MonteCarloSim(Ab, iterations=Ab.args.monte_carlo_iter,
                                        writer=writer, init_states = s_init)
                else:
                    Ab.mc = MonteCarloSim(Ab, iterations=Ab.args.monte_carlo_iter,
                                        writer=writer)

            writer.save()
        else:
            Ab.blref.set_values(Ab.results['optimal_reward'])

    pickle_results(Ab)

    if harm_osc:
        print('-- Monte Carlo simulations to determine controller safety...')
        harm_osc.add_iteration(Ab, case_id)
        
# Save overall data in Excel (for all iterations combined)   
exporter.save_to_excel(Ab.setup.directories['outputF'] + \
    Ab.setup.time['datetime'] + '_iterative_results.xlsx')

print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
print('APPLICATION FINISHED AT', datetime.now().strftime("%m-%d-%Y %H-%M-%S"))

#-----------------------------------------------------------------------------
# Create plots
#-----------------------------------------------------------------------------

# Plots for AAAI 2023 paper
if harm_osc:
    print('-- Export results for longitudinal drone dynamics as in paper...')
    
    harm_osc.export(Ab)

    harm_osc.plot_trace(Ab, spring=Ab.model.spring_nom, 
                        state_center=(-6.5,-2.5))
    
if Ab.model.name == 'anaesthesia_delivery':

    centers = Ab.partition['R']['center']
    values = Ab.results['optimal_reward']

    heatmap_3D(Ab.setup, centers, values)

# Plots for JAIR paper / AAAI 2022
if args.plot:
    from RunPlots import plot
    plot(path = Ab.setup.directories['outputF']+'data_dump.p')