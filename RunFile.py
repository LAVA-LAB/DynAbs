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

# %run "~/documents/sample-abstract/RunFile.py"

# Load general packages
from datetime import datetime   # Import Datetime to get current date/time
import pandas as pd             # Import Pandas to store data in frames
import numpy as np              # Import Numpy for computations
import matplotlib.pyplot as plt # Import Pyplot to generate plots
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

preset = 'spacecraft_2D'
models_file = 'JAIR22_models'

if preset == 'uav_2D':
    args.model = 'UAV'
    args.UAV_dim = 2
    args.noise_samples = 3200
    args.confidence = 0.01
    args.prism_java_memory = 8
    args.monte_carlo_iter = 10

elif preset == 'uav_3D':
    args.model = 'UAV'
    args.UAV_dim = 3
    args.noise_factor = 1
    args.noise_samples = 1600
    args.confidence = 0.01
    args.prism_java_memory = 8
    args.nongaussian_noise = True
    args.monte_carlo_iter = 100
    args.x_init = [-14,0,6,0,-2,0]
    
elif preset == 'spacecraft_2D':
    args.model = 'spacecraft_2D'
    args.noise_samples = 3200
    args.confidence = 0.01
    args.prism_java_memory = 8
    args.monte_carlo_iter = 1000
    args.x_init = np.array([1.2, 19.9, 0, 0]) #, 0, 0])
    
elif preset == 'spacecraft_3D':
    args.model = 'spacecraft'
    args.noise_samples = 3200
    args.confidence = 0.01
    args.prism_java_memory = 32
    args.monte_carlo_iter = 1000
    args.x_init = np.array([0.8, 16, 0, 0, 0, 0])
    
args.block_refinement = False

print(vars(args))

with open(os.path.join(args.base_dir, 'path_to_prism.txt')) as f:
    args.prism_folder = str(f.readlines()[0])
    print('-- Path to PRISM is:', args.prism_folder)

#-----------------------------------------------------------------------------
# Load model and set specification
#-----------------------------------------------------------------------------

# Define model
print('Run `'+args.model+'` model...')

# Retreive a list of all available models
models = importlib.import_module("models."+models_file, package=None)
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

# %%

#-----------------------------------------------------------------------------
# Define actions (only required once outside iterative scheme)
#-----------------------------------------------------------------------------

# Create directories
createDirectory(Ab.setup.directories['outputF']) 
        
# Create target points associated with actions
Ab.define_target_points()

# Determine enabled state-action paris
Ab.define_enabled_actions()

# %%

from core.block_refinement import block_refinement
if args.block_refinement:
    Ab.blref = block_refinement(100, Ab.partition['goal'], 
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

done = False

# For every iteration... (or once, if iterations are disabled)
for case_id in range(0, Ab.args.iterations):

    # Set directories for this iteration
    Ab.setup.prepare_iteration(N, case_id)

    # Export the results of the current iteration
    writer = exporter.create_writer(Ab, N)

    while not done:

        if args.block_refinement:
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

        if not args.block_refinement or Ab.blref.decrease_time():
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

######################

if harm_osc:
    print('-- Export results for longitudinal drone dynamics as in paper...')
    
    harm_osc.export(Ab)

    harm_osc.plot_trace(Ab, spring=Ab.model.spring_nom, 
                        state_center=(-6.5,-2.5))
    
if Ab.model.name == 'anaesthesia_delivery':

    centers = Ab.partition['R']['center']
    values = Ab.results['optimal_reward']

    heatmap_3D(Ab.setup, centers, values)
    
assert False
# %%

import pickle

infile = open(Ab.setup.directories['outputF']+'data_dump.p','rb')
# infile = open('/home/thom/documents/sample-abstract/output/Ab_spacecraft_2D_08-29-2022_15-08-09/data_dump.p', 'rb')
data = pickle.load(infile)
infile.close()

from plotting.createPlots import reachability_plot
if 'mc' in data:
    print('Create plot with Monte Carlo results')
    reachability_plot(data['setup'], data['results'], data['mc'])
else:
    reachability_plot(data['setup'], data['results'])

from plotting.createPlots import heatmap_3D_view
heatmap_3D_view(data['model'], data['setup'], data['spec'], data['regions']['center'], data['results'])

from plotting.createPlots import heatmap_2D
heatmap_2D(data['args'], data['model'], data['setup'], data['regions']['c_tuple'], data['spec'], data['results']['optimal_reward'])

from plotting.uav_plots import UAV_plot_2D, UAV_3D_plotLayout
from core.define_partition import state2region

if data['model'].name == 'shuttle':

    if len(data['args'].x_init) == data['model'].n:
        s_init = state2region(data['args'].x_init, data['spec'].partition, data['regions']['c_tuple'])[0]
        traces = data['mc'].traces[s_init]

        UAV_plot_2D((0,1), (2,3), data['setup'], data['args'], data['regions'], data['goal_regions'], data['critical_regions'], 
                    data['spec'], traces, cut_idx = [0,0], traces_to_plot=10, line=True)
    else:
        print('-- No initial state provided')

if data['model'].name == 'UAV' and data['model'].modelDim == 3:

    if len(data['args'].x_init) == data['model'].n:
        s_init = state2region(data['args'].x_init, data['spec'].partition, data['regions']['c_tuple'])[0]
        traces = data['mc'].traces[s_init]
        
        UAV_3D_plotLayout(data['setup'], data['args'], data['model'], data['regions'], 
                          data['goal_regions'], data['critical_regions'], traces, data['spec'])
    else:
        print('-- No initial state provided')