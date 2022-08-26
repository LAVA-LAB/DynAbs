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

# Load main classes and methods
from core.abstraction_default import abstraction_default
from core.abstraction_epistemic import abstraction_epistemic
from core.monte_carlo import monte_carlo, MonteCarloSim
from core.preprocessing.set_model_class import set_model_class
from core.commons import createDirectory
from core.export import result_exporter, pickle_results

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

args.model = 'shuttle'
#args.UAV_dim = 2
#args.noise_factor = 1
args.noise_samples = 1600
args.confidence = 0.01
args.prism_java_memory = 8
# args.nongaussian_noise = False
args.monte_carlo_iter = 100

with open(os.path.join(args.base_dir, 'path_to_prism.txt')) as f:
    args.prism_folder = str(f.readlines()[0])
    print('-- Path to PRISM is:', args.prism_folder)

#-----------------------------------------------------------------------------
# Create model and set specification
#-----------------------------------------------------------------------------

# Define model
print('Run `'+args.model+'` model...')
method = set_model_class(args.model)
model  = method(args)

# Define specification
spec = model.set_spec()

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

# %%

#-----------------------------------------------------------------------------
# Define actions (only required once outside iterative scheme)
#-----------------------------------------------------------------------------

# Create directories
createDirectory(Ab.setup.directories['outputF']) 
        
# Create actions and determine which ones are enabled
Ab.define_target_points()
Ab.define_enabled_actions()

# %%
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
        
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    print('START ITERATION ID:', case_id)

    # Set directories for this iteration
    Ab.setup.prepare_iteration(N, case_id)

    # Calculate transition probabilities
    Ab.define_probabilities()
    
    # Build and solve interval MDP
    model_size = Ab.build_iMDP()
    Ab.solve_iMDP()
    
    # Export the results of the current iteration
    writer = exporter.create_writer(Ab, model_size, case_id, N)
    
    if Ab.args.monte_carlo_iter > 0:
        Ab.mc = MonteCarloSim(Ab, iterations=Ab.args.monte_carlo_iter,
                              writer=writer)
    
    # Store run times of current iterations        
    time_df = pd.DataFrame( data=Ab.time, index=[case_id] )
    time_df.to_excel(writer, sheet_name='Run time')
    writer.save()
    plt.close('all')
    
    exporter.add_to_df(pd.DataFrame(data=N, index=[case_id], columns=['N']), 
                       'general')
    exporter.add_to_df(time_df, 'run_times')
    exporter.add_to_df(pd.DataFrame(data=model_size, index=[case_id]), 
                       'model_size')

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
    
# %%

import pickle

infile = open(Ab.setup.directories['outputF']+'data_dump.p','rb')
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
