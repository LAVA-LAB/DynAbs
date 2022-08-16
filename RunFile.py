#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 ______________________________________
|                                      |
|  SCENARIO-BASED ABSTRACTION PROGRAM  |
|______________________________________|
______________________________________________________________________________
"""

# %run "/home/thom/Documents/aaai23_2/RunFile.py"

# Load general packages
from datetime import datetime   # Import Datetime to get current date/time
import pandas as pd             # Import Pandas to store data in frames
import numpy as np              # Import Numpy for computations
import matplotlib.pyplot as plt # Import Pyplot to generate plots
import os
import matplotlib as mpl

# Load main classes and methods
from core.define_model import define_model
from core.scenarioBasedAbstraction import scenarioBasedAbstraction
from core.monte_carlo import monte_carlo
from core.preprocessing.user_interface import load_PRISM_result_file
from core.preprocessing.set_model_class import set_model_class
from core.commons import createDirectory

from core.preprocessing.master_classes import settings, result_exporter
from core.preprocessing.argument_parser import parse_arguments

from core.postprocessing.harmonic_oscillator import oscillator_experiment

np.random.seed(1)
mpl.rcParams['figure.dpi'] = 300
base_dir = os.path.dirname(os.path.abspath(__file__))
print('Base directory:', base_dir)

#-----------------------------------------------------------------------------
# Parse arguments
#-----------------------------------------------------------------------------

args = parse_arguments()
args.prism_folder = "~/Documents/prism-imc/prism/"

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
setup = settings(application=model.name, base_dir = base_dir)

print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n' +
      'PROGRAM STARTED AT \n'+setup.time['datetime'] +
      '\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

# Create the main object for the current instance
ScAb = scenarioBasedAbstraction(args, setup, define_model(setup, model, spec))
ScAb.define_states()

#-----------------------------------------------------------------------------
# Define actions (only required once outside iterative scheme)
#-----------------------------------------------------------------------------

# Create directories
createDirectory(ScAb.setup.directories['outputF']) 
        
# Create actions and determine which ones are enabled
ScAb.define_target_points()
ScAb.define_actions()

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

if ScAb.model.name == 'oscillator':
    harm_osc = oscillator_experiment(f_min=0, f_max=2.01, f_step=0.2, monte_carlo_iterations=100)
else:
    harm_osc = False

# For every iteration... (or once, if iterations are disabled)
for case_id in range(0, ScAb.args.iterations):   
        
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    print('START ITERATION ID:', case_id)

    # Set directories for this iteration
    ScAb.setup.prepare_iteration(N, case_id)

    # Calculate transition probabilities
    ScAb.define_probabilities()
    
    # Build and solve interval MDP
    model_size = ScAb.build_iMDP()
    ScAb.solve_iMDP()
    
    # Export the results of the current iteration
    writer = exporter.create_writer(ScAb, model_size, case_id, N)
    
    if ScAb.args.monte_carlo_iter > 0:
        ScAb.mc = monte_carlo(ScAb, writer=writer)
    
    # Plot results
    ScAb.generate_probability_plots()
    ScAb.generate_heatmap()
    
    # Store run times of current iterations        
    time_df = pd.DataFrame( data=ScAb.time, index=[case_id] )
    time_df.to_excel(writer, sheet_name='Run time')
    writer.save()
    plt.close('all')
    
    exporter.add_to_df(pd.DataFrame(data=N, index=[case_id], columns=['N']), 'general')
    exporter.add_to_df(time_df, 'run_times')
    exporter.add_to_df(pd.DataFrame(data=model_size, index=[case_id]), 'model_size')
    
    if harm_osc:
        harm_osc.add_iteration(ScAb, case_id)
        
# Save overall data in Excel (for all iterations combined)   
exporter.save_to_excel(ScAb.setup.directories['outputF'] + \
    ScAb.setup.time['datetime'] + '_iterative_results.xlsx')
    
print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
print('APPLICATION FINISHED AT', datetime.now().strftime("%m-%d-%Y %H-%M-%S"))

if harm_osc:
    harm_osc.export(ScAb)

    harm_osc.plot_trace(ScAb, spring=ScAb.model.spring_nom, 
                        state_center=(-6.5,-2.5))