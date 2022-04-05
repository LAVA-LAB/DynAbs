#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 ______________________________________
|                                      |
|  SCENARIO-BASED ABSTRACTION PROGRAM  |
|______________________________________|
______________________________________________________________________________
"""

# %run "/home/thom/Documents/Abstractions/sample-abstract/RunFile.py"
# %run "/home/tbadings/Documents/SBA/sample-abstract/RunFile.py"

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

np.random.seed(1)
mpl.rcParams['figure.dpi'] = 300
base_dir = os.path.dirname(os.path.abspath(__file__))

#-----------------------------------------------------------------------------
# Create model and set specification
#-----------------------------------------------------------------------------

# Define model
model = set_model_class()

# Define specification
spec = model.set_spec()

spec.problem_type = 'reachavoid'

#-----------------------------------------------------------------------------
# Load default settings and overwrite by options provided by the user
#-----------------------------------------------------------------------------

# Create settings object
setup = settings(application=model.name, base_dir = base_dir)

# Manual changes in general settings
setup.setOptions(category       = 'plotting', 
        exportFormats           = ['pdf'], 
        partitionPlot           = True,
        partitionPlot_plotHull  = True)
setup.parametric = False

# Give some user prompts
setup.set_monte_carlo()
setup.set_new_abstraction()

print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n' +
      'PROGRAM STARTED AT \n'+setup.time['datetime'] +
      '\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    
# Create the main object for the current instance
ScAb = scenarioBasedAbstraction(setup, define_model(setup, model, spec))
ScAb.define_states()

#-----------------------------------------------------------------------------
# Define actions (only required once outside iterative scheme)
#-----------------------------------------------------------------------------

if ScAb.setup.main['newRun'] is True:        
    # Create directories
    createDirectory(ScAb.setup.directories['outputF']) 
    
# Create actions and determine which ones are enabled
ScAb.define_actions()

#-----------------------------------------------------------------------------
# Code below is repeated every iteration of the iterative scheme
#-----------------------------------------------------------------------------

# Initialize case ID at zero
case_id = 0

# Create empty DataFrames to store iterative results
exporter = result_exporter()

# For every iteration... (or once, if iterations are disabled)
while case_id < ScAb.setup.scenarios['maxIters']:
    # Shortcut to the number of samples
    N = ScAb.setup.scenarios['samples']        
        
    # Only perform code below is a new run is chosen
    if ScAb.setup.main['newRun'] is True:
        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        print('START ITERATION WITH NUMBER OF SAMPLES N = '+str(N))
    
        # Set directories for this iteration
        ScAb.setup.prepare_iteration(N, case_id)
    
        # Calculate transition probabilities
        ScAb.define_probabilities()
        
        # Build and solve interval MDP
        model_size = ScAb.build_iMDP()
        ScAb.solve_iMDP()
        
        # Export the results of the current iteration
        writer = exporter.create_writer(ScAb, model_size, case_id, N)
        
    # If no new run was chosen, load the results from existing data files
    else:
        # Load results from existing PRISM results files
        ScAb.setup.directories['outputFcase'], policy_file, vector_file = \
            load_PRISM_result_file(ScAb.setup.directories['output'], 
                                   ScAb.model.name, N)
    
        # Save case-specific data in Excel
        output_file = ScAb.setup.directories['outputFcase'] + \
            ScAb.setup.time['datetime'] + '_N='+str(N)+'_data_export.xlsx'
        
        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    
        # Load results
        ScAb.loadPRISMresults(policy_file, vector_file)
    
    if ScAb.setup.montecarlo['enabled']:
        ScAb.mc = monte_carlo(ScAb, writer=writer)
    
    # Plot results
    ScAb.generate_probability_plots()
    ScAb.generate_UAV_plots(case_id, writer, exporter)
    ScAb.generate_heatmap()
    
    # Store run times of current iterations        
    time_df = pd.DataFrame( data=ScAb.time, index=[case_id] )
    time_df.to_excel(writer, sheet_name='Run time')
    writer.save()
    plt.close('all')
    
    # If iterative approach is enabled...
    if ScAb.setup.main['iterative'] is True:
        exporter.add_to_df(pd.DataFrame(data=N, index=[case_id], columns=['N']), 'general')
        exporter.add_to_df(time_df, 'run_times')
        exporter.add_to_df(pd.DataFrame(data=model_size, index=[case_id]), 'model_size')
        case_id += 1
        ScAb.setup.scenarios['samples'] = \
            int(ScAb.setup.scenarios['samples']*ScAb.setup.scenarios['gamma'])
        
        if ScAb.setup.scenarios['samples'] > ScAb.setup.scenarios['samples_max']:
            break
        
    else:
        print('\nITERATIVE SCHEME DISABLED, SO TERMINATE LOOP')
        break

# Save overall data in Excel (for all iterations combined)
if ScAb.setup.main['iterative'] and ScAb.setup.main['newRun']:    
    exporter.save_to_excel(ScAb.setup.directories['outputF'] + \
        ScAb.setup.time['datetime'] + '_iterative_results.xlsx')
        
print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
print('APPLICATION FINISHED AT', datetime.now().strftime("%m-%d-%Y %H-%M-%S"))
