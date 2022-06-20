#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 ______________________________________
|                                      |
|  SCENARIO-BASED ABSTRACTION PROGRAM  |
|______________________________________|

Implementation of the method proposed in the paper:

  Thom Badings, Alessandro Abate, David Parker, Nils Jansen, Hasan Poonawala & 
  Marielle Stoelinga (2021). Sampling-based Robust Control of Autonomous 
  Systems with Non-Gaussian Noise. AAAI 2022.

Originally coded by:        Thom S. Badings
Contact e-mail address:     thom.badings@ru.nl>
______________________________________________________________________________
"""

# Load general packages
from datetime import datetime   # Import Datetime to get current date/time
import pandas as pd             # Import Pandas to store data in frames
import numpy as np              # Import Numpy for computations
import math                     # Import Math for mathematical operations
import matplotlib.pyplot as plt # Import Pyplot to generate plots
from inspect import getmembers, isclass # To get list of all available models

# Load main classes and methods
from core.scenarioBasedAbstraction import scenarioBasedAbstraction
from core.preprocessing.user_interface import user_choice, \
    load_PRISM_result_file
from core.commons import printWarning, createDirectory
from core import modelDefinitions
from core.masterClasses import settings, loadOptions

# Retreive a list of all available models
modelClasses = np.array(getmembers(modelDefinitions, isclass))
application, application_id  = user_choice('application',
                                           list(modelClasses[:,0]))

np.random.seed(10)

#-----------------------------------------------------------------------------
# Create model object
#-----------------------------------------------------------------------------

# Create model object
model = modelClasses[application_id, 1]()

#-----------------------------------------------------------------------------
# Create settings object + change manual settings
#-----------------------------------------------------------------------------

# Create settings object
setup = settings(application=model.name)
setup.deltas = model.setup['deltas']

loadOptions('options.txt', setup)

# %%

# Manual changes in general settings
setup.setOptions(category='plotting', 
        exportFormats=['pdf'], 
        partitionPlot=False,
        partitionPlot_plotHull=False)

print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

#-----------------------------------------------------------------------------
# Settings related to Monte Carlo simulations
#-----------------------------------------------------------------------------

# If TRUE monte carlo simulations are performed
setup.montecarlo['enabled'], _ = user_choice( \
                                'Monte Carlo simulations', [True, False])
if setup.montecarlo['enabled']:
    setup.montecarlo['iterations'], _ = user_choice( \
                                'Monte Carlo iterations', 'integer')
else:
    setup.montecarlo['iterations'] = 0

print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
print('PROGRAM STARTED AT \n'+setup.time['datetime'])
print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

#-----------------------------------------------------------------------------
# Initialize the main abstraction method
#-----------------------------------------------------------------------------

if setup.main['iterative'] is False:
    setup.scenarios['samples_max'] = setup.scenarios['samples']

# Dictionary for all instances
ScAb = dict()

# Set LTI model in main object
model.setModel(observer=False)

# If TRUE monte carlo simulations are performed
_, choice = user_choice( \
    'Start a new abstraction or load existing PRISM results?', 
    ['New abstraction', 'Load existing results'])
setup.main['newRun'] = not choice

# Create noise samples
if model.name in ['UAV'] and model.modelDim == 3:
    setup.setOptions(category='scenarios', gaussian=False)
    model.setTurbulenceNoise(setup.scenarios['samples_max'])

if setup.main['iterative'] is True and setup.main['newRun'] is False:
    printWarning("Iterative scheme cannot be combined with loading existing "+
                 "PRISM results, so iterative scheme disabled")
    setup.main['iterative'] = False
    setup.scenarios['samples_max'] = setup.scenarios['samples']

print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

# Create the main object for the current instance
ScAb = scenarioBasedAbstraction(setup=setup, basemodel=model)

# Remove initial variable dictionaries (reducing data usage)
del setup
del model

#-----------------------------------------------------------------------------
# Define actions (only required once outside iterative scheme)
#-----------------------------------------------------------------------------

if ScAb.setup.main['newRun'] is True:        

    # Create directories
    createDirectory(ScAb.setup.directories['outputF'])    

    assert False

    # Create actions and determine which ones are enabled
    ScAb.defActions()

#-----------------------------------------------------------------------------
# Code below is repeated every iteration of the iterative scheme
#-----------------------------------------------------------------------------

# Initialize case ID at zero
case_id = 0

# Create empty DataFrames to store iterative results
iterative_results = dict()
iterative_results['general'] = pd.DataFrame()
iterative_results['run_times'] = pd.DataFrame()
iterative_results['performance'] = pd.DataFrame()
iterative_results['model_size'] = pd.DataFrame()

# For every iteration... (or once, if iterations are disabled)
while (ScAb.setup.scenarios['samples'] <= ScAb.setup.scenarios['samples_max'] \
    and case_id < ScAb.setup.scenarios['maxIters']) or ScAb.setup.main['iterative'] is False:
        
    # Shortcut to sample complexity
    N = ScAb.setup.scenarios['samples']
    general_df = pd.DataFrame(data=N, index=[case_id], columns=['N'])        
        
    # Only perform code below is a new run is chosen
    if ScAb.setup.main['newRun'] is True:
    
        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        print('START ITERATION WITH NUMBER OF SAMPPLES N = '+str(N))
    
        # Set name for the seperate output folders of different instances
        ScAb.setup.directories['outputFcase'] = \
            ScAb.setup.directories['outputF'] + 'N='+str(N)+'_'+str(case_id)+'/' 
        
        # Create folder to save results
        createDirectory( ScAb.setup.directories['outputFcase'] )    
    
        # Compute factorial of the sample complexity upfront
        ScAb.setup.scenarios['log_factorial_N'] = \
            math.log(math.factorial(ScAb.setup.scenarios['samples']))
        
        # Calculate transition probabilities
        ScAb.defTransitions()
        
        # Save case-specific data in Excel
        output_file = ScAb.setup.directories['outputFcase'] + \
            ScAb.setup.time['datetime'] + '_N='+str(N)+'_data_export.xlsx'
        
        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
        
        # Build MDP
        model_size = ScAb.buildMDP()
        
        # Write model size results to Excel
        model_size_df = pd.DataFrame(model_size, index=[case_id])
        model_size_df.to_excel(writer, sheet_name='Model size')
        
        # Solve the MDP
        ScAb.solveMDP()
        
        # Write results to dataframes
        horizon_len = int(ScAb.N/min(ScAb.setup.deltas))
        
        # Load data into dataframes
        policy_df   = pd.DataFrame( ScAb.results['optimal_policy'], 
         columns=range(ScAb.abstr['nr_regions']), index=range(horizon_len)).T
        delta_df    = pd.DataFrame( ScAb.results['optimal_delta'], 
         columns=range(ScAb.abstr['nr_regions']), index=range(horizon_len)).T
        reward_df   = pd.DataFrame( ScAb.results['optimal_reward'], 
         columns=range(ScAb.abstr['nr_regions']), index=range(horizon_len)).T
        
        # Write dataframes to a different worksheet
        policy_df.to_excel(writer, sheet_name='Optimal policy')
        delta_df.to_excel(writer, sheet_name='Optimal delta')
        reward_df.to_excel(writer, sheet_name='Optimal reward')
        
    # If no new run was chosen, load the results from existing data files
    else:
        
        # Load results from existing PRISM results files
        output_folder, policy_file, vector_file = load_PRISM_result_file(
            ScAb.setup.directories['output'], ScAb.basemodel.name, N)
        
        # Retreive output folder
        ScAb.setup.directories['outputFcase'] = output_folder
        
        print(' -- Load policy file:',policy_file)
        print(' -- Load vector file:',vector_file)
    
        # Save case-specific data in Excel
        output_file = ScAb.setup.directories['outputFcase'] + \
            ScAb.setup.time['datetime'] + '_N='+str(N)+'_data_export.xlsx'
        
        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    
        # Load results
        ScAb.loadPRISMresults(policy_file, vector_file)
    
    # Initialize plotting
    ScAb.preparePlots()
    
    if ScAb.setup.montecarlo['enabled']:
        # Perform monte carlo simulation for validation purposes
        
        # setup.setOptions(category='montecarlo', init_states=[7])
        ScAb.monteCarlo()
        
        # Store Monte Carlo results as dataframe
        cols = ScAb.setup.montecarlo['init_timesteps']
        MCsims_df = pd.DataFrame( 
            ScAb.mc['results']['reachability_probability'], \
            columns=cols, index=range(ScAb.abstr['nr_regions']) )
            
        # Write Monte Carlo results to Excel
        MCsims_df.to_excel(writer, sheet_name='Empirical reach.')
    
    # Plot results
    ScAb.generatePlots( delta_value = ScAb.setup.deltas[0], 
                       max_delta = max(ScAb.setup.deltas) )
    
    # %%
    
    # The code below plots the trajectories for the UAV benchmark
    if ScAb.basemodel.name in ['UAV', 'shuttle']:
        
        from core.postprocessing.createPlots import UAVplots
    
        # Create trajectory plot
        performance_df = UAVplots(ScAb, case_id, writer)
        
        if ScAb.setup.main['iterative'] is True:
            iterative_results['performance'] = pd.concat(
                [iterative_results['performance'], performance_df], axis=0)

    # The code below plots the heat map for the BAS benchmark
    if ScAb.basemodel.name in ['building_1room','building_2room'] or \
        (ScAb.basemodel.name == 'UAV' and ScAb.basemodel.modelDim == 2):
        
        from core.postprocessing.createPlots import reachabilityHeatMap
        
        # Create heat map
        reachabilityHeatMap(ScAb)
    
    # Store run times of current iterations        
    time_df = pd.DataFrame( data=ScAb.time, index=[case_id] )
    time_df.to_excel(writer, sheet_name='Run time')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    plt.close('all')
    
    # If iterative approach is enabled...
    if ScAb.setup.main['iterative'] is True:
        
        # Add current results to general dictionary
        iterative_results['general'] = pd.concat(
            [iterative_results['general'], general_df], axis=0)
        iterative_results['run_times'] = pd.concat(
            [iterative_results['run_times'], time_df], axis=0)
        iterative_results['model_size'] = pd.concat(
            [iterative_results['model_size'], model_size_df], axis=0)
        
        case_id += 1
        
        ScAb.setup.scenarios['samples'] = \
            int(ScAb.setup.scenarios['samples']*ScAb.setup.scenarios['gamma'])
        
    else:
        print('\nITERATIVE SCHEME DISABLED, SO TERMINATE LOOP')
        break

# %%

if ScAb.setup.main['iterative'] and ScAb.setup.main['newRun']:
    # Save overall data in Excel (for all iterations combined)
    output_file = ScAb.setup.directories['outputF'] + \
        ScAb.setup.time['datetime'] + '_iterative_results.xlsx'
    
    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    
    for key,df in iterative_results.items():
        df.to_excel(writer, sheet_name=str(key))
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
datestring_end = datetime.now().strftime("%m-%d-%Y %H-%M-%S")             
print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
print('APPLICATION FINISHED AT', datestring_end)
