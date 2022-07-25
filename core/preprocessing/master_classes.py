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

import os                       # Import OS to allow creation of folders
import pandas as pd             # Import Pandas to store data in frames
import matplotlib.pyplot as plt # Import to generate plos using Pyplot
import seaborn as sns           # Import Seaborn to plot heat maps
from datetime import datetime   # Import Datetime to retreive current date/time
import math                     # Import Math for mathematical operations

from core.preprocessing.user_interface import user_choice
from core.commons import createDirectory

def loadOptions(file, setup):
    '''
    Load user options from the provided file

    Parameters
    ----------
    file : stri
        Filename to load options from.
    setup : object
        Object containing all setup data.

    Returns
    -------
    setup
        Modified setup object.

    '''
    
    def str2bool(v):
        return v.lower() in ("True","true")

    # Read options file
    if os.path.isfile(file):
        options = open(file, 'r')
        for line in options.readlines():
            if line[0] != '#':
                line_cut = line.rstrip()
                frags = line_cut.split(' = ')
                frags0 = frags[0].split('.')
                
                category = frags0[0]
                key = frags0[1]
                value = frags[1]
                
                try:
                    value = float(value)
                    
                    if value == int(value):
                        value = int(value)
                except:
                    if value in ['True', 'true']:
                        value = True
                    elif value in ['False', 'false']:
                        value = False
                    else:
                        value = str(value)
                
                category_upd = getattr(setup, category)
                    
                category_upd[str(key)] = value
                    
                print(' >> Changed "'+str(key)+'" in "'+str(category)+'" to "'+str(value)+'"')
                    
                setattr(setup, category, category_upd)
    
    return setup

class settings(object):
    
    def setOptions(self, category=None, **kwargs):
        '''
        Change options in the main 'options' object

        Parameters
        ----------
        category : str, optional
            Category (i.e. dictionary entry) for which to make changes. 
            The default is None.
        **kwargs : <multiple arguments>
            Multiple arguments for which to make changes in the settings.

        Returns
        -------
        None.

        '''
        
        # Function to set values in the setup dictionary
            
        category_upd = getattr(self, category)
        
        for key, value in kwargs.items():
            
            category_upd[str(key)] = value
            
            print(' >> Changed "'+str(key)+'" in "'+str(category)+'" to "'+str(value)+'"')
            
        setattr(self, category, category_upd)
    
    def __init__(self, application, base_dir):
        '''
        Initialize 'options' object

        Parameters
        ----------
        application : str
            Name of the application / benchmark to initialize for.

        Returns
        -------
        None.

        '''
        
        directories = dict()
        directories['base'] = base_dir
        
        sns.set_style("ticks")
        
        # Default pyplot style (font size, template, etc.)
        plt.close('all')
        plt.ion()
        plt.style.use('seaborn-deep')
        plt.rcParams.update({'font.size': 7, 
                             'pgf.texsystem' : "xelatex"})
        
        # Plot font family and size
        plt.rc('font', family='serif')
        SMALL_SIZE = 7
        MEDIUM_SIZE = 9
        BIGGER_SIZE = 9
        
        # Make sure the matplotlib generates editable text (e.g. for Illustrator)
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        
        # Set font sizes
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        # Default scenario approach settings
        sa = dict()
        gaussian = dict()
        
        sa['samples'] = 25 # Sample complexity used in scenario approach
        sa['confidence']   = 1e-1 # Confidence level (beta)
        sa['gaussian'] = True # Use Gaussian noise if true
        
        # Default MDP and prism settings
        mdp = dict()
        mdp['filename'] = 'Abstraction'
        mdp['mode'] = ['estimate','interval'][1]
        mdp['prism_java_memory'] = 1 # PRISM java memory allocation in GB
        mdp['prism_model_writer'] = ['default','explicit'][1]
        mdp['prism_folder'] = "/Users/..."
        
        # Default time/date settings
        timing = dict()
        # Retreive datetime string
        timing['datetime'] = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        
        # Default folder/directory settings
        # Retreive working folder
        directories['output']   = directories['base']+'/output/'
        directories['outputF']  = directories['output']+'ScAb_'+application+'_'+ \
                                        timing['datetime']+'/'
        
        # Default plotting settings
        plot = dict()
        # TRUE/FALSE setup whether plots should be generated
        plot['partitionPlot']           = False
        plot['3D_UAV']                  = False
        plot['partitionPlot_plotHull']  = True
        plot['probabilityPlots']        = True
        plot['exportFormats']           = ['pdf','png']
        
        # Default Monte Carelo settings
        # Set which Monte Carlo simulations to perform (False means inherent)
        mc = dict()
        mc['init_states']           = False
        
        # Main settings
        main = dict()
        main['verbose']             = True
        main['iterations']          = 1
        
        self.mdp = mdp
        self.plotting = plot
        self.montecarlo = mc
        self.gaussian = gaussian
        self.scenarios = sa
        self.time = timing
        self.directories = directories
        self.main = main
        
        self.cvx = {'solver': 'ECOS'}
        
        loadOptions(base_dir+'/options.txt', self)

    def set_monte_carlo(self, iterations=None):
        
        if iterations is None:
        
            # If TRUE monte carlo simulations are performed
            self.montecarlo['enabled'], _ = user_choice( \
                                            'Monte Carlo simulations', [True, False])
            if self.montecarlo['enabled']:
                self.montecarlo['iterations'], _ = user_choice( \
                                            'Monte Carlo iterations', 'integer')
            else:
                self.montecarlo['iterations'] = 0
                
        else:
            self.montecarlo['enabled']=True
            self.montecarlo['iterations']=int(iterations)
            
    def set_new_abstraction(self):
        
        _, choice = user_choice( \
            'Start a new abstraction or load existing PRISM results?', 
            ['New abstraction', 'Load existing results'])
        self.main['newRun'] = not choice

    def set_output_directories(self, N, case_id):
        
        # Set name for the seperate output folders of different instances
        self.directories['outputFcase'] = \
            self.directories['outputF'] + 'N='+str(N)+'_'+str(case_id)+'/' 
        
        # Create folder to save results
        createDirectory( self.directories['outputFcase'] )
        
    def prepare_iteration(self, N, case_id):
        
        self.set_output_directories(N, case_id)
        self.scenarios['log_factorial_N'] = math.log(math.factorial(N))

class LTI_master(object):
    
    def setOptions(self, category=None, **kwargs):
        '''
        Change options in the model-specific 'options' object

        Parameters
        ----------
        category : str, optional
            Category (i.e. dictionary entry) for which to make changes. 
            The default is None.
        **kwargs : <multiple arguments>
            Multiple arguments for which to make changes in the settings.

        Returns
        -------
        None.

        '''
        
        # Function to set values in the setup dictionary

        # If category is none, settings are set in main dictionary
        if category==None:
            for key, value in kwargs.items():
                self.setup[str(key)] = value
                
                print(' >> Changed',key,'to',value)
                
        # Otherwise, settings are set within sub-dictionary 'category'
        else:
            for key, value in kwargs.items():
                self.setup[str(category)][str(key)] = value
    
                print(' >> Changed "'+str(key)+'" in "'+str(category)+'" to "'+str(value)+'"')
    
    def __init__(self):
        '''
        Initialize the model object

        Returns
        -------
        None.

        '''
        
        self.setup = {}
        
        self.name = type(self).__name__
        
class spec_master(object):
    
    def __init__(self):
        '''
        Initialize the model object

        Returns
        -------
        None.

        '''
        
        self.partition = {}
        self.specification = {}
        self.targets = {}
        self.noise = {}
        self.control = {}
        self.error = {}
        
class result_exporter(object):
    
    def __init__(self):
        
        self.it_results = self.init_dataframes()
        
    def init_dataframes(self):
        
        iterative_results = dict()
        iterative_results['general'] = pd.DataFrame()
        iterative_results['run_times'] = pd.DataFrame()
        iterative_results['performance'] = pd.DataFrame()
        iterative_results['model_size'] = pd.DataFrame()
        
        return iterative_results
    
    def create_writer(self, ScAb, model_size, case_id, N):
        
        # Save case-specific data in Excel
        output_file = ScAb.setup.directories['outputFcase'] + \
            ScAb.setup.time['datetime'] + '_N='+str(N)+'_data_export.xlsx'
        
        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
        
        # Write model size results to Excel
        model_size_df = pd.DataFrame(model_size, index=[case_id])
        model_size_df.to_excel(writer, sheet_name='Model size')
        
        # Load data into dataframes
        policy_df   = pd.DataFrame( ScAb.results['optimal_policy'], 
         columns=range(ScAb.partition['nr_regions']), index=range(ScAb.N)).T
        reward_df   = pd.Series( ScAb.results['optimal_reward'], 
         index=range(ScAb.partition['nr_regions'])).T
        
        # Write dataframes to a different worksheet
        policy_df.to_excel(writer, sheet_name='Optimal policy')
        reward_df.to_excel(writer, sheet_name='Optimal reward')
        
        return writer
    
    def add_to_df(self, df, key):
        
        self.it_results[key] = pd.concat([self.it_results[key], df], axis=0)
        
    def save_to_excel(self, output_file):
        
        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
        
        for key,df in self.it_results.items():
            df.to_excel(writer, sheet_name=str(key))
        
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()