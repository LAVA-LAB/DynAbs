#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 ______________________________________
|                                      |
|  SCENARIO-BASED ABSTRACTION PROGRAM  |
|______________________________________|

Implementation of the method proposed in the paper:
 "Sampling-Based Robust Control of Autonomous Systems with Non-Gaussian Noise"

Originally coded by:        <anonymized>
Contact e-mail address:     <anonymized>
______________________________________________________________________________
"""

import os                       # Import OS to allow creation of folders
import matplotlib.pyplot as plt # Import to generate plos using Pyplot
import seaborn as sns           # Import Seaborn to plot heat maps
from datetime import datetime   # Import Datetime to retreive current date/time

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
    
    def __init__(self, application):
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
        sa['gamma'] = 2 # Factor by which N is multiplied in every iteration
        sa['maxIters'] = 10 # Maximum number of iterations performed, before quitting
        sa['samples_max'] = 6400 # Maximum number of samples in iterative scheme
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
        directories = dict()
        # Retreive working folder
        directories['base']     = os.getcwd()
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
        mc['init_timesteps']        = False
        
        # Main settings
        main = dict()
        main['verbose']             = True
        main['iterative']           = True
        
        self.mdp = mdp
        self.plotting = plot
        self.montecarlo = mc
        self.gaussian = gaussian
        self.scenarios = sa
        self.time = timing
        self.directories = directories
        self.main = main

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
        
        partition = dict()
        specification = dict()
        targets = dict()
        noise = dict()
        control = dict()
        control['limits'] = dict()
        
        self.setup = {
                'partition' :       partition,
                'specification' :   specification,
                'targets' :         targets,
                'noise' :           noise,
                'control' :         control,
                'endTime' :         32
            }
        
        self.name = type(self).__name__