#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import math

from core.commons import createDirectory

def loadOptions(file, setup):
    '''
    Load user options from the provided file

    Parameters
    ----------
    file : str
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
        base_dir : str
            Root directory

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
        
        # Default MDP and prism settings
        mdp = dict()
        mdp['filename'] = 'Abstraction'
        
        # Default time/date settings
        timing = {'datetime': datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}
        
        # Default folder/directory settings
        # Retreive working folder
        directories['output']   = directories['base']+'/output/'
        directories['outputF']  = directories['output']+'Ab_'+application+'_'+ \
                                        timing['datetime']+'/'
        
        # Default plotting settings
        plot = dict()
        # TRUE/FALSE setup whether plots should be generated
        plot['exportFormats']           = ['pdf','png']
        
        self.mdp = mdp
        self.plotting = plot
        self.sampling = {}
        self.time = timing
        self.directories = directories
        self.cvx = {'solver': 'ECOS'}
        
        loadOptions(base_dir+'/options.txt', self)

    def set_output_directories(self, N, case_id):
        
        # Set name for the seperate output folders of different instances
        self.directories['outputFcase'] = \
            self.directories['outputF'] + 'N='+str(N)+'_'+str(case_id)+'/' 
        
        # Create folder to save results
        createDirectory( self.directories['outputFcase'] )
        
    def prepare_iteration(self, N, case_id):
        
        self.set_output_directories(N, case_id)
        self.sampling['log_factorial_N'] = math.log(math.factorial(N))

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
        