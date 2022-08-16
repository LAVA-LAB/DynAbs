# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 07:29:04 2022

@author: Thom Badings
"""

import numpy as np
import core.preprocessing.master_classes as master

class oscillator_spec(master.spec_master):
    
    def __init__(self):

        # Initialize superclass
        master.spec_master.__init__(self)        
        
        # Step-bound on spec
        self.end_time = 12

        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [-5]
        self.control['uMax'] = [5]
        
        self.partition['boundary']  = np.array([[-10, 14], 
                                                [-10, 10]])
        self.partition['number']    = [24, 20]
        
        self.targets['boundary']    = np.array([[-9.5, 13.5], 
                                                [-9.5, 9.5]])
        self.targets['number']      = [24, 20]
        
        self.targets['extra']       = np.array([[11, 0]])
        
        self.goal = [
            np.array([[8, 14], [-14, 14]])
            ]
        
        self.critical = None #[
            # np.array([[-2, 2], [-2, 2]]),
            # np.array([[-13, 13], [9,13]]),
            #np.array([[-13,-5], [-13,13]])
            # ]
        
        self.error['max_control_error'] = {
            'default': np.array([[-1.2, 1.2], [-1.2, 1.2]]),
            'extra': np.array([[-1.5, 1.5], [-6, 6]])
            }
        self.error['max_action_distance'] = np.array([80,80])
        
class UAV_2D_spec(master.spec_master):
    
    def __init__(self):
        
        # Initialize superclass
        master.spec_master.__init__(self)   
        
        self.end_time = 32
        
        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [-4, -4]
        self.control['uMax'] = [4, 4]        
        
        # Partition size
        self.partition['boundary']  = np.array([[-11, 11], 
                                                [-6, 6], 
                                                [-11, 11], 
                                                [-6, 6]])
        self.partition['number']    = [11, 9, 11, 9]
        
        self.targets['boundary']    = np.array([[-9, 9], 
                                                [-6+8/3, 6-8/3], 
                                                [-9, 9], 
                                                [-6+8/3, 6-8/3]])
        self.targets['number']      = [9, 5, 9, 5]
        
        # Specification information
        self.goal = [
            np.array([[6, 10], 'all', [6, 10], 'all'])
            ]
        self.critical = None #[
            #np.array([[-8, -6], 'all', [-2, 0], 'all'])
            #]
        
        self.x0 = np.array([-6,0,-8,0]) 
        
        self.error['max_control_error'] = np.array([
                                                [-2, 2], 
                                                [-1.5, 1.5],
                                                [-2, 2], 
                                                [-1.5, 1.5]])
        self.error['max_action_distance'] = np.array([6,8,6,8])
        
class building_2room_spec(master.spec_master):
    
    def __init__(self, T_boiler):
        
        # Initialize superclass
        master.spec_master.__init__(self)   
        
        self.end_time = 32
        
        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [14, 14, T_boiler-10, T_boiler-10]
        self.control['uMax'] = [26, 26, T_boiler+10, T_boiler+10]     

        # Partition size
        self.partition['boundary']  = np.array([[20-2.1, 20+2.1], 
                                                [20-2.1, 20+2.1], 
                                                [38.3-0.9, 38.3+0.9], 
                                                [38.3-0.9, 38.3+0.9]])
        self.partition['number']  = [21,21,9,9]
        
        self.targets['boundary']  = self.partition['boundary']
        self.targets['number']    = self.partition['number']
        
        # Specification information
        self.goal = [
            np.array([[20, 20.1], [20, 20.1], 'all', 'all'])
            ]
        self.critical = None
        
class building_1room_spec(master.spec_master):
    
    def __init__(self, scenario):

        # Initialize superclass
        master.spec_master.__init__(self)        
        
        # Step-bound on spec
        self.end_time = 15

        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [15]
        self.control['uMax'] = [30]
        
        self.partition['boundary']  = np.array([[18.5, 23.5], [39, 46]])
        
        # Partition size
        if scenario == 0:
            self.partition['number']  = [15, 25]
            
            self.error['max_control_error'] = {
                'default': np.array([[-.2, .2], [-.5, .5]]),
                }
            
        elif scenario == 1:
            self.partition['number']  = [25, 35]
            
            self.error['max_control_error'] = {
                'default': np.array([[-.1, .1], [-.3, .3]]),
                }
           
        elif scenario == 2:
            self.partition['number']  = [35, 45]
            
            self.error['max_control_error'] = {
                'default': np.array([[-.1, .1], [-.3, .3]]),
                }
            
        elif scenario == 3:
            self.partition['number']  = [50, 70]
            
            self.error['max_control_error'] = {
                'default': np.array([[-.05, .05], [-.15, .15]]),
                }
            
        width = (self.partition['boundary'][:,1] - self.partition['boundary'][:,0]) / self.partition['number']
        
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.targets['boundary']    = np.array([[18.5+width[0]*0.5, 23.5-width[0]*0.5], 
                                                [39+width[1]*0.5, 46-width[1]*0.5]])
        self.targets['number']      = self.partition['number']
        
        
        self.goal = None
        self.critical = None
        
        
        
class shuttle_spec(master.spec_master):
    
    def __init__(self):

        # Initialize superclass
        master.spec_master.__init__(self)        
        
        # Step-bound on spec
        self.end_time = 64

        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [-0.1, -0.1]
        self.control['uMax'] = [0.1, 0.1]
        
        self.partition['boundary']  = np.array([[-1, 1], [-1, 0], [-0.02, 0.02], [-0.02, 0.02]])
        self.partition['number']  = [20, 10, 4, 4]
        
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.targets['boundary']    = self.partition['boundary']
        self.targets['number']      = self.partition['number']
            
        self.goal = [
            np.array([[-0.05, 0.05], [-0.05, -0.04], 'all', 'all'])
            ]
        
        #TODO: set critical states
        self.critical = None
        
        # self.critical = np.vstack((
        #         setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.05, .1)]).flatten(), b=[-.05], c='all', d='all'),
        #         setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.05, .1)]).flatten(), b=[-.15], c='all', d='all'),
        #         setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.15, .1)]).flatten(), b=[-.25], c='all', d='all'),
        #         setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.25, .1)]).flatten(), b=[-.35], c='all', d='all'),
        #         setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.35, .1)]).flatten(), b=[-.45], c='all', d='all'),
        #         setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.45, .1)]).flatten(), b=[-.55], c='all', d='all'),
        #         setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.55, .1)]).flatten(), b=[-.65], c='all', d='all'),
        #         setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.65, .1)]).flatten(), b=[-.75], c='all', d='all'),
        #         setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.75, .1)]).flatten(), b=[-.85], c='all', d='all'),
        #         setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.85, .1)]).flatten(), b=[-.95], c='all', d='all')
        #         ))
        
        self.end_time = 16
        
class anaesthesia_delivery_spec(master.spec_master):
    
    def __init__(self):

        # Initialize superclass
        master.spec_master.__init__(self)        
        
        # Step-bound on spec
        self.end_time = 10

        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [0]
        self.control['uMax'] = [40]
        
        # Partition size
        self.partition['boundary']  = np.array([[1, 6], [0, 10], [0, 10]])
        self.partition['number']  = [10, 30, 30]
        
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.targets['boundary']    = 'auto' #self.partition['boundary']
        self.targets['number']      = 'auto' #self.partition['number']
        
        self.goal = [
            np.array([[4, 6], [0, 10], [0, 10]])
            ]
        
        self.critical = None
        
        self.error['max_control_error'] = {
            'default': np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]) 
            }