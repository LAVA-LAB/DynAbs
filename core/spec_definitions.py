# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 07:29:04 2022

@author: Thom Badings
"""

import numpy as np
import core.preprocessing.master_classes as master

class robot_spec(master.spec_master):
    
    def __init__(self, mode = 1):

        # Initialize superclass
        master.spec_master.__init__(self)        
        
        # Step-bound on spec
        self.end_time = 32 

        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [-4]
        self.control['uMax'] = [4]
        
        # Partition size
        if mode == 0:
            self.partition['boundary']  = np.array([[-15, 15], [-15, 15]])
            self.partition['number']  = [15, 15]
            
            self.specification['goal'] = [
                np.array([[8, 12], 'all'])
                ]
            
            # Actions per dimension (if 'auto', equal to nr of regions)
            self.targets['boundary']    = self.partition['boundary']
            self.targets['number']      = [15,15]
            
        elif mode == 1:
            self.partition['boundary']  = np.array([[-11, 11], 
                                                             [-6, 6]])
            self.partition['number']    = [11, 9]
            
            self.targets['boundary']    = np.array([[-9, 9], 
                                                             [-6+8/3, 6-8/3]])
            self.targets['number']      = [9, 5]
            
            self.goal = [
                np.array([[6, 10], 'all'])
                ]
        
        self.critical = None
        
        self.error['max_control_error'] = np.array([[-2, 2], [-4/3, 4/3]])
        self.error['max_action_distance'] = np.array([6,8])
        
class UAV_2D_spec(master.spec_master):
    
    def __init__(self):
        
        # Initialize superclass
        master.spec_master.__init__(self)   
        
        self.end_time = 32
        
        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [-4, -4]
        self.control['uMax'] = [4, 4]        
        
        # Partition size
        self.partition['boundary']  = np.array([[-13, 13], 
                                                [-6-4/3, 6+4/3], 
                                                [-13, 13], 
                                                [-6-4/3, 6+4/3]])
        self.partition['number']  = [13, 11, 13, 11]
        
        self.targets['boundary']    = np.array([[-9, 9], 
                                                [-6+4/3, 6-4/3], 
                                                [-9, 9], 
                                                [-6+4/3, 6-4/3]])
        self.targets['number']      = [9, 7, 9, 7]
        
        # Specification information
        self.goal = [
            np.array([[-10, -6], 'all', [6, 10], 'all'])
            ]
        self.critical = None #[
            # np.array([[-10, -2], 'all', [-2, 2], 'all'])
            # ]
        
        self.x0 = np.array([-8,0,-8,0]) 
        
        # # Partition size
        # self.partition['boundary']  = np.array([[-7, 7], 
        #                                         [-6, 6], 
        #                                         [-11, 5], 
        #                                         [-6, 6]])
        # self.partition['number']  = [7, 9, 8, 9]
        
        # self.targets['boundary']    = self.partition['boundary']
        # self.targets['number']      = self.partition['number']
        
        # # Specification information
        # self.goal = [
        #     np.array([[-2, 2], 'all', [0, 4], 'all'])
        #     ]
        # self.critical = None
        
        self.error['max_control_error'] = np.array([[-2, 2], 
                                                [-1.5, 1.5],
                                                [-2, 2], 
                                                [-1.5, 1.5]])
        #self.error['max_action_distance'] = np.array([6,8,6,8])
        
        self.x0 = np.array([-2,0,-2,0])  
        
class UAV_3D_spec(master.spec_master):
    
    def __init__(self):
        
        # Initialize superclass
        master.spec_master.__init__(self)   
        
        self.end_time = 32
        
        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [-4, -4, -4]
        self.control['uMax'] = [4, 4, 4]        

        # Partition size
        self.partition['boundary']  = np.array([[-15, 15], 
                                                [-4.5, 4.5], 
                                                [-9, 9], 
                                                [-4.5, 4.5],
                                                [-7, 7],
                                                [-3, 3]])
        self.partition['number']  = [15, 3, 9, 3, 7, 3]
        
        self.targets['boundary']    = self.partition['boundary']
        self.targets['number']      = self.partition['number']
        
        # Specification information
        self.goal = [
            np.array([[12, 14], 'all', [2, 4], 'all', [-6, -4], 'all'])
            ]
        self.critical = [
            # Hole 1
            np.array([[-10, 6], 'all', [0, 8], 'all', [-6, 2], 'all']),
            np.array([[-10, 6], 'all', [6, 8], 'all', [2, 4], 'all']),
            np.array([[-10, 6], 'all', [4, 4.2], 'all', [-6, -5.8], 'all']),
            # Hole 2
            np.array([[0, 2], 'all', [2, 8], 'all', [-6, 4], 'all']),
            np.array([[0, 2], 'all', [2, 8], 'all', [0, 2], 'all']),
            # Tower
            np.array([[0, 2], 'all', [-2, 0], 'all', [-6, 6], 'all']),
            # Wall between routes
            np.array([[4, 8], 'all', [-2, 0], 'all', [-6, -2], 'all']),
            # Long route obstacles
            np.array([[-10, 8], 'all', [-4, 2], 'all', [-6, 0], 'all']),
            np.array([[0, 2], 'all', [-8, -4], 'all', [-6, -5.8], 'all']),
            # Overhanging
            np.array([[0, 2], 'all', [-8, -4], 'all', [4, 6], 'all']),
            # Small last obstacle
            np.array([[12, 14], 'all', [-8, -6], 'all', [-6, -5.8], 'all']),
            # Obstacle next go goal
            np.array([[10, 14], 'all', [6, 8], 'all', [-6, 0], 'all']),
            ]
        
        self.x0 = [ np.array([[-10, -2], 'all', [-2, 2], 'all']) ]
        
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
        self.end_time = 64

        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [15]
        self.control['uMax'] = [30]
        
        # Partition size
        if scenario == 0:
            self.partition['boundary']  = np.array([[19.1, 22.9], [36, 40]])
            self.partition['number']  = [19, 20]
            
            # Actions per dimension (if 'auto', equal to nr of regions)
            self.targets['boundary']    = np.array([[19.1, 22.9], [36, 40]])
            self.targets['number']      = [19, 40]
            
        elif scenario == 1:
            self.partition['boundary']  = np.array([[19.1, 22.9], [36, 40]])
            self.partition['number']    = [190, 600]
            
            self.targets['boundary']    = np.array([[19.1, 22.9], [36, 40]])
            self.targets['number']      = [76, 200]
        
        self.goal = [
            np.array([[20.8, 21.2], 'all'])
            ]
        
        self.critical = None
        
        self.error['max_control_error'] = np.array([[-.1, .1], [-.3, .3]])     
        
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
        self.partition['boundary']  = np.array([[1, 6], [0, 10], [-0.5, 10.5]])
        self.partition['number']  = [20, 40, 11]
        
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.targets['boundary']    = self.partition['boundary']
        self.targets['number']      = self.partition['number']
        
        self.goal = [
            np.array([[4.25, 5.75], [8.5, 9.5], [8.5, 9.5]])
            ]
        
        self.critical = None
        
        self.error['max_control_error'] = np.array([[-.25, .25], [-.25, .25], [-1, 1]]) 