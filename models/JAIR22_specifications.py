#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Models used in the paper:
 "Robust Control for Dynamical Systems with Non-Gaussian Noise via
 Formal Abstractions"

Originally coded by:        Thom Badings
Contact e-mail address:     thombadings@gmail.com
______________________________________________________________________________
"""

import numpy as np
import sys
import core.preprocessing.master_classes as master

class robot_spec(master.spec_master):
    
    def __init__(self):

        # Initialize superclass
        master.spec_master.__init__(self)        

        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [-5]
        self.control['uMax'] = [5]
        
        self.partition['boundary']  = np.array([[-21, 21], [-21, 21]])
        self.partition['number']    = [21, 21]
        
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.targets['boundary']    = 'auto'
        self.targets['number']      = 'auto'
            
        self.goal = [
            np.array([[-1, 1], [-1 ,1]], dtype='object')
            ]
        
        self.critical = None



class building_2room_spec(master.spec_master):
    
    def __init__(self, T_boiler):

        # Initialize superclass
        master.spec_master.__init__(self)        

        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [14, 14, T_boiler-10, T_boiler-10]
        self.control['uMax'] = [26, 26, T_boiler+10, T_boiler+10]
        
        self.partition['boundary']  = np.array([[17.9, 22.1], [17.9, 22.1], [37.4, 39.2], [37.4, 39.2]])
        self.partition['number']    = [21,21,9,9]
        
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.targets['boundary']    = 'auto'
        self.targets['number']      = 'auto'
            
        self.goal = [
            np.array([[19.9, 20.1], [19.9, 20.1], 'all', 'all'], dtype='object')
            ]
        
        self.critical = None



class building_1room_spec(master.spec_master):
    
    def __init__(self):

        # Initialize superclass
        master.spec_master.__init__(self)        

        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [14, -10]
        self.control['uMax'] = [28, 10]
        
        self.partition['boundary']  = np.array([[19.1, 22.9], [36, 40]])
        self.partition['number']    = [19, 20]
        
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.targets['boundary']    = 'auto'
        self.targets['number']      = 'auto'
            
        self.goal = [
            np.array([[20.9, 21.1], 'all'], dtype='object')
            ]
        
        self.critical = None



class shuttle_spec(master.spec_master):
    
    def __init__(self):

        # Initialize superclass
        master.spec_master.__init__(self)

        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [-0.1, -0.1]
        self.control['uMax'] = [0.1, 0.1]
        
        self.partition['boundary']  = np.array([[-1, 1], [-1, 0], [-0.02, 0.02], [-0.02, 0.02]])
        self.partition['number']    = [20, 10, 4, 4]
        
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.targets['boundary']    = 'auto'
        self.targets['number']      = 'auto'
            
        self.goal = [
            np.array([[-0.05, 0.05], [-0.05, -0.04], 'all', 'all'], dtype='object')
            ]
        
        self.critical = np.vstack((
                np.array([[-1, -0.1], [-0.2, 0], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.2], [-0.3, -0.2], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.3], [-0.4, -0.3], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.4], [-0.5, -0.4], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.5], [-0.6, -0.5], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.6], [-0.7, -0.6], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.7], [-0.8, -0.7], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.8], [-0.9, -0.8], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.9], [-1.0, -0.9], 'all', 'all'], dtype='object'),
                np.array([[0.1, 1], [-0.2, 0], 'all', 'all'], dtype='object'),
                np.array([[0.2, 1], [-0.3, -0.2], 'all', 'all'], dtype='object'),
                np.array([[0.3, 1], [-0.4, -0.3], 'all', 'all'], dtype='object'),
                np.array([[0.4, 1], [-0.5, -0.4], 'all', 'all'], dtype='object'),
                np.array([[0.5, 1], [-0.6, -0.5], 'all', 'all'], dtype='object'),
                np.array([[0.6, 1], [-0.7, -0.6], 'all', 'all'], dtype='object'),
                np.array([[0.7, 1], [-0.8, -0.7], 'all', 'all'], dtype='object'),
                np.array([[0.8, 1], [-0.9, -0.8], 'all', 'all'], dtype='object'),
                np.array([[0.9, 1], [-1.0, -0.9], 'all', 'all'], dtype='object')
        ))



class UAV_spec(master.spec_master):
    
    def __init__(self, args, modelDim):

        # Initialize superclass
        master.spec_master.__init__(self)        

        # Authority limit for the control u, both positive and negative
        if modelDim == 2:

            # Authority limit for the control u, both positive and negative
            self.control['uMin'] = [-4, -4]
            self.control['uMax'] = [4, 4]
        
            self.partition['boundary']  = np.array([[-7, 7], [-3, 3], [-7, 7], [-3, 3]])
            self.partition['number']    = [7, 4, 7, 4]
        
            # Actions per dimension (if 'auto', equal to nr of regions)
            self.targets['boundary']    = 'auto'
            self.targets['number']      = 'auto'
            
            self.goal = [
                np.array([[5, 7], 'all', [5, 7], 'all'], dtype='object')
                ]
            
            self.critical = np.vstack((
                    np.array([[-7, -1], 'all', [1, 3], 'all'], dtype='object'),
                    np.array([[3, 7], 'all', [-7, -3], 'all'], dtype='object')
            ))

        elif modelDim == 3:

            # Let the user make a choice for the model dimension
            self.noiseMultiplier = args.noise_factor
            
            # Authority limit for the control u, both positive and negative
            self.control['uMin'] = [-4, -4, -4]
            self.control['uMax'] = [4, 4, 4]
        
            self.partition['boundary']  = np.array([[-15, 15], [-2.25, 2.25], 
                                                    [-9,   9], [-2.25, 2.25], 
                                                    [-7,   7], [-2.25, 2.25]])
            self.partition['number']    = [15, 3, 9, 3, 7, 3]
        
            # Actions per dimension (if 'auto', equal to nr of regions)
            self.targets['boundary']    = 'auto'
            self.targets['number']      = 'auto'
        
            self.goal = [
                np.array([[11, 15], 'all', [1, 5], 'all', [-7,-3], 'all'], dtype='object')
                ]

            self.critical = [
                # Hole 1
                np.array([[-11, -5], 'all', [-1, 9], 'all', [-7, -5], 'all'], dtype='object'),
                np.array([[-11, -5], 'all', [5, 9], 'all', [-5, 5], 'all'], dtype='object'),
                np.array([[-11, -5], 'all', [-1, 3], 'all', [-5, 3], 'all'], dtype='object'),

                # Hole 2
                np.array([[-1, 3], 'all', [1, 9], 'all', [-7, -1], 'all'], dtype='object'),
                np.array([[-1, 3], 'all', [1, 9], 'all', [3, 5], 'all'], dtype='object'),
                np.array([[-1, 3], 'all', [1, 3], 'all', [-1, 3], 'all'], dtype='object'),
                np.array([[-1, 3], 'all', [7, 9], 'all', [-1, 3], 'all'], dtype='object'),

                # Tower
                np.array([[-1, 3], 'all', [-3, 1], 'all', [-7, 7], 'all'], dtype='object'),

                # Wall between routes
                np.array([[3, 9], 'all', [-3, 1], 'all', [-7, -1], 'all'], dtype='object'),

                # Long route obstacles
                np.array([[-11, -7], 'all', [-5, -1], 'all', [-7, 1], 'all'], dtype='object'),
                np.array([[-1, 3], 'all', [-9, -3], 'all', [-7, -5], 'all'], dtype='object'),

                # Overhanging
                np.array([[-1, 3], 'all', [-9, -3], 'all', [3, 7], 'all'], dtype='object'),

                # Small last obstacle
                np.array([[11, 15], 'all', [-9, -5], 'all', [-7, -5], 'all'], dtype='object'),

                # Obstacle next to goal
                np.array([[9, 15], 'all', [5, 9], 'all', [-7, 1], 'all'], dtype='object'),
                ]

        else:
            print('No valid dimension for the drone model was provided')
            sys.exit()



class spacecraft_spec(master.spec_master):
    
    def __init__(self, args):

        # Initialize superclass
        master.spec_master.__init__(self)        
    
        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [-2, -2, -2]
        self.control['uMax'] = [2, 2, 2]
    
        self.partition['boundary']  = np.array([[-3.4, 1], 
                                                [-2, 16.4], 
                                                [-2, 2],
                                                [-4, 4], 
                                                [-4, 4],
                                                [-2, 2]])
        self.partition['number']    = [11, 23, 5, 5, 5, 5]
    
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.targets['boundary']    = 'auto'
        self.targets['number']      = 'auto'
    
        self.goal = [
            np.array([[-0.2, 0.2], [-0.4, 0.4], [-0.4, 0.4], 'all', 'all', 'all'], dtype='object')
            ]

        self.critical = [
            np.array([[-1, 1], [12.4-8*0.8, 12.4], 'all', 'all', 'all', 'all'], dtype='object')
        ]



class spacecraft_2D_spec(master.spec_master):
    
    def __init__(self, args):

        # Initialize superclass
        master.spec_master.__init__(self)        
    
        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [-2, -2]
        self.control['uMax'] = [2, 2]

        self.partition['boundary']  = np.array([[-4.6, 2.2], [-2, 21.2], [-4, 4], [-4, 4]])
        self.partition['number']    = [17, 29, 5, 5]
    
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.targets['boundary']    = 'auto'
        self.targets['number']      = 'auto'
    
        self.goal = [
            np.array([[-0.2, 0.2], [-0.4, 0.4], 'all', 'all'], dtype='object')
            ]

        self.critical = [
            np.array([[-1, 1], [8, 12], 'all', 'all'], dtype='object')
        ]



class spacecraft_1D_spec(master.spec_master):
    
    def __init__(self, args):

        # Initialize superclass
        master.spec_master.__init__(self)        
    
        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [-2]
        self.control['uMax'] = [2]
    
        self.partition['boundary']  = np.array([[-2, 2],
                                                [-2, 2]])
        self.partition['number']    = [5, 5]
    
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.targets['boundary']    = 'auto'
        self.targets['number']      = 'auto'
    
        self.goal = [
            np.array([[-0.2, 0.2], [-0.2, 0.2]], dtype='object')
            ]

        self.critical = None