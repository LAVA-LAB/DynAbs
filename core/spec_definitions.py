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

import numpy as np
import core.preprocessing.master_classes as master

class drone_spec(master.spec_master):
    
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
        
        self.critical = None
        
        self.error['target_set_size'] = {
            'default': np.array([[-1.2, 1.2], [-1.2, 1.2]]),
            'extra': np.array([[-1.5, 1.5], [-6, 6]])
            }
        self.error['max_action_distance'] = np.array([80,80])
        
        
        
class building_temp_spec(master.spec_master):
    
    def __init__(self, args):

        # Initialize superclass
        master.spec_master.__init__(self)        
        
        # Step-bound on spec
        self.end_time = 15

        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [15]
        self.control['uMax'] = [30]
        
        self.partition['boundary']  = np.array([[18.5, 23.5], [39, 46]])
        
        self.partition['number'] = list(args.bld_partition)
        self.error['target_set_size'] = {'default': np.array(args.bld_target_size)}
        
        print('-- Partition:', args.bld_partition)
        print('-- Size of target size:', args.bld_target_size)
            
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
        self.partition['number']    = [20, 10, 4, 4]
        
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.targets['boundary']    = 'auto'
        self.targets['number']      = 'auto'
            
        self.goal = [
            np.array([[-0.05, 0.05], [-0.05, -0.04], 'all', 'all'], dtype='object')
            ]
        
        self.critical = np.vstack((
                np.array([[-1, -0.05], [-0.2, 0], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.15], [-0.3, -0.2], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.25], [-0.4, -0.3], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.35], [-0.5, -0.4], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.45], [-0.6, -0.5], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.55], [-0.7, -0.6], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.65], [-0.8, -0.7], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.75], [-0.9, -0.8], 'all', 'all'], dtype='object'),
                np.array([[-1, -0.85], [-1.0, -0.9], 'all', 'all'], dtype='object')
        ))

                # setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.05, .1)]).flatten(), b=[-.05], c='all', d='all'),
                # setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.05, .1)]).flatten(), b=[-.15], c='all', d='all'),
                # setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.15, .1)]).flatten(), b=[-.25], c='all', d='all'),
                # setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.25, .1)]).flatten(), b=[-.35], c='all', d='all'),
                # setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.35, .1)]).flatten(), b=[-.45], c='all', d='all'),
                # setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.45, .1)]).flatten(), b=[-.55], c='all', d='all'),
                # setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.55, .1)]).flatten(), b=[-.65], c='all', d='all'),
                # setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.65, .1)]).flatten(), b=[-.75], c='all', d='all'),
                # setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.75, .1)]).flatten(), b=[-.85], c='all', d='all'),
                # setStateBlock(self.partition, a=np.array([[i, -i] for i in np.arange(-.95, -.85, .1)]).flatten(), b=[-.95], c='all', d='all')
                # ))
        
        self.end_time = 16
        
class anaesthesia_delivery_spec(master.spec_master):
    
    def __init__(self, args):

        # Initialize superclass
        master.spec_master.__init__(self)        
        
        # Step-bound on spec
        self.end_time = 20

        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [-10]
        self.control['uMax'] = [40]
        
        # Partition size
        self.partition['boundary']  = np.array([[1, 6], [0, 10], [0, 10]])
        self.partition['number']  = list(args.drug_partition) #[20, 20, 20]
        
        width = self.partition['boundary'] @ np.array([-1, 1]) / self.partition['number']
        
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.targets['boundary'] = np.vstack((
                self.partition['boundary'][:,0] + 1.5*width,
                self.partition['boundary'][:,1] - 1.5*width
                )).T
        
        self.targets['number']      = list(np.array(self.partition['number'])-2)
        
        # self.goal = [
        #     np.array([[4, 6], [0, 10], [0, 10]])
        #     ]
        
        self.goal = None
        self.critical = None
        
        self.error['target_set_size'] = {
            'default': np.vstack(([-1.4, -1.4, -1.4]*width, [1.4, 1.4, 1.4]*width)).T
            }