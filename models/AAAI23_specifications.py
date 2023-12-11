#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Reach-avoid specifications used for the models in the paper:
 "Probabilities Are Not Enough: Formal Controller Synthesis for Stochastic 
  Dynamical Models with Epistemic Uncertainty"

Originally coded by:        <anonymized>
Contact e-mail address:     <anonymized>
______________________________________________________________________________
"""

import numpy as np
import core.preprocessing.master_classes as master

class drone_spec(master.spec_master):
    
    def __init__(self, args):

        # Initialize superclass
        master.spec_master.__init__(self)

        # Step-bound on spec
        self.end_time = 12

        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = args.input_min
        self.control['uMax'] = args.input_max

        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = [-5]
        self.control['uMax'] = [5]
        
        self.partition['boundary']  = np.array([[-10, 14], 
                                                [-10, 10]])
        self.partition['number']    = args.partition_num_elem

        self.targets['boundary']    = np.array([[-9.5, 13.5], 
                                                [-9.5, 9.5]])
        self.targets['number']      = args.partition_num_elem
        
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
        self.control['uMin'] = args.input_min
        self.control['uMax'] = args.input_max
        
        self.partition['boundary']  = np.array([[18.5, 23.5], [39, 46]])
        self.partition['number'] = args.partition_num_elem
        self.error['target_set_size'] = {'default': np.array(args.bld_target_size)}
        print('-- Size of target size:', args.bld_target_size)
            
        width = (self.partition['boundary'][:,1] - self.partition['boundary'][:,0]) / self.partition['number']
        
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.targets['boundary']    = np.array([[18.5+width[0]*0.5, 23.5-width[0]*0.5], 
                                                [39+width[1]*0.5, 46-width[1]*0.5]])
        self.targets['number']      = self.partition['number']
        
        self.goal = None
        self.critical = None
       
        
       
class anaesthesia_delivery_spec(master.spec_master):
    
    def __init__(self, args):

        # Initialize superclass
        master.spec_master.__init__(self)        
        
        # Step-bound on spec
        self.end_time = 20

        # Authority limit for the control u, both positive and negative
        self.control['uMin'] = args.input_min
        self.control['uMax'] = args.input_max
        
        # Partition size
        self.partition['boundary']  = np.array([[1, 6], [0, 10], [0, 10]])
        self.partition['number'] = args.partition_num_elem
        
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