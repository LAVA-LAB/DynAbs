#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Models used in the paper:
 "Probabilities Are Not Enough: Formal Controller Synthesis for Stochastic 
  Dynamical Models with Epistemic Uncertainty"

Originally coded by:        Thom Badings
Contact e-mail address:     thombadings@gmail.com
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