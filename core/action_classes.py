# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 08:49:42 2022

@author: Thom Badings
"""

import itertools
import numpy as np

from .compute_actions import def_backward_reach, find_backward_inflated

class action(object):
    '''
    Action object
    ''' 
    
    def __init__(self, center, idx_tuple, backreachset=None):
        
        self.center         = center
        self.tuple          = idx_tuple
        self.backreachset  = backreachset
        
class backreachset(object):
    '''
    Backward reachable set
    '''
    
    def __init__(self, name, max_control_error):
        
        self.name = name
        self.max_control_error = max_control_error
        
    def compute_default_set(self, model):
        
        G_zero = def_backward_reach(model)
        alphas = np.eye(len(G_zero))
        
        BRS_inflated = []
        
        # Compute the control error
        for err in itertools.product(*self.max_control_error):
            for alpha in alphas:
                prob,x,_ = find_backward_inflated(model.A, np.array(err), 
                                                  alpha, G_zero)
                
                BRS_inflated += [x]
        
        self.vert = np.array(BRS_inflated)
    