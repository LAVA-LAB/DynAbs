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

import itertools
import numpy as np

from .compute_actions import def_backward_reach, find_backward_inflated

class action(object):
    '''
    Action object
    ''' 
    
    def __init__(self, idx, model, center, idx_tuple, backreach_obj=None):
        
        '''
        Initialize action and compute the (inflated) backward reachable set 
        
        Parameters
        ----------
        model : Model object
        '''
        
        self.idx                = idx
        self.center             = center
        self.tuple              = idx_tuple
        self.backreach_obj      = backreach_obj
        
        self.error              = None
        self.enabled_in         = set()
          
        shift = model.A_inv @ self.center
        self.backreach = self.backreach_obj.verts + shift
        
        if not backreach_obj is None:
            self.backreach_infl = self.backreach_obj.verts_infl + shift
        
        
        
class backreachset(object):
    '''
    Backward reachable set
    '''
    
    def __init__(self, name, max_control_error=None):
        
        self.name = name
        self.max_control_error = max_control_error
        
    def compute_default_set(self, model):
        '''
        Compute the default (inflated) backward reachable set for a target
        point at the origin (zero).
        
        Parameters
        ----------
        model : Model object

        '''
        
        self.verts = def_backward_reach(model)
        
        if not self.max_control_error is None:
            
            alphas = np.eye(len(self.verts))
            
            BRS_inflated = []
            
            # Compute the control error
            for err in itertools.product(*self.max_control_error):
                for alpha in alphas:
                    prob,x,_ = find_backward_inflated(model.A, np.array(err), 
                                                      alpha, self.verts)
                    
                    BRS_inflated += [x]
            
            self.verts_infl = np.array(BRS_inflated)
    