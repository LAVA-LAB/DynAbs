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

import sys
import numpy as np
from core import model_definitions
from inspect import getmembers, isclass # To get list of all available models

def set_model_class(input_model_name):

    # Retreive a list of all available models
    modelClasses = np.array(getmembers(model_definitions, isclass))
    names   = modelClasses[:,0]
    methods = modelClasses[:,1]
    
    # Check if the input model name is also present in the list
    if sum(names == input_model_name) > 0:
        
        # Get method corresponding to this model
        method = methods[names == input_model_name][0]
        
    else:
        sys.exit("There does not exist a model with name `"+str(input_model_name)+"`.")
    
    return method