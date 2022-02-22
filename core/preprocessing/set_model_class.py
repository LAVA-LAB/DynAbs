#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 ______________________________________
|                                      |
|  SCENARIO-BASED ABSTRACTION PROGRAM  |
|______________________________________|

Implementation of the method proposed in the paper:

  Thom Badings, Alessandro Abate, David Parker, Nils Jansen, Hasan Poonawala & 
  Marielle Stoelinga (2021). Sampling-based Robust Control of Autonomous 
  Systems with Non-Gaussian Noise. AAAI 2022.

Originally coded by:        Thom S. Badings
Contact e-mail address:     thom.badings@ru.nl>
______________________________________________________________________________
"""

import numpy as np
from core import modelDefinitions
from core.preprocessing.user_interface import user_choice
from inspect import getmembers, isclass # To get list of all available models

def set_model_class():

    # Retreive a list of all available models
    modelClasses = np.array(getmembers(modelDefinitions, isclass))
    application, application_id  = user_choice('application',
                                               list(modelClasses[:,0]))
    model = modelClasses[application_id, 1]()
    
    return model