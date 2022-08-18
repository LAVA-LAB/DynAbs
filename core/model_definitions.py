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

import numpy as np              # Import Numpy for computations
from .preprocessing.define_gears_order import discretizeGearsMethod        
import core.preprocessing.master_classes as master

class oscillator(master.LTI_master):
    
    def __init__(self, args):
        '''
        Initialize oscillator model class, which is a 1-dimensional dummy problem,
        modelled as a double integrator.

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Number of time steps to lump together (can be used to make the model
        # fully actuated)
        self.lump = 1
        
        # Discretization step size
        self.tau = 1
        
        if args.osc_spring:
            print('-- Enable spring coefficient')
            
            self.spring_min = 0.40
            self.spring_nom = 0.50
            self.spring_max = 0.60
            
            self.mass_min = 0.80
            self.mass_nom = 1.00
            self.mass_max = 1.20
            
        else:
            print('-- Do not enable spring coefficient')
            
            self.spring_min = 0.00
            self.spring_nom = 0.00
            self.spring_max = 0.00
            
            self.mass_min = 0.75
            self.mass_nom = 1.00
            self.mass_max = 1.25

        if args.osc_par_uncertainty:
            print('-- Enable parameter uncertainty')
            
            self.A_set = [
                        self.set_A(self.mass_min, self.spring_min),
                        self.set_A(self.mass_min, self.spring_max),
                        self.set_A(self.mass_max, self.spring_min),
                        self.set_A(self.mass_max, self.spring_max)
                        ]
            
            self.B_set = [
                        self.set_B(self.mass_min),
                        self.set_B(self.mass_min),
                        self.set_B(self.mass_max),
                        self.set_B(self.mass_max)
                        ]
            
        else:
            print('-- Do not enable parameter uncertainty')
                
        # State transition matrix
        self.A     = self.set_A(self.mass_nom, self.spring_nom)
        
        # Input matrix
        self.B     = self.set_B(self.mass_nom)
        
        # Disturbance matrix
        self.Q  = np.array([[0],[0]])
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)
        
        # Covariance of the process noise
        self.noise = dict()
        self.noise['w_cov'] = np.diag([0.0000001, 0.0000001]) #np.diag([0.01, 0.01])
        
    def set_spec(self):
        
        from core.spec_definitions import oscillator_spec
        
        spec = oscillator_spec()
        spec.problem_type = 'reachavoid'
        
        return spec
    
    def set_true_model(self, mass, spring):
        
        # State transition matrix
        self.A_true     = self.set_A(mass, spring)
        # Input matrix
        self.B_true     = self.set_B(mass)
        
    def set_A(self, mass, spring):
        
        A = np.array([
            [1, self.tau],
            [-self.tau*spring/mass, 1-self.tau*0.05/mass]
            ])
        
        return A
    
    def set_B(self, mass):
        
        B = np.array([[self.tau**2/(2*mass)],
                      [self.tau/mass]])
        
        return B
        
        
class building_1room(master.LTI_master):
    
    def __init__(self, args):
        '''
        Initialize the 1-zone building automation system (BAS) model class.
        Note that this is a downscaled version of the 2-zone model above.

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        self.args = args
        
        # Number of time steps to lump together (can be used to make the model
        # fully actuated)
        self.lump = 1
        
        # Discretization step size
        self.tau = 20 # NOTE: in minutes for BAS!
        
        import core.BAS.parameters as BAS_class
        BAS = BAS_class.parameters()
        
        # Steady state values
        Tswb    = 55 #BAS.Boiler['Tswbss']
        Twss    = BAS.Zone1['Twss']
        Pout1   = BAS.Radiator['Zone1']['Prad']      
        
        w       = BAS.Radiator['w_r']
        
        BAS.Zone1['Cz'] = BAS.Zone1['Cz']
        
        m1      = BAS.Zone1['m'] # Proportional factor for the air conditioning
        
        k1_a    = BAS.Radiator['k1']
        k0_a    = BAS.Radiator['k0'] #Proportional factor for the boiler temp. on radiator temp.
        
        # Defining Deterministic Model corresponding matrices
        A_cont      = np.zeros((2,2));
        A_cont[0,0] = -(1/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))-((Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])) - ((m1*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz']))
        A_cont[0,1] = (Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])
        A_cont[1,0] = (k1_a)
        A_cont[1,1] = -(k0_a*w) - k1_a
        
        B_cont = np.array([
            [(m1*BAS.Materials['air']['Cpa'])/BAS.Zone1['Cz']], 
            [0] ])
        
        W_cont  = np.array([
                [ (Twss/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))+ (BAS.Radiator['alpha1'])/(BAS.Zone1['Cz']) ],
                [ (k0_a*w*Tswb) ],
                ])
        
        self.A = np.eye(2) + self.tau*A_cont
        
        self.B = B_cont*self.tau
        self.Q = W_cont*self.tau
        
        if args.bld_par_uncertainty:
            print('-- Enable parameter uncertainty')
          
            f1 = 0.8
            A0_cont      = np.zeros((2,2));
            A0_cont[0,0] = -(1/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))-((f1*Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])) - ((m1*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz']))
            A0_cont[0,1] = (f1*Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])
            A0_cont[1,0] = (k1_a)
            A0_cont[1,1] = -(k0_a*w) - k1_a
            
            f2 = 1.2
            A1_cont      = np.zeros((2,2));
            A1_cont[0,0] = -(1/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))-((f2*Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])) - ((m1*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz']))
            A1_cont[0,1] = (f2*Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])
            A1_cont[1,0] = (k1_a)
            A1_cont[1,1] = -(k0_a*w) - k1_a
            
            self.A_set = [
                np.eye(2) + self.tau*A0_cont,
                np.eye(2) + self.tau*A1_cont
                        ]
            
            self.B_set = [
                self.B,
                self.B
                ]
            
            infl = (1/(BAS.Zone1['Rn']*BAS.Zone1['Cz']*4)) * self.tau
            max_temp_diff = 5 
            
            self.Q_uncertain = {'min': np.array([-infl * max_temp_diff, 0]),
                                'max': np.array([infl * max_temp_diff,  0])} 
            
        else:
            print('-- Do not enabled parameter uncertainty')
            
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = np.diag([ BAS.Zone1['Tz']['sigma'], BAS.Radiator['rw']['sigma'] ]) * 0.1

    def set_spec(self):
        
        from core.spec_definitions import building_1room_spec
        spec = building_1room_spec(self.args)        
            
        spec.problem_type = 'avoid'
        
        return spec


class shuttle(master.LTI_master):
    
    def __init__(self, args):
        '''
        Initialize the spaceshuttle rendezvous model class, adapted from
        a problem in the SReachTools MATLAB toolbox (see 
        https://sreachtools.github.io/ for details)

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Number of time steps to lump together (can be used to make the model
        # fully actuated)
        self.lump = 2
        
        # Discretization step size
        self.tau = 1 # NOTE: in minutes for BAS!
        
        self.modelDim = 2
        
        # Defining Deterministic Model corresponding matrices
        self.A = np.array([[1.000631, 0, 19.9986, 0.410039],
                            [8.62e-6, 1, -0.41004, 19.9944],
                            [6.30e-05, 0, 0.99979, 0.041002],
                            [-1.29e-6, 0, -0.041, 0.999159]])
        
        self.A = (self.A - np.eye(4)) / 2 + np.eye(4)
        
        self.B = np.array([[0.666643, 0.009112],
                           [-0.00911, 0.666573],
                           [0.66662, 0.001367],
                           [-0.00137, 0.66648]]) / 2
        
        self.Q = np.zeros((4,1))
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = 10*np.diag([ 1e-4, 1e-4, 5e-8, 5e-8 ])
        
    def set_spec(self):
        
        from core.spec_definitions import shuttle_spec
        spec = shuttle_spec()        
            
        spec.problem_type = 'reachavoid'
        
        return spec
        
    
class anaesthesia_delivery(master.LTI_master):
    
    def __init__(self, args):
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        self.args = args
        
        # Number of time steps to lump together (can be used to make the model
        # fully actuated)
        self.lump = 1
        
        # Discretization step size
        self.tau = 1 #3 * 20 / 60 # 20 seconds (base is 1 minute)

        # State transition matrix
        self.A = np.array([
                        [0.8192,    0.03412,    0.01265],
                        [0.01646,   0.9822,     0.0001],
                        [0.0009,    0.00002,    0.9989]
                    ])
        
        # Input matrix
        self.B  = np.array([[0.01883],
                            [0.005],
                            [0.003] ])
        
        self.A = np.eye(3) + self.tau * (self.A - np.eye(3))
        self.B = self.tau * self.B
        
        # Disturbance matrix
        self.Q  = np.array([[0],[0],[0]])
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)
        
        # Covariance of the process noise
        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A,1))*1e-3
        
    def set_spec(self):
        
        from core.spec_definitions import anaesthesia_delivery_spec
        spec = anaesthesia_delivery_spec(self.args)     
        
        spec.problem_type = 'reachavoid'
            
        return spec