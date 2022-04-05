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

import numpy as np              # Import Numpy for computations
import scipy.linalg             # Import to enable matrix operations
import sys                      # Import to allow terminating the script

from .preprocessing.define_gears_order import discretizeGearsMethod        
import core.preprocessing.user_interface as ui
import core.preprocessing.master_classes as master
from .commons import setStateBlock

class robot(master.LTI_master):
    
    def __init__(self):
        '''
        Initialize robot model class, which is a 1-dimensional dummy problem,
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
        self.tau = 2
        
        # State transition matrix
        self.A  = np.array([[1, self.tau],
                            [0, .9]])
        
        mass_min = 0.95
        mass_max = 1.05
        
        # self.A_set = [
        #             np.array([[1, self.tau],[0,0.7]]),
        #             np.array([[1, self.tau],[0,0.9]])
        #             ]
        
        self.A_set = [
                    np.array([[1, self.tau],[0,1-0.1/mass_min]]),
                    np.array([[1, self.tau],[0,1-0.1/mass_max]])
                    ]
        
        self.B_set = [
                    np.array([[self.tau**2/2], [self.tau]]) * mass_min,
                    np.array([[self.tau**2/2], [self.tau]]) * mass_max
                    ]
        
        # Input matrix
        self.B  = np.array([[self.tau**2/2],
                            [self.tau]])
        
        # Disturbance matrix
        self.Q  = np.array([[0],[0]])
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)
        
        # Covariance of the process noise
        self.noise = dict()
        self.noise['w_cov'] = np.diag([0.1, 0.01]) 
        
    def set_spec(self):
        
        from core.spec_definitions import robot_spec
        
        spec = robot_spec()
        
        return spec
        
class UAV(master.LTI_master):
    
    def __init__(self):
        '''
        Initialize the UAV model class, which can be 2D or 3D. The 3D case
        corresponds to the UAV benchmark in the paper.

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Number of time steps to lump together (can be used to make the model
        # fully actuated)
        self.lump = 1
        
        # Let the user make a choice for the model dimension
        self.modelDim, _  = ui.user_choice('model dimension',[2,3])
        
        # Discretization step size
        self.tau = 2.0
        
        # State transition matrix
        Ablock = np.array([[1, self.tau],
                          [0, 0.9]])
        
        # Input matrix
        Bblock = np.array([[self.tau**2/2],
                           [self.tau]])
        
        if self.modelDim==3:
            self.A  = scipy.linalg.block_diag(Ablock, Ablock, Ablock)
            self.B  = scipy.linalg.block_diag(Bblock, Bblock, Bblock)
            
            # Disturbance matrix
            self.Q  = np.array([[0],[0],[0],[0],[0],[0]])
                
        elif self.modelDim==2:
            self.A  = scipy.linalg.block_diag(Ablock, Ablock)
            self.B  = scipy.linalg.block_diag(Bblock, Bblock)
            
            mass_min = 0.95
            mass_max = 1.05
            
            self.A_set = [
                        self.A + np.diag([0, -0.9 + 1-0.1/mass_min, 0, -0.9 + 1-0.1/mass_min]),
                        self.A + np.diag([0, -0.9 + 1-0.1/mass_max, 0, -0.9 + 1-0.1/mass_max])
                        ]
            
            B_min = np.array([[self.tau**2/2], [self.tau]]) * mass_min
            B_max = np.array([[self.tau**2/2], [self.tau]]) * mass_max
            
            self.B_set = [
                        scipy.linalg.block_diag(B_min, B_min),
                        scipy.linalg.block_diag(B_max, B_max)
                        ]
        
            # Disturbance matrix
            self.Q  = np.array([[0],[0],[0],[0]])
            
        else:
            print('No valid dimension for the drone model was provided')
            sys.exit()
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        # Covariance of the process noise
        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A,1))*0.001
           
    def set_spec(self):
    
        if self.modelDim == 2:
            
            from core.spec_definitions import UAV_2D_spec
            spec = UAV_2D_spec()        
            
        elif self.modelDim == 3:
            
            # Let the user make a choice for the model dimension
            self.setup['noiseMultiplier'], _  = ui.user_choice('process noise multiplier',[1,0.1])
            
            from core.spec_definitions import UAV_3D_spec
            spec = UAV_3D_spec(self.setup['noiseMultiplier'])   
            
        return spec
        
    def setTurbulenceNoise(self, folder, N):
        '''
        Set the turbulence noise samples for N samples

        Parameters
        ----------
        N : int
            Number of samples used.

        Returns
        -------
        None.

        '''
        
        samples = np.genfromtxt(folder + '/input/TurbulenceNoise_N=1000.csv', 
                                delimiter=',')
        
        self.noise['samples'] = samples
        
class building_2room(master.LTI_master):
    
    def __init__(self):
        '''
        Initialize the 2-zone building automation system (BAS) model class,
        which corresponds to the BAS benchmark in the paper.

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Load building automation system (BAS) parameters
        import core.BAS.parameters as BAS_class
        self.BAS = BAS_class.parameters()
        
        # Number of time steps to lump together (can be used to make the model
        # fully actuated)
        self.lump = 1
        
        # Discretization step size
        self.tau = 15 # NOTE: in minutes for BAS!

        BAS = self.BAS
        
        # Steady state values
        Twss        = BAS.Zone1['Twss'] + 5
        Pout1       = BAS.Radiator['Zone1']['Prad'] * 1.5
        Pout2       = BAS.Radiator['Zone2']['Prad'] * 1.5
        
        w1          = BAS.Radiator['w_r'] * 1.5
        w2          = BAS.Radiator['w_r'] * 1.5
        
        BAS.Zone1['Cz'] = BAS.Zone1['Cz']
        BAS.Zone1['Rn'] = BAS.Zone1['Rn']
        
        BAS.Zone2['Cz'] = BAS.Zone2['Cz']
        BAS.Zone2['Rn'] = BAS.Zone2['Rn']
        
        m1          = BAS.Zone1['m']
        m2          = BAS.Zone2['m']
        
        Rad_k1_z1   = BAS.Radiator['k1'] * 5
        Rad_k1_z2   = BAS.Radiator['k1'] * 5
        
        Rad_k0_z1   = BAS.Radiator['k0']
        Rad_k0_z2   = BAS.Radiator['k0']
        
        alpha1_z1   = BAS.Radiator['alpha1']
        alpha1_z2   = BAS.Radiator['alpha1']
        
        alpha2_z1   = BAS.Radiator['alpha1']
        alpha2_z2   = BAS.Radiator['alpha1']
        
        # Defining Deterministic Model corresponding matrices
        A_cont      = np.zeros((4,4));
        
        # Room 1
        A_cont[0,0] = ( -(1/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))-((Pout1*alpha2_z1 )/(BAS.Zone1['Cz'])) - ((m1*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz'])) - (1/(BAS.Zone1['Rn']*BAS.Zone1['Cz'])) )
        A_cont[0,2] = (Pout1*alpha2_z1 )/(BAS.Zone1['Cz'])
        
        # Room 2
        A_cont[1,1] = ( -(1/(BAS.Zone2['Rn']*BAS.Zone2['Cz']))-((Pout2*alpha2_z2 )/(BAS.Zone2['Cz'])) - ((m2*BAS.Materials['air']['Cpa'])/(BAS.Zone2['Cz'])) - (1/(BAS.Zone2['Rn']*BAS.Zone2['Cz'])) )
        A_cont[1,3] = (Pout2*alpha2_z2 )/(BAS.Zone2['Cz'])
        
        # Heat transfer room 1 <-> room 2
        A_cont[0,1] = ( (1/(BAS.Zone1['Rn']*BAS.Zone1['Cz'])) )
        A_cont[1,0] = ( (1/(BAS.Zone2['Rn']*BAS.Zone2['Cz'])) )
        
        # Radiator 1
        A_cont[2,0] = (Rad_k1_z1)
        A_cont[2,2] = ( -(Rad_k0_z1*w1) - Rad_k1_z1 )
        
        # Radiator 2
        A_cont[3,1] = (Rad_k1_z2)
        A_cont[3,3] = ( -(Rad_k0_z2*w2) - Rad_k1_z2 )

        B_cont      = np.zeros((4,4))
        B_cont[0,0] = (m1*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz'])
        B_cont[1,1] = (m2*BAS.Materials['air']['Cpa'])/(BAS.Zone2['Cz'])
        B_cont[2,2] = (Rad_k0_z1*w1) # < Allows to change the boiler temperature
        B_cont[3,3] = (Rad_k0_z2*w2) # < Allows to change the boiler temperature

        W_cont  = np.array([
                [ ((Twss)/(BAS.Zone1['Rn']*BAS.Zone1['Cz'])) + (alpha1_z1)/(BAS.Zone1['Cz']) ],
                [ ((Twss-2)/(BAS.Zone2['Rn']*BAS.Zone2['Cz'])) + (alpha1_z2)/(BAS.Zone1['Cz']) ],
                [ 0 ],
                [ 0 ]
                ])
        
        self.A = np.eye(4) + self.tau*A_cont
        self.B = B_cont*self.tau
        self.Q = W_cont*self.tau
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = 0.05*np.diag([0.2, 0.2, 0.2, 0.2])
                
        self.A_cont = A_cont
        self.B_cont = B_cont
        self.Q_cont = W_cont
        
    def set_spec(self):
        
        # Shortcut to boiler temperature        
        T_boiler = self.BAS.Boiler['Tswbss']
        
        from core.spec_definitions import building_2room_spec
        spec = building_2room_spec(T_boiler)        
            
        return spec
        
class building_1room(master.LTI_master):
    
    def __init__(self):
        '''
        Initialize the 1-zone building automation system (BAS) model class.
        Note that this is a downscaled version of the 2-zone model above.

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Let the user make a choice for the model dimension
        _, self.scenario  = ui.user_choice('Select the scenario to run',['Underactuated (original)','Fully actuated (modified)'])
           
        if self.scenario == 0:
            # Number of time steps to lump together (can be used to make the model
            # fully actuated)
            self.lump = 1
            
        else:
            # Number of time steps to lump together (can be used to make the model
            # fully actuated)
            self.lump = 2
            
            self.setup['partition_plot_action'] = 7700
        
        
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
        
        if self.lump == 1:
          # Let the user make a choice for the model dimension
          _, uncertainty  = ui.user_choice('Do you want to enable uncertainty about the radiator power output?',['No', 'Yes'])
        
          if uncertainty == 1:
            
            f1 = 0.75
            A0_cont      = np.zeros((2,2));
            A0_cont[0,0] = -(1/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))-((f1*Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])) - ((m1*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz']))
            A0_cont[0,1] = (f1*Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])
            A0_cont[1,0] = (k1_a)
            A0_cont[1,1] = -(k0_a*w) - k1_a
            
            f2 = 1.25
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
            
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = np.diag([ BAS.Zone1['Tz']['sigma'], BAS.Radiator['rw']['sigma'] ])

    def set_spec(self):
        
        from core.spec_definitions import building_1room_spec
        spec = building_1room_spec(self.scenario)        
            
        return spec

class shuttle(master.LTI_master):
    
    def __init__(self):
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
            
        return spec
        
class anaesthesia_delivery(master.LTI_master):
    
    def __init__(self):
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
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
                            [0.0002],
                            [0.00001] ])
        
        # Disturbance matrix
        self.Q  = np.array([[0], [0],[0]])
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)
        
        # Covariance of the process noise
        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A,1))*1e-3
        
    def set_spec(self):
        
        from core.spec_definitions import anaesthesia_delivery_spec
        spec = anaesthesia_delivery_spec()        
            
        return spec