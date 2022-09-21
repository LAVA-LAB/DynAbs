#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Models used in the paper:
 "Robust Control for Dynamical Systems with Non-Gaussian Noise via
 Formal Abstractions"

Originally coded by:        Thom Badings
Contact e-mail address:     thom.badings@ru.nl
______________________________________________________________________________
"""

import numpy as np              # Import Numpy for computations
from core.preprocessing.define_gears_order import discretizeGearsMethod        
import core.preprocessing.master_classes as master
import scipy
from pathlib import Path

class robot(master.LTI_master):
    
    def __init__(self, args):
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
        self.lump = 2
        
        # Discretization step size
        self.tau = 1
        
        # State transition matrix
        self.A  = np.array([[1, self.tau],
                                [0, 1]])
        
        # Input matrix
        self.B  = np.array([[self.tau**2/2],
                                [self.tau]])
        
        self.Q = np.array([[0],[0]]) #np.zeros((2,1))
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A,1))*0.15
        
    def set_spec(self):
        
        from models.JAIR22_specifications import robot_spec
        spec = robot_spec()        
            
        spec.problem_type = 'reachavoid'
        
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
        self.tau = 1
        
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
        self.noise['w_cov'] = np.diag([ 1e-4, 1e-4, 5e-8, 5e-8 ])
        
    def set_spec(self):
        
        from models.JAIR22_specifications import shuttle_spec
        spec = shuttle_spec()        
            
        spec.problem_type = 'reachavoid'
        
        return spec



class building_2room(master.LTI_master):
    
    def __init__(self, args):
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

        # Shortcut to boiler temperature        
        self.T_boiler = self.BAS.Boiler['Tswbss']
        
        # Discretization step size
        self.tau = 15 # NOTE: in minutes for BAS!
        
        # Steady state values
        Twss        = self.BAS.Zone1['Twss'] + 5
        Pout1       = self.BAS.Radiator['Zone1']['Prad'] * 1.5
        Pout2       = self.BAS.Radiator['Zone2']['Prad'] * 1.5
        
        w1          = self.BAS.Radiator['w_r'] * 1.5
        w2          = self.BAS.Radiator['w_r'] * 1.5
        
        self.BAS.Zone1['Cz'] = self.BAS.Zone1['Cz']
        self.BAS.Zone1['Rn'] = self.BAS.Zone1['Rn']
        
        self.BAS.Zone2['Cz'] = self.BAS.Zone2['Cz']
        self.BAS.Zone2['Rn'] = self.BAS.Zone2['Rn']
        
        m1          = self.BAS.Zone1['m']
        m2          = self.BAS.Zone2['m']
        
        Rad_k1_z1   = self.BAS.Radiator['k1'] * 5
        Rad_k1_z2   = self.BAS.Radiator['k1'] * 5
        
        Rad_k0_z1   = self.BAS.Radiator['k0']
        Rad_k0_z2   = self.BAS.Radiator['k0']
        
        alpha1_z1   = self.BAS.Radiator['alpha1']
        alpha1_z2   = self.BAS.Radiator['alpha1']
        
        alpha2_z1   = self.BAS.Radiator['alpha1']
        alpha2_z2   = self.BAS.Radiator['alpha1']
        
        # Defining Deterministic Model corresponding matrices
        A_cont      = np.zeros((4,4));
        
        # Room 1
        A_cont[0,0] = ( -(1/(self.BAS.Zone1['Rn']*self.BAS.Zone1['Cz']))-((Pout1*alpha2_z1 )/(self.BAS.Zone1['Cz'])) - ((m1*self.BAS.Materials['air']['Cpa'])/(self.BAS.Zone1['Cz'])) - (1/(self.BAS.Zone1['Rn']*self.BAS.Zone1['Cz'])) )
        A_cont[0,2] = (Pout1*alpha2_z1 )/(self.BAS.Zone1['Cz'])
        
        # Room 2
        A_cont[1,1] = ( -(1/(self.BAS.Zone2['Rn']*self.BAS.Zone2['Cz']))-((Pout2*alpha2_z2 )/(self.BAS.Zone2['Cz'])) - ((m2*self.BAS.Materials['air']['Cpa'])/(self.BAS.Zone2['Cz'])) - (1/(self.BAS.Zone2['Rn']*self.BAS.Zone2['Cz'])) )
        A_cont[1,3] = (Pout2*alpha2_z2 )/(self.BAS.Zone2['Cz'])
        
        # Heat transfer room 1 <-> room 2
        A_cont[0,1] = ( (1/(self.BAS.Zone1['Rn']*self.BAS.Zone1['Cz'])) )
        A_cont[1,0] = ( (1/(self.BAS.Zone2['Rn']*self.BAS.Zone2['Cz'])) )
        
        # Radiator 1
        A_cont[2,0] = (Rad_k1_z1)
        A_cont[2,2] = ( -(Rad_k0_z1*w1) - Rad_k1_z1 )
        
        # Radiator 2
        A_cont[3,1] = (Rad_k1_z2)
        A_cont[3,3] = ( -(Rad_k0_z2*w2) - Rad_k1_z2 )

        B_cont      = np.zeros((4,4))
        B_cont[0,0] = (m1*self.BAS.Materials['air']['Cpa'])/(self.BAS.Zone1['Cz'])
        B_cont[1,1] = (m2*self.BAS.Materials['air']['Cpa'])/(self.BAS.Zone2['Cz'])
        B_cont[2,2] = (Rad_k0_z1*w1) # < Allows to change the boiler temperature
        B_cont[3,3] = (Rad_k0_z2*w2) # < Allows to change the boiler temperature

        W_cont  = np.array([
                [ ((Twss)/(self.BAS.Zone1['Rn']*self.BAS.Zone1['Cz'])) + (alpha1_z1)/(self.BAS.Zone1['Cz']) ],
                [ ((Twss-2)/(self.BAS.Zone2['Rn']*self.BAS.Zone2['Cz'])) + (alpha1_z2)/(self.BAS.Zone1['Cz']) ],
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
        
        from models.JAIR22_specifications import building_2room_spec
        spec = building_2room_spec(self.T_boiler)        
            
        spec.problem_type = 'reachavoid'
        
        return spec



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
        
        # Load building automation system (BAS) parameters
        import core.BAS.parameters as BAS_class
        self.BAS = BAS_class.parameters()

        # Number of time steps to lump together (can be used to make the model
        # fully actuated)
        self.lump = 1
        
        # Discretization step size
        self.tau = 15 # NOTE: in minutes for BAS!
        
        # Steady state values
        Tswb    = self.BAS.Boiler['Tswbss'] - 20
        Twss    = self.BAS.Zone1['Twss']
        Pout1   = self.BAS.Radiator['Zone1']['Prad']      
        
        w       = self.BAS.Radiator['w_r']
        
        self.BAS.Zone1['Cz'] = self.BAS.Zone1['Cz']
        
        m1      = self.BAS.Zone1['m'] # Proportional factor for the air conditioning
        
        k1_a    = self.BAS.Radiator['k1']
        k0_a    = self.BAS.Radiator['k0'] #Proportional factor for the boiler temp. on radiator temp.
        
        # Defining Deterministic Model corresponding matrices
        A_cont      = np.zeros((2,2));
        A_cont[0,0] = -(1/(self.BAS.Zone1['Rn']*self.BAS.Zone1['Cz']))-((Pout1*self.BAS.Radiator['alpha2'] )/(self.BAS.Zone1['Cz'])) - ((m1*self.BAS.Materials['air']['Cpa'])/(self.BAS.Zone1['Cz']))
        A_cont[0,1] = (Pout1*self.BAS.Radiator['alpha2'] )/(self.BAS.Zone1['Cz'])
        A_cont[1,0] = (k1_a)
        A_cont[1,1] = -(k0_a*w) - k1_a
        
        B_cont      = np.zeros((2,2))
        B_cont[0,0] = (m1*self.BAS.Materials['air']['Cpa'])/(self.BAS.Zone1['Cz'])
        B_cont[1,1] = (k0_a*w) # < Allows to change the boiler temperature

        
        W_cont  = np.array([
                [ (Twss/(self.BAS.Zone1['Rn']*self.BAS.Zone1['Cz']))+ (self.BAS.Radiator['alpha1'])/(self.BAS.Zone1['Cz']) ],
                [ (k0_a*w*Tswb) ],
                ])
        
        self.A = np.eye(2) + self.tau*A_cont
        self.B = B_cont*self.tau
        self.Q = W_cont*self.tau
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = np.diag([ self.BAS.Zone1['Tz']['sigma'], self.BAS.Radiator['rw']['sigma'] ])
        
    def set_spec(self):
        
        from models.JAIR22_specifications import building_1room_spec
        spec = building_1room_spec()        
            
        spec.problem_type = 'reachavoid'
        
        return spec



class UAV(master.LTI_master):
    
    def __init__(self, args):
        '''
        Initialize the UAV model class, which can be 2D or 3D. The 3D case
        corresponds to the UAV benchmark in the paper.

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        self.args = args

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 2
        
        # Let the user make a choice for the model dimension
        self.modelDim = args.UAV_dim
        
        # Discretization step size
        self.tau = 1.0

        # State transition matrix
        Ablock = np.array([[1, self.tau],
                          [0, 1]])
        
        # Input matrix
        Bblock = np.array([[self.tau**2/2],
                           [self.tau]])
        
        if self.modelDim==3:
            self.A  = scipy.linalg.block_diag(Ablock, Ablock, Ablock)
            self.B  = scipy.linalg.block_diag(Bblock, Bblock, Bblock)
            
            # Disturbance matrix
            self.Q  = np.array([[0],[0],[0],[0],[0],[0]])
                
        else:
            self.A  = scipy.linalg.block_diag(Ablock, Ablock)
            self.B  = scipy.linalg.block_diag(Bblock, Bblock)
        
            # Disturbance matrix
            self.Q  = np.array([[0],[0],[0],[0]])
            
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        # Covariance of the process noise
        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A,1))*0.15

        if args.nongaussian_noise:
            self.setTurbulenceNoise(args)

    def set_spec(self):
        
        from models.JAIR22_specifications import UAV_spec
        spec = UAV_spec(self.args, self.modelDim)     
        
        spec.problem_type = 'reachavoid'
            
        return spec
           
    def setTurbulenceNoise(self, args):
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
        
        path = Path(args.base_dir, 'input/TurbulenceNoise_N=1000.csv')

        samples = np.genfromtxt(path, delimiter=',')
        
        self.noise['samples'] = args.noise_factor * np.vstack(
            (2*samples[:,0],
                0.2*samples[:,0],
                2*samples[:,1],
                0.2*samples[:,1],
                2*samples[:,2],
                0.2*samples[:,2])).T



class spacecraft(master.LTI_master):
    
    def __init__(self, args):
        '''
        Initialize the spacecraft model class.

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        self.args = args

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 2
        
        # Discretization step size
        self.tau = 1.0
        T = self.tau

        mu = 3.986e14 / 1e9
        r0 = 42164
        n = np.sqrt(mu / r0**3)

        self.A = np.array([
            [4-3*np.cos(n*T),       0,          0,              1/n*np.sin(n*T),        2/n*(1-np.cos(n*T)),        0],
            [6*(np.sin(n*T) - n*T), 1,          0,              -2/n*(1-np.cos(n*T)),   1/n*(4*np.sin(n*T)-3*n*T),  0],
            [0,                     0,          np.cos(n*T),    0,                      0,                          1/n*np.sin(n*T)],
            [3*n*np.sin(n*T),       0,          0,              np.cos(n*T),            2*np.sin(n*T),              0],
            [-6*n*(1-np.cos(n*T)),  0,          0,              -2*np.sin(n*T),         4*np.cos(n*T)-3,            0],
            [0,                     0,          -n*np.sin(n*T), 0,                      0,                          np.cos(n*T)]
        ])

        self.B = np.array([
            [1/n*np.sin(n*T),       2/n*(1-np.cos(n*T)),        0],
            [-2/n*(1-np.cos(n*T)),  1/n*(4*np.sin(n*T)-3*n*T),  0],
            [0,                     0,                          1/n*np.sin(n*T)],
            [np.cos(n*T),           2*np.sin(n*T),              0],
            [-2*np.sin(n*T),        4*np.cos(n*T)-3,            0],
            [0,                     0,                          np.cos(n*T)]
        ])

        self.Q  = np.array([[0],[0],[0],[0],[0],[0]])
            
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        # Covariance of the process noise
        self.noise = dict()
        self.noise['w_cov'] = np.diag([.1, .1, .01, .01, .01, .01])
        
    def set_spec(self):
        
        from models.JAIR22_specifications import spacecraft_spec
        spec = spacecraft_spec(self.args)     
        
        spec.problem_type = 'reachavoid'
            
        return spec



class spacecraft_2D(master.LTI_master):
    
    def __init__(self, args):
        '''
        Initialize the spacecraft model class.

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        self.args = args

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 2
        
        # Discretization step size
        self.tau = 1
        T = self.tau

        mu = 3.986e14 / 1e9
        r0 = 42164
        n = np.sqrt(mu / r0**3)

        self.A = np.array([
            [4-3*np.cos(n*T),       0,          1/n*np.sin(n*T),        2/n*(1-np.cos(n*T))       ],
            [6*(np.sin(n*T) - n*T), 1,          -2/n*(1-np.cos(n*T)),   1/n*(4*np.sin(n*T)-3*n*T) ],
            [3*n*np.sin(n*T),       0,          np.cos(n*T),            2*np.sin(n*T)             ],
            [-6*n*(1-np.cos(n*T)),  0,          -2*np.sin(n*T),         4*np.cos(n*T)-3           ]
        ])

        self.B = np.array([
            [1/n*np.sin(n*T),       2/n*(1-np.cos(n*T))       ],
            [-2/n*(1-np.cos(n*T)),  1/n*(4*np.sin(n*T)-3*n*T) ],
            [np.cos(n*T),           2*np.sin(n*T)             ],
            [-2*np.sin(n*T),        4*np.cos(n*T)-3           ],
        ])

        self.Q  = np.array([[0],[0],[0],[0]])
            
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        # Covariance of the process noise
        self.noise = dict()
        self.noise['w_cov'] = np.diag([.01, .01, 0.001, .001]) #np.eye(np.size(self.A,1))*0.01

        if args.nongaussian_noise:
            self.setTurbulenceNoise(args)

    def set_spec(self):
        
        from models.JAIR22_specifications import spacecraft_2D_spec
        spec = spacecraft_2D_spec(self.args)     
        
        spec.problem_type = 'reachavoid'
            
        return spec



class spacecraft_1D(master.LTI_master):
    
    def __init__(self, args):
        '''
        Initialize the spacecraft model class.

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        self.args = args

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 2
        
        # Discretization step size
        self.tau = 1.0
        T = self.tau

        mu = 3.986e14 / 1e9
        r0 = 42164
        n = np.sqrt(mu / r0**3)

        self.A = np.array([
            [np.cos(n*T),    1/n*np.sin(n*T)],
            [-n*np.sin(n*T), np.cos(n*T)]
        ])

        self.B = np.array([
            [1/n*np.sin(n*T)],
            [np.cos(n*T)]
        ])

        self.Q  = np.array([[0],[0]])
            
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        # Covariance of the process noise
        self.noise = dict()
        self.noise['w_cov'] = np.diag([.001, .001])

    def set_spec(self):
        
        from models.JAIR22_specifications import spacecraft_1D_spec
        spec = spacecraft_1D_spec(self.args)     
        
        spec.problem_type = 'reachavoid'
            
        return spec