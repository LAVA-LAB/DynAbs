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
        self.setup['lump'] = 1
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] =  [-5]
        self.setup['control']['limits']['uMax'] =  [5]
        
        mode = 1
        
        # Partition size
        if mode == 0:
            self.setup['partition']['boundary']  = np.array([[-11, 11], [-11, 11]])
            self.setup['partition']['number']  = [11, 11]
            
            self.setup['specification']['goal'] = [
                np.array([[3, 9], 'all'])
                ]
            
            # Actions per dimension (if 'auto', equal to nr of regions)
            self.setup['targets']['boundary']    = self.setup['partition']['boundary']
            self.setup['targets']['number']      = self.setup['partition']['number']
            
        elif mode == 1:
            self.setup['partition']['boundary']  = np.array([[-10.5, 10.5], [-10.5, 10.5]])
            self.setup['partition']['number']  = [41, 41]
            
            self.setup['specification']['goal'] = [
                np.array([[-2, 2], [-2, 2]])
                ]
            
            # Actions per dimension (if 'auto', equal to nr of regions)
            self.setup['targets']['boundary']    = self.setup['partition']['boundary']
            self.setup['targets']['number']      = [21, 21]

        self.setup['specification']['critical'] = None #setStateBlock(self.setup['partition'], a=[-6,-4], b=[-4,-2]) #[[]]
        
        # Discretization step size
        self.tau = 1
        
        # Step-bound on property
        self.setup['endTime'] = 32 
    
    def setModel(self, observer):
        '''
        Set linear dynamical system.

        Parameters
        ----------
        observer : Boolean
            If True, an observer is created for the model.

        Returns
        -------
        None.

        '''
        
        # State transition matrix
        self.A  = np.array([[1, self.tau],
                            [0, .8]])
        
        self.A_set = [
                    np.array([[1, self.tau],[0,0.7]]),
                    np.array([[1, self.tau],[0,0.9]])
                    ]
        
        # Input matrix
        self.B  = np.array([[self.tau**2/2],
                                [self.tau]])
        
        if observer:
            # Observation matrix
            self.C          = np.array([[1, 0]])
            self.r          = len(self.C)
        
        # Disturbance matrix
        self.Q  = np.array([[0],[0]])
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)
        
        # Covariance of the process noise
        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A,1)) * 0.0001
        
        
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
        self.setup['lump'] = 1
        
        # Let the user make a choice for the model dimension
        self.modelDim, _  = ui.user_choice('model dimension',[2,3])
    
        if self.modelDim == 2:
    
            # Authority limit for the control u, both positive and negative
            self.setup['control']['limits']['uMin'] = [-2, -2]
            self.setup['control']['limits']['uMax'] = [2, 2]        
    
            self.setup['max_control_error'] = np.array([1, 10, 1, 10])
    
            V = 1
    
            if V == 1:
                
                # Partition size
                self.setup['partition']['boundary']  = np.array([[-11, 11], 
                                                                 [-7, 7], 
                                                                 [-11, 11], 
                                                                 [-7, 7]])
                self.setup['partition']['number']  = [11, 7, 11, 7]
                
                self.setup['targets']['boundary']    = self.setup['partition']['boundary']
                self.setup['targets']['number']      = self.setup['partition']['number']
                
                # Specification information
                self.setup['specification']['goal'] = [
                    np.array([[2, 6], 'all', [2, 6], 'all'])
                    ]
                
                self.setup['specification']['critical'] = None
                # np.vstack((
                #     setStateBlock(self.setup['partition'], a=[-6,-4,-2], b='all', c=[2], d='all'),
                #     setStateBlock(self.setup['partition'], a=[4,6], b='all', c=[-8,-6], d='all')
                #     ))
                
                self.setup['x0'] = np.array([-6,0,-6,0])
                
            elif V == 2:
                
                # Partition size
                self.setup['partition']['nrPerDim']  = [15,7,15,7]
                self.setup['partition']['width']     = [1, 1.5, 1, 1.5]
                self.setup['partition']['origin']    = [0, 0, 0, 0]
                
                # Actions per dimension (if 'auto', equal to nr of regions)
                self.setup['targets']['nrPerDim']    = 'auto'
                self.setup['targets']['domain']      = 'auto'
                
                # Specification information
                self.setup['specification']['goal'] = setStateBlock(self.setup['partition'], a=[4,5,6], b='all', c=[4,5,6], d='all')
                
                self.setup['specification']['critical'] = np.vstack((
                    setStateBlock(self.setup['partition'], a=[-6,-5,-4,-3,-2], b='all', c=[2,3], d='all'),
                    setStateBlock(self.setup['partition'], a=[5,6,7], b='all', c=[-7,-6,-5], d='all')
                    ))
                
                self.setup['x0'] = setStateBlock(self.setup['partition'], a=[-5], b=[0], c=[-5], d=[0])
            
        elif self.modelDim == 3:
            
            # Let the user make a choice for the model dimension
            self.setup['noiseMultiplier'], _  = ui.user_choice('process noise multiplier',[1,0.1])
            
            # Authority limit for the control u, both positive and negative
            self.setup['control']['limits']['uMin'] = [-4, -4, -4]
            self.setup['control']['limits']['uMax'] = [4, 4, 4]
            
            # Actions per dimension (if 'auto', equal to nr of regions)
            self.setup['targets']['nrPerDim']    = 'auto'
            self.setup['targets']['domain']      = 'auto'
            
            ## "Hole-in-the-wall" scenario (low vs. high noise)
            
            # Partition size
            self.setup['partition']['nrPerDim']  = [15, 3, 9, 3, 7, 3]
            self.setup['partition']['width']     = [2, 1.5, 2, 1.5, 2, 1.5]
            self.setup['partition']['origin']    = [0, 0, 0, 0, 0, 0]
            
            self.setup['x0'] = setStateBlock(self.setup['partition'], a=[-14], b=[0], c=[6], d=[0], e=[-2], f=[0])
            
            # Specification information
            self.setup['specification']['goal'] = setStateBlock(self.setup['partition'], a=[12,14], b='all', c=[2,4], d='all', e=[-6,-4], f='all')
            
            self.setup['specification']['critical']   = np.vstack((
                # Hole 1
                setStateBlock(self.setup['partition'], a=[-10,-8,-6], b='all', c=[0,2,6,8], d='all', e=[-6,-4,-2,0,2], f='all'),
                setStateBlock(self.setup['partition'], a=[-10,-8,-6], b='all', c=[6,8], d='all', e=[2,4], f='all'),
                setStateBlock(self.setup['partition'], a=[-10,-8,-6], b='all', c=[4], d='all', e=[-6], f='all'),
                
                # Hole 2
                setStateBlock(self.setup['partition'], a=[0,2], b='all', c=[2,4,6,8], d='all', e=[-6,-4,-2,4], f='all'),
                setStateBlock(self.setup['partition'], a=[0,2], b='all', c=[2,8], d='all', e=[0,2], f='all'),
                
                # Tower
                setStateBlock(self.setup['partition'], a=[0,2], b='all', c=[-2,0], d='all', e=[-6,-4,-2,-0,2,4,6], f='all'),
                
                # Wall between routes
                setStateBlock(self.setup['partition'], a=[4,6,8], b='all', c=[-2,0], d='all', e=[-6,-4,-2], f='all'),
                
                # Long route obstacles
                setStateBlock(self.setup['partition'], a=[-10,-8], b='all', c=[-4,-2], d='all', e=[-6,-4,-2,0], f='all'),
                setStateBlock(self.setup['partition'], a=[0,2], b='all', c=[-8,-6,-4], d='all', e=[-6], f='all'),
                
                # Overhanging
                setStateBlock(self.setup['partition'], a=[0,2], b='all', c=[-8,-6,-4], d='all', e=[4,6], f='all'),
                
                # Small last obstacle
                setStateBlock(self.setup['partition'], a=[12,14], b='all', c=[-8,-6], d='all', e=[-6], f='all'),
                
                # Obstacle next to goal
                setStateBlock(self.setup['partition'], a=[10,12,14], b='all', c=[6,8], d='all', e=[-6,-4,-2,0], f='all')
                ))
        
        else:
            print('No valid dimension for the drone model was provided')
            sys.exit()
        
        # Discretization step size
        self.tau = 2.0
        
        # Step-bound on property
        self.setup['endTime'] = 32

    def setModel(self, observer):
        '''
        Set linear dynamical system.

        Parameters
        ----------
        observer : Boolean
            If True, an observer is created for the model.

        Returns
        -------
        None.

        '''
        
        # State transition matrix
        Ablock = np.array([[1, self.tau],
                          [0, 0.8]])
        
        # Input matrix
        Bblock = np.array([[self.tau**2/2],
                           [self.tau]])
        
        if self.modelDim==3:
            self.A  = scipy.linalg.block_diag(Ablock, Ablock, Ablock)
            self.B  = scipy.linalg.block_diag(Bblock, Bblock, Bblock)
            
            # Disturbance matrix
            self.Q  = np.array([[0],[0],[0],[0],[0],[0]])
            
            if observer:
                # Observation matrix
                self.C          = np.array([[1, 0, 1, 0, 1, 0]])
                self.r          = len(self.C)
                
        else:
            self.A  = scipy.linalg.block_diag(Ablock, Ablock)
            self.B  = scipy.linalg.block_diag(Bblock, Bblock)
            
            # self.A_set = [
            #             self.A - np.diag([0, -0.1, 0, -0.1]),
            #             self.A - np.diag([0, 0.1, 0, 0.1])
            #             ]
        
            # Disturbance matrix
            self.Q  = np.array([[0],[0],[0],[0]])
        
            if observer:
                # Observation matrix
                self.C          = np.array([[1, 0, 1, 0]])
                self.r          = len(self.C)
            
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        # Covariance of the process noise
        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A,1))*0.1
           
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
        self.setup['lump'] = 1
        
        # Shortcut to boiler temperature        
        T_boiler = self.BAS.Boiler['Tswbss']
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] = [14, 14, T_boiler-10, T_boiler-10]
        self.setup['control']['limits']['uMax'] = [26, 26, T_boiler+10, T_boiler+10]
            
        # Partition size
        self.setup['partition']['nrPerDim']  = [21,21,9,9]
        self.setup['partition']['width']     = [0.2, 0.2, 0.2, 0.2]
        self.setup['partition']['origin']    = [20, 20, 38.3, 38.3]
        
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.setup['targets']['nrPerDim']    = 'auto'
        self.setup['targets']['domain']      = 'auto'
        
        # Specification information
        self.setup['specification']['goal'] = setStateBlock(self.setup['partition'], 
                                    a=[20], 
                                    b=[20], 
                                    c='all', d='all')
        
        self.setup['specification']['critical'] = [[]]

        # Discretization step size
        self.tau = 15 # NOTE: in minutes for BAS!
        
        # Step-bound on property
        self.setup['endTime'] = 32

    def setModel(self, observer):           
        '''
        Set linear dynamical system.

        Parameters
        ----------
        observer : Boolean
            If True, an observer is created for the model.

        Returns
        -------
        None.

        '''
        
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
        
        # Number of time steps to lump together (can be used to make the model
        # fully actuated)
        self.setup['lump'] = 1
        
        # Let the user make a choice for the model dimension
        _, gridType  = ui.user_choice('grid size',['19x20','40x40'])
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] = [14, -10]
        self.setup['control']['limits']['uMax'] = [28, 10]
            
        if gridType == 0:
            nrPerDim = [19, 20]
            width = [0.2, 0.2]
            goal = [21]
        else:
            nrPerDim = [40, 40]
            width = [0.1, 0.1]
            goal = [20.95, 21.05]
        
        # Partition size
        self.setup['partition']['nrPerDim']  = nrPerDim
        self.setup['partition']['width']     = width
        self.setup['partition']['origin']    = [21, 38]
        
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.setup['targets']['nrPerDim']    = 'auto'
        self.setup['targets']['domain']      = 'auto'
        
        # Specification information
        self.setup['specification']['goal'] = setStateBlock(self.setup['partition'], a=goal, b='all')
        
        self.setup['specification']['critical'] = [[]]
        
        # Discretization step size
        self.tau = 15 # NOTE: in minutes for BAS!
        
        # Step-bound on property
        self.setup['endTime'] = 64

    def setModel(self, observer):           
        '''
        Set linear dynamical system.

        Parameters
        ----------
        observer : Boolean
            If True, an observer is created for the model.

        Returns
        -------
        None.

        '''
        
        import core.BAS.parameters as BAS_class
        
        BAS = BAS_class.parameters()
        
        # Steady state values
        Tswb    = BAS.Boiler['Tswbss'] - 20
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
        
        B_cont      = np.zeros((2,2))
        B_cont[0,0] = (m1*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz'])
        B_cont[1,1] = (k0_a*w) # < Allows to change the boiler temperature

        
        W_cont  = np.array([
                [ (Twss/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))+ (BAS.Radiator['alpha1'])/(BAS.Zone1['Cz']) ],
                [ (k0_a*w*Tswb) ],
                ])
        
        self.A = np.eye(2) + self.tau*A_cont
        self.B = B_cont*self.tau
        self.Q = W_cont*self.tau
        
        self.A_set = [
            self.A - np.array([[0.01,0],[0,0]]),
            self.A + np.array([[0.01,0],[0,0]])
            ]
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = np.diag([ BAS.Zone1['Tz']['sigma'], BAS.Radiator['rw']['sigma'] ])
        
class building_1room_1control(master.LTI_master):
    
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
        _, scenario  = ui.user_choice('Select the scenario to run',['Underactuated (original)','Fully actuated (modified)'])
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] = [15]
        self.setup['control']['limits']['uMax'] = [30]
           
        if scenario == 0:
            # Number of time steps to lump together (can be used to make the model
            # fully actuated)
            self.setup['lump'] = 1
            
            partition_boundary    = np.array([[19.1, 22.9], [36, 40]])
            partition_number      = [19, 20]
            
            target_boundary       = np.array([[19.1, 22.9], [36, 40]])
            target_number         = [19, 40]
        else:
            # Number of time steps to lump together (can be used to make the model
            # fully actuated)
            self.setup['lump'] = 2
            
            partition_boundary    = np.array([[19.1, 22.9], [36, 40]])
            partition_number      = [190, 600]
            
            target_boundary       = np.array([[19.1, 22.9], [36, 40]])
            target_number         = [76, 200]
            
            self.setup['partition_plot_action'] = 7700
        
        self.setup['partition']['boundary'] = partition_boundary
        self.setup['partition']['number']   = partition_number
        self.setup['targets']['boundary']   = target_boundary
        self.setup['targets']['number']     = target_number
        
        
        self.setup['specification']['goal'] = [
            np.array([[20.8, 21.2], 'all'])
            ]
        self.setup['specification']['critical'] = None
        
        # Discretization step size
        self.tau = 20 # NOTE: in minutes for BAS!
        
        # Step-bound on property
        self.setup['endTime'] = 64
        
        # self.setup['max_control_error'] = np.array([0.1, 10])

    def setModel(self, observer):           
        '''
        Set linear dynamical system.

        Parameters
        ----------
        observer : Boolean
            If True, an observer is created for the model.

        Returns
        -------
        None.

        '''
        
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
        
        if self.setup['lump'] == 1:
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
            
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = np.diag([ BAS.Zone1['Tz']['sigma'], BAS.Radiator['rw']['sigma'] ])

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
        self.setup['lump'] = 2
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] = [-0.1, -0.1]
        self.setup['control']['limits']['uMax'] = [0.1, 0.1]
        
        # Partition size
        self.setup['partition']['nrPerDim']  = [20, 10, 4, 4]
        self.setup['partition']['width']     = [0.1, 0.1, 0.01, 0.01]
        self.setup['partition']['origin']    = [0, -0.5, 0.01, 0.01]
        
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.setup['targets']['nrPerDim']    = 'auto'
        self.setup['targets']['domain']      = 'auto'
        
        # Specification information
        self.setup['specification']['goal'] = setStateBlock(self.setup['partition'], a=[-0.05, 0.05], b=[-0.05], c=[0], d=[0])
        
        self.setup['specification']['critical'] = np.vstack((
                setStateBlock(self.setup['partition'], a=np.array([[i, -i] for i in np.arange(-.95, -.05, .1)]).flatten(), b=[-.05], c='all', d='all'),
                setStateBlock(self.setup['partition'], a=np.array([[i, -i] for i in np.arange(-.95, -.05, .1)]).flatten(), b=[-.15], c='all', d='all'),
                setStateBlock(self.setup['partition'], a=np.array([[i, -i] for i in np.arange(-.95, -.15, .1)]).flatten(), b=[-.25], c='all', d='all'),
                setStateBlock(self.setup['partition'], a=np.array([[i, -i] for i in np.arange(-.95, -.25, .1)]).flatten(), b=[-.35], c='all', d='all'),
                setStateBlock(self.setup['partition'], a=np.array([[i, -i] for i in np.arange(-.95, -.35, .1)]).flatten(), b=[-.45], c='all', d='all'),
                setStateBlock(self.setup['partition'], a=np.array([[i, -i] for i in np.arange(-.95, -.45, .1)]).flatten(), b=[-.55], c='all', d='all'),
                setStateBlock(self.setup['partition'], a=np.array([[i, -i] for i in np.arange(-.95, -.55, .1)]).flatten(), b=[-.65], c='all', d='all'),
                setStateBlock(self.setup['partition'], a=np.array([[i, -i] for i in np.arange(-.95, -.65, .1)]).flatten(), b=[-.75], c='all', d='all'),
                setStateBlock(self.setup['partition'], a=np.array([[i, -i] for i in np.arange(-.95, -.75, .1)]).flatten(), b=[-.85], c='all', d='all'),
                setStateBlock(self.setup['partition'], a=np.array([[i, -i] for i in np.arange(-.95, -.85, .1)]).flatten(), b=[-.95], c='all', d='all')
                ))
        
        # Discretization step size
        self.tau = 1 # NOTE: in minutes for BAS!
        
        # Step-bound on property
        self.setup['endTime'] = 16
        
        self.modelDim = 2

    def setModel(self, observer):           
        '''
        Set linear dynamical system.

        Parameters
        ----------
        observer : Boolean
            If True, an observer is created for the model.

        Returns
        -------
        None.

        '''
        
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
        
class anaesthesia_delivery(master.LTI_master):
    
    def __init__(self):
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Number of time steps to lump together (can be used to make the model
        # fully actuated)
        self.setup['lump'] = 1
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] =  [0]
        self.setup['control']['limits']['uMax'] =  [40]
        
        # Partition size
        self.setup['partition']['nrPerDim']  = [20, 40, 11]
        self.setup['partition']['width']     = [0.25, 0.25, 1]
        self.setup['partition']['origin']    = [3.5, 5, 5]
        
        # Actions per dimension (if 'auto', equal to nr of regions)
        self.setup['targets']['nrPerDim']    = 'auto'
        self.setup['targets']['domain']      = 'auto'

        # Specification information
        self.setup['specification']['goal']           = setStateBlock(self.setup['partition'], a=[4.25, 4.75, 5.25, 5.75], b=[8.5, 9.5], c=[8.5, 9.5])
        self.setup['specification']['critical']       = [[]]
        
        # Discretization step size
        self.tau = 1 #3 * 20 / 60 # 20 seconds (base is 1 minute)
        
        # Step-bound on property
        self.setup['endTime'] = 10
    
    def setModel(self, observer):
        '''
        Set linear dynamical system.

        Parameters
        ----------
        observer : Boolean
            If True, an observer is created for the model.

        Returns
        -------
        None.

        '''
        
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
        
        if observer:
            # Observation matrix
            self.C          = np.array([[1, 0]])
            self.r          = len(self.C)
        
        # Disturbance matrix
        self.Q  = np.array([[0], [0],[0]])
        
        # k10 = 0.4436
        # k12 = 0.1140
        # k13 = 0.0419
        # k21 = 0.0550
        # k31 = 0.0033
        # V1  = 16.044
        
        # A_cont = np.array([
        # [-(k10 + k12 + k13),    k12,       k13],
        # [k21,                  -k21,       0],
        # [k31,                   0,         -k31]
        # ])
        
        # B_cont  = np.array([[1/V1],
        #                     [0],
        #                     [0]])
        
        # # Disturbance matrix
        # Q_cont = np.array([[0], [0],[0]])
        
        # self.A, self.B, self.Q = discretizeGearsMethod(A_cont, B_cont, Q_cont, self.tau)
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)
        
        # Covariance of the process noise
        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A,1))*1e-3