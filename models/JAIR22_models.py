import numpy as np              # Import Numpy for computations
from core.preprocessing.define_gears_order import discretizeGearsMethod        
import core.preprocessing.master_classes as master
import scipy
from pathlib import Path

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
        self.noise['w_cov'] = np.diag([.01, .01, 0.001, .001, .001, .001])

        if args.nongaussian_noise:
            self.setTurbulenceNoise(args)

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