from core.abstraction import Abstraction
from core.define_model import define_model

import numpy as np              # Import Numpy for computations
import itertools                # Import to create iterators
from copy import deepcopy       # Import to copy variables in Python
from progressbar import progressbar # Import to create progress bars
from scipy.spatial import Delaunay # Import to create convex hulls
from operator import itemgetter

from .action_classes import backreachset, partial_model
from .compute_probabilities import compute_intervals_default

from core.define_partition import computeRegionIdx
from core.commons import tic, ticDiff, tocDiff, table, in_hull
from core.scenario_approach import load_scenario_table

class abstraction_default(Abstraction):

    def __init__(self, args, setup, model_raw, spec_raw):
        '''
        Initialize scenario-based abstraction (Ab) object

        Parameters
        ----------
        setup : dict
            Setup dictionary.
        model : dict
            Base model for which to create the abstraction.

        Returns
        -------
        None.

        '''
        
        model, spec = define_model(setup, model_raw, spec_raw)

        # Copy setup to internal variable
        self.setup = setup
        self.args  = args
        self.model = model
        self.spec  = spec
        
        # Start timer
        tic()
        ticDiff()
        self.time = dict()
        self.flags = {}
        
        Abstraction.__init__(self)
    


    def define_backreachsets(self):

        # For a default abstraction, the target of an action is a point,
        # so there is only a single backreachset.

        # Compute the backward reachable set objects
        self.actions['backreach_obj']['default'] = backreachset(name='default')
            
        # Compute the zero-shifted inflated backward reachable set
        self.actions['backreach_obj']['default'].compute_default_set(self.model)    



    def get_enabled_actions(self, model, spec, 
                            dim_n=False, dim_p=False, verbose=False,
                            print_every=39):
        
        # Compute the backward reachable set (not accounting for target point yet)    
        if dim_n is False or dim_p is False or len(dim_n) == model.n:
            dim_n = np.arange(model.n)
            dim_p = np.arange(model.p)
            compositional = False
            
        else:
            dim_excl = np.array([i for i in range(model.n) if i not in dim_n])
            
            model, spec = partial_model(self.flags, deepcopy(model), deepcopy(spec), dim_n, dim_p)
            compositional = True
            
        
        enabled = {}
        enabled_inv = {}   

        #############################

        from .commons import angle_between
        
        def f7(seq):
            seen = set()
            seen_add = seen.add
            return [x for x in seq if not (x in seen or seen_add(x))]

        # Compute inverse reachability area        
        x_inv_area = self.actions['backreach_obj']['default'].verts[:, dim_n]
        x_inv_area = np.unique(x_inv_area, axis=0)

        enabled_polypoints = dict()

        nr_corners = 2**model.n

        # Check if dimension of control area equals that if the state vector
        dimEqual = model.p == model.n

        # Determine the index tuples of all possible successor states
        upper = np.ones(self.model.n)
        upper[dim_n] = spec.partition['number']
        
        # Find list of tuples of the states that must be considered for this partial model
        state_tuples = list(itertools.product(*map(range, 
                            np.zeros(self.model.n, dtype=int), 
                            upper.astype(int))))

        # Also find the corresponding absolute indices of these states
        state_idxs   = itemgetter(*state_tuples)(self.partition['R']['idx'])
        
        # Retrieve all corners corresponding to these regions
        selected_regions = [np.unique(r[:, dim_n], axis=0) for r in self.partition['allCorners'][state_idxs,:,:]]
        region_corners = np.concatenate(selected_regions)

        if dimEqual:
            
            print(' -- Computing inverse basis vectors...')
            # Use preferred method: map back the skewed image to squares
            
            basis_vectors = defBasisVectors(model, verbose=verbose)   
            
            if verbose:
                for i,v1 in enumerate(basis_vectors):
                    for j,v2 in enumerate(basis_vectors):
                        if i != j:
                            print(' ---- Angle between control',i,'and',j,':',
                                  angle_between(v1,v2) / np.pi * 180)
            
            parralelo2cube = np.linalg.inv( basis_vectors )
            
            x_inv_area_normalized = x_inv_area @ parralelo2cube
            
            predSet_originShift = -np.average(x_inv_area_normalized, axis=0)
            if verbose:
                print('Transformation:',parralelo2cube)            
                print('Normal inverse area:',x_inv_area)
                
                print('Normalized hypercube:',x_inv_area_normalized)
                
                print('Off origin:',predSet_originShift)
            
                print('Shifted normalized hypercube:',x_inv_area @ parralelo2cube
                        + predSet_originShift)
                
            allRegionVertices = region_corners @ parralelo2cube \
                    - predSet_originShift

        else:
            
            print(' -- Creating inverse hull...')
            
            # Use standard method: check if points are in (skewed) hull
            x_inv_hull = Delaunay(x_inv_area, qhull_options='QJ')
            
            if verbose:
                print('Normal inverse area:',x_inv_area)
        
            allRegionVertices = region_corners

        # For every action
        for a_tup,a_idx in self.actions['tup2idx'].items(): #progressbar(enumerate(actions['tup2idx']), redirect_stdout=True):
            
            # If we are in compositional mode, only check this action if all 
            # excluded dimensions are zero        
            if compositional and any(np.array(a_tup)[dim_excl] != 0):
                continue
            
            # Get reference to current action object
            act   = self.actions['obj'][a_idx]
            
            if dimEqual:

                # Shift the origin points (instead of the target point)
                A_inv_d = model.A_inv @ np.array(act.center[dim_n])
                
                # Python implementation
                allVerticesNormalized = (A_inv_d @ parralelo2cube) - \
                                         allRegionVertices

                # Reshape the whole matrix that we obtain
                poly_reshape = np.reshape( allVerticesNormalized,
                                (len(state_idxs), 
                                 nr_corners*model.n))
                
                # Enabled actions are ones that have all corner points within
                # the origin-centered hypercube with unit length
                enabled_in = np.maximum(np.max(poly_reshape, axis=1), 
                                        -np.min(poly_reshape, axis=1)) <= 1.0

            else:
                
                # Shift the origin points (instead of the target point)
                A_inv_d = model.A_inv @ np.array(act.center[dim_n])
            
                # Subtract the shift from all corner points
                allVertices = A_inv_d - allRegionVertices
            
                # Check which points are in the convex hull
                polypoints_vec = in_hull(allVertices, x_inv_hull)
            
                # Map enabled vertices of the partitions to actual partitions
                enabled_polypoints = np.reshape(  polypoints_vec, 
                              (len(state_idxs), nr_corners))
            
                # Polypoints contains the True/False results for all vertices
                # of every partition. An action is enabled in a state if it 
                # is enabled in all its vertices
                enabled_in = np.all(enabled_polypoints == True, 
                                    axis = 1)

            enabled_in_tups = [tuple(j) for j in np.array(state_tuples)[enabled_in]]
            enabled_in_idxs = np.array(state_idxs)[enabled_in]

            enabled_inv[a_tup] = set(enabled_in_tups)

            for s_tup in enabled_in_tups:
                # Enable the current action in the current state
                if s_tup in enabled:
                    enabled[s_tup].add(a_tup)
                else:
                    enabled[s_tup] = {a_tup}
                
            if a_idx % print_every == 0:
                print('Action to',act.center[dim_n],'enabled in',len(enabled_in_tups),'states') 
                #print(self.partition['R']['center'][enabled_in_idxs])
                
        return enabled, enabled_inv, None



    def set_extra_actions(self):

        return



    def _computeProbabilityBounds(self, tab, k):
        '''
        Compute transition probability intervals (bounds)

        Parameters
        ----------
        tab : dict
            Table dictionary.
        k : int
            Discrete time step.

        Returns
        -------
        prob : dict
            Dictionary containing the computed transition probabilities.

        '''
        
        prob = dict()
        printEvery = min(100, max(1, int(self.actions['nr_actions']/10)))

        # Compute Gaussian noise samples
        noise_samples = np.random.multivariate_normal(
                        np.zeros(self.model.n), self.model.noise['w_cov'], 
                        size=self.args.noise_samples)

        # For every action (i.e. target point)
        for a_idx, act in self.actions['obj'].items():
            
            # Check if action a is available in any state at all
            if len(act.enabled_in) > 0:
                    
                successor_samples = act.center + noise_samples
                    
                prob[a_idx] = compute_intervals_default(self.args,
                      self.spec.partition, self.partition, self.trans,
                      successor_samples)
                
                # Print normal row in table
                if a_idx % printEvery == 0:
                    nr_transitions = len(prob[a_idx]['successor_idxs'])
                    tab.print_row([k, a_idx, 
                       'Probabilities computed (transitions: '+
                       str(nr_transitions)+')'])
                
        return prob

    
    
    def define_probabilities(self):
        '''
        Define the transition probabilities of the finite-state abstraction 
        (perform for every iteration of the iterative scheme).

        Returns
        -------
        None.

        '''
           
        # Column widths for tabular prints
        col_width = [8,8,8,46]
        tab = table(col_width)
        
        self.trans = {'prob': {}}
                
        print(' -- Loading scenario approach table...')
        
        tableFile = self.setup.directories['base'] + '/input/SaD_probabilityTable_N='+ \
                        str(self.args.noise_samples)+'_beta='+ \
                        str(self.args.confidence)+'.csv'
        
        # Load scenario approach table
        self.trans['memory'] = load_scenario_table(tableFile = tableFile,
                                                   k = self.args.noise_samples)
        
        print('Computing transition probabilities...')
        
        self.trans['prob'] = {0: self._computeProbabilityBounds(tab, 0)}
        
        self.time['3_probabilities'] = tocDiff(False)
        print('Transition probabilities calculated - time:',
              self.time['3_probabilities'])
        


def defBasisVectors(model, verbose=False):
        '''
        Compute the basis vectors of the predecessor set, computed from the
        average control inputs to the maximum in every dimension of the
        control space.
        Note that the drift does not play a role here.  

        Parameters
        ----------
        verbose : Bool
            If True, provide verbose output.

        Returns
        -------
        basis_vectors : 2D Numpy array
            Numpy array of basis vectors (every row is a vector).

        '''
        
        u_avg = np.array(model.uMax + model.uMin)/2    

        # Compute basis vectors (with average of all controls as origin)
        u = np.tile(u_avg, (model.n,1)) + \
            np.diag(model.uMax - u_avg)

        origin = model.A_inv @ \
                (model.B @ np.array(u_avg).T)   
                
        basis_vectors = np.zeros((model.n, model.n))
        
        for i,elem in enumerate(u):
            
            # Calculate inverse image of the current extreme control input
            point = model.A_inv @ \
                (model.B @ elem.T)    
            
            basis_vectors[i,:] = point - origin
        
            if verbose:
                print(' ---- Length of basis',i,':',
                  np.linalg.norm(basis_vectors[i,:]))
        
        return basis_vectors


        
def exclude_samples(samples, width):
    
    N,n = samples.shape
    
    S = np.reshape(samples, (N,n,1))
    diff = S - S.T
    width_tile = np.tile(width, (N,1)).T
    boolean = np.any(diff > width_tile, axis=1) | np.any(diff < -width_tile, axis=1)
    
    mp = map(np.nonzero, boolean)
    exclude = [set(m[0]) for m in mp]
    
    return exclude