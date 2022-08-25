from core.abstraction import Abstraction
from core.define_model import define_model

import numpy as np              # Import Numpy for computations
import itertools
from copy import deepcopy
from progressbar import progressbar # Import to create progress bars

from .action_classes import backreachset, partial_model, epistemic_error, rotate_2D_vector
from .compute_probabilities import compute_intervals_error

from core.define_partition import computeRegionIdx
from core.commons import overapprox_box, tic, ticDiff, tocDiff, table
from core.cvx_opt import LP_vertices_contained
from core.scenario_approach import load_scenario_table

class abstraction_epistemic(Abstraction):

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

        for name, target_set in self.spec.error['target_set_size'].items():
            # Compute the backward reachable set objects
            self.actions['backreach_obj'][name] = backreachset(name, target_set)
            
            # Compute the zero-shifted inflated backward reachable set
            self.actions['backreach_obj'][name].compute_default_set(self.model)    



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
        error = {}
        
        # Create LP object
        BRS_0 = self.actions['backreach_obj']['default'].verts_infl
        LP = LP_vertices_contained(model, 
                                np.unique(BRS_0[:, dim_n], axis=0).shape, 
                                solver=self.setup.cvx['solver'])
        
        if self.flags['parametric']:
            epist = epistemic_error(model)
        else:
            epist = None
        
        # If the (noninflated) backward reachable set is a line, compute the rot.
        # matrix. Used to make computation of enabled actions faster, by reducing
        # the number of potentially active actions.
        if len(self.actions['backreach_obj']['default'].verts) == 2 and len(dim_n) == 2:
            verts = self.actions['backreach_obj']['default'].verts[:, dim_n]
            brs_0_shift = verts - verts[0]
            BRS_0_shift = BRS_0[:, dim_n] - verts[0]
            
            ROT = rotate_2D_vector(brs_0_shift[1] - brs_0_shift[0])
            distances = (ROT @ (BRS_0_shift[:, dim_n]).T).T
            max_distance_to_brs = np.max(np.abs(distances[:, 1:]))
            
        else:
            ROT = None
        
        # For every action
        for a_tup,a_idx in self.actions['tup2idx'].items(): #progressbar(enumerate(actions['tup2idx']), redirect_stdout=True):
            
            # If we are in compositional mode, only check this action if all 
            # excluded dimensions are zero        
            if compositional and any(np.array(a_tup)[dim_excl] != 0):
                continue
            
            # Get reference to current action object
            act   = self.actions['obj'][a_idx]
            
            # Get backward reachable set
            BRS = np.unique(act.backreach_infl[:, dim_n], axis=0)
            brs_basis = act.backreach[0, dim_n]
            
            # Set current backward reachable set as parameter
            LP.set_backreach(BRS)
            
            # Overapproximate the backward reachable set
            BRS_overapprox_box = overapprox_box(BRS)
            
            if 'max_action_distance' in spec.error:
                BRS_overapprox_box = np.maximum(np.minimum(BRS_overapprox_box - act.center[dim_n], 
                                        spec.error['max_action_distance']), 
                                       -spec.error['max_action_distance']) + act.center[dim_n]
            
            # Determine set of regions that potentially intersect with G
            _, idx_edgeV = computeRegionIdx(BRS_overapprox_box, 
                                            spec.partition,
                                            borderOutside=[False,True])
            
            # Transpose because we want the min/max combinations per dimension
            # (not per point)
            idx_edge = np.zeros((model.n, 2))
            idx_edge[dim_n] = idx_edgeV.T
            
            idx_mat = [np.arange(low, high+1).astype(int) for low,high in idx_edge]
            
            s_min_list = []
            
            # Try each potential predecessor state
            for si, s_tup in enumerate(itertools.product(*idx_mat)):
                tocDiff(False)
                
                # Retrieve current state index
                s_min = self.partition['R']['idx'][s_tup]
                
                # Skip if this is a critical state
                if s_min in self.partition['critical'] and not compositional:
                    continue
                
                unique_verts = np.unique(self.partition['allCorners'][s_min][:, dim_n], axis=0)
                
                # if not ROT is None:
                if not ROT is None:
                    rotated_region = (ROT @ (unique_verts - brs_basis).T).T
                    
                    distance = max(np.abs(rotated_region[:,1:]))
                    if distance > max_distance_to_brs:
                        continue
                
                # If the problem is feasible, then action is enabled in this state
                if LP.solve(unique_verts):
                    
                    # Add state to the list of enabled states
                    s_min_list += [s_min]
                    
                    # Enable the current action in the current state
                    if s_tup in enabled:
                        enabled[s_tup].add(a_tup)
                    else:
                        enabled[s_tup] = {a_tup}
                        
                    if a_tup in enabled_inv:
                        enabled_inv[a_tup].add(s_tup)
                    else:
                        enabled_inv[a_tup] = {s_tup}
                    
            # Retrieve control error negative/positive
            control_error = np.zeros((model.n, 2))
            control_error[dim_n] = act.backreach_obj.target_set_size[dim_n, :]
            
            # If a parametric model is used
            if not epist is None and len(s_min_list) > 0:
                
                # Retrieve list of unique vertices of predecessor states
                s_vertices = self.partition['allCorners'][s_min_list][:, :, dim_n]
                s_vertices_unique = np.unique(np.vstack(s_vertices), axis=0)
            
                # Compute the epistemic error
                epist_error_neg, epist_error_pos = epist.compute(s_vertices_unique)
                
                # Store the control error for this action
                error[a_tup] = {'neg': control_error[:,0] + epist_error_neg,
                                'pos': control_error[:,1] + epist_error_pos}
                
            elif len(s_min_list) > 0:
                
                # Store the control error for this action
                error[a_tup] = {'neg': control_error[:,0],
                                'pos': control_error[:,1]}        
                
            if a_idx % print_every == 0:
                print('Action to',act.center,'enabled in',len(s_min_list),'states') 
                print(self.partition['R']['center'][s_min_list])
                
        return enabled, enabled_inv, error


    def set_extra_actions(self):

        # Add extra actions
        BRS_0 = self.actions['backreach_obj']['default'].verts_infl
        LP = LP_vertices_contained(self.model, BRS_0.shape, 
                                   solver=self.setup.cvx['solver'])

        if self.flags['parametric']:
                epist = epistemic_error(self.model)
        else:
            epist = None

        for act in self.actions['extra_act']:
                
            # Set current backward reachable set as parameter
            LP.set_backreach(act.backreach_infl)
            
            s_min_list = []
            
            # Try each potential predecessor state
            for s_min in range(self.partition['nr_regions']):
                
                # Skip if this is a critical state
                if s_min in self.partition['critical']:
                    continue
                
                unique_verts = np.unique(self.partition['allCorners'][s_min], axis=0)
                
                # If the problem is feasible, then action is enabled in this state
                if LP.solve(unique_verts):
                    
                    # Add state to the list of enabled states
                    s_min_list += [s_min]
                    
                    # Enable the current action in the current state
                    self.actions['enabled'][s_min].add(act.idx)
                    
                    act.enabled_in.add(s_min)
                    
            # Retrieve control error negative/positive
            control_error = act.backreach_obj.target_set_size
                    
            # If a parametric model is used
            if not epist is None and len(s_min_list) > 0:
                
                # Retrieve list of unique vertices of predecessor states
                s_vertices = self.partition['allCorners'][s_min_list]
                s_vertices_unique = np.unique(np.vstack(s_vertices), axis=0)
            
                # Compute the epistemic error
                epist_error_neg, epist_error_pos = epist.compute(s_vertices_unique)
                
                # Store the control error for this action
                act.error = {'neg': control_error[:,0] + epist_error_neg,
                            'pos': control_error[:,1] + epist_error_pos}
                
            else:
                
                # Store the control error for this action
                act.error = {'neg': control_error[:,0],
                            'pos': control_error[:,1]}

        # Add extra actions
        BRS_0 = self.actions['backreach_obj']['default'].verts_infl
        LP = LP_vertices_contained(self.model, BRS_0.shape, 
                                   solver=self.setup.cvx['solver'])

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
        samples = np.random.multivariate_normal(
                        np.zeros(self.model.n), self.model.noise['w_cov'], 
                        size=self.args.noise_samples)
        
        # Cluster samples
        if self.args.sample_clustering > 0:
            
            max_radius = float(self.args.sample_clustering)
            
            remaining_samples       = samples
            remaining_samples_i     = np.arange(len(samples))
            
            clusters0 = {
                'value': [],
                'lb': [],
                'ub': []
                }
            
            while len(remaining_samples) > 0:
                
                # Compute distance between first samples and all others
                distances = np.linalg.norm( remaining_samples[0] - 
                                            remaining_samples, 
                                            axis=1 )
                
                # Add the samples closer than `max_radius` to the first
                # sample to a new cluster
                distances_below = distances < max_radius
                
                cluster_samples = remaining_samples[ distances_below ]
                
                clusters0['value'] += [int(len(remaining_samples_i[ distances_below ]))]
                clusters0['lb'] += [np.min(cluster_samples, axis=0)]
                clusters0['ub'] += [np.max(cluster_samples, axis=0)]
                
                remaining_samples = remaining_samples[ ~distances_below ]
                remaining_samples_i = remaining_samples_i[ ~distances_below ]
                
            clusters0['value']       = np.array(clusters0['value'])
            clusters0['lb']          = np.array(clusters0['lb'])
            clusters0['ub']          = np.array(clusters0['ub'])
            
            print('--',len(samples),'samples clustered into',
                  len(clusters0['value']),'clusters')
            
            assert sum(clusters0['value']) == self.args.noise_samples
            
        else:
            
            clusters0 = {
                'value': np.ones(len(samples)),
                'lb': samples,
                'ub': samples
                }
        
        # For every action (i.e. target point)
        for a_idx, act in self.actions['obj'].items():
            
            # Shift samples by the center of the target set of this action
            clusters = {
                'value': clusters0['value'],
                'lb':    clusters0['lb'] + act.center,
                'ub':    clusters0['ub'] + act.center
                }
            
            # Check if action a is available in any state at all
            if len(act.enabled_in) > 0:
                
                # Checking which samples cannot be contained in a region
                # at the same time is of quadratic complexity in the number
                # of samples. Thus, we disable this above a certain limit.
                if True:
                    exclude = []
                else:
                    exclude = exclude_samples(samples, 
                                      self.spec.partition['width'])
                
                prob[a_idx] = compute_intervals_error(self.args, 
                      self.spec.partition, self.partition, self.trans, 
                      clusters, act.error, exclude, verbose=False)
                
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
        
        
        
def exclude_samples(samples, width):
    
    N,n = samples.shape
    
    S = np.reshape(samples, (N,n,1))
    diff = S - S.T
    width_tile = np.tile(width, (N,1)).T
    boolean = np.any(diff > width_tile, axis=1) | np.any(diff < -width_tile, axis=1)
    
    mp = map(np.nonzero, boolean)
    exclude = [set(m[0]) for m in mp]
    
    return exclude