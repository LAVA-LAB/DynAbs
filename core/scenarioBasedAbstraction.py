#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 ______________________________________
|                                      |
|  SCENARIO-BASED ABSTRACTION PROGRAM  |
|______________________________________|

Implementation of the method proposed in the paper:
 "Sampling-Based Robust Control of Autonomous Systems with Non-Gaussian Noise"

Originally coded by:        <anonymized>
Contact e-mail address:     <anonymized>
______________________________________________________________________________
"""

import numpy as np              # Import Numpy for computations
import itertools                # Import to crate iterators
import copy                     # Import to copy variables in Python
import csv                      # Import to create/load CSV files
import sys                      # Allows to terminate the code at some point
import os                       # Import OS to allow creationg of folders
import random                   # Import to use random variables
from scipy.spatial import Delaunay # Import to create convex hulls
from progressbar import progressbar # Import to create progress bars

from .mainFunctions import definePartitions, \
    in_hull, makeModelFullyActuated, computeScenarioBounds_sparse, \
    computeRegionCenters
from .commons import tic, ticDiff, tocDiff, table, printWarning
from .postprocessing.createPlots import createPartitionPlot

from .createMDP import mdp

'''
------------------------------------------------------------------------------
Main filter-based abstraction object definition
------------------------------------------------------------------------------
'''

class Abstraction(object):
    '''
    Main abstraction object    
    '''
    
    def _defModel(self, delta):
        '''
        Define model within abstraction object for given value of delta

        Parameters
        ----------
        delta : int
            Value of delta to create model for.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        '''
        
        if delta == 0:
            model = makeModelFullyActuated(copy.deepcopy(self.basemodel), 
                       manualDimension = 'auto', observer=False)
        else:
            model = makeModelFullyActuated(copy.deepcopy(self.basemodel), 
                       manualDimension = delta, observer=False)
            
        # Determine inverse A matrix
        model.A_inv  = np.linalg.inv(model.A)
        
        # Determine pseudo-inverse B matrix
        model.B_pinv = np.linalg.pinv(model.B)
        
        # Retreive system dimensions
        model.p      = np.size(model.B,1)   # Nr of inputs
    
        # Control limitations
        model.uMin   = np.array( model.setup['control']['limits']['uMin'] * \
                                int(model.p/self.basemodel.p) )
        model.uMax   = np.array( model.setup['control']['limits']['uMax'] * \
                                int(model.p/self.basemodel.p) )
        
        # If noise samples are used, recompute them
        if self.setup.scenarios['gaussian'] is False:
            
            f = self.basemodel.setup['noiseMultiplier']
            
            model.noise['samples'] = f * np.vstack(
                (2*model.noise['samples'][:,0],
                 0.2*model.noise['samples'][:,0],
                 2*model.noise['samples'][:,1],
                 0.2*model.noise['samples'][:,1],
                 2*model.noise['samples'][:,2],
                 0.2*model.noise['samples'][:,2])).T
        
        uAvg = (model.uMin + model.uMax) / 2
        
        if np.linalg.matrix_rank(np.eye(model.n) - model.A) == model.n:
            model.equilibrium = np.linalg.inv(np.eye(model.n) - model.A) @ \
                (model.B @ uAvg + model.Q_flat)
        
        return model
    
    def _defPartition(self):
        '''
        Define the partition of the state space

        Returns
        -------
        abstr : dict
            Dictionay containing all information of the abstraction.

        '''
        
        # Define partitioning of state-space
        abstr = dict()
        
        abstr['P'] = definePartitions(self.basemodel.n,
                            self.basemodel.setup['partition']['nrPerDim'],
                            self.basemodel.setup['partition']['width'],
                            self.basemodel.setup['partition']['origin'],
                            onlyCenter = False)
        
        abstr['nr_regions'] = len(abstr['P'])
        
        centerTuples = [tuple(abstr['P'][i]['center']) for i in 
                        range(abstr['nr_regions'])] 
        
        abstr['allCenters'] = dict(zip(centerTuples, 
                                       range(abstr['nr_regions'])))
        
        # Determine goal regions
        abstr['goal'] = self._defStateLabelSet(abstr['allCenters'], 
            self.basemodel.setup['partition'], 
            self.basemodel.setup['specification']['goal'])
        
        # Determine critical regions
        abstr['critical'] = self._defStateLabelSet(abstr['allCenters'], 
            self.basemodel.setup['partition'], 
            self.basemodel.setup['specification']['critical'])
        
        return abstr
    
    def _defStateLabelSet(self, allCenters, partition, subset):
        '''
        Return the indices of regions associated with the unique centers.

        Parameters
        ----------
        allCenters : List
            List of the center coordinates for all regions.
        partition : Dict
            Partition dictionary.
        subset : List
            List of points to return the unique centers for.

        Returns
        -------
        list
            List of unique center points.

        '''
        
        if np.shape(subset)[1] == 0:
            return []
        
        else:
        
            # Retreive list of points and convert to array
            points = np.array( subset ) 
        
            # Compute all centers of regions associated with points
            centers = computeRegionCenters(points, partition)
            
            # Filter to only keep unique centers
            centers_unique = np.unique(centers, axis=0)

            # Return the ID's of regions associated with the unique centers            
            return [allCenters[tuple(c)] for c in centers_unique 
                               if tuple(c) in allCenters]
    
    def _defAllCorners(self):
        '''
        Returns the vertices of every region in the partition (as nested list)

        Returns
        -------
        list
            Nested list of all vertices of all regions.

        '''
            
        # Create all combinations of n bits, to reflect combinations of all 
        # lower/upper bounds of partitions
        bitCombinations = list(itertools.product([0, 1], 
                               repeat=self.basemodel.n))
        bitRelation     = ['low','upp']
        
        # Calculate all corner points of every partition. Every partition has 
        # an upper and lower bounnd in every state (dimension). Hence, the 
        # every partition has 2^n corners, with n the number of states.
        
        allOriginPointsNested = [[[
            self.abstr['P'][i][bitRelation[bit]][bitIndex] 
                for bitIndex,bit in enumerate(bitList)
            ] for bitList in bitCombinations 
            ] for i in self.abstr['P']
            ]
        
        return np.array(allOriginPointsNested)
    
    def _defRegionHull(self, points):
        '''
        Compute the convex hull for the given points

        Parameters
        ----------
        points : 2D Numpy array
            Numpy array, with every row being a point to include in the hull.

        Returns
        -------
        Convex hull
            Convex hull object.

        '''
        
        return Delaunay(points, qhull_options='QJ')
    
    def _createTargetPoints(self, cornerPoints):
        '''
        Create target points, based on the vertices given

        Parameters
        ----------
        cornerPoints : list
            Vertices to compute target points for.

        Returns
        -------
        target : dict
            Dictionary with all data about the target points.

        '''
        
        target = dict()
        
        if self.basemodel.setup['targets']['nrPerDim'] != 'auto':

            # Width (span) between target points in every dimension
            if self.basemodel.setup['targets']['domain'] != 'auto':
                targetWidth = np.array(
                    self.basemodel.setup['targets']['domain'])*2 / \
                    self.basemodel.setup['targets']['nrPerDim']
            else:
                targetWidth = np.array(
                    self.basemodel.setup['partition']['nrPerDim']) * \
                    self.basemodel.setup['partition']['width'] / \
                    self.basemodel.setup['targets']['nrPerDim']
        
            # Create target points (similar to a partition of the state space)
            target['d'] = definePartitions(self.basemodel.n,
                    self.basemodel.setup['targets']['nrPerDim'],
                    targetWidth,
                    self.basemodel.setup['partition']['origin'],
                    onlyCenter = True)
        
        else:
        
            # Set default target points to the center of every region
            target['d'] = [self.abstr['P'][j]['center'] for j in 
                           range(self.abstr['nr_regions'])]
        
        targetPointTuples = [tuple(point) for point in target['d']]        
        target['inv'] = dict(zip(targetPointTuples, range(len(target['d']))))
        
        return target
    
    def _defInvArea(self, delta):
        '''
        Compute the predecessor set (without the shift due to the target
        point). This acccounts to computing, for all u_k, the set
        A^-1 (B u_k - q_k)

        Parameters
        ----------
        delta : int
            Delta value of the model which is used.

        Returns
        -------
        x_inv_area : 2D Numpy array
            Predecessor set (every row is a vertex).

        '''
        
        # Determine the set of extremal control inputs
        u = [[self.model[delta].uMin[i], self.model[delta].uMax[i]] for i in 
              range(self.model[delta].p)]
        
        # Determine the inverse image of the extreme control inputs
        x_inv_area = np.zeros((2**self.model[delta].p, self.model[delta].n))
        for i,elem in enumerate(itertools.product(*u)):
            list_elem = list(elem)
            
            # Calculate inverse image of the current extreme control input
            x_inv_area[i,:] = self.model[delta].A_inv @ \
                (self.model[delta].B @ np.array(list_elem).T + 
                 self.model[delta].Q_flat)  
    
        return x_inv_area
    
    def _defInvHull(self, x_inv_area):
        '''
        Define the convex hull object of the predecessor set        

        Parameters
        ----------
        x_inv_area : 2D Numpy array
            Predecessor set (every row is a vertex).

        Returns
        -------
        x_inv_hull : hull
            Convex hull object.

        '''
        
        x_inv_hull = Delaunay(x_inv_area, qhull_options='QJ')
        
        return x_inv_hull
    
    def _defBasisVectors(self, delta, verbose=False):
        '''
        Compute the basis vectors of the predecessor set, computed from the
        average control inputs to the maximum in every dimension of the
        control space.
        Note that the drift does not play a role here.  

        Parameters
        ----------
        delta : int
            Delta value of the model which is used.

        Returns
        -------
        basis_vectors : 2D Numpy array
            Numpy array of basis vectors (every row is a vector).

        '''
        
        u_avg = np.array(self.model[delta].uMax + self.model[delta].uMin)/2    

        # Compute basis vectors (with average of all controls as origin)
        u = np.tile(u_avg, (self.basemodel.n,1)) + \
            np.diag(self.model[delta].uMax - u_avg)
        
        origin = self.model[delta].A_inv @ \
                (self.model[delta].B @ np.array(u_avg).T)   
                
        basis_vectors = np.zeros((self.basemodel.n, self.basemodel.n))
        
        for i,elem in enumerate(u):
            
            # Calculate inverse image of the current extreme control input
            point = self.model[delta].A_inv @ \
                (self.model[delta].B @ elem.T)    
            
            basis_vectors[i,:] = point - origin
        
            if verbose:
                print(' ---- Length of basis',i,':',
                  np.linalg.norm(basis_vectors[i,:]))
        
        return basis_vectors
    
    def _defEnabledActions(self, delta, verbose=False):
        '''
        Define dictionaries to sture points in the preimage of a state, and
        the corresponding polytope points.

        Parameters
        ----------
        delta : int
            Delta value of the model which is used.

        Returns
        -------
        total_actions_enabled : int
            Total number of enabled actions.
        enabled_actions : list
            Nested list of actions enabled in every region.
        enabledActions_inv : list
            Nested list of inverse actions enabled in every region.
            
        '''
        
        from .commons import angle_between
        
        def f7(seq):
            seen = set()
            seen_add = seen.add
            return [x for x in seq if not (x in seen or seen_add(x))]
        
        # Compute inverse reachability area
        x_inv_area = self._defInvArea(delta)
        
        total_actions_enabled = 0
        
        enabled_polypoints = dict()
        
        enabled_in_states = [[] for i in range(self.abstr['nr_actions'])]
        enabled_actions   = [[] for i in range(self.abstr['nr_regions'])]
        
        nr_corners = 2**self.basemodel.n
        
        printEvery = min(100, max(1, int(self.abstr['nr_actions']/10)))
        
        # Check if dimension of control area equals that if the state vector
        dimEqual = self.model[delta].p == self.basemodel.n
        
        if dimEqual:
            
            print(' -- Computing inverse basis vectors...')
            # Use preferred method: map back the skewed image to squares
            
            basis_vectors = self._defBasisVectors(delta, verbose=verbose)   
            
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
                
            allRegionVertices = self.abstr['allCornersFlat'] @ parralelo2cube \
                    - predSet_originShift
            
        else:
            
            print(' -- Creating inverse hull for delta =',delta,'...')
            
            # Use standard method: check if points are in (skewed) hull
            x_inv_hull = self._defInvHull(x_inv_area)
            
            if verbose:
                print('Normal inverse area:',x_inv_area)
        
            allRegionVertices = self.abstr['allCornersFlat'] 
        
        import sys
        np.set_printoptions(threshold=sys.maxsize)
        
        action_range = f7(np.concatenate(( self.abstr['goal'],
                           np.arange(self.abstr['nr_actions']) )))
        
        # For every action
        for action_id in progressbar(action_range, redirect_stdout=True):
            
            targetPoint = self.abstr['target']['d'][action_id]
            
            if dimEqual:
            
                # Shift the origin points (instead of the target point)
                A_inv_d = self.model[delta].A_inv @ np.array(targetPoint)
                
                # Python implementation
                allVerticesNormalized = (A_inv_d @ parralelo2cube) - \
                                         allRegionVertices
                                
                # Reshape the whole matrix that we obtain
                poly_reshape = np.reshape( allVerticesNormalized,
                                (self.abstr['nr_regions'], 
                                 nr_corners*self.basemodel.n))
                
                # Enabled actions are ones that have all corner points within
                # the origin-centered hypercube with unit length
                enabled_in = np.maximum(np.max(poly_reshape, axis=1), 
                                        -np.min(poly_reshape, axis=1)) <= 1.0
                
            else:
                
                # Shift the origin points (instead of the target point)
                A_inv_d = self.model[delta].A_inv @ np.array(targetPoint)
            
                # Subtract the shift from all corner points
                allVertices = A_inv_d - allRegionVertices
            
                # Check which points are in the convex hull
                polypoints_vec = in_hull(allVertices, x_inv_hull)
            
                # Map enabled vertices of the partitions to actual partitions
                enabled_polypoints[action_id] = np.reshape(  polypoints_vec, 
                              (self.abstr['nr_regions'], nr_corners))
            
                # Polypoints contains the True/False results for all vertices
                # of every partition. An action is enabled in a state if it 
                # is enabled in all its vertices
                enabled_in = np.all(enabled_polypoints[action_id] == True, 
                                    axis = 1)
            
            # Shift the inverse hull to account for the specific target point
            if self.setup.plotting['partitionPlot'] and \
                action_id == int(self.abstr['nr_regions']/2):

                if verbose:
                    print('x_inv_area:',x_inv_area)
                    print('origin shift:',A_inv_d)       
                    print('targetPoint:',targetPoint,' - drift:',
                          self.model[delta].Q_flat)
                
                    # Partition plot for the goal state, also showing pre-image
                    print('Create partition plot...')
                
                predecessor_set = A_inv_d - x_inv_area
                    
                createPartitionPlot((0,1), (2,3), self.abstr['goal'], 
                    delta, self.setup, self.model[delta], self.abstr, 
                    self.abstr['allCorners'], predecessor_set)
                
                createPartitionPlot((0,2), (1,3), self.abstr['goal'], 
                    delta, self.setup, self.model[delta], self.abstr, 
                    self.abstr['allCorners'], predecessor_set)
            
            # Retreive the ID's of all states in which the action is enabled
            enabled_in_states[action_id] = np.nonzero(enabled_in)[0]
            
            # Remove critical states from the list of enabled actions
            enabled_in_states[action_id] = np.setdiff1d(
                enabled_in_states[action_id], self.abstr['critical'])
            
            if action_id % printEvery == 0:
                if action_id in self.abstr['goal']:
                    if verbose:
                        print(' -- GOAL action',str(action_id),'enabled in',
                              str(len(enabled_in_states[action_id])),
                              'states - target point:',
                              str(targetPoint))
                    
                else:
                    if verbose:
                        print(' -- Action',str(action_id),'enabled in',
                              str(len(enabled_in_states[action_id])),
                              'states - target point:',
                              str(targetPoint))
            
            if len(enabled_in_states[action_id]) > 0:
                total_actions_enabled += 1
            
            for origin in enabled_in_states[action_id]:
                enabled_actions[origin] += [action_id]
                
        enabledActions_inv = [enabled_in_states[i] 
                              for i in range(self.abstr['nr_actions'])]
        
        return total_actions_enabled, enabled_actions, enabledActions_inv
                
    def __init__(self):
        '''
        Initialize the scenario-based abstraction object.

        Returns
        -------
        None.

        '''
        
        print('Base model loaded:')
        print(' -- Dimension of the state:',self.basemodel.n)
        print(' -- Dimension of the control input:',self.basemodel.p)
        
        # Simulation end time is the end time
        self.T  = self.basemodel.setup['endTime']
        
        # Number of simulation time steps
        self.N = int(self.T) #/self.basemodel.tau)
        
        self.model = dict()
        for delta in self.setup.deltas:
            
            # Define model object for current delta value
            self.model[delta] = self._defModel(delta)
        
        self.time['0_init'] = tocDiff(False)
        print('Abstraction object initialized - time:',self.time['0_init'])
        
    def definePartition(self):
        ''' 
        Define the discrete state space partition and target points
        
        Returns
        -------
        None.
        '''
        
        # Create partition
        print('\nComputing partition of the state space...')
        
        self.abstr = self._defPartition()
        
        print(' -- Number of regions:',self.abstr['nr_regions'])
        print(' -- Number of goal regions:',len(self.abstr['goal']))
        print(' -- Number of critical regions:',len(self.abstr['critical']))

        self.time['1_partition'] = tocDiff(False)
        print('Discretized states defined - time:',self.time['1_partition'])
        
        # Create target points
        print('\nCreating actions (target points)...')
        
        self.abstr['allCorners']     = self._defAllCorners()
        self.abstr['allCornersFlat'] =np.concatenate(self.abstr['allCorners'])
        
        # Create the target point for every action (= every state)
        self.abstr['target']=self._createTargetPoints(self.abstr['allCorners'])
        self.abstr['nr_actions'] = len(self.abstr['target']['d'])
        
        print(' -- Number of actions (target points):',
              self.abstr['nr_actions'])
        
        # Create hull of the full state space
        origin = np.array(self.basemodel.setup['partition']['origin'])
        halfWidth = np.array(self.basemodel.setup['partition']['nrPerDim']) \
            / 2 * self.basemodel.setup['partition']['width']
        stateSpaceBox = np.vstack((origin-halfWidth, origin+halfWidth))
        
        outerCorners = list( itertools.product(*stateSpaceBox.T) )
        
        print(' -- Creating hull for the full state space...')
        
        self.abstr['stateSpaceHull'] = self._defRegionHull(outerCorners)

    def defineActions(self):
        ''' 
        Determine which actions are actually enabled.
        
        Returns
        -------
        None.
        '''
        
        print('\nComputing set of enabled actions...')
        
        self.abstr['actions']     = dict()
        self.abstr['actions_inv'] = dict()
        
        # For every time step grouping value in the list
        for delta_idx,delta in enumerate(self.setup.deltas):
            
            nr_A, self.abstr['actions'][delta], \
             self.abstr['actions_inv'][delta] = self._defEnabledActions(delta)
                        
            print(nr_A,'actions enabled')
            if nr_A == 0:
                printWarning('No actions enabled at all, so terminate')
                sys.exit()
        
        self.time['2_enabledActions'] = tocDiff(False)
        print('Enabled actions define - time:',self.time['2_enabledActions'])
        
###############################

class scenarioBasedAbstraction(Abstraction):
    def __init__(self, setup, basemodel):
        '''
        Initialize scenario-based abstraction (ScAb) object

        Parameters
        ----------
        setup : dict
            Setup dictionary.
        basemodel : dict
            Base model for which to create the abstraction.

        Returns
        -------
        None.

        '''
        
        # Copy setup to internal variable
        self.setup = setup
        self.basemodel = basemodel
        
        # Define empty dictionary for monte carlo sims. (even if not used)
        self.mc = dict()   
        
        # Start timer
        tic()
        ticDiff()
        self.time = dict()
        
        Abstraction.__init__(self)
        
        Abstraction.definePartition(self)
        
    def _loadScenarioTable(self, tableFile):
        '''
        Load tabulated bounds on the transition probabilities (computed using
        the scenario approach).

        Parameters
        ----------
        tableFile : str
            File from which to load the table.

        Returns
        -------
        memory : dict
            Dictionary containing all loaded probability bounds / intervals.

        '''
        
        memory = dict()
        
        if not os.path.isfile(tableFile):
            sys.exit('ERROR: the following table file does not exist:'+
                     str(tableFile))
        
        with open(tableFile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            
            # Skip header row
            next(reader)
            
            for i,row in enumerate(reader):
                
                strSplit = row[0].split(',')
                
                # key = tuple( [int(float(i)) for i in strSplit[0:2]] + 
                #              [float(strSplit[2])] )
                
                value = [float(i) for i in strSplit[-2:]]
                memory[int(strSplit[0])] = value
                    
        return memory
        
    def _computeProbabilityBounds(self, tab, k, delta):
        '''
        Compute transition probability intervals (bounds)

        Parameters
        ----------
        tab : dict
            Table dictionary.
        k : int
            Discrete time step.
        delta : int
            Value of delta to use the model for.

        Returns
        -------
        prob : dict
            Dictionary containing the computed transition probabilities.

        '''
        
        prob = dict()

        printEvery = min(100, max(1, int(self.abstr['nr_actions']/10)))

        # For every action (i.e. target point)
        for a in range(self.abstr['nr_actions']):
            
            # Check if action a is available in any state at all
            if len(self.abstr['actions_inv'][delta][a]) > 0:
                    
                prob[a] = dict()
            
                mu = self.abstr['target']['d'][a]
                Sigma = self.model[delta].noise['w_cov']
                
                if self.setup.scenarios['gaussian'] is True:
                    # Compute Gaussian noise samples
                    samples = np.random.multivariate_normal(mu, Sigma, 
                                    size=self.setup.scenarios['samples'])
                    
                else:
                    # Determine non-Gaussian noise samples (relative from 
                    # target point)
                    samples = mu + np.array(
                        random.choices(self.model[delta].noise['samples'], 
                        k=self.setup.scenarios['samples']) )
                    
                prob[a] = computeScenarioBounds_sparse(self.setup, 
                      self.basemodel.setup['partition'], 
                      self.abstr, self.trans, samples)
                
                # Print normal row in table
                if a % printEvery == 0:
                    nr_transitions = len(prob[a]['interval_idxs'])
                    tab.print_row([delta, k, a, 
                       'Probabilities computed (transitions: '+
                       str(nr_transitions)+')'])
                
        return prob
    
    def defActions(self):
        '''
        Define the actions of the finite-state abstraction (performed once,
        outside the iterative scheme).

        Returns
        -------
        None.

        '''
        
        Abstraction.defineActions(self)
    
    def defTransitions(self):
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
        
        tableFile = 'input/SaD_probabilityTable_N='+ \
                        str(self.setup.scenarios['samples'])+'_beta='+ \
                        str(self.setup.scenarios['confidence'])+'.csv'
        
        # Load scenario approach table
        self.trans['memory'] = self._loadScenarioTable(tableFile = tableFile)
        
        # Retreive type of horizon
        k_range = [0]
        
        print('Computing transition probabilities...')
        
        # For every delta value in the list
        for delta_idx,delta in enumerate(self.setup.deltas):
            self.trans['prob'][delta] = dict()
            
            # For every time step in the horizon
            for k in k_range:
                
                # Print header row
                tab.print_row(['DELTA','K','ACTION','STATUS'], head=True)    
                
                self.trans['prob'][delta][k] = \
                    self._computeProbabilityBounds(tab, k, delta)
                
            # Delete iterable variables
            del k
        del delta
        
        self.time['3_probabilities'] = tocDiff(False)
        print('Transition probabilities calculated - time:',
              self.time['3_probabilities'])

    def buildMDP(self):
        '''
        Build the (i)MDP and create all respective PRISM files.

        Returns
        -------
        model_size : dict
            Dictionary describing the number of states, choices, and 
            transitions.

        '''
        
        # Initialize MDP object
        self.mdp = mdp(self.setup, self.N, self.abstr)
        
        if self.setup.mdp['prism_model_writer'] == 'explicit':
            
            # Create PRISM file (explicit way)
            model_size, self.mdp.prism_file, self.mdp.spec_file, \
            self.mdp.specification = \
                self.mdp.writePRISM_explicit(self.abstr, self.trans, 
                                             self.setup.mdp['mode'])   
        
        else:
        
            # Create PRISM file (default way)
            self.mdp.prism_file, self.mdp.spec_file, self.mdp.specification = \
                self.mdp.writePRISM_scenario(self.abstr, self.trans, 
                                             self.setup.mdp['mode'])  
              
            model_size = {'States':None,'Choices':None,'Transitions':None}

        self.time['4_MDPcreated'] = tocDiff(False)
        print('MDP created - time:',self.time['4_MDPcreated'])
        
        return model_size
            
    def solveMDP(self):
        '''
        Solve the (i)MDP usign PRISM

        Returns
        -------
        None.

        '''
            
        # Solve the MDP in PRISM (which is called via the terminal)
        policy_file, vector_file = self._solveMDPviaPRISM()
        
        # Load PRISM results back into Python
        self.loadPRISMresults(policy_file, vector_file)
            
        self.time['5_MDPsolved'] = tocDiff(False)
        print('MDP solved in',self.time['5_MDPsolved'])
        
    def preparePlots(self):
        '''
        Initializing function to prepare for plotting

        Returns
        -------
        None.

        '''
        
        # Process results
        self.plot           = dict()
    
        for delta_idx, delta in enumerate(self.setup.deltas):
            self.plot[delta] = dict()
            self.plot[delta]['N'] = dict()
            self.plot[delta]['T'] = dict()
            
            self.plot[delta]['N']['start'] = 0
            
            # Convert index to absolute time (note: index 0 is time tau)
            self.plot[delta]['T']['start'] = \
                int(self.plot[delta]['N']['start'] * self.basemodel.tau)
        
    def _solveMDPviaPRISM(self):
        '''
        Call PRISM to solve (i)MDP while executing the Python codes.

        Returns
        -------
        policy_file : str
            Name of the file in which the optimal policy is stored.
        vector_file : str
            Name of the file in which the optimal rewards are stored.

        '''
        
        import subprocess

        prism_folder = self.setup.mdp['prism_folder'] 
        
        print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        
        print('Starting PRISM...')
        
        spec = self.mdp.specification
        mode = self.setup.mdp['mode']
        java_memory = self.setup.mdp['prism_java_memory']
        
        print(' -- Running PRISM with specification for mode',
              mode.upper()+'...')
    
        file_prefix = self.setup.directories['outputFcase'] + "PRISM_" + mode
        policy_file = file_prefix + '_policy.csv'
        vector_file = file_prefix + '_vector.csv'
    
        options = ' -ex -exportadv "'+policy_file+'"'+ \
                  ' -exportvector "'+vector_file+'"'
    
        # Switch between PRISM command for explicit model vs. default model
        if self.setup.mdp['prism_model_writer'] == 'explicit':
    
            print(' --- Execute PRISM command for EXPLICIT model description')        
    
            model_file      = '"'+self.mdp.prism_file+'"'             
        
            # Explicit model
            command = prism_folder+"bin/prism -javamaxmem "+ \
                str(java_memory)+"g -importmodel "+model_file+" -pf '"+ \
                spec+"' "+options
        else:
            
            print(' --- Execute PRISM command for DEFAULT model description')
            
            model_file      = '"'+self.mdp.prism_file+'"'
            
            # Default model
            command = prism_folder+"bin/prism -javamaxmem "+ \
                str(java_memory)+"g "+model_file+" -pf '"+spec+"' "+options    
        
        subprocess.Popen(command, shell=True).wait()    
        
        return policy_file, vector_file
        
    def loadPRISMresults(self, policy_file, vector_file):
        '''
        Load results from existing PRISM output files.

        Parameters
        ----------
        policy_file : str
            Name of the file to load the optimal policy from.
        vector_file : str
            Name of the file to load the optimal policy from.

        Returns
        -------
        None.

        '''
        
        import pandas as pd
        
        self.results = dict()
        
        # Read policy CSV file
        policy_all = pd.read_csv(policy_file, header=None).iloc[:, 1:].\
            fillna(-1).to_numpy()
            
        # Flip policy upside down (PRISM generates last time step at top!)
        policy_all = np.flipud(policy_all)
        
        self.results['optimal_policy'] = np.zeros(np.shape(policy_all))
        self.results['optimal_delta'] = np.zeros(np.shape(policy_all))
        self.results['optimal_reward'] = np.zeros(np.shape(policy_all))
        
        rewards_k0 = pd.read_csv(vector_file, header=None).iloc[1:].to_numpy()
        self.results['optimal_reward'][0,:] = rewards_k0.flatten()
        
        # Split the optimal policy between delta and action itself
        for i,row in enumerate(policy_all):
            
            for j,value in enumerate(row):
                
                # If value is not -1 (means no action defined)
                if value != -1:
                    # Split string
                    value_split = value.split('_')
                    # Store action and delta value separately
                    self.results['optimal_policy'][i,j] = int(value_split[1])
                    self.results['optimal_delta'][i,j] = int(value_split[3])
                else:
                    # If no policy is known, set to -1
                    self.results['optimal_policy'][i,j] = int(value)
                    self.results['optimal_delta'][i,j] = int(value) 
        
    def generatePlots(self, delta_value, max_delta):
        '''
        Generate (optimal reachability probability) plots

        Parameters
        ----------
        delta_value : int
            Value of delta for the model to plot for.
        max_delta : int
            Maximum value of delta used for any model.

        Returns
        -------
        None.
        '''
        
        print('\nGenerate plots')
        
        if self.abstr['nr_regions'] <= 1000:
        
            from .postprocessing.createPlots import createProbabilityPlots
            
            if self.setup.plotting['probabilityPlots']:
                createProbabilityPlots(self.setup, self.plot[delta_value], 
                                       self.N, self.model[delta_value],
                                       self.results, self.abstr, self.mc)
                    
        else:
            
            printWarning("Omit probability plots (nr. of regions too large)")
            
    def monteCarlo(self, iterations='auto', init_states='auto', 
                   init_times='auto', printDetails=False):
        '''
        Perform Monte Carlo simulations to validate the obtained results

        Parameters
        ----------
        iterations : str or int, optional
            Number of Monte Carlo iterations. The default is 'auto'.
        init_states : str or list, optional
            Initial states to start simulations from. The default is 'auto'.
        init_times : str or list, optional
            Initial time steps to start sims. from. The default is 'auto'.

        Returns
        -------
        None.

        '''
        
        tocDiff(False)
        if self.setup.main['verbose']:
            print(' -- Starting Monte Carlo simulations...')
        
        self.mc['results']  = dict()
        self.mc['traces']   = dict()
        
        if iterations != 'auto':
            self.setup.montecarlo['iterations'] = int(iterations)
            
        if init_states != 'auto':
            self.setup.montecarlo['init_states'] = list(init_states)
            
        if init_times != 'auto':
            self.setup.montecarlo['init_timesteps'] = list(init_times)
        elif self.setup.montecarlo['init_timesteps'] is False:
            
            # Determine minimum action time step width
            min_delta = min(self.setup.deltas)
            
            self.setup.montecarlo['init_timesteps'] = [ n for n in 
                                       self.plot[min_delta]['N'].values() ]
            
        n_list = self.setup.montecarlo['init_timesteps']
        
        self.mc['results']['reachability_probability'] = \
            np.zeros((self.abstr['nr_regions'],len(n_list)))
        
        # Column widths for tabular prints
        if self.setup.main['verbose']:
            col_width = [6,8,6,6,46]
            tab = table(col_width)

            print(' -- Computing required Gaussian random variables...')
        
        if self.setup.montecarlo['init_states'] == False:
            init_state_idxs = np.arange(self.abstr['nr_regions'])
            
        else:
            init_state_idxs = self.setup.montecarlo['init_states']
        
        # The gaussian random variables are precomputed to speed up the code
        if self.setup.scenarios['gaussian'] is True:
            w_array = dict()
            for delta in self.setup.deltas:
                w_array[delta] = np.random.multivariate_normal(
                    np.zeros(self.model[delta].n), 
                    self.model[delta].noise['w_cov'],
                   ( len(n_list), len(init_state_idxs), 
                     self.setup.montecarlo['iterations'], self.N ))
        
        # For each starting time step in the list
        for n0id,n0 in enumerate(n_list):
            
            self.mc['results'][n0] = dict()
            self.mc['traces'][n0] = dict()
        
            # For each initial state
            for i_abs,i in enumerate(init_state_idxs):
                
                regionA = self.abstr['P'][i]
                
                # Check if we should perform Monte Carlo sims for this state
                if self.setup.montecarlo['init_states'] is not False and \
                            i not in self.setup.montecarlo['init_states']:
                    print('Warning: this should not happen!')
                    # If current state is not in the list of initial states,
                    # continue to the next iteration
                    continue
                # Otherwise, continue with the Monte Carlo simulation
                
                if self.setup.main['verbose']:
                    tab.print_row(['K0','STATE','ITER','K','STATUS'], 
                                  head=True)
                else:
                    print(' -- Monte Carlo for start time',n0,'and state',i)
                
                # Create dictionaries for results related to partition i
                self.mc['results'][n0][i]  = dict()
                self.mc['results'][n0][i]['goalReached'] = np.full(
                    self.setup.montecarlo['iterations'], False, dtype=bool)
                self.mc['traces'][n0][i]  = dict()    
                
                # For each of the monte carlo iterations
                if self.setup.main['verbose']:
                    loop = range(self.setup.montecarlo['iterations'])
                else:
                    loop = progressbar(
                            range(self.setup.montecarlo['iterations']), 
                            redirect_stdout=True)
                    
                for m in loop:
                    
                    self.mc['traces'][n0][i][m] = []
                    
                    # Retreive the initial action time-grouping to be chosen
                    # (given by the optimal policy to the MDP)
                    delta = self.results['optimal_delta'][n0,i]
                    
                    if i in self.abstr['goal']:
                        # If initial state is already the goal state, succes
                        # Then, abort the iteration, as we reached the goal
                        self.mc['results'][n0][i]['goalReached'][m] = True
                        
                        if printDetails:
                            tab.print_row([n0, i, m, n0, 
                               'Initial state is goal state'], sort="Success")
                    elif delta == 0:
                        # If delta=0, no policy known, and reachability is 0
                        if self.setup.main['verbose']:
                            tab.print_row([n0, i, m, n0, 
                               'No initial policy known, so abort'], 
                               sort="Warning")
                    else:
                        if self.setup.main['verbose']:
                            tab.print_row([n0, i, m, n0, 
                               'Start Monte Carlo iteration'])
                                            
                        # Initialize the current simulation
                        x = np.zeros((self.N, self.basemodel.n))
                        x_goal = [None]*self.N
                        x_region = np.zeros(self.N).astype(int)
                        u = [None]*self.N
                        
                        actionToTake = np.zeros(self.N).astype(int)
                        deltaToTake = np.zeros(self.N).astype(int)
                        
                        # Set initial time
                        k = n0
                        
                        # True state model dynamics
                        x[n0] = np.array(regionA['center'])
                        
                        # Add current state to trace
                        self.mc['traces'][n0][i][m] += [x[n0]]
                        
                        # For each time step in the finite time horizon
                        while k < self.N / min(self.setup.deltas):
                            
                            if self.setup.main['verbose']:
                                tab.print_row([n0, i, m, k, 'New time step'])
                            
                            # Compute all centers of regions associated with points
                            center_coord = computeRegionCenters(x[k], 
                                self.basemodel.setup['partition']).flatten()
                            
                            if tuple(center_coord) in self.abstr['allCenters']:
                                # Save that state is currently in region ii
                                x_region[k] = self.abstr['allCenters'][
                                    tuple(center_coord)]
                                
                                # Retreive the action from the policy
                                actionToTake[k] = self.results[
                                    'optimal_policy'][k, x_region[k]]
                            else:
                                x_region[k] = -1
                            
                            # If current region is the goal state ... 
                            if x_region[k] in self.abstr['goal']:
                                # Then abort the current iteration, as we have achieved the goal
                                self.mc['results'][n0][i]['goalReached'][m] = True
                                
                                if self.setup.main['verbose']:
                                    tab.print_row([n0, i, m, k, 'Goal state reached'], sort="Success")
                                break
                            # If current region is in critical states...
                            elif x_region[k] in self.abstr['critical']:
                                # Then abort current iteration
                                if self.setup.main['verbose']:
                                    tab.print_row([n0, i, m, k, 'Critical state reached, so abort'], sort="Warning")
                                break
                            elif x_region[k] == -1:
                                if self.setup.main['verbose']:
                                    tab.print_row([n0, i, m, k, 'Absorbing state reached, so abort'], sort="Warning")
                                break
                            elif actionToTake[k] == -1:
                                if self.setup.main['verbose']:
                                    tab.print_row([n0, i, m, k, 'No policy known, so abort'], sort="Warning")
                                break
                            
                            # If loop was not aborted, we have a valid action
    
                            # Update the value of the time-grouping of the action
                            # dictated by the optimal policy
                            deltaToTake[k] = self.results['optimal_delta'][k, x_region[k]]
                            delta = deltaToTake[k]
                            
                            # If delta is zero, no policy is known, and reachability is zero
                            if delta == 0:
                                if self.setup.main['verbose']:
                                    tab.print_row([n0, i, m, k, 'Action type undefined, so abort'], sort="Warning")
                                break
                            
                            if self.setup.main['verbose']:
                                tab.print_row([n0, i, m, k, 'In state: '+str(x_region[k])+', take action: '+str(actionToTake[k])+' (delta='+str(delta)+')'])
                            
                            # Only perform another movement if k < N-tau (of base model)
                            if k < self.N:
                            
                                # Move predicted mean to the future belief to the target point of the next state
                                x_goal[k+delta] = self.abstr['target']['d'][actionToTake[k]]
        
                                # Reconstruct the control input required to achieve this target point
                                # Note that we do not constrain the control input; we already know that a suitable control exists!
                                u[k] = np.array(self.model[delta].B_pinv @ ( x_goal[k+delta] - self.model[delta].A @ x[k] - self.model[delta].Q_flat ))
                                
                                # Implement the control into the physical (unobservable) system
                                x_hat = self.model[delta].A @ x[k] + self.model[delta].B @ u[k] + self.model[delta].Q_flat
                                
                                if self.setup.scenarios['gaussian'] is True:
                                    # Use Gaussian process noise
                                    x[k+delta] = x_hat + w_array[delta][n0id, i_abs, m, k]
                                else:
                                    # Use generated samples                                    
                                    disturbance = random.choice(self.model[delta].noise['samples'])
                                    
                                    x[k+delta] = x_hat + disturbance
                                    
                                # Add current state to trace
                                self.mc['traces'][n0][i][m] += [x[k+delta]]
                            
                            # Increase iterator variable by the value of delta associated to chosen action
                            k += delta
                        
                # Set of monte carlo iterations completed for specific initial state
                
                # Calculate the overall reachability probability for initial state i
                self.mc['results']['reachability_probability'][i,n0id] = \
                    np.sum(self.mc['results'][n0][i]['goalReached']) / self.setup.montecarlo['iterations']
                    
        self.time['6_MonteCarlo'] = tocDiff(False)
        print('Monte Carlo simulations finished:',self.time['6_MonteCarlo'])