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

from .commons import writeFile, printSuccess

from progressbar import progressbar # Import to create progress bars

class mdp(object):
    def __init__(self, setup, N, partition, actions):
        '''
        Create the MDP model in Python for the current instance.
    
        Parameters
        ----------
        setup : dict
            Setup dictionary.
        N : int
            Finite time horizon.
        partition : dict
            Partition dictionary
    
        Returns
        -------
        mdp : dict
            MDP dictionary.
    
        '''
        
        self.setup = setup
        self.N = N
        
        self.nr_regions = partition['nr_regions']
        self.nr_actions = actions['nr_actions']
        self.nr_states  = self.nr_regions
        
        # Specify goal state set
        self.goodStates = partition['goal']
        
        # Specify sets that can never reach target set
        self.badStates  = partition['critical']
    
    def writePRISM_specification(self, mode, problem_type):
        '''
        Write the PRISM specificatoin / property to a file

        Parameters
        ----------
        mode : str
            Is either 'estimate' or 'interval'.
        problem_type : str
            Is either 'reachavoid' or 'avoid'.

        Returns
        -------
        specfile : str
            The name of the file in which the specification is stored.
        specification : str
            The specification itself given as a string.

        '''
        
        if mode == 'estimate':
            # If mode is default, set maximum probability as specification
            if problem_type == 'avoid':
                specification = 'Pmin=? [ F<='+str(self.N)+' "failed" ]'    
            else:
                specification = 'Pmax=? [ F<='+str(self.N)+' "reached" ]'
            
        elif mode == 'interval':
            # If mode is interval, set lower bound of max. prob. as spec.
            if problem_type == 'avoid':
                specification = 'Pminmax=? [ F<='+str(self.N)+' "failed" ]'
            else:
                specification = 'Pmaxmin=? [ F<='+str(self.N)+' "reached" ]'            
            
        # Define specification file
        specfile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".pctl"
    
        # Write specification file
        writeFile(specfile, 'w', specification)
        
        return specfile, specification
    
    def writePRISM_explicit(self, actions, partition, trans, problem_type, mode='estimate', 
                            verbose=False):
        '''
        Converts the model to the PRISM language, and write the model in
        explicit form to files (meaning that every transition is already
        enumerated explicitly).
    
        Parameters
        ----------
        abstr : dict
            Abstraction dictionary
        trans : dict
            Dictionary of transition probabilities        
        mode : str, optional
            Is either 'estimate' or 'interval'.
    
        Returns
        -------
        None.
    
        '''
        
        print('\nExport abstraction as PRISM model...')
        
        # Define PRISM filename
        PRISM_allfiles = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".all"
        
        print(' --- Writing PRISM states file')
        
        head = 3
        
        ### Write states file
        PRISM_statefile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".sta"
        
        state_file_string = '\n'.join(['(x)\n0:(-3)\n1:(-2)\n2:(-1)'] + 
              [str(i+head)+':('+str(i)+')' for i in range(self.nr_regions)])
        
        # Write content to file
        writeFile(PRISM_statefile, 'w', state_file_string)
        
        print(' --- Writing PRISM label file')
        
        ### Write label file
        PRISM_labelfile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".lab"
            
        label_file_list = ['0="init" 1="deadlock" 2="reached" 3="failed"'] + \
                          ['0: 1 3'] + ['1: 1 3'] + ['2: 2'] + \
                          ['' for i in range(self.nr_regions)]
        
        for i in range(self.nr_regions):
            substring = str(i+head)+': 0'
            
            # Check if region is a deadlock state
            if len(actions['enabled'][i]) == 0:
                substring += ' 1'
            
            # Check if region is in goal set
            if i in self.goodStates:
                substring += ' 2' 
            elif i in self.badStates: # or len(actions['enabled'][i]) == 0:
                substring += ' 3'
            
            label_file_list[i+head+1] = substring
            
        label_file_string = '\n'.join(label_file_list)
           
        # Write content to file
        writeFile(PRISM_labelfile, 'w', label_file_string)
        
        print(' --- Writing PRISM transition file')
        
        ### Write transition file
        PRISM_transitionfile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".tra"
            
        transition_file_list = ['' for i in range(self.nr_regions)]
            
        nr_choices_absolute = 0
        nr_transitions_absolute = 0
        
        printEvery = min(100, max(1, int(self.nr_regions/10)))
        
        # For every state
        for s in progressbar(range(self.nr_regions), redirect_stdout=True):
            
            if s in partition['goal']:
                # print(' ---- Skip',s,'because it is a goal region')
                continue
            if s in partition['critical']:
                # print(' ---- Skip',s,'because it is a critical region')
                continue
            
            if s % printEvery == 0 and verbose:
                print(' ---- Write for region',s)
            
            choice = 0
            selfloop = False
            
            if len(actions['enabled'][s]) > 0:
            
                substring = ['' for i in 
                             range(len(actions['enabled'][s]))]
                    
                # For every enabled action                
                for a_idx,a in enumerate(actions['enabled'][s]):
                    
                    # Define name of action
                    actionLabel = "a_"+str(a)
                    
                    substring_start = str(s+head) +' '+ str(choice)
                    
                    if self.setup.parametric:
                        P = trans['prob'][0][a][s]
                        
                    else:
                        P = trans['prob'][0][a]
                    
                    if mode == 'interval':
                    
                        # Add probability to end in absorbing state
                        deadlock_string = P['deadlock_interval_string']
                    
                        # Retreive probability intervals (and 
                        # corresponding state ID's)
                        probability_strings = P['interval_strings']
                        probability_idxs    = P['successor_idxs']
                    
                        # Absorbing state has index zero
                        substring_a = [substring_start+' 0 '+
                                  deadlock_string+' '+actionLabel]
                        
                        # Add resulting entries to the list
                        substring_b = [substring_start +" "+str(i+head)+
                                          " "+intv+" "+actionLabel 
                          for (i,intv) in zip(probability_idxs,
                                              probability_strings) if intv]
                        
                    else:
                        
                        # Add probability to end in absorbing state
                        deadlock_string = str(P['deadlock_approx'])
                        
                        # Retreive probability intervals (and 
                        # corresponding state ID's)
                        probability_strings = P['approx_strings']
                        probability_idxs    = P['successor_idxs']
                        
                        if float(deadlock_string) > 0:
                            # Absorbing state has index zero
                            substring_a = [substring_start+' 0 '+
                                      deadlock_string+' '+actionLabel]
                        else:
                            substring_a = []
                            
                        # Add resulting entries to the list
                        substring_b = [substring_start +" "+str(i+head)+
                                          " "+intv+" "+actionLabel 
                          for (i,intv) in zip(probability_idxs,
                                              probability_strings) if intv]
                        
                    # Increase choice counter
                    choice += 1
                    nr_choices_absolute += 1
                    
                    nr_transitions_absolute += len(substring_a) + \
                                               len(substring_b)
                    
                    substring[a_idx] = '\n'.join(substring_a + substring_b)
                
            else:
                
                # No actions enabled, so only add self-loop
                if selfloop is False:
                    if mode == 'interval':
                        selfloop_prob = '[1.0,1.0]'
                    else:
                        selfloop_prob = '1.0'
                        
                    substring = [str(s+head) +' 0 '+str(s+head)+' '+
                                    selfloop_prob]
        
                    selfloop = True
                    
                    # Increment choices and transitions both by one
                    nr_choices_absolute += 1
                    nr_transitions_absolute += 1
                    choice += 1
                
                else:
                    substring = []
              
            transition_file_list[s] = substring
            
        flatten = lambda t: [item for sublist in t 
                                  for item in sublist]
        transition_file_list = '\n'.join(flatten(transition_file_list))
        
        print(' ---- String ready; write to file...')
        
        # Header contains nr of states, choices, and transitions
        size_states = self.nr_regions+head
        size_choices = nr_choices_absolute+head
        size_transitions = nr_transitions_absolute+head
        model_size = {'States': size_states, 
                      'Choices': size_choices, 
                      'Transitions':size_transitions}
        header = str(size_states)+' '+str(size_choices)+' '+ \
                 str(size_transitions)+'\n'
        
        if mode == 'interval':
            firstrow = '0 0 0 [1.0,1.0]\n1 0 1 [1.0,1.0]\n2 0 2 [1.0,1.0]\n'
        else:
            firstrow = '0 0 0 1.0\n1 0 1 1.0\n1 0 1 1.0\n'
        
        # Write content to file
        writeFile(PRISM_transitionfile, 'w', header+firstrow+
                                             transition_file_list)
            
        ### Write specification file
        specfile, specification = self.writePRISM_specification(mode, problem_type)
        
        return model_size, PRISM_allfiles, specfile, specification