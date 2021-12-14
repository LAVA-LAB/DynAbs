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
    def __init__(self, setup, N, abstr):
        '''
        Create the MDP model in Python for the current instance.
    
        Parameters
        ----------
        setup : dict
            Setup dictionary.
        N : int
            Finite time horizon.
        abstr : dict
            Abstraction dictionary
    
        Returns
        -------
        mdp : dict
            MDP dictionary.
    
        '''
        
        self.setup = setup
        self.N = N
        
        self.nr_regions = abstr['nr_regions']
        self.nr_actions = abstr['nr_actions']
        self.nr_states  = self.nr_regions
        
        # Specify goal state set
        self.goodStates = abstr['goal']
        
        # Specify sets that can never reach target set
        self.badStates  = abstr['critical']

    def writePRISM_scenario(self, abstr, trans, mode='estimate', 
                            horizon='infinite'):
        '''
        Converts the model to the PRISM language, and write the model to file.
    
        Parameters
        ----------
        abstr : dict
            Abstraction dictionary
        trans : dict
            Dictionary of transition probabilities        
        mode : str, optional
            Is either 'estimate' or 'interval'.
        horizon : str, optional
            Is either 'finite' or 'infinite'.
    
        Returns
        -------
        None.
    
        '''
    
        if horizon == 'infinite':
            min_delta = self.N
        else:
            min_delta = min(self.setup.deltas)                
    
        # Define PRISM filename
        PRISM_file = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".prism"
    
        if mode == "default":
            modeltype = "MDP"
        else:
            modeltype = "iMDP"
    
        # Write header for the PRISM file
        if horizon == 'infinite':
            header = [
                "// "+modeltype+" (filter-based abstraction method) \n",
                "// Infinite horizon version \n\n"
                "mdp \n\n",
                # "const int xInit; \n\n",
                "const int regions = "+str(int(self.nr_regions-1))+"; \n\n",
                "module "+modeltype+"_abstraction \n\n",
                ]
            
            # Define variables in module
            variable_defs = [
                "\tx : [-1..regions]; \n\n",
                ]
            
        else:
            header = [
                "// "+modeltype+" (scenario-based abstraction method) \n\n",
                "mdp \n\n",
                # "const int xInit; \n\n",
                "const int Nhor = "+str(int(self.N / min_delta))+"; \n",
                "const int regions = "+str(int(self.nr_regions-1))+"; \n\n",
                "module iMDP \n\n",
                ]
        
            # Define variables in module
            variable_defs = [
                "\tk : [0..Nhor]; \n",
                "\tx : [-1..regions]; \n\n",
                ]
        
        # Add initial header to PRISM file
        writeFile(PRISM_file, 'w', header+variable_defs)
            
        #########
        
        # Define actions
        for k in range(0, self.N, min_delta):
            
            # Add comment to list
            if horizon == 'finite':
                action_defs = ["\t// Actions for k="+str(k)+"\n"]
                
            else:
                action_defs = []    
            
            # For every delta value in the list
            for delta_idx,delta in enumerate(self.setup.deltas):
                
                action_defs += ["\t// Delta="+str(delta)+"\n"]
                
                # Only continue if resulting time is below horizon
                boolean = True
                if k % delta != 0:
                    boolean = False
                # Always proceed if horizon is infinite
                if (k + delta <= self.N and boolean) or horizon == 'infinite':
                    
                    # For every action (i.e. target state)
                    for a in range(self.nr_actions):
                        
                        # Define name of action
                        actionLabel = "[a_"+str(a)+"_d_"+str(delta)+"]"
                        
                        # Retreive in which states this action is enabled
                        enabledIn = abstr['actions_inv'][delta][a]
                        
                        # Only continue if this action is enabled anywhere
                        if len(enabledIn) > 0:
                            
                            # Write guards
                            guardPieces = ["x="+str(i) for i in enabledIn]
                            sep = " | "
                            
                            if horizon == 'infinite':
                                # Join individual pieces and write full guard
                                guard = sep.join(guardPieces)
                                
                                # For infinite time horizon, kprime not used
                                kprime = ""
                                
                            else:                
                                # Join  individual pieces
                                guardStates = sep.join(guardPieces)
                                
                                # Write full guard
                                guard = "k="+str(int(k/min_delta))+" & ("+guardStates+")"
                                
                                # Compute successor state time step
                                kprime = "&(k'=k+"+str(int(delta/min_delta))+")"
                            
                            if mode == 'interval':
                                
                                # Retreive probability intervals (and corresponding state ID's)
                                interval_idxs    = trans['prob'][delta][k][a]['interval_idxs']
                                interval_strings = trans['prob'][delta][k][a]['interval_strings']
                                
                                # If mode is interval, use intervals on probs.
                                succPieces = [intv+" : (x'="+
                                              str(i)+")"+kprime 
                                              for (i,intv) in zip(interval_idxs,interval_strings)]
                                
                                # Use absorbing state to make sure that probs sum to one
                                deadlock_string = trans['prob'][delta][k][a]['deadlock_interval_string']
                                succPieces += [deadlock_string+" : (x'=-1)"+kprime]
                                
                            else:
                                # Write resulting states with their probabilities
                                succProbStrings = trans['prob'][delta][k][a]['approx_strings']
                                succProbIdxs    = trans['prob'][delta][k][a]['approx_idxs']
                                
                                # If mode is default, use concrete probabilities
                                succPieces = [str(p)+":(x'="+str(i)+")"+kprime
                                              for (i,p) in zip(succProbIdxs,succProbStrings)]
                                
                                # Use absorbing state to make sure that probs sum to one
                                deadlockProb = trans['prob'][delta][k][a]['deadlock_approx']
                                if float(deadlockProb) > 0:
                                    succPieces += [str(deadlockProb)+":(x'=-1)"+kprime]                                
                                
                            # Join  individual pieces
                            sep = " + "
                            successors = sep.join(succPieces)
                            
                            # Join pieces to write full action definition
                            action_defs += "\t"+actionLabel+" " + guard + \
                                " -> " + successors + "; \n"
            
            # Insert extra white lines
            action_defs += ["\n\n"]
            
            # Add actions to PRISM file
            writeFile(PRISM_file, "a", action_defs)
            
        #########
        
        if horizon == 'infinite':
            footer = [
                "endmodule \n\n",
                "init x > -1 endinit \n\n"
                ]
            
        else:
            footer = [
                "endmodule \n\n",
                "init k=0 endinit \n\n"
                ]
        
        labelPieces = ["(x="+str(x)+")" for x in abstr['goal']]
        sep = "|"
        labelGuard = sep.join(labelPieces)
        labels = [
            "// Labels \n",
            "label \"reached\" = "+labelGuard+"; \n"
            ]
        
        # Add footer and label definitions to PRISM file
        writeFile(PRISM_file, "a", footer + labels)
        
        #########
        specfile, specification = self.writePRISM_specification(mode, horizon)
        
        if mode == 'estimate':
            printSuccess('MDP ('+horizon+' horizon) exported as PRISM file')
        else:
            printSuccess('iMDP ('+horizon+' horizon) exported as PRISM file')
        
        return PRISM_file, specfile, specification
    
    def writePRISM_specification(self, mode, horizon):
        '''
        Write the PRISM specificatoin / property to a file

        Parameters
        ----------
        mode : str
            Is either 'estimate' or 'interval'.
        horizon : str
            Is either 'finite' or 'infinite'.

        Returns
        -------
        specfile : str
            The name of the file in which the specification is stored.
        specification : str
            The specification itself given as a string.

        '''
        
        if horizon == 'infinite':
            # Infer number of time steps in horizon (at minimum delta value)
            horizonLen = int(self.N/min(self.setup.deltas))
            
            if mode == 'estimate':
                # If mode is default, set maximum probability as specification
                specification = 'Pmax=? [ F<='+str(horizonLen)+' "reached" ]'
                
            elif mode == 'interval':
                # If mode is interval, set lower bound of max. prob. as spec.
                specification = 'Pmaxmin=? [ F<='+str(horizonLen)+' "reached" ]'
            
        else:
            if mode == 'estimate':
                # If mode is default, set maximum probability as specification
                specification = 'Pmax=? [ F "reached" ]'
                
            elif mode == 'interval':
                # If mode is interval, set lower bound of max. prob. as spec.
                specification = 'Pmaxmin=? [ F "reached" ]'
            
        # Define specification file
        specfile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".pctl"
    
        # Write specification file
        writeFile(specfile, 'w', specification)
        
        return specfile, specification
    
    def writePRISM_explicit(self, abstr, trans, mode='estimate', 
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
        
        ### Write states file
        PRISM_statefile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".sta"
        
        state_file_string = '\n'.join(['(x)\n0:(-1)'] + 
              [str(i+1)+':('+str(i)+')' for i in range(self.nr_regions)])
        
        # Write content to file
        writeFile(PRISM_statefile, 'w', state_file_string)
        
        print(' --- Writing PRISM label file')
        
        ### Write label file
        PRISM_labelfile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".lab"
            
        label_file_list = ['0="init" 1="deadlock" 2="reached"'] + \
                          ['0: 1'] + \
                          ['' for i in range(self.nr_regions)]
        
        for i in range(self.nr_regions):
            substring = str(i+1)+': 0'
            
            # Check if region is a deadlock state
            for delta in self.setup.deltas:
                if len(abstr['actions'][delta][i]) == 0:
                    substring += ' 1'
                    break
            
            # Check if region is in goal set
            if i in self.goodStates:
                substring += ' 2'
            
            label_file_list[i+2] = substring
            
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
            
            if s % printEvery == 0 and verbose:
                print(' ---- Write for region',s)
            
            substring = ['' for i in range(len(self.setup.deltas))]
            
            choice = 0
            selfloop = False
            
            # Check if region is a deadlock state
            for delta_idx,delta in enumerate(self.setup.deltas):    
            
                if len(abstr['actions'][delta][s]) > 0:
            
                    subsubstring = ['' for i in 
                                    range(len(abstr['actions'][delta][s]))]
            
                    # For every enabled action                
                    for a_idx,a in enumerate(abstr['actions'][delta][s]):
                        
                        # Define name of action
                        actionLabel = "a_"+str(a)+"_d_"+str(delta)
                        
                        substring_start = str(s+1) +' '+ str(choice)
                        
                        if mode == 'interval':
                        
                            # Add probability to end in absorbing state
                            deadlock_string = trans['prob'][delta][0][a][
                                                'deadlock_interval_string']
                        
                            # Retreive probability intervals (and 
                            # corresponding state ID's)
                            probability_strings = trans['prob'][delta][0][a][
                                                'interval_strings']
                            probability_idxs    = trans['prob'][delta][0][a][
                                                'interval_idxs']
                        
                            # Absorbing state has index zero
                            subsubstring_a = [substring_start+' 0 '+
                                      deadlock_string+' '+actionLabel]
                            
                            # Add resulting entries to the list
                            subsubstring_b = [substring_start +" "+str(i+1)+
                                              " "+intv+" "+actionLabel 
                              for (i,intv) in zip(probability_idxs,
                                                  probability_strings) if intv]
                            
                        else:
                            
                            # Add probability to end in absorbing state
                            deadlock_string = str(trans['prob'][delta][0][a][
                                                'deadlock_approx'])
                            
                            # Retreive probability intervals (and 
                            # corresponding state ID's)
                            probability_strings = trans['prob'][delta][0][a][
                                                'approx_strings']
                            probability_idxs    = trans['prob'][delta][0][a][
                                                'approx_idxs']
                            
                            if float(deadlock_string) > 0:
                                # Absorbing state has index zero
                                subsubstring_a = [substring_start+' 0 '+
                                          deadlock_string+' '+actionLabel]
                            else:
                                subsubstring_a = []
                                
                            # Add resulting entries to the list
                            subsubstring_b = [substring_start +" "+str(i+1)+
                                              " "+intv+" "+actionLabel 
                              for (i,intv) in zip(probability_idxs,
                                                  probability_strings) if intv]
                            
                        # Increase choice counter
                        choice += 1
                        nr_choices_absolute += 1
                        
                        nr_transitions_absolute += len(subsubstring_a) + \
                                                   len(subsubstring_b)
                        
                        subsubstring[a_idx] = '\n'.join(subsubstring_a + 
                                                        subsubstring_b)
                    
                else:
                    
                    # No actions enabled, so only add self-loop
                    if selfloop is False:
                        if mode == 'interval':
                            selfloop_prob = '[1.0,1.0]'
                        else:
                            selfloop_prob = '1.0'
                            
                        subsubstring = [str(s+1) +' 0 '+str(s+1)+' '+
                                        selfloop_prob]
            
                        selfloop = True
                        
                        # Increment choices and transitions both by one
                        nr_choices_absolute += 1
                        nr_transitions_absolute += 1
                        choice += 1
                    
                    else:
                        subsubstring = []
                        
                substring[delta_idx] = subsubstring
                
            transition_file_list[s] = substring
            
        flatten = lambda t: [item for sublist in t 
                                  for subsublist in sublist 
                                  for item in subsublist]
        transition_file_list = '\n'.join(flatten(transition_file_list))
        
        print(' ---- String ready; write to file...')
        
        # Header contains nr of states, choices, and transitions
        size_states = self.nr_regions+1
        size_choices = nr_choices_absolute+1
        size_transitions = nr_transitions_absolute+1
        model_size = {'States': size_states, 
                      'Choices': size_choices, 
                      'Transitions':size_transitions}
        header = str(size_states)+' '+str(size_choices)+' '+ \
                 str(size_transitions)+'\n'
        
        if mode == 'interval':
            firstrow = '0 0 0 [1.0,1.0]\n'
        else:
            firstrow = '0 0 0 1.0\n'
        
        # Write content to file
        writeFile(PRISM_transitionfile, 'w', header+firstrow+
                                             transition_file_list)
            
        ### Write specification file
        specfile, specification = self.writePRISM_specification(mode, 
                                                        horizon='infinite')
        
        return model_size, PRISM_allfiles, specfile, specification