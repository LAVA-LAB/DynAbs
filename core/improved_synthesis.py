#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


class improved_synthesis(object):
    '''
    Improved policy synthesis scheme, which aggregates state based on their
    reach-avoid probability, creating multiple smaller iMDPs.
    '''

    def __init__(self, num_states, goal, num_regions, N):

        self.num_states = num_states
        self.step_size = 1 / num_states
        self.state_ids = np.arange(0, num_states)

        self.partition = np.arange(0, 1, self.step_size)

        # Initialize general policy
        self.general_policy = np.full((N, num_regions), fill_value=-1, dtype=int)

        # Initially set time step equal to horizon
        self.k = N - 1
        self.initial = True

        # Initially, values are one in goal regions; zero elsewhere
        values = np.zeros(num_regions)
        values[goal] = 1

        self.set_values(values)

    def append_policy(self, policy):
        # Append policy for current time step into the general policy
        print(' - Store (partial) policy for time step', self.k)
        self.general_policy[self.k] = policy[0]

        return

    def set_values(self, values):

        # Set values and determine which regions belong to which partitioned values
        self.values = values
        self.state_relation = (self.values // self.step_size).astype(int)

        # Determine lowest value (i.e. worst-case in each partitioned value)
        self.lb_values = [min(self.values[self.values >= v]) for v in self.partition]

        # Number of partitioned values taht are actually used
        self.num_lb_used = len(np.unique(self.lb_values))

    def decrease_time(self):

        self.k -= 1
        self.initial = False

        if self.k < 0:
            return True
        else:
            return False
