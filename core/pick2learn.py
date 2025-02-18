#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .define_partition import computeRegionCenters
from .cvx_opt import Controller
from scipy.stats import beta as betaF

from core.define_partition import state2region
from core.monte_carlo import MonteCarloSim


class P2L:
    '''
    Class to run Monte Carlo simulations under a derived controller
    '''

    def __init__(self, Ab, noise_samples, enabled=True, **kwargs):
        '''
        Initialization function

        Parameters
        ----------
        Ab : abstraction instance
            Full object of the abstraction being plotted for

        Returns
        -------
        None.

        '''

        print('\nInitialize Pick2Learn meta algorithm...')

        self.iteration = 1
        self.enabled = bool(enabled)

        # Create order (tie-break rule) over the samples
        self.order = np.arange(len(noise_samples))

        if enabled:
            # If P2L is enabled
            self.num_pretrain_samples = int(np.round(len(noise_samples) * Ab.args.P2L_pretrain_fraction))

            # Unused samples
            self.D = noise_samples[self.num_pretrain_samples:]
            self.Di = self.order[self.num_pretrain_samples:]

            # Used samples
            self.T = noise_samples[:self.num_pretrain_samples]
            self.Ti = self.order[:self.num_pretrain_samples]

        else:
            # Otherwise, use all samples
            self.T = noise_samples
            self.Ti = self.order

        if Ab.flags['underactuated']:
            self.controller = Controller(Ab.model)

        self.results = dict()
        self.traces = dict()

        # Copy necessary data from abstraction object
        self.flags = Ab.flags
        self.model = Ab.model
        self.partition = Ab.partition
        self.actions = Ab.actions
        self.args = Ab.args
        self.spec = Ab.spec

        self.horizon = Ab.N
        if self.horizon == np.inf:
            max_horizon = int(100)
            print(' -- Horizon of abstraction is infinite; set to {} for simulations'.format(max_horizon))
            self.horizon = max_horizon

        print('')

    def run(self, x0, policy, noise_samples):

        self.policy = policy

        # Preference scores
        score = np.zeros(len(noise_samples))

        for i, noise in enumerate(noise_samples):
            _, score[i] = self.simulate(x0, noise)

        return score

    def update(self, Ab):

        # Run Monte Carlo simulations
        score = self.run(policy=Ab.results['optimal_policy'],
                         noise_samples=self.D,
                         x0=Ab.args.x_init)

        # First sort the failed sample idxs according to the ordering Di
        idxs_failed = np.arange(len(self.D))[score == 0]
        idxs_sorted = idxs_failed[np.argsort(self.Di[score == 0])]

        print(f'\n=== Pick2Learn iteration {self.iteration} ===')
        print(f'Number of violating samples: {len(idxs_failed)}')

        if len(idxs_sorted) == 0 or len(idxs_sorted) < Ab.args.P2L_add_per_iteration:
            # If there are fewer violations than the max we add per iteration, then we are done
            done = True

            # Add all remaining failed samples (sorted by tie-break rule)
            add_idx = idxs_sorted

        else:
            # If there are more violations than the max we add per iteration, then we are not done yet
            done = False
            self.iteration += 1

            # Add the max. nr. of remaining failed samples (sorted by tie-break rule)
            add_idx = idxs_sorted[:Ab.args.P2L_add_per_iteration]

        # Append sample to data used by algorithm
        self.T = np.concatenate((self.T, self.D[add_idx]))
        self.Ti = np.concatenate((self.Ti, self.Di[add_idx]))

        # Remove these samples from the unused dataset
        bool_remove = np.full(len(self.Di), False)
        bool_remove[add_idx] = True
        self.D = self.D[~bool_remove]
        self.Di = self.Di[~bool_remove]

        print(f' - Added {len(add_idx)} samples')
        print(f' - Training set has now size {len(self.T)}\n')

        return done

    def compute_bound(self, Ab):

        # Calculate confidence leven on reach-avoid probability
        samples_added = len(self.Ti) - self.num_pretrain_samples  # Only count samples that were not in initial pretrain set
        epsL, epsU = bound_risk(samples_added, Ab.args.noise_samples - self.num_pretrain_samples, Ab.args.P2L_delta)

        self.generalization_bound = 1 - epsU
        print(f' > With a confidence of {(1 - Ab.args.P2L_delta):.6f}, the reach-avoid probability is at least {self.generalization_bound:.6f}')

        self.empirical_bound = -1
        if Ab.args.monte_carlo_iter > 0 and len(Ab.args.x_init) == Ab.model.n:
            s_init = state2region(Ab.args.x_init, Ab.spec.partition, Ab.partition['R']['c_tuple'])
            Ab.mc = MonteCarloSim(Ab, iterations=Ab.args.monte_carlo_iter, init_states=s_init)

            self.empirical_bound = Ab.mc.results['reachability_probability'][s_init[0]]
            print(f" > Empirical reach-avoid probability (over {Ab.args.monte_carlo_iter} simulations): {self.empirical_bound:.6f}\n")

        return self.generalization_bound, self.empirical_bound

    def simulate(self, x0, noise):

        # Initialize variables at start of iteration
        success = False
        trace = {'k': [], 'x': [], 'action': []}
        k = 0

        # Initialize the current simulation
        x = np.zeros((self.horizon + 1, self.model.n))
        x_target = np.zeros((self.horizon + 1, self.model.n))
        x_region = np.zeros(self.horizon + 1).astype(int)
        u = np.zeros((self.horizon, self.model.p))
        action = np.zeros(self.horizon).astype(int)

        # Determine nitial state
        x[0] = x0

        # Add current state, belief, etc. to trace
        trace['k'] += [0]
        trace['x'] += [x[0]]

        ######

        # For each time step in the finite time horizon
        while k <= self.horizon:

            # Determine to which region the state belongs
            region_center = computeRegionCenters(x[k],
                                                 self.spec.partition).flatten()

            if tuple(region_center) in self.partition['R']['c_tuple']:
                # Save that state is currently in region ii
                x_region[k] = self.partition['R']['c_tuple'][tuple(region_center)]

            else:
                # Absorbing region reached
                x_region[k] = -1
                return trace, success

            # If current region is the goal state ... 
            if x_region[k] in self.partition['goal']:
                # Then abort the current iteration, as we have achieved the goal
                success = True
                return trace, success

            # If current region is in critical states...
            elif x_region[k] in self.partition['critical']:
                # Then abort current iteration
                return trace, success

            # Check if we can still perform another action within the horizon
            elif k >= self.horizon:
                return trace, success

            # Retreive the action from the policy
            if self.policy.shape[0] == 1:
                # If infinite horizon, policy does not have a time index
                action[k] = self.policy[0, x_region[k]]
            else:
                # If finite horizon, use action for the current time step k
                action[k] = self.policy[k, x_region[k]]

            if action[k] == -1:
                return trace, success

            ###

            # If loop was not aborted, we have a valid action            
            # Set target state equal to the center of the target set
            x_target[k + 1] = self.actions['obj'][action[k]].center

            # Reconstruct the control input required to achieve this target point
            # Note that we do not constrain the control input; we already know that a suitable control exists!
            if self.flags['underactuated']:
                success, _, u[k] = self.controller.solve(x_target[k + 1], x[k],
                                                         self.actions['obj'][action[k]].backreach_obj.target_set_size)

                if not success:
                    print('  >> Failed to compute control input <<')
                    assert False

                # Implement the control into the physical (unobservable) system
                x_hat = self.model.A_true @ x[k] + self.model.B_true @ u[k] + self.model.Q_flat
            else:
                x_nom = x[k]

                u[k] = np.array(self.model.B_pinv @ (x_target[k + 1] - self.model.A @ x_nom.flatten() - self.model.Q_flat))

                # Implement the control into the physical (unobservable) system
                x_hat = self.model.A @ x[k] + self.model.B @ u[k] + self.model.Q_flat

            x[k + 1] = x_hat + noise[k]

            if (any(self.model.uMin > u[k]) or any(self.model.uMax < u[k])) and self.args.verbose:
                print(' - Warning: Control input ' + str(u[k]) + ' outside limits')

            # Add current state, belief, etc. to trace
            trace['k'] += [k + 1]
            trace['x'] += [x[k + 1]]
            trace['action'] += [action[k]]

            # Increase iterator variable by one
            k += 1

        ######

        return trace, success


def bound_risk(k, N, delta):
    t1 = 0
    t2 = k / N
    while t2 - t1 > 1e-10:
        t = (t1 + t2) / 2
        left = delta / 3 * betaF.cdf(t, k + 1, N - k) + delta / 6 * betaF.cdf(t, k + 1, 4 * N + 1 - k)
        right = (1 + delta / 6 / N) * t * N * (betaF.cdf(t, k, N - k + 1) - betaF.cdf(t, k + 1, N - k))
        if left > right:
            t1 = t
        else:
            t2 = t

    epsL = t1

    if k == N:
        epsU = 1
    else:
        t1 = k / N
        t2 = 1
        while t2 - t1 > 1e-10:
            t = (t1 + t2) / 2
            left = (delta / 2 - delta / 6) * betaF.cdf(t, k + 1, N - k) + delta / 6 * betaF.cdf(t, k + 1, 4 * N + 1 - k)
            right = (1 + delta / 6 / N) * t * N * (betaF.cdf(t, k, N - k + 1) - betaF.cdf(t, k + 1, N - k))
            if left > right:
                t2 = t
            else:
                t1 = t
        epsU = t2

    return [epsL, epsU]
