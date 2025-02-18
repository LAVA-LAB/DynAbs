#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load main classes and methods
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

from core.commons import printWarning, cm2inch
from core.monte_carlo import MonteCarloSim
from core.define_partition import define_partition


def oscillator_heatmap(Ab, title='auto'):
    '''
    Create heat map for the reachability probability from any initial state.

    Parameters
    ----------
    Ab : abstraction instance
        Full object of the abstraction being plotted for

    Returns
    -------
    None.

    '''

    x_nr = Ab.spec.partition['number'][0]
    y_nr = Ab.spec.partition['number'][1]

    cut_centers = define_partition(Ab.model.n, [x_nr, y_nr],
                                   Ab.spec.partition['width'],
                                   Ab.spec.partition['origin'])['center']

    cut_values = np.zeros((x_nr, y_nr))
    cut_coords = np.zeros((x_nr, y_nr, Ab.model.n))

    cut_idxs = [Ab.partition['R']['c_tuple'][tuple(c)] for c in cut_centers
                if tuple(c) in Ab.partition['R']['c_tuple']]

    for i, (idx, center) in enumerate(zip(cut_idxs, cut_centers)):

        j = i % y_nr
        k = i // y_nr

        difference = Ab.mc.results['reachability_probability'][idx] - Ab.results['optimal_reward'][idx]

        if difference >= 0:
            # Guarantees safe (model checking >= empirical)
            cut_values[k, j] = 1
        else:
            # Guarantees unsafe (model checking < empirical)
            cut_values[k, j] = 0
        cut_coords[k, j, :] = center

    plot_dataframe = pd.DataFrame(cut_values, index=cut_coords[:, 0, 0],
                                  columns=cut_coords[0, :, 1])

    # Compute the fraction of states for which the empirical reachability
    # guarantees (i.e. simulated performance) are lower than the guarantees
    # obtained from the iMDP model checking
    average_value = plot_dataframe.mean().mean()

    fig = plt.figure(figsize=cm2inch(9, 8))
    ax = sns.heatmap(plot_dataframe.T, cmap="jet",  # YlGnBu
                     vmin=0, vmax=1)
    ax.figure.axes[-1].yaxis.label.set_size(20)
    ax.invert_yaxis()

    ax.set_xlabel('Var 1', fontsize=15)
    ax.set_ylabel('Var 2', fontsize=15)
    if title == 'auto':
        ax.set_title("N = " + str(Ab.args.noise_samples), fontsize=20)
    else:
        ax.set_title(str(title), fontsize=20)

    # Set tight layout
    fig.tight_layout()

    # Save figure
    filename = Path(Ab.setup.directories['outputFcase'], 'safeset_N=' + str(Ab.args.noise_samples))
    for form in Ab.setup.plotting['exportFormats']:
        plt.savefig(filename.with_suffix('.' + str(form)), format=form, bbox_inches='tight')

    plt.show()

    return average_value


def oscillator_traces(Ab, traces, plot_trace_ids=None,
                      line=True, stateLabels=False, title='auto', case=0):
    '''
    Create 2D trajectory plots for the harmonic oscillator benchmark
    
    Returns
    -------
    None.

    '''

    from scipy.interpolate import interp1d

    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))

    plt.xlabel('$x$', labelpad=0)
    plt.ylabel('$y$', labelpad=0)

    width = np.array(Ab.spec.partition['width'])
    domainMax = width * np.array(Ab.spec.partition['number']) / 2

    min_xy = Ab.spec.partition['origin'] - domainMax
    max_xy = Ab.spec.partition['origin'] + domainMax

    major_ticks_x = np.arange(min_xy[0] + 1, max_xy[0] + 1, 5 * width[0])
    major_ticks_y = np.arange(min_xy[1] + 1, max_xy[1] + 1, 5 * width[1])

    minor_ticks_x = np.arange(min_xy[0], max_xy[0] + 1, width[0])
    minor_ticks_y = np.arange(min_xy[1], max_xy[1] + 1, width[1])

    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(minor_ticks_y, minor=True)

    for axi in (ax.xaxis, ax.yaxis):
        for tic in axi.get_minor_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)

    plt.grid(which='minor', color='#CCCCCC', linewidth=0.3)
    plt.grid(which='major', color='#CCCCCC', linewidth=0.3)

    # Goal x-y limits
    ax.set_xlim(min_xy[0] - width[0], max_xy[0] + width[0])
    ax.set_ylim(min_xy[1] - width[1], max_xy[1] + width[1])

    if title == 'auto':
        ax.set_title("N = " + str(Ab.args.noise_samples), fontsize=10)
    else:
        ax.set_title(str(title), fontsize=10)

    # Draw goal states
    for goal in Ab.partition['goal']:
        goal_lower = Ab.partition['R']['low'][goal, [0, 1]]
        goalState = Rectangle(goal_lower, width=width[0],
                              height=width[1], color="green",
                              alpha=0.3, linewidth=None)
        ax.add_patch(goalState)

    # Draw critical states
    for crit in Ab.partition['critical']:
        critStateLow = Ab.partition['R']['low'][crit, [0, 1]]
        criticalState = Rectangle(critStateLow, width=width[0],
                                  height=width[1], color="red",
                                  alpha=0.3, linewidth=None)
        ax.add_patch(criticalState)

    # Show boundary of the partition
    rect = patches.Rectangle(min_xy, max_xy[0] - min_xy[0], max_xy[1] - min_xy[1],
                             linewidth=1.5, edgecolor='gray', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    # Add traces
    for i, trace in traces.items():

        if not plot_trace_ids is None and i not in plot_trace_ids:
            continue

        if len(trace['x']) < 2:
            printWarning('Warning: trace ' + str(i) +
                         ' has length of ' + str(len(trace['x'])))
            continue

        # Convert nested list to 2D array
        trace_array = np.array(trace['x'])

        # Extract x,y coordinates of trace
        x = trace_array[:, 0]
        y = trace_array[:, 1]
        points = np.array([x, y]).T

        if line:
            # Linear length along the line:
            distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2,
                                                axis=1)))
            distance = np.insert(distance, 0, 0) / distance[-1]

            # Interpolation for different methods:
            alpha = np.linspace(0, 1, 75)

            if len(points) > 2:
                interpolator = interp1d(distance, points, kind='quadratic',
                                        axis=0)
            elif len(points) == 2:
                interpolator = interp1d(distance, points, kind='linear',
                                        axis=0)
            else:
                continue
            interpolated_points = interpolator(alpha)

            # Plot trace
            plt.plot(*interpolated_points.T, '-', color="blue", linewidth=1.0, alpha=0.5);

        # Plot precise points
        plt.plot(*points.T, 'o', markersize=2, color="black");

        action_centers = np.array([Ab.actions['obj'][a].center
                                   for a in trace['action'][:len(trace) - 1]])
        action_errors = np.array([Ab.actions['obj'][a].error
                                  for a in trace['action'][:len(trace) - 1]])
        plt.plot(*action_centers.T, 'o', markersize=2, color="red");

        for center, error in zip(action_centers, action_errors):
            low = center + error['neg']
            diff = error['pos'] - error['neg']

            rect = patches.Rectangle(low, diff[0], diff[1],
                                     linewidth=1.0, edgecolor='red',
                                     linestyle='dashed', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

    # Set font sizes
    plt.rc('axes', titlesize=18)  # fontsize of the axes title
    plt.rc('axes', labelsize=18)  # fontsize of the x and y
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels

    # Set tight layout
    fig.tight_layout()

    # Save figure
    filename = Path(Ab.setup.directories['outputFcase'], 'oscillator_traces' + str(case))
    for form in Ab.setup.plotting['exportFormats']:
        plt.savefig(filename.with_suffix('.' + str(form)), format=form, bbox_inches='tight')

    plt.show()


class oscillator_experiment(object):

    def __init__(self, f_min, f_max, f_step, monte_carlo_iterations):
        '''
        Initialize object for collecting and exporting the fraction of safe
        controllers (as reported in the paper)

        Returns
        -------
        None.

        '''

        self.fraction_safe = []

        self.f_min = f_min
        self.f_max = f_max
        self.f_step = f_step

        self.monte_carlo_iterations = monte_carlo_iterations

    def add_iteration(self, Ab, case_id):
        '''
        Perform Monte Carlo simulations and add the resulting fraction of
        states for which the resulting controller is safe to the list        

        Parameters
        ----------
        Ab : Abstraction object
        case_id : Index (int) of the case being run

        Returns
        -------
        None.

        '''

        f_list = np.arange(self.f_min, self.f_max + 0.01, self.f_step)
        frac_sr = pd.Series(index=np.round(f_list, 3), dtype=float, name=case_id)

        for f in f_list:
            f = np.round(f, 3)
            spring = np.round(Ab.model.spring_nom * (1 - f) + Ab.model.spring_max * f, 3)
            mass = np.round(Ab.model.mass_nom * (1 - f) + Ab.model.mass_min * f, 3)

            Ab.model.set_true_model(mass=mass, spring=spring)
            Ab.mc = MonteCarloSim(Ab, iterations=self.monte_carlo_iterations,
                                  random_initial_state=True)

            # reachabilityHeatMap(Ab, montecarlo = True, title='Monte Carlo, mass='+str(mass)+'; spring='+str(spring))
            frac_sr[f] = oscillator_heatmap(Ab, title='Monte Carlo, mass=' + str(mass) + '; spring=' + str(spring))

        self.fraction_safe += [frac_sr]

    def export(self, Ab):

        fraction_safe_df = pd.concat(self.fraction_safe, axis=1)
        fraction_safe_df.index.name = 'x'
        df = fraction_safe_df.T.describe().T

        fraction_safe_df.to_csv(Path(Ab.setup.directories['outputF'], 'fraction_safe_parametric=' + str(Ab.flags['parametric']) + '.csv'),
                                sep=';', encoding='utf-8-sig')

        df.to_csv(Path(Ab.setup.directories['outputF'], 'fraction_safe_parametric=' + str(Ab.flags['parametric']) + '_stats.csv'),
                  sep=';', encoding='utf-8-sig')

    def plot_trace(self, Ab, spring, state_center, number_to_plot=1):
        '''
        Plot a simulated trace

        Parameters
        ----------
        Ab : Abstraction object
        spring : Actual/true spring coefficient
        state_center : Tuple of center of the state to plot
        number_to_plot : Number of traces to plot

        Returns
        -------
        None.

        '''

        df = pd.DataFrame(columns=['guaranteed', 'simulated', 'ratio'], dtype=float)
        df.index.name = 'mass'

        # eval_state = Ab.partition['R']['c_tuple'][(-9.5,0.5)]
        eval_state = Ab.partition['R']['c_tuple'][state_center]

        f_list = np.arange(self.f_min, self.f_max + 0.01, self.f_step)

        for f in f_list:
            f = np.round(f, 3)
            spring = np.round(Ab.model.spring_nom * (1 - f) + Ab.model.spring_max * f, 3)
            mass = np.round(Ab.model.mass_nom * (1 - f) + Ab.model.mass_min * f, 3)

            mass = np.round(mass, 3)
            spring = np.round(spring, 3)

            Ab.model.set_true_model(mass=mass, spring=spring)
            Ab.mc = MonteCarloSim(Ab, iterations=self.monte_carlo_iterations,
                                  init_states=[eval_state], random_initial_state=True)

            df.loc[mass, 'guaranteed'] = np.round(Ab.results['optimal_reward'][eval_state], 4)
            df.loc[mass, 'simulated'] = np.round(Ab.mc.results['reachability_probability'][eval_state], 4)
            df.loc[mass, 'ratio'] = np.round(Ab.mc.results['reachability_probability'][eval_state] / Ab.results['optimal_reward'][eval_state], 4)

            oscillator_traces(Ab,
                              Ab.mc.traces[eval_state],
                              plot_trace_ids=[0],
                              title='Traces for mass=' + str(mass) + '; spring=' + str(spring),
                              case=f)

        csv_file = Path(Ab.setup.directories['outputF'], 'gauranteed_vs_simulated.csv')
        df.to_csv(csv_file, sep=',')
