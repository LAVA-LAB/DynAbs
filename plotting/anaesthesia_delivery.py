#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from matplotlib import cm


def heatmap_3D(setup, centers, values, ev = 2):
    '''
    Plot 3D heatmap of the iMDP verification results (only works for a system
    with a 3D state space)

    Parameters
    ----------
    centers : NumPy array of region centers
    values : Optimal reward under iMDP policy
    ev : Number of regions to skip (to improve readability of plot), optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    None.

    '''
    
    assert centers.shape[1] == 3
    
    X = centers[:,0][::ev]
    Y = centers[:,1][::ev]
    Z = centers[:,2][::ev]
    
    values = values[::ev]
    
    MAP = cm.viridis
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=MAP(values), alpha=1, s=2.5)
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
                 
    colmap = cm.ScalarMappable(cmap=MAP)
    colmap.set_array(values)
    
    fig.colorbar(colmap, shrink=0.6, aspect=7, pad=0.1)
    
    plt.tight_layout()

    # Save figure
    filename = setup.directories['outputFcase']+'3D_heatmap'
    for form in setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    plt.show(block = False)