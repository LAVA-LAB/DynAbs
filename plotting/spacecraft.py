from re import I
import numpy as np              # Import Numpy for computations
import pandas as pd             # Import Pandas to store data in frames
import matplotlib.pyplot as plt # Import Pyplot to generate plots
import seaborn as sns
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage,AnnotationBbox
import os
from pathlib import Path
import matplotlib as mpl

from core.commons import printWarning, cm2inch

mpl.rcParams['figure.dpi'] = 300

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def hill2cart(trace,x,y,phi):
    
    xx = x + np.cos(phi) * trace[:,0] - np.sin(phi) * trace[:,1]
    yy = y + np.sin(phi) * trace[:,0] + np.cos(phi) * trace[:,1]

    return xx, yy

def add_image(ax, img, pos, zoom):
    im = OffsetImage(img, zoom)
    im.image.axes = ax
    ab = AnnotationBbox(im, pos, xycoords='data', frameon=False) #,  xybox=(-100, 0.0), frameon=False, xycoords='data',  boxcoords="offset points", pad=0.4)
    ax.add_artist(ab)

def spacecraft(setup, trace):
    
    from scipy.interpolate import interp1d
    
    N = len(trace)

    # Determine path of target
    tg_omega0 = .6*np.pi # Initial angle of target
    tg_dist   = 10 # Initial distance of target from earth
    tg_omega  = -.3 # Angular velocity of target

    tg_angle  = np.array([tg_omega0 + k*tg_omega for k in range(N)])

    # Compute target trajectory
    tg_trajectory = pol2cart( tg_dist, tg_angle )
    
    # Compute chaser trajectory
    ch_trajectory = hill2cart(trace, tg_trajectory[0], tg_trajectory[1], tg_angle)

    tg_trajectory = np.array(tg_trajectory).T
    ch_trajectory = np.array(ch_trajectory).T

    fig, ax = plt.subplots(figsize=cm2inch(6, 6))
    
    # Add figure of earth and satellite
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    earth = plt.imread(Path(cwd, 'earth.png'))
    add_image(ax, earth, pos=(0.0,0.0), zoom=0.04)
    
    satellite = plt.imread(Path(cwd, 'satellite.png'))
    
    add_image(ax, satellite, pos=tuple(ch_trajectory[0]), zoom=0.006)
    
    ###
    
    plt.xlabel('$x$', labelpad=0)
    plt.ylabel('$y$', labelpad=0)
    
    ### PLOT TARGET
    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(tg_trajectory, axis=0)**2, 
                                          axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]
    
    # Interpolation for different methods:
    alpha = np.linspace(0, 1, 75)
    
    if len(tg_trajectory) == 2:
        kind = 'linear'
    else:
        kind = 'quadratic'

    interpolator =  interp1d(distance, tg_trajectory, kind=kind, 
                             axis=0)
    interpolated_points = interpolator(alpha)
    
    # Plot trace
    plt.plot(*interpolated_points.T, ls='dotted', color="red", linewidth=1);
    
    ### PLOT CHASER
    
    # Plot precise points
    plt.plot(*ch_trajectory.T, 'o', markersize=1, color="black");
    
    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(ch_trajectory, axis=0)**2, 
                                          axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]
    
    # Interpolation for different methods:
    alpha = np.linspace(0, 1, 75)
    
    if len(ch_trajectory) == 2:
        kind = 'linear'
    else:
        kind = 'quadratic'

    interpolator =  interp1d(distance, ch_trajectory, kind=kind, 
                             axis=0)
    interpolated_points = interpolator(alpha)
    
    # Plot trace
    plt.plot(*interpolated_points.T, color="blue", linewidth=1);

    ###########

    plt.xlim(-1.5*tg_dist, 1.5*tg_dist)
    plt.ylim(-1.5*tg_dist, 1.5*tg_dist)

    plt.gca().set_aspect('equal', adjustable='box')
    
    # Set tight layout
    fig.tight_layout()
    
    # Save figure
    filename = setup.directories['outputFcase']+'spacecraft_orbit'
    for form in setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    plt.show()
    
    plt.show()