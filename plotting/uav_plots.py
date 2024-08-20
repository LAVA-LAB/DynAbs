from re import I
import numpy as np              # Import Numpy for computations
import pandas as pd             # Import Pandas to store data in frames
import matplotlib.pyplot as plt # Import Pyplot to generate plots
import seaborn as sns
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

from core.commons import printWarning, cm2inch

def UAV_plot_2D(i_show, setup, args, regions, goal_regions, critical_regions, 
                spec, traces, cut_idx, traces_to_plot = 10, line=False):
    '''
    Create 2D trajectory plots for the 2D UAV benchmark

    Parameters
    ----------

    Returns
    -------
    None.

    '''
    
    from scipy.interpolate import interp1d
    
    is1, is2 = i_show
    i_hide = np.array([i for i in range(len(spec.partition['width'])) 
                       if i not in i_show], dtype=int)
    
    print('Show state variables',i_show,'and hide',i_hide)
    
    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))
    
    plt.xlabel('$x$', labelpad=0)
    plt.ylabel('$y$', labelpad=0)

    width = np.array(spec.partition['width'])
    domainMax = width * np.array(spec.partition['number']) / 2
    
    min_xy = spec.partition['origin'] - domainMax
    max_xy = spec.partition['origin'] + domainMax
    
    major_ticks_x = np.arange(min_xy[is1]+1, max_xy[is1]+1, 4*width[is1])
    major_ticks_y = np.arange(min_xy[is2]+1, max_xy[is2]+1, 4*width[is2])
    
    minor_ticks_x = np.arange(min_xy[is1], max_xy[is1]+1, width[is1])
    minor_ticks_y = np.arange(min_xy[is2], max_xy[is2]+1, width[is2])
    
    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(minor_ticks_y, minor=True)
    
    for axi in (ax.xaxis, ax.yaxis):
        for tic in axi.get_minor_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
    
    plt.grid(which='minor', color='#CCCCCC', linewidth=0.3)
    
    # Goal x-y limits
    ax.set_xlim(min_xy[is1], max_xy[is1])
    ax.set_ylim(min_xy[is2], max_xy[is2])
    
    ax.set_title("N = "+str(args.noise_samples),fontsize=10)
    
    keys = list( regions['idx'].keys() )
    # Draw goal states
    for goal in goal_regions:
        
        goalIdx   = np.array(keys[goal])
        if all(goalIdx[i_hide] == cut_idx):

            goal_lower = [regions['low'][goal][is1], regions['low'][goal][is2]]
            goalState = Rectangle(goal_lower, width=width[is1], 
                                  height=width[is2], color="green", 
                                  alpha=0.3, linewidth=None)
            ax.add_patch(goalState)
    
    keys = list( regions['idx'].keys() )
    # Draw critical states
    for crit in critical_regions:
        
        critIdx   = np.array(keys[crit])
        
        if all(critIdx[i_hide] == cut_idx):
        
            critStateLow = [regions['low'][crit][is1], regions['low'][crit][is2]]
            criticalState = Rectangle(critStateLow, width=width[is1], 
                                  height=width[is2], color="red", 
                                  alpha=0.3, linewidth=None)
            ax.add_patch(criticalState)
            
    # Add traces
    i = 0
    for trace in traces.values():

        state_traj = trace['x']

        # Only show trace if there are at least two points
        if len(state_traj) < 2:
            printWarning('Warning: trace '+str(i)+
                         ' has length of '+str(len(state_traj)))
            continue
        else:
            i+= 1

        # Stop at desired number of traces
        if i >= traces_to_plot:
            break
        
        # Convert nested list to 2D array
        trace_array = np.array(state_traj)
        
        # Extract x,y coordinates of trace
        x = trace_array[:, is1]
        y = trace_array[:, is2]
        points = np.array([x,y]).T
        
        # Plot precise points
        plt.plot(*points.T, 'o', markersize=1, color="black");
        
        if line:
        
            # Linear length along the line:
            distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, 
                                                  axis=1 )) )
            distance = np.insert(distance, 0, 0)/distance[-1]
            
            # Interpolation for different methods:
            alpha = np.linspace(0, 1, 75)
            
            if len(state_traj) == 2:
                kind = 'linear'
            else:
                kind = 'quadratic'

            interpolator =  interp1d(distance, points, kind=kind, 
                                     axis=0)
            interpolated_points = interpolator(alpha)
            
            # Plot trace
            plt.plot(*interpolated_points.T, '-', color="blue", linewidth=1);
    
    # Set tight layout
    fig.tight_layout()
    
    # Save figure
    filename = setup.directories['outputFcase']+'drone_trajectory'
    for form in setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    plt.show()



def UAV_3D_plotLayout(setup, args, model, regions, 
                      goal_regions, critical_regions, traces, spec):
    '''
    Create a plot that shows the layout of the UAV problem (without results)
    
    Parameters
    ----------
    ScAb : abstraction instance
        Full object of the abstraction being plotted for
        
    Returns
    -------
    None.
        
    '''
    
    cut_value = np.zeros(3)
    for i,d in enumerate(range(1, model.n, 2)):
        if spec.partition['number'][d]/2 != round( 
                spec.partition['number'][d]/2 ):
            cut_value[i] = 0
        else:
            cut_value[i] = spec.partition['number'][d] / 2    
    
    UAVplot3d_visvis(setup, args, model, regions, goal_regions, 
                     critical_regions, spec, traces=traces, 
                     cut_value=cut_value ) 



def UAVplot3d_visvis(setup, args, model, regions, goal_regions, 
                     critical_regions, spec, traces, cut_value, 
                     traces_to_plot = 10):
    '''
    Create 3D trajectory plots for the 3D UAV benchmark

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    model : dict
        Main dictionary of the LTI system model.
    abstr : dict
        Dictionay containing all information of the finite-state abstraction.
    traces : list
        Nested list containing the trajectories (traces) to plot for
    cut_value : array
        Values to create the cross-section for

    Returns
    -------
    None.

    '''
    
    print('Create 3D UAV plot using Visvis')

    from scipy.interpolate import interp1d
    import visvis as vv
    
    print('-- Visvis imported')

    fig = vv.figure()
    f = vv.clf()
    a = vv.cla()
    fig = vv.gcf()
    ax = vv.gca()
    
    ix = 0
    iy = 2
    iz = 4
    
    regionWidth_xyz = np.array([spec.partition['width'][0], 
                                spec.partition['width'][2], 
                                spec.partition['width'][4]])    
    
    print('-- Visvis initialized')

    # Draw goal states
    for i,goal in enumerate(goal_regions):

        goalState = regions['center'][goal]
        if goalState[1] == cut_value[0] and \
            goalState[3] == cut_value[1] and \
            goalState[5] == cut_value[2]:

            center_xyz = np.array([goalState[0], 
                                   goalState[2], 
                                   goalState[4]])
            
            goal = vv.solidBox(tuple(center_xyz), 
                               scaling=tuple(regionWidth_xyz))
            goal.faceColor = (0,1,0,0.8)

    print('-- Goal regions drawn')

    # Draw critical states
    for i,crit in enumerate(critical_regions):

        critState = regions['center'][crit]
        if critState[1] == cut_value[0] and \
            critState[3] == cut_value[1] and \
            critState[5] == cut_value[2]:
        
            center_xyz = np.array([critState[0], 
                                    critState[2], 
                                    critState[4]])    
        
            critical = vv.solidBox(tuple(center_xyz), 
                                    scaling=tuple(regionWidth_xyz))
            critical.faceColor = (1,0,0,0.8)
    
    print('-- Critical regions drawn')

    # Add traces
    i = 0
    for trace in traces.values():

        state_traj = trace['x']

        # Set color and line style
        if i == 0:
            clr = 'b'
            ms = '.'
        else:
            clr = (1,0.647,0)
            ms = 'x'

        # Only show trace if there are at least two points
        if len(state_traj) < 2:
            printWarning('Warning: trace '+str(i)+
                         ' has length of '+str(len(state_traj)))
            continue
        else:
            i+= 1

        # Stop at desired number of traces
        if i >= traces_to_plot:
            break

        # Convert nested list to 2D array
        trace_array = np.array(state_traj)
        
        # Extract x,y coordinates of trace
        x = trace_array[:, ix]
        y = trace_array[:, iy]
        z = trace_array[:, iz]
        points = np.array([x,y,z]).T
        
        # Plot precise points
        vv.plot(x,y,z, ms=ms, lw=0, mc=clr, markerWidth=20)
        
        # Linear length along the line:
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, 
                                              axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
        
        # Interpolation for different methods:
        alpha = np.linspace(0, 1, 75)
        
        if len(trace_array) == 2:
                kind = 'linear'
        else:
            kind = 'cubic'

        interpolator =  interp1d(distance, points, kind=kind, axis=0)
        interpolated_points = interpolator(alpha)
        
        xp = interpolated_points[:,0]
        yp = interpolated_points[:,1]
        zp = interpolated_points[:,2]
        
        # Plot trace
        vv.plot(xp,yp,zp, lw=5, lc=clr)

    print('-- Traces regions drawn')

    ax.axis.xLabel = 'X'
    ax.axis.yLabel = 'Y'
    ax.axis.zLabel = 'Z'
    
    # Hide ticks labels and axis labels
    ax.axis.xLabel = ax.axis.yLabel = ax.axis.zLabel = ''    
    ax.axis.xTicks = ax.axis.yTicks = ax.axis.zTicks = []
    
    a.axis.axisColor = 'k'
    a.axis.showGrid = True
    a.axis.edgeWidth = 10
    a.bgcolor = 'w'
    
    app = vv.use()
    
    f.relativeFontSize = 1.6
    # ax.position.Correct(dh=-5)
    
    vv.axis('tight', axes=ax)
    
    fig.position.w = 1000
    fig.position.h = 600
    
    im = vv.getframe(vv.gcf())
    
    ax.SetView({'zoom':0.042, 'elevation':25, 'azimuth':-35})
    
    print('-- Plot configured')

    if 'outputFcase' in setup.directories:
    
        filename = setup.directories['outputFcase'] + \
                    'UAV_paths_screenshot.png'
        
    else:
        
        filename = setup.directories['outputF'] + 'UAV_paths_screenshot.png'
    
    vv.screenshot(filename, sf=3, bg='w', ob=vv.gcf())
    app.Run()