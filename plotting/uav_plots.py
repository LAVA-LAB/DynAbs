from re import I
import numpy as np              # Import Numpy for computations
import pandas as pd             # Import Pandas to store data in frames
import matplotlib.pyplot as plt # Import Pyplot to generate plots
import seaborn as sns
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

from core.commons import printWarning, cm2inch
from core.define_partition import state2region

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



def plot_uav_layout(data, static = False, export = True, fname = 'uav_layout'):
    '''
    Plot UAV planning layout without resulting traces / simulations
    '''
    
    print('- Plot UAV planning environment without trajectories...')
    
    UAV = plot_uav(data['model'], data['regions'], data['goal_regions'], data['critical_regions'], data['spec'])
    
    if static:
        
        UAV.render(data['spec'])
        UAV.render_screenshot(data['setup'], fname = str(fname)+'.png')
        
    else:
        
        UAV.render(data['spec'])
        UAV.render_rotate(export, data['setup'], fname = str(fname)+'.gif')

    return
    


def plot_uav_sim(data, num = 10, static = False, export = True, fname = 'uav_trajectories'):
    '''
    Plot UAV problem withresulting traces (if static) or simulations (othw.)
    '''
    
    print('- Plot UAV planning environment with {} trajectories...'.format(num))
    
    s_init = state2region(data['args'].x_init, data['spec'].partition, data['regions']['c_tuple'])[0]
    
    UAV = plot_uav(data['model'], data['regions'], data['goal_regions'], data['critical_regions'], data['spec'])
    
    if static:
    
        for i in range(num):
            trace = np.array(data['mc'].traces[s_init][i]['x'])
            
            # if i == 0:
            color = 'b'
            marker = '.'
            # else:
            #     color = (1,0.647,0)
            #     marker = 'x'
                
            UAV.add_trace_static(trace, color, marker)
        
        UAV.render(data['spec'])
        UAV.render_screenshot(data['setup'], fname = str(fname)+'.png')
        
    else:
        
        for i in range(num):
            trace = np.array(data['mc'].traces[s_init][i]['x'])
            
            UAV.add_trace_animated(trace)
            
        UAV.render(data['spec'])
        UAV.render_dynamic(export, data['setup'], fname = str(fname)+'.gif')

    return



from scipy.interpolate import interp1d
import visvis as vv

class animate(object):
    
    def __init__(self, points, obj):
        
        self.points = points
        self.i = 0
        self.obj = obj
        
        return
    
    def move(self, reset):
        
        if reset:
            self.i = 0
        else:
            self.i += 1
        
        if self.i < len(self.points):
        
            self.obj.dx = self.points[self.i, 0]
            self.obj.dy = self.points[self.i, 1]
            self.obj.dz = self.points[self.i, 2]
        
            done = False
        
        else:        
            
            done = True
        
        return done
    
    

class plot_uav(object):
    '''
    Main abstraction object    
    '''

    def __init__(self, model, regions, goal_regions, critical_regions, spec):
        
        self.cut_value = np.zeros(3)
        for i,d in enumerate(range(1, model.n, 2)):
            if spec.partition['number'][d]/2 != round( 
                    spec.partition['number'][d]/2 ):
                self.cut_value[i] = 0
            else:
                self.cut_value[i] = spec.partition['number'][d] / 2
        
        print('Create 3D UAV plot using Visvis')

        
        
        print('-- Visvis imported')

        self.fig = vv.figure()
        self.f = vv.clf()
        self.a = vv.cla()
        self.fig = vv.gcf()
        self.ax = vv.gca()
        
        self.obj_anim_list = []
        self.reset = False
        self.times_reset = 0
        
        self.ix = 0
        self.iy = 2
        self.iz = 4
        
        self.vx = 1
        self.vy = 3
        self.vz = 5
        
        regionWidth_xyz = np.array([spec.partition['width'][self.ix], 
                                    spec.partition['width'][self.iy], 
                                    spec.partition['width'][self.iz]])    
        
        print('-- Visvis initialized')
        
        # Draw goal states
        for i,goal in enumerate(goal_regions):

            goalState = regions['center'][goal]
            if goalState[self.vx] == self.cut_value[0] and \
                goalState[self.vy] == self.cut_value[1] and \
                goalState[self.vz] == self.cut_value[2]:

                # print('--- Draw goal region',i)
            
                center_xyz = np.array([goalState[self.ix], 
                                        goalState[self.iy], 
                                        goalState[self.iz]])
                
                goal = vv.solidBox(tuple(center_xyz), 
                                    scaling=tuple(regionWidth_xyz))
                goal.faceColor = (0,1,0,0.8)

        print('-- Goal regions drawn')

        # Draw critical states
        for i,crit in enumerate(critical_regions):

            critState = regions['center'][crit]
            if critState[self.vx] == self.cut_value[0] and \
                critState[self.vy] == self.cut_value[1] and \
                critState[self.vz] == self.cut_value[2]:
            
                # print('--- Draw critical region',i)

                center_xyz = np.array([critState[self.ix], 
                                        critState[self.iy], 
                                        critState[self.iz]])    
            
                critical = vv.solidBox(tuple(center_xyz), 
                                        scaling=tuple(regionWidth_xyz))
                critical.faceColor = (1,0,0,0.8)
        
        print('-- Critical regions drawn')


    def _get_smooth_curve(self, points, steps = 25):
        
        # Linear length along the line:
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, 
                                              axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
        
        # Interpolation for different methods:
        alpha = np.linspace(0, 1, steps*len(points))
        
        if len(points) == 2:
                kind = 'linear'
        else:
            kind = 'cubic'

        interpolator =  interp1d(distance, points, kind=kind, axis=0)
        interpolated_points = interpolator(alpha)
        
        x = interpolated_points[:,0]
        y = interpolated_points[:,1]
        z = interpolated_points[:,2]
        
        return x,y,z

    
    def add_trace_static(self, trace, color, marker):
            
            # Extract x,y coordinates of trace
            x = trace[:, self.ix]
            y = trace[:, self.iy]
            z = trace[:, self.iz]
            points = np.array([x,y,z]).T
            
            # Plot precise points
            vv.plot(x,y,z, ms=marker, lw=0, mc=color, markerWidth=20)
            
            xp,yp,zp = self._get_smooth_curve(points)
            
            # Plot trace
            vv.plot(xp,yp,zp, lw=5, lc=color)
            
            
    def add_trace_animated(self, trace):
        
        # Extract x,y coordinates of trace
        x = trace[:, self.ix]
        y = trace[:, self.iy]
        z = trace[:, self.iz]
        points = np.array([x,y,z]).T
        
        xp,yp,zp = self._get_smooth_curve(points)
        points_intp = np.array([xp,yp,zp]).T
        
        bm = vv.meshRead('drone.obj')
        obj = vv.OrientableMesh(self.ax, bm)
        
        droneRot = vv.Transform_Rotate(90, ax=1, ay=0, az=0)
        droneScale = vv.Transform_Scale(0.2, 0.2, 0.2)
        obj.transformations.insert(1,droneRot)
        obj.transformations.insert(2,droneScale)
        
        # obj = vv.solidSphere(scaling = 0.3)
        objTrans = obj.transformations[0]
        obj_anim = animate(points_intp, objTrans)
        
        self.obj_anim_list += [obj_anim]
        
        return obj_anim
        
    
    def render(self, spec):

        self.ax.axis.xLabel = 'X'
        self.ax.axis.yLabel = 'Y'
        self.ax.axis.zLabel = 'Z'
        
        # Hide ticks labels and axis labels
        self.ax.axis.xLabel = self.ax.axis.yLabel = self.ax.axis.zLabel = ''    
        self.ax.axis.xTicks = self.ax.axis.yTicks = self.ax.axis.zTicks = []
        
        self.a.axis.axisColor = 'k'
        self.a.axis.showGrid = True
        self.a.axis.edgeWidth = 10
        self.a.bgcolor = 'w'
        
        self.f.relativeFontSize = 1.6
        # ax.position.Correct(dh=-5)
        
        vv.axis('tight', axes=self.ax)
        
        self.fig.position.w = 500
        self.fig.position.h = 500
        
        self.im = vv.getframe(vv.gcf())
        
        bndr = spec.partition['boundary']
        
        self.ax.SetLimits(rangeX=tuple(bndr[self.ix]), 
                          rangeY=tuple(bndr[self.iy]), 
                          rangeZ=tuple(bndr[self.iz]))
        
        self.ax.SetView({'zoom':0.026, 'elevation':65, 'azimuth':-20})
        
        print('-- Plot configured')


    def render_screenshot(self, setup, fname):

        if 'outputFcase' in setup.directories:
        
            filename = setup.directories['outputFcase'] + str(fname)
            
        else:
            
            filename = setup.directories['outputF'] + str(fname)
        
        vv.screenshot(filename, sf=3, bg='w', ob=vv.gcf())
        # app = vv.use()
        # app.Run()
        
        
    def onTimer(self):
        
        done = [obj.move(self.reset) for obj in self.obj_anim_list]        
        self.ax.Draw()
        self.reset = False
        
        if all(done):
            self.times_reset += 1
            self.reset = True
        
        
    def render_dynamic(self, export, setup, fname):
        
        if export:
            rec = vv.record(vv.gcf())
        
        done = False
        max_reps = 2
        while not done:
            self.onTimer()
            
            self.ax.Draw() # Tell the axes to redraw
            self.fig.DrawNow() # Draw the figure NOW, instead of waiting for GUI event loop
            
            if self.times_reset >= max_reps:
                done = True
        
        if export:
            self._export_recording(rec, setup, fname)
        
        
    def render_rotate(self, export, setup, fname):
        
        if export:
            rec = vv.record(vv.gcf())
        
        Nangles = 720
        
        for i in range(Nangles):
            
            self.ax.camera.azimuth = 360 * float(i) / Nangles
            if self.ax.camera.azimuth>180:
                self.ax.camera.azimuth -= 360
            
            self.ax.Draw() # Tell the axes to redraw
            self.fig.DrawNow() # Draw the figure NOW, instead of waiting for GUI event loop
        
        if export:
            self._export_recording(rec, setup, fname)
        
        
    def _export_recording(self, rec, setup, fname):
        
        rec.Stop()
    
        if 'outputFcase' in setup.directories:
        
            filename = setup.directories['outputFcase'] + str(fname)
            
        else:
            
            filename = setup.directories['outputF'] + str(fname)
        
        rec.Export(filename, duration = 1/30)
        
        # app = vv.use()
        # app.Run()