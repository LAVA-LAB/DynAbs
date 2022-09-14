
# %run "~/documents/sample-abstract/RunPlots.py"

import pickle
import numpy as np

# infile = open(Ab.setup.directories['outputF']+'data_dump.p','rb')
# path = '/home/thom/documents/sample-abstract/output/Ab_spacecraft_2D_09-02-2022_16-27-19/data_dump.p'
path = 'C:\\Users\\Thom Badings\\surfdrive\RU\\2. Research\\Abstractions\\6. JAIR 2022\\Results\\Spacecraft\\data_dump_obstacle.p'
infile = open(path, 'rb')
data = pickle.load(infile)
infile.close()

from plotting.createPlots import reachability_plot
if 'mc' in data:
    print('Create plot with Monte Carlo results')
    reachability_plot(data['setup'], data['results'], data['mc'])
else:
    reachability_plot(data['setup'], data['results'])

from plotting.createPlots import heatmap_3D_view
heatmap_3D_view(data['model'], data['setup'], data['spec'], data['regions']['center'], data['results'])

from plotting.createPlots import heatmap_2D
heatmap_2D(data['args'], data['model'], data['setup'], data['regions']['c_tuple'], data['spec'], data['results']['optimal_reward'])

from plotting.uav_plots import UAV_plot_2D, UAV_3D_plotLayout
from core.define_partition import state2region

if data['model'].name in ['shuttle', 'spacecraft_2D'] :

    if len(data['args'].x_init) == data['model'].n:
        s_init = state2region(data['args'].x_init, data['spec'].partition, data['regions']['c_tuple'])[0]
        traces = data['mc'].traces[s_init]

        UAV_plot_2D((0,1), data['setup'], data['args'], data['regions'], data['goal_regions'], data['critical_regions'], 
                    data['spec'], traces, cut_idx = [0,0], traces_to_plot=10, line=True)
    else:
        print('-- No initial state provided')

if data['model'].name == 'UAV' and data['model'].modelDim == 3:

    if len(data['args'].x_init) == data['model'].n:
        s_init = state2region(data['args'].x_init, data['spec'].partition, data['regions']['c_tuple'])[0]
        traces = data['mc'].traces[s_init]
        
        UAV_3D_plotLayout(data['setup'], data['args'], data['model'], data['regions'], 
                          data['goal_regions'], data['critical_regions'], traces, data['spec'])
    else:
        print('-- No initial state provided')
    
# %%

if data['model'].name == 'spacecraft':
    
    from plotting.uav_plots import UAV_plot_2D
    from core.define_partition import state2region
    
    # Spacecraft trajectory plot in the Hill's frame
    if len(data['args'].x_init) == data['model'].n:
        s_init = state2region(data['args'].x_init, data['spec'].partition, data['regions']['c_tuple'])[0]
        traces = data['mc'].traces[s_init]

        UAV_plot_2D((0,1), data['setup'], data['args'], data['regions'], data['goal_regions'], data['critical_regions'], 
                    data['spec'], traces, cut_idx = [0,0,0,0], traces_to_plot=10, line=True)
    else:
        print('-- No initial state provided')    
    
# %%
        
if data['model'].name == 'spacecraft_2D':
    
    from plotting.spacecraft import spacecraft
    
    key = list(data['mc'].traces.keys())[0]
    trace = np.array(data['mc'].traces[key][0]['x'])
    
    trace = trace[:,[1,0]] / 10
    trace_3D = np.vstack((trace.T, np.linspace(1, 0, len(trace)))).T
    
    spacecraft(data['setup'], trace_3D)
    
if data['model'].name == 'spacecraft':
    
    # 3D spacecraft orbital plot
    from plotting.spacecraft import spacecraft_3D
    
    key = list(data['mc'].traces.keys())[0]
    trace = np.array(data['mc'].traces[key][0]['x'])
    
    scaling = np.array([0.2, 0.2, 1])
    trace = trace[:,0:3] * scaling
    
    trace[0,0] = 0
    
    obstacle_rel_state = np.tile(np.array([0, 12.4-8*0.8/2, 0]) * scaling,
                                 (len(trace),1))
    
    spacecraft_3D(data['setup'], trace, cam_elevation = 20, cam_azimuth = -50, obstacle_rel_state = obstacle_rel_state)