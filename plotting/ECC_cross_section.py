import numpy as np              # Import Numpy for computations
import pandas as pd
from core.define_partition import define_partition

def get_cross_section(model, setup, c_tuple, spec, values, suffix = False):
    '''
    Extract the satisfaction probabilities on a 1D cross-section of the state space

    Usage:
    get_cross_section(data['model'], data['setup'], data['regions']['c_tuple'], data['spec'], data['results']['optimal_reward'])

    Returns
    -------
    Array and DataFrame with cross-section

    '''

    x_nr = spec.partition['number'][0]
    y_nr = 1

    cut_centers = define_partition(model.n, [x_nr, y_nr],
                     spec.partition['width'],
                     spec.partition['origin'])['center']

    cross_section_array = np.zeros(x_nr)

    cut_idxs = [c_tuple[tuple(c)] for c in cut_centers
                if tuple(c) in c_tuple]

    for i, idx in enumerate(cut_idxs):
        cross_section_array[i] = values[idx]

    dict = {
        'x': np.array(cut_centers)[:,0],
        'y': cross_section_array
    }

    print(dict)
    df = pd.DataFrame(dict).reset_index(drop=True)

    if suffix is not False:
        filename = setup.directories['outputFcase'] + 'cross_section_'+str(suffix)+'.csv'
    else:
        filename = setup.directories['outputFcase'] + 'cross_section.csv'
    df.to_csv(filename, sep='\t', index=False)

    return cross_section_array, df
