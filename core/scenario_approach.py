import os
import sys
import csv
import numpy as np

def load_scenario_table(tableFile, k):
    '''
    Load tabulated bounds on the transition probabilities (computed using
    the scenario approach).

    Parameters
    ----------
    tableFile : str
        File from which to load the table.

    Returns
    -------
    memory : dict
        Dictionary containing all loaded probability bounds / intervals.

    '''
    
    if not os.path.isfile(tableFile):
        sys.exit('ERROR: the following table file does not exist:'+
                    str(tableFile))
    
    with open(tableFile, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        
        # Skip header row
        next(reader)
        
        memory = np.full((k+1, 2), fill_value = -1, dtype=float)
            
        for i,row in enumerate(reader):
            
            strSplit = row[0].split(',')
            
            value = [float(i) for i in strSplit[-2:]]
            memory[int(strSplit[0])] = value
                
    return memory