#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Implementation of the method proposed in the paper:
 "Probabilities Are Not Enough: Formal Controller Synthesis for Stochastic 
  Dynamical Models with Epistemic Uncertainty"

Originally coded by:        <anonymized>
Contact e-mail address:     <anonymized>

Module containing smaller ancillary functions called repeatedly by other 
functions
______________________________________________________________________________
"""

import numpy as np              # Import Numpy for computations
import math                     # Import Math for mathematical operations
import time                     # Import to create tic/toc functions
import sys                      # Allows to terminate the code at some point
import itertools                # Import to crate iterators
import os                       # Import OS to allow creationg of folders
from scipy.spatial import Delaunay

class table(object):
    '''
    Table object, to print structured output in the console.
    '''
    
    def __init__(self, col_width):
        '''
        Initialize the table.

        Parameters
        ----------
        col_width : list
            List of column widths for the table to be initialized.

        Returns
        -------
        None.

        '''
        self.col_width = col_width
        
    def print_row(self, row, head=False, sort=False):
        '''
        Print a row in the table.

        Parameters
        ----------
        row : list
            List of strings to print in the table.
        head : boolean, optional
            Boolean whether to print a header row. The default is False.
        sort : string, optional
            Can be "warning", "success", or "False". The default is False.

        Returns
        -------
        None.

        '''
        if head:
            print('\n'+'='*sum(self.col_width))
            
        # Define string
        string = "".join(str(word).ljust(self.col_width[i]) 
                         for i,word in enumerate(row))
        
        if sort == "Warning":
            print("\u001b[35m"+string+"\x1b[0m")
        elif sort == "Success":
            print("\u001b[32m"+string+"\x1b[0m")
        else:
            print(string)
            
        if head:
            print('-'*sum(self.col_width))

def in_hull(p, hull):
    '''
    Test if points in `p` are in `hull`.

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    '''
    
    if not isinstance(hull,Delaunay):
        print(' -- Creating hull...')
        hull = Delaunay(hull, qhull_options='QJ')

    boolArray = hull.find_simplex(p) >= 0

    return boolArray

def overapprox_box(brs):
    
    brs_min = np.min(brs, axis=0)
    brs_max = np.max(brs, axis=0)
    
    backreach_heur = np.vstack((brs_min, brs_max))
    
    return backreach_heur

def createDirectory(folder):
    '''
    Helpeer function to create a directory if it not exists yet

    Parameters
    ----------
    folder : str
        Folder to create.

    Returns
    -------
    None.

    '''
    if not os.path.exists(folder):
        os.makedirs(folder)

def TicTocGenerator():
    ''' Generator that returns the elapsed run time '''
    ti = time.time() # initial time
    tf = time.time() # final time
    while True:
        tf = time.time()
        yield tf-ti # returns the time difference
        
def TicTocDifference():
    ''' Generator that returns time differences '''
    tf0 = time.time() # initial time
    tf = time.time() # final time
    while True:
        tf0 = tf
        tf = time.time()
        yield tf-tf0 # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator
TicTocDiff = TicTocDifference() # create an instance of the TicTocGen generator

def toc(tempBool=True):
    ''' Print current time difference '''
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds." %tempTimeInterval )

def tic():
    ''' Start time recorder '''
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
    
def tocDiff(tempBool=True):
    ''' Print current time difference '''
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicTocDiff)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %np.round(tempTimeInterval, 5) )
    else:
        return np.round(tempTimeInterval, 12)
        
    return tempTimeInterval

def ticDiff():
    ''' Start time recorder '''
    # Records a time in TicToc, marks the beginning of a time interval
    tocDiff(False)
    
def nchoosek(n, k):
    '''
    Binomial coefficient or all combinations
    n C k = n! / ( (n-k)! * k! )
    '''
    
    if k == 0:
        r = 1
    else:
        r = n/k * nchoosek(n-1, k-1)
    return round(r)
    
def is_invertible(a):
    '''
    Check if matrix `a` is invertibe

    Parameters
    ----------
    a : ndarray
        A square matrix.

    Returns
    -------
    boolean
        Boolean which is True if the matrix is invertible.

    '''
    
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def printWarning(text):
    '''
    Print a warning

    Parameters
    ----------
    text : str
        Text to print as warning.

    Returns
    -------
    None.

    '''
    
    print("\u001b[35m>>> "+str(text)+" <<<\x1b[0m")
    
def printSuccess(text):
    '''
    Print a success message

    Parameters
    ----------
    text : str
        Text to print as success message.

    Returns
    -------
    None.

    '''
    
    print("\u001b[32m>>> "+str(text)+" <<<\x1b[0m")
    
def mat_to_vec(inp):
    '''
    Convert `inp` from a matrix to a vector

    Parameters
    ----------
    inp : ndarray
        A matrix.

    Returns
    -------
    ndarray
        A vector.

    '''
    
    return np.reshape(inp, np.size(inp))

def dot(v,w):
    x,y = v
    X,Y = w
    return x*X + y*Y

def length(v):
    x,y = v
    return math.sqrt(x*x + y*y)

def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)

def unit(v):
    x,y = v
    mag = length(v)
    return (x/mag, y/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y = v
    return (x * sc, y * sc)

def add(v,w):
    x,y = v
    X,Y = w
    return (x+X, y+Y)

def pnt2line(pnt, start, end):
    '''
    Map a point `pnt` to a line from `start` to `end`.
    '''
    
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return dist, nearest

def point_in_poly(x,y,poly):
    '''
    Determine which points (x,y) are in the polytope `poly`
    '''

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def cm2inch(*tupl):
    '''
    Convert centimeters to inches
    '''
    
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
            
def floor_decimal(a, precision=0):
    '''
    Floor function, but than with a specific precision
    '''
    
    return np.round(a - 0.5 * 10**(-precision), precision)

def writeFile(file, operation="w", content=[""]):
    '''
    Create a filehandle and store content in it.

    Parameters
    ----------
    file : str
        Filename to store the content in.
    operation : str, optional
        Type of operation to perform on the file. The default is "w".
    content : list, optional
        List of strings to store in the file. The default is [""].

    Returns
    -------
    None.

    '''
    filehandle = open(file, operation)
    filehandle.writelines(content)
    filehandle.close()


def setStateBlock(partition, **kwargs):
    '''
    Create a block of discrete regions for the given partition (can be used
    for the goal / critical regions)

    Parameters
    ----------
    partition : dict
        Dictionary of the partition of the abstraction.
    **kwargs : (multiple) lists
        Multiple arguments that give the lists of (center) coordinates to 
        include in the block in every dimension.

    Returns
    -------
    2D Numpy array
        Array with every row being the center coordinates of a region in the 
        block.

    '''
    
    nrArgs = len(kwargs)
    stateDim = len(partition['number'])
    
    if nrArgs != len(partition['number']):
        print('State dimension is',stateDim,'but only',nrArgs,
              'arguments given.')
        sys.exit()
    
    row = [None for i in range(stateDim)]
    
    center_domain = (np.array(partition['number'])-1) * 0.5 * \
                     np.array(partition['width'])
    
    for i,value in enumerate(kwargs.values()):
        
        if value == 'all':
            
            row[i] = np.linspace(
                        -center_domain[i] + np.array(partition['origin'][i]), 
                         center_domain[i] + np.array(partition['origin'][i]), 
                         partition['number'][i])
            
        else:
            
            row[i] = list(value)
            
    return np.array(list(itertools.product(*row)))

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def flatten(t):
    return [item for sublist in t for item in sublist]