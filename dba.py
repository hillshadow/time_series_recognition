#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Philippenko

This is an implementation of the DBA algorithm presented in:
A global averaging method for dynamic time warping, with applications to clustering, Petitjean et. al.
(http://dpt-info.u-strasbg.fr/~fpetitjean/Research/Petitjean2011-PR.pdf)
"""

from __future__ import absolute_import, division
import numbers
import numpy as np
from collections import defaultdict

try:
    range = xrange
except NameError:
    pass

from scipy.spatial.distance import euclidean
from statistics import mean, variance



def DBA(segments, iter):
    """
    Computes the medoid segment of a bunch of segments based on the Petitjean method.
    
    Parameters
    ----------
    segments: list of list-like
        the bunch of segments
    iter: int-like
        the number of iterations
    """
    print("DBA Initialisation ...")
    medoid=initial_medoid(segments)
    print("Iteration :", 0)
    (medoid,w)=DBA_update(medoid,segments)
    for i in range(1,iter):
        print("Iteration :", i)
        (medoid,w)=DBA_update(medoid,segments)
    return (medoid,w)
 
def initial_medoid(segments):
    """
    Initializes the medoid by taking the segments with the smallest sum of squares
    
    Parameter
    ---------
    segments: list of list-like
        the bunch of segments
    """
    minSSQ=float("Inf") #Minimum sum of square
    for s1 in segments:
        # Computing sum of square for s1
        tmpSSQ=0
        for s2 in segments:
            distance, path = fastdtw(s1, s2, dist=euclidean)
            tmpSSQ+=distance**2
        if tmpSSQ<minSSQ:
            medoid=s1
            minSSQ=tmpSSQ
    return medoid
 
def DBA_update(medoid,segments):
    """
    Updates the medoid.
    
    Parameters
    ----------
    medoid: list-like
    segments: list of list-like
        the bunch of segments
    """
    n=len(medoid)
    nb_segments=len(segments)
    #Step 1 : compute the multiple alignment for medoid
    alignment=[[] for i in range(n)]
    path=[[] for i in range(len(segments))]
    for i in range(nb_segments):
        s=segments[i]
        (alignment_for_s, path_for_s)=DTW_multiple_alignment(medoid,s)
        path[i]=path_for_s
        for i in range(n):
            alignment[i].extend(alignment_for_s[i])
#     the_path=[get_the_path(path[i], n, len(segments[i])) for i in range(nb_segments)]
#     w=compute_weight(the_path, segments, medoid, w)
    #Step 2 : compute the multiple alignment for the alignment
    return ([mean(a) for a in alignment],[variance(a) for a in alignment])
 
def DTW_multiple_alignment(medoid,s):
    """
    Compute the shif for each segment according to the medoid.
    
    Parameters
    ----------
    medoid: list-like
    segments: list of list-like
        the bunch of segments
    """
    #Step 1: compute the accumulated cost matrix of DTW
    d, cost, path = fastdtw(medoid,s)
    #Step 2 : store the elements associated with the medoid
    n = len(medoid)
    alignment=[[] for i in range(n)]
    i=len(medoid)-1
    j=len(s)-1
    while(True):
        alignment[i].append(s[j])
        if path[i][j]==0:
            i-=1
            j-=1
        elif path[i][j]==1:
            j-=1
        elif path[i][j]==2:
            i-=1
        else:
            break          
    return (alignment,path)
 
def DTWCumulMat(medoid, s):
    """
    Computes the cost/path matrix of a medoid/segment doublet.
    
    Parameters
    ----------
    medoid: list-like
    s: list-like
        a segment
    """
    cost = [[0] * len(s) for _ in range(len(medoid))]
    weight= [[0] * len(s) for _ in range(len(medoid))]
    path = [[0] * len(s) for _ in range(len(medoid))]
    cost[0][0]=distanceTo(medoid[0],s[0])
    weight[0][0]=distanceTo(medoid[0],s[0])
    path[0][0] = -1;
    n=len(medoid)
    m=len(s)
    for i in range(1,n):
        cost[i][0]=cost[i-1][0]+distanceTo(medoid[i],s[0])
        weight[i][0]=distanceTo(medoid[i],s[0])
        path[i][0]=2
    for j in range(1,m):
        cost[0][j]=cost[0][j-1]+distanceTo(s[j],medoid[0])
        weight[0][j]=distanceTo(s[j],medoid[0])
        path[0][j]=1
    for i in range(1,n):
        for j in range(1,m):
            tab=[cost[i-1][j-1],cost[i][j-1],cost[i-1][j]]
            indiceRes=tab.index(min(tab))
            path[i][j]=indiceRes
            if indiceRes==0:
                res=cost[i-1][j-1]
            elif indiceRes==1:
                res=cost[i][j-1]
            else:
                res=cost[i-1][j]
            cost[i][j]=res+distanceTo(medoid[i],s[j])
            weight[i][j]=distanceTo(medoid[i],s[j])
    return (cost,path,weight)

def optimal_path(n,m, path, cost):
    """
    Parameters
    ----------
    n: int-like
        the length of the medoid
    m: int-like
        the length of the segment
    """
    the_path=[]
    path_cost=[]
    i=n-1
    j=m-1
    while(True):
        # First : the coordinate of the segment (which is on the abscissa)
        # Secondly : the coordinate of the medoid (which is on the vertical axes)
        the_path.append([j,i])
        path_cost.append(cost[i][j])
        if path[i][j]==0:
            i-=1
            j-=1
        elif path[i][j]==1:
            j-=1
        elif path[i][j]==2:
            i-=1
        else:
            return (list(reversed(the_path)),list(reversed(path_cost)))
     
def distanceTo(a,b):
    """
    The considered distance between two scalars
    
    Parameters
    ----------
    a,b: float-like
    
    Examples
    ---------
    >>> distanceTo(0,0)
    0
    >>> distanceTo(4,2)
    4
    """
    return (a-b)**2   
        




def fastdtw(x, y, radius=1, dist=None):
    ''' return the approximate distance between 2 time series with O(N)
        time and memory complexity
        Parameters
        ----------
        x : array_like
            input array 1
        y : array_like
            input array 2
        radius : int
            size of neighborhood when expanding the path. A higher value will
            increase the accuracy of the calculation but also increase time
            and memory consumption. A radius equal to the size of x and y will
            yield an exact dynamic time warping calculation.
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.
        Returns
        -------
        distance : float
            the approximate distance between the 2 time series
        path : list
            list of indexes for the inputs x and y
        Examples
        --------
        >>> import numpy as np
        >>> import fastdtw
        >>> x = np.array([1, 2, 3, 4, 5], dtype='float')
        >>> y = np.array([2, 3, 4], dtype='float')
        >>> fastdtw.fastdtw(x, y)
        (2.0, [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)])
    '''
    x, y, dist = __prep_inputs(x, y, dist)
    return __fastdtw(x, y, radius, dist)


def __difference(a, b):
    return abs(a - b)


def __norm(p):
    return lambda a, b: np.linalg.norm(a - b, p)


def __fastdtw(x, y, radius, dist):
    min_time_size = radius + 2

    if len(x) < min_time_size or len(y) < min_time_size:
        return dtw(x, y, dist=dist)

    x_shrinked = __reduce_by_half(x)
    y_shrinked = __reduce_by_half(y)
    distance, path, cost = \
        __fastdtw(x_shrinked, y_shrinked, radius=radius, dist=dist)
    window = __expand_window(path, len(x), len(y), radius)
    return __dtw(x, y, window, dist=dist)


def __prep_inputs(x, y, dist):
    x = np.asanyarray(x, dtype='float')
    y = np.asanyarray(y, dtype='float')

    if x.ndim == y.ndim > 1 and x.shape[1] != y.shape[1]:
        raise ValueError('second dimension of x and y must be the same')
    if isinstance(dist, numbers.Number) and dist <= 0:
        raise ValueError('dist cannot be a negative integer')

    if dist is None:
        if x.ndim == 1:
            dist = __difference
        else: 
            dist = __norm(p=1)
    elif isinstance(dist, numbers.Number):
        dist = __norm(p=dist)

    return x, y, dist


def dtw(x, y, dist=None):
    ''' return the distance between 2 time series without approximation
        Parameters
        ----------
        x : array_like
            input array 1
        y : array_like
            input array 2
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.
        Returns
        -------
        distance : float
            the approximate distance between the 2 time series
        path : list
            list of indexes for the inputs x and y
        Examples
        --------
        >>> import numpy as np
        >>> import fastdtw
        >>> x = np.array([1, 2, 3, 4, 5], dtype='float')
        >>> y = np.array([2, 3, 4], dtype='float')
        >>> fastdtw.dtw(x, y)
        (2.0, [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)])
    '''
    x, y, dist = __prep_inputs(x, y, dist)
    return __dtw(x, y, None, dist)


def __dtw(x, y, window, dist):
    len_x, len_y = len(x), len(y)
    if window is None:
        window = [(i, j) for i in range(len_x) for j in range(len_y)]
    window = ((i + 1, j + 1) for i, j in window)
    D = defaultdict(lambda: (float('inf'),))
    D[0, 0] = (0, 0, 0)
    for i, j in window:
        dt = dist(x[i-1], y[j-1])
        D[i, j] = min((D[i-1, j][0]+dt, i-1, j), (D[i, j-1][0]+dt, i, j-1),
                      (D[i-1, j-1][0]+dt, i-1, j-1), key=lambda a: a[0])
    path = []
    cost = []
    i, j = len_x, len_y
    while not (i == j == 0):
        path.append((i-1, j-1))
        cost.append(D[i-1, j-1][0])
        i, j = D[i, j][1], D[i, j][2]
    path.reverse()
    cost.reverse()
    return (D[len_x, len_y][0], path, cost)


def __reduce_by_half(x):
    return [(x[i] + x[1+i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]


def __expand_window(path, len_x, len_y, radius):
    path_ = set(path)
    for i, j in path:
        for a, b in ((i + a, j + b)
                     for a in range(-radius, radius+1)
                     for b in range(-radius, radius+1)):
            path_.add((a, b))

    window_ = set()
    for i, j in path_:
        for a, b in ((i * 2, j * 2), (i * 2, j * 2 + 1),
                     (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1)):
            window_.add((a, b))

    window = []
    start_j = 0
    for i in range(0, len_x):
        new_start_j = None
        for j in range(start_j, len_y):
            if (i, j) in window_:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j

    return window
        

        

   
    
    
    