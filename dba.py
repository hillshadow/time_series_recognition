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
    from random import randint
    init_segments=[segments[i] for i in range(0,len(segments), int(len(segments)/10.0))]
    print("DBA Initialisation with ", len(init_segments), "segments")
    medoid=initial_medoid(init_segments)#segments[randint(0, len(segments)-1)]
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
            distance, cost, D, path, path_weight = fastdtw(s1, s2, dist=euclidean)
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
    Compute the shift for a segment according to the medoid.
    
    This is the first part of the DTW algorithm
    
    Parameters
    ----------
    medoid: list-like
    segments: list of list-like
        the bunch of segments
        
    Return
    ------
    alignment: list-like
        the shift for the segment "s" w.r.t the medoid
    path: list-like of tuplets
        the optimal path between the two series
    """
    #Step 1: compute the accumulated cost matrix of DTW
    (cost,path,weight) = DTWCumulMat(medoid,s)
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
    
    This is the second part of the DTW algorithm
    
    Parameters
    ----------
    medoid: list-like
    s: list-like
        a segment
        
    Return
    ------
    cost: matrix
        the accrued weights
    path:  list-like of tuplets
        the optimal path between the two series
    weight: matrix
        the weights
    
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
    cost: matrix
        the accrued weights
    path:  list-like of tuplets
        the optimal path between the two series
        
    Return
    ------
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

from numpy import array, zeros, argmin, inf, equal, ndim
from scipy.spatial.distance import cdist

def dtw(x, y, dist=distanceTo):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    path_weigth=[]
    for i in range(len(path[0])):
        path_weigth.append(C[path[0][i]][path[1][i]])
    return D1[-1, -1] / sum(D1.shape), C, D1, path, path_weigth

        

def fastdtw(x, y, dist=distanceTo):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    x=np.array(x)
    y=np.array(y)
    assert len(x)
    assert len(y)
    if ndim(x)==1:
        x = x.reshape(-1,1)
    if ndim(y)==1:
        y = y.reshape(-1,1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:,1:] = cdist(x,y,dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
        
    path_weigth=[]
    for i in range(len(path[0])):
        path_weigth.append(C[path[0][i]][path[1][i]])
        
    return D1[-1, -1] / sum(D1.shape), C, D1, path, path_weigth

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

from math import sqrt

def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):
        
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    
    return sqrt(LB_sum)
     