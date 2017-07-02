# coding: utf-8

import statistics as stat
from scipy.signal import argrelextrema
import numpy as np
import statistics as st

def smoothing(y, j):
    """
    Smooth y using the median value of the j-values before and after the considered point.
    The used median is based on the common “mean of middle two” method.
    
    Parameters 
    ----------
    y : list-like
        the time serie to be smoothed
    j : int-like
        the smoothing order
        
    Examples 
    ----------
    >>> # Giving an order in float have no sense, it is automatically cast in int.
    >>> smoothing([],0.5) 
    []
    >>> smoothing([1,1,10,1,1,1],0) # No smoothing.
    [1, 1, 10, 1, 1, 1]
    >>> smoothing([1,1,10,1,1,1],1)
    [1, 1.0, 5.5, 5.5, 1.0, 1.0]
    >>> smoothing([1,1,10,1,1,1],2)
    [1, 1, 1.0, 1.0, 1.0, 1]
    >>> smoothing([1,1,10,1,1,1],10)
    Traceback (most recent call last):
    AssertionError: The order should not be superior than the length of the serie.
    """
    j=int(j)
    if j==0:
        return y
    T=len(y)
    try:
        assert j<T
    except AssertionError:
        raise AssertionError("The order should not be superior than the length of the serie.")
    m=[]
    for e in y[:j]:
        m.append(e)
    for t in range(j,T-j+1):
        m.append(stat.median([y[i] for i in range(t-j,t+j)]))
    for e in y[T-j+1:]:
        m.append(e)
    return m

def differentiation(m, j):
    """
    Differentiate m with an order j.
    
    Parameters
    ----------
    m: list-like
        the serie to be differentiated
        in the segmentation process, the serie is already smoothed with an order j
    j : int-like
        the smoothing order
        
    Examples 
    ----------
    # Giving an order in float have no sense, it is automatically cast in int.
    >>> differentiation([],0.5)
    []
    >>> differentiation([1.0,2.0,3.0,1.0,2.0,3.0],1)
    [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5]
    >>> differentiation([1.0,2.0,3.0,1.0,2.0,3.0],2)
    [0, -0.5, 0.0, 0.5, -0.5, 0.0]
    >>> # Differentiating with an order equal to the length implies a null serie.
    >>> differentiation([1.0,2.0,3.0,1.0,2.0,3.0],6) 
    [0, 0, 0, 0, 0, 0.0]
    """
    j=int(j)
    m=[float(i) for i in m] # The division needs float !
    if j==0:
        return m
    T=len(m)
    d=[0 for e in m[0:j-1]]
    if j%2==0:
        minus=int(j/2)
    else:
        minus=int((j+1)/2)
    for t in range(T-j+1):
        k=t-minus
        # The division is not possible, we savesave maximum float in Python
        if m[t-k]==0:
            import sys
            d.append(sys.float_info.max)
        else:
            d.append((m[t]-m[t-k])/m[t-k])
    return d
    
def preparation(y, j):
    """
    Prepare the serie for the segmentation. 
    That is to say : a smoothing followed by a differentiation, both with an order j.
    
    Parameters
    -----------
    m: list-like
        the serie to be differentiated
        in the segmentation process, the serie is already smoothed with an order j
    j : int-like
        the smoothing order
        
    Examples
    ---------
    >>> preparation([],0)
    []
    >>> preparation([0,1,2,1,0,-1,0,1,2,1,0,-1,0],1)
    [-1.0, 0.0, 2.0, 2.0, 0.0, -2.0, -2.0, 0.0, 2.0, 2.0, 0.0, -2.0, -2.0]
    >>> preparation([0,1,2,1,0,-1,0,1,2,1,0,-1,0],8)
    [0, 0, 0, 0, 0, 0, 0, 1.7976931348623157e+308, 1.7976931348623157e+308, 1.7976931348623157e+308, 1.7976931348623157e+308, 1.7976931348623157e+308, 1.7976931348623157e+308, 1.7976931348623157e+308, 1.7976931348623157e+308]
    """
    m=smoothing(y, j)
    d=differentiation(m, j)
    return d

# The first and the last points are included in the breaking points set.
def compute_breaking_points(y, d):
    """
    Return the breaking points of a serie considering it smoothed and differentiated serie
    
    Parameters
    ----------
    y: list-like
        the serie
    d: list-like
        y smoothed and differentiated
        
    Examples
    ---------
    >>> compute_breaking_points([0,1,2,1,0,-1,0,1,2,1,0,-1,0],
    ... [-1.0, 0.0, 2.0, 2.0, 0.0, -2.0, -2.0, 0.0, 2.0, 2.0, 0.0, -2.0, -2.0])
    [0, 1, 7, 13]
    """
    d=[e-np.average(d) for e in d]
    points=[0]
    for i in range(1,len(d)):
        if (d[i]>= 0 and d[i-1]<=0):
            points.append(i)
    points.append(len(d))
    return points
            
def local_extremums(serie):
    """
    Return the local maximums/minimums of a serie.
    The returned object is two list of list of size two. 
    A local extrema is characterized by a value and an index.
    
    Warning : The extremum is greter/less or equal !
    
    Examples:  
    >>> local_extremums([0,1,1,1,0,-1,2,2,1,1,5,])
    (array([[ 0,  0],
           [ 2,  1],
           [ 5, -1],
           [ 8,  1],
           [ 9,  1]]), array([[ 1,  1],
           [ 2,  1],
           [ 3,  1],
           [ 6,  2],
           [ 7,  2],
           [10,  5]]))
    """
    max_locaux=argrelextrema(np.array(serie), np.greater_equal)[0]
    min_locaux=argrelextrema(np.array(serie), np.less_equal)[0]
    mins=np.array([[min_locaux[i],serie[min_locaux[i]]] for i in range(len(min_locaux))])
    maxs=np.array([[max_locaux[i],serie[max_locaux[i]]] for i in range(len(max_locaux))])  
    return (mins,maxs)
    
def next_minimum(pos,mins):
    """
    Return the nearest next minimum of a given position
    
    Parameters
    ----------
    pos: int-like
        the position from where the search begins
    mins: list-like
        the position of the minimums
        
    Examples:
    ---------
    >>> next_minimum(2,[0,5,10,12])
    5
    >>> next_minimum(10,[0,5,10,12])
    10
    """
    if pos in mins:
        return pos
    next=[mins[i] for i in range(1,len(mins)) if mins[i]>pos and mins[i-1]<pos]
    if not next:
        return pos
    return next[0]

def prev_minimum(pos,mins):
    """
    Return the nearest previous minimum of a given position
    
    Parameters
    ----------
    pos: int-like
        the position from where the search begins
    mins: list-like
        the position of the minimums
        
    Examples:
    ---------
    >>> prev_minimum(5,[0,5,10,12])
    5
    >>> prev_minimum(11,[0,5,10,12])
    10
    >>> prev_minimum(19,[0,5,10,12])
    12
    """
    if pos in mins:
        return pos
    if pos>mins[len(mins)-1]:
        return mins[len(mins)-1]
    previous=[mins[i] for i in range(len(mins)-1) if mins[i+1]>pos and mins[i]<pos]
    if not previous:
        return pos
    return previous[0]


def compute_segments(points,serie):
    '''
    Return all the sub-segments of a serie given the breaking points.
    There will be p-1 segments returned, where p is the length of the breaking points set.
    
    Parameters
    ----------
    points: list-like
        the breaking points set
    serie: list-like
    
    Examples:
    ---------
    >>> compute_segments([0,5,10,15],[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])
    [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]
    '''
    segments=[]
    for i in range(len(points)-1):
        segments.append([serie[j] for j in range(points[i],points[i+1])])
    return segments

# Nettoyage des points de rupture :
# Entre deux PdR soit la distance est trop grande (trop d'information), soit trop petite (information redondante)
# Dans le deuxi�me cas, on peut agir et filtrer l'information la plus pertinente.

def little_variation(M,m,p):
    """
    Return True if two numbers are near under a given percentage.
    
    Parameters:
    -----------
    M,m: float
        the number to be compared
    p: int
        the percentage
    Example
    >>> little_variation(10,10,10)
    True
    >>> little_variation(10,12,10)
    False
    >>> little_variation(10,10.5,10)
    True
    """
    return ( M*(1-float(p)/100) < m < M*(1+float(p)/100) )

def selection_relevant_points(points):
    """
    Select the most relevant points of the breaking points based on statistical properties.
    The distance between breaking points must be as regular as possible. 
    That means that we want a dispersion as little as possible.
    
    There is two case:
        - the distance between two breaking points are too big (lack of information)
        - the distance between two breaking points are too small (too much informations)
    In the second case we can filter the information
    
    Warning : This function has been design to work with a list of position which means
            that the result will be incoherent with a list of float !
    
    
    Parameters
    -----------
    points: list-like of int
        the breaking points
    
    Examples
    ---------
    >>> selection_relevant_points([0,5,11,12,15,21,27,30,31,35,36,40])
    [0, 5, 12, 15, 21, 27, 30, 35, 40]
    """
    suppression=True
    while suppression==True:
        suppression=False
        distance=[points[i+1]-points[i] for i in range(len(points)-1)]  
        moy=sum(distance)/len(distance)
        var=np.var(distance)    
        minimum=min(distance)
        idx_min=distance.index(min(distance))
        if minimum==1: 
            suppression=True
            points=remove(idx_min,points, moy, var)
        else:
            if relevant_suppression(distance,points,idx_min, moy, var):
                suppression=True
                points=remove(idx_min,points, moy, var)
    return points

def remove(idx, points, moy, var):
    """
    Remove a breaking points.
    
    Parameters
    ----------
    idx: int-like
        the checked index
    points: list of int-like
        the breaking points
    moy: float-like
        the average of the breaking points distances before the suppression
    var: float-like
        the variance of the breaking points distances before the suppression
    """
    if points_extremes(idx, points):
        return remove_other(idx, points)
    else:
        return remove_best_points(points, idx, moy, var)
        
def remove_best_points(points, idx, moy, var):
    """
    Remove the best breaking points with regards to the statistical properties of the 
    breaking points set.
    
    Parameters
    ----------
    points: list-like
        the breaking points
    idx: int-like
        the checked index, either we delete the point idx or we delete the point idx+1
    moy: float-like
        the average of the breaking points distances before the suppression
    var: float-like
        the variance of the breaking points distances before the suppression
    
    """
    new_points1=[points[i] for i in range(len(points)) if i!=idx]
    moy1=sum(new_points1)/len(new_points1)
    var1=np.var(new_points1)
    new_points2=[points[i] for i in range(len(points)) if i!=idx+1]
    moy2=sum(new_points2)/len(new_points2)
    var2=np.var(new_points2)
    if var1==var2 and moy1==moy2: # choix arbtraire
        return new_points1
    elif var1==var2: # plus petite variation de moyenne
        if abs(moy-moy1)<abs(moy-moy2):
            return new_points1
        else:
            return new_points2
    elif var1<var2:
        return new_points1
    return new_points2

def points_extremes(idx, points):
    """
    Return a boolean indicating if the consider points is the first or the last of the 
    breaking points.
        
    Warning : the last point of the serie is automatically incorporated in the 
    breaking point list, thus the last braking point is in reality the one before 
    the last.
    
    Parameters
    -----------
    idx: int-like
        the checked breaking point 
    points: list-like of int
        the breaking points
        
    Examples
    ---------
    >>> points_extremes(0,[0,1,2,3,4])
    True
    >>> points_extremes(2,[0,1,2,3,4])
    False
    >>> points_extremes(4,[0,1,2,3,4])
    False
    >>> points_extremes(3,[0,1,2,3,4])
    True
    """
    return idx==0 or idx==len(points)-2

def remove_other(idx, points):
    if idx==0:
        return remove_points(idx+1, points)
    else:
        return remove_points(idx, points)       
    
def remove_points(idx, points):
    """
    Remove a breaking point.
    
    Parameters
    ----------
    idx: int-like
        the breaking point to be removed
    points: list-like of int
        the breaking points
        
    Examples
    ---------
    >>> remove_points(2,[0,1,2,3,4])
    [0, 1, 3, 4]
    >>> remove_points(20,[0,1,2,3,4])
    [0, 1, 2, 3, 4]
    """
    return [points[i] for i in range(len(points)) if i!=idx]
    
def relevant_suppression(distance,points,idx_min, moy, var):
    """
    Return a boolean indicating if the suppression will be relevant
    """
    points1=[points[i] for i in range(len(points)) if i!=idx_min]
    points2=[points[i] for i in range(len(points)) if i!=idx_min+1]
    return relevant_point_suppression(points1,idx_min,  moy, var) or relevant_point_suppression(points2, idx_min+1, moy, var)

def relevant_point_suppression(points, idx, moy, var):
    """
    Return a boolean indicating if it is relevant to remove the point at the position `idx`.
    Based on the variance and average variation.
    """
    new_points=[points[i] for i in range(len(points)) if i!=idx]
    moyP=sum(new_points)/len(new_points)
    varP=np.var(new_points)
    return (varP<var or little_variation(var, varP, 10)) and little_variation(moy, moyP, 25)
            

def compute_average_segment(segments):
    """
    Warning : Deprecate function which calculated the average segment of a set of segments
    
    This method has been replaced by the DBA method.
    
    Parameters
    ----------
    segments: list of list-like
        the set of semgents to be averaged
    
    Example
    --------
    >>> # The averaging could be done only if there is at least four points.
    >>> compute_average_segment([[1,1,1],[1,1.1,1],[0.9,0.9,1]])
    []
    >>> compute_average_segment([[1,1,1,1.1,1,1,1,1,1,1,1,],[1,1.1,1,1,1,1,1,1,1,1],
    ... [0.9,0.9,1,1,1,1,1,1],[1,1,1,1.1,1,1,1,1,1,1,1,]])
    [0.975, 1.0, 1, 1.05, 1, 1, 1, 1]
    """
    longueur_max=max([len(s) for s in segments])
    longueur_segments=len(segments)
    s=[]
    for i in range(longueur_max):
        somme=0
        cpt=0
        for j in range(longueur_segments):
            if len(segments[j])>i:
                somme=somme+segments[j][i]
                cpt+=1
        # Need to have at least 3 time series for the averaging process !
        if (cpt>3):
            s.append(somme/cpt)
    return s

def compute_dispersion_segment(segments, moyenne):
    """
    Warning : Deprecate function which calculated the dispersion segment of a set of segments
    
    This method has been replaced by the DBA method.
    
    Parameters
    ----------
    segments: list of list-like
        the set of semgents one wants to computcomputeispersion
    
    Example
    --------
    >>> # The dispersion calcul could be done only if there is at least four points.
    >>> compute_dispersion_segment([[1,1,1],[1,1.1,1],[0.9,0.9,1]],[])
    []
    >>> compute_dispersion_segment([[1,1,1,1.1,1,1,1,1,1,1,1,],[1,1.1,1,1,1,1,1,1,1,1],
    ... [0.9,0.9,1,1,1,1,1,1],[1,1,1,1.1,1,1,1,1,1,1,1,]], [0.975, 1.0, 1, 1.05, 1, 1, 1, 1])
    [0.0018749999999999993, 0.005000000000000003, 0, 0.0025000000000000044, 0, 0, 0, 0]
    """
    longueur_moyenne=len(moyenne)
    longueur_segments=len(segments)
    s=[]
    for i in range(longueur_moyenne):
        somme=0
        cpt=0
        for j in range(longueur_segments):
            if len(segments[j])>i:
                dump=(segments[j][i]-moyenne[i])
                somme+=dump*dump
                cpt+=1
        # Need to have at least 3 time series for the averaging process !
        if (cpt>3):
            s.append(somme/cpt)
    return s

def segmentation(serie, j):    
    """
    Performs the segmentation of a serie with a order j.
    That means, it computes ;
        - the smoothed and differentiated serie
        - the breaking points
        - the relevant breaking points
        - the segments
        
    Parameters
    ----------
    serie: list-like
        the serie to be segmented
    j: int-like
        the order of the differenciation.
    """
    d=preparation(serie, j)
    points=compute_breaking_points(serie,d)
    points=selection_relevant_points(points)
    segments=compute_segments(points,serie)
    return [d,segments,points]
        
def normalization(serie):
    """
    Return the normalized serie.
    Ie each of its element is subtracted from the mean and divided by the variance.
    
    Parameter
    ----------
    serie: list-like
        the serie to be normalized
        
    Examples:
    ----------
    >>> normalization([0,1,2,3,4,5,4,3,2,1,0,0,0,1,1])
    [0.0, 0.6038073644245598, 1.2076147288491197, 1.8114220932736798, 2.4152294576982394, 3.0190368221227994, 2.4152294576982394, 1.8114220932736798, 1.2076147288491197, 0.6038073644245598, 0.0, 0.0, 0.0, 0.6038073644245598, 0.6038073644245598]
    """
    sigma=st.stdev(serie)
    m=min(serie)
    serie=[(s-m)/sigma for s in serie]
    return serie

def union(sgmtt1,sgmtt2):
    """
    Join two segmentation.
    
    Parameters
    -----------
    sgmtt1, sgmtt2: segmentation-like
        the segmentation to be joined
        
    Return
    -------
    The parameter of the segmentation needed by the constructor of the Segmentation class
    """
    # TODO: recompute_segment, prunning_breaking_points !
    if sgmtt1=='':
        return sgmtt2.get_all()
    elif sgmtt2=='':
        return sgmtt1.get_all()
    print(sgmtt1)
    print(sgmtt2)
    absc1=sgmtt1.get_absc()
    absc2=sgmtt2.get_absc()
    absc=absc1+absc2
    serie=sgmtt1.get_serie()+sgmtt2.get_serie()
    order=0
    activity=""
    sd_serie=sgmtt1.get_sd_serie()+sgmtt2.get_sd_serie()
    bp1=sgmtt1.get_breaking_points()
    bp2=sgmtt2.get_breaking_points()
    breaking_points=bp1+[bp+bp1[len(bp1)-1] for bp in bp2]
    segments=[]
    s1=sgmtt1.get_segments()
    s2=sgmtt2.get_segments()
    for s in s1:
        segments.append(s)
    for s in s2:
        segments.append(s)
    average_segment=[]
    dispersion_segment=[]
    return (absc,serie,order,activity,sd_serie,breaking_points,segments,average_segment,dispersion_segment)

# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()
