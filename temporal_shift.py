# coding: utf-8

"""
@author: Philippenko

Let c a template and s a segment.
Thus, one wants to write : c(t) ~ w1 * s(w3*t + w2) + w0

This module focuses on the temporal shift characterization ie on the computation of w0 and w1

"""

import matplotlib.pyplot as plt
from numpy import array, inf
from dba import DTWCumulMat, optimal_path, fastdtw
from sklearn import linear_model
from save import save_double_list
import sys

def rising_too_strong(x,y):
    # Si sur les trois premiers on a grimp� de plus de trois.
#     nb_points_a_3=len([x[i] for i in range(len(x)) if x[i]<3])
#     return y[nb_points_a_3]>4
    return True

def tray_too_long(x,y):
#     nb_points_a_3=len([y[i] for i in range(len(y)) if y[i]<3])
#     return x[nb_points_a_3]>4
    return True

def compute_temporel_shift_parameters(template, serie, plot=False):
    """
    Computes the temporal shift parameters between a time serie and a template, that is to say w2 and w3.
    In that manner:
        1) Compute the cost matrix and search for the best path
        2) Try to found the best linear regression of the path by looking at sub-intervals
        3) w3 is the director coefficients of this straight line
        
    .. warning:: w2 is the number of shift points ! As a result, it an int !
    .. todo:: The selection of the best sub-interval regression is not optimized and very long !
    .. todo:: Dynamic choose of sub-intervals depending of the trays, risings ... and of the the points concentration
    """
    template=[template[i] for i in range(0, len(template),1)]
    serie=[serie[i] for i in range(0, len(serie),1)]
    reg = linear_model.LinearRegression()
    reg_min = linear_model.LinearRegression()
    reg_select = linear_model.LinearRegression()
        
#     (cost,path,weight)=DTWCumulMat(medoid=template,s=serie)
#     (opt_path,weight_opt_path)=optimal_path(len(template), len(serie),path,weight)

    (d,opt_path,cost_path)=fastdtw(template, serie)
    opt_path=array(opt_path)
        
    #Data preparation
    X=array(opt_path[:,0]).reshape(len(opt_path[:,0]),1)
    y=opt_path[:,1]
    
    level=len(X)/10
    
    # Suppresion des pics
    (new_y,new_X)=remove_front(y,X,level)
    
    # Regression initialization
    reg.fit(new_X,new_y)
    R_max=reg.score(new_X,new_y)
    reg_min=reg
    w3=reg.coef_
    if w3==0:
        w2=-sys.maxint
    else:
        w2=-reg.intercept_/w3
        
    plt.subplot(131)
    plt.scatter(X,y,color="b")
    plt.scatter(new_X,new_y, color="r")
    plt.plot(new_X,reg.predict(new_X), color="k")
    
    (X_min,y_min)=(new_X,new_y)
    
     #Choose of the sub set
    n=len(new_X)
    # TODO : on peux en enlever autant que l'on veut � condition qu'il y ait suffisament de point % au nombre de points r�el
    minus=4*n/10
    
    # Searching of the best sub-intervalle regression.
    for i in reversed(range(1,minus,2)):
        for j in reversed(range(1,minus,2)):
            Xprime=new_X[minus-i:n-minus+j]
            yPrime=new_y[minus-i:n-minus+j]
            reg = linear_model.LinearRegression()
            reg.fit(Xprime,yPrime)
            score=reg.score(Xprime,yPrime)
            if R_max < score:
                reg_min=reg
                R_max=score
                w3=reg.coef_
                if w3==0:
                    w2=-sys.maxint
                else:
                    w2=-reg.intercept_/w3
                X_min=Xprime
                y_min=yPrime
                
    print("w2,w3,R2=",-w2,w3,R_max)
                
    plt.subplot(132)
    plt.scatter(X,y,color="b")
    plt.scatter(new_X,new_y, color="r")
    plt.scatter(X_min,y_min, color="g")
    plt.plot(X_min,reg_min.predict(X_min), color="k")
    
    
    # Plot the series
    plt.subplot(133)
    plt.plot(template, label="Template")
    plt.plot(serie, label="Serie")
    plt.legend()
    plt.show()
    
    if w2!=sys.maxint:
        print("Front removing attempt")
        (X_select,y_select)=remove_front(X,y,level)
    else: 
        (X_select,y_select)=(X_min, y_min)
        
    try:
        reg_select.fit(X_select,y_select)
        score=reg_select.score(X_select,y_select)
    except(ValueError):
        print(X_select,y_select)
        (X_select,y_select)=(X_min, y_min)
        reg_select=reg
        score=0
        
    if score>R_max:
        R_max=score
        w3=reg.coef_
        if w3==0:
            w2=-sys.maxint
        else:
            w2=-reg.intercept_/w3
            
    #if plot:
        #print("w2,w3,R2=",w2,w3,R_max)
        #plot_temporal_shift(template, serie, X, y, X_min, y_min, X_select, y_select, reg, reg_select)
    # w2 is the number of points shift ! Does not have any sense to return a float !
    
    if w2!=-array([ inf]):
        return (int(w2),w3[0],R_max,rising_too_strong(X, y), tray_too_long(X, y))
    return (sys.float_info.max,w3[0],R_max,rising_too_strong(X, y), tray_too_long(X, y))

def remove_tray_and_peack(x,y):
    """
    Examples
    --------
    >>> remove_tray_and_peack([0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,8,8],[1,1,1,1,1,1,1,1,2,3,4,5,6,7,8,9,10,10])
    ([], [])
    """
    save_double_list([int(x[i]) for i in range(len(x))],y,"exemple_qui_foire.csv")
    level=len(x)/10
    (X_select,y_select)=remove_front(x,y,level)
    (y_select, X_select)=remove_front(y_select,X_select,level)
    return (X_select,y_select)

def remove_front(x,y,level):
    # Remove peack
    cluster=[[x[i] for i in range(len(x)) if y[i]==j] for j in range(len(y))]
    new_x=x
    new_y=y
    for c in cluster:
        if len(c)>level:
            new_x=[x[i] for i in range(len(x)) if x[i] not in c]
            new_y=[y[i] for i in range(len(y)) if x[i] not in c]
            x=new_x
            y=new_y
    return (new_x,new_y)
            
    

def plot_temporal_shift(template, serie, X, y, X_min, y_min, X_select, y_select, reg, reg_select):
    """
    Plot the most useful informations of the temporal shift
    
    Parameters
    ----------
    template, series: list-like
    cost: list-like with len(template) lines and len(series) columns. 
        the cost matrix computed via the DBA algorithm.
    X: numpy.ndarray-like
        the abscissa of the optimal path.
    y: list-like
        the ordinates of the optimal path.
    X_min: numpy.ndarray-like
        the sub-abscissa of the optimal path where the regression is performed.
    y_min: list-like
        the sub-ordinates of the optimal path where the regression is performed.
    reg: linear_model.LinearRegression()
        the regression model. 
    """
    
#     # Plot the cost matrix
#     plt.matshow(cost)
#     plt.colorbar()
#     plt.show()
        
    # Plot the shift computation
    plt.subplot(131)
    plt.scatter(X,y,color="b")
    plt.scatter(X_min,y_min, color="r")
    plt.plot(X_min,reg.predict(X_min), color="k")
    
    # Plot the shift computation
    plt.subplot(132)
    plt.scatter(X,y,color="b")
    plt.scatter(X_min,y_min, color="r")
    plt.scatter(X_select,y_select, color="g")
    plt.plot(X_select,reg_select.predict(X_select), color="k")

    # Plot the series
    plt.subplot(133)
    plt.plot(template, label="Template")
    plt.plot(serie, label="Serie")
    plt.legend()
    plt.show()
