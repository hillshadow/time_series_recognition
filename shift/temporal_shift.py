# coding: utf-8

"""
@author: Philippenko

Let c a template and s a segment.
Thus, one wants to write : c(t) ~ w1 * s(w3*t + w2) + w0

This module focuses on the temporal shift characterization ie on the segmentation_construction of w0 and w1

"""

import matplotlib.pyplot as plt
from numpy import array, inf, piecewise, linspace   
from dba import DTWCumulMat, optimal_path, fastdtw, dtw
from sklearn import linear_model
from scipy import optimize
from storage.save import save_double_list
import sys
from segmentation.segmentation_construction import local_extremums, little_variation
from statistics import median   

import logging


def rising_too_strong(x,y):
    # Si sur les trois premiers on a grimp� de plus de trois.
#     nb_points_a_3=len([x[i] for i in range(len(x)) if x[i]<3])
#     return y[nb_points_a_3]>4
    return True

def tray_too_long(x,y):
#     nb_points_a_3=len([y[i] for i in range(len(y)) if y[i]<3])
#     return x[nb_points_a_3]>4
    return True

def sequence_to_matrix(len_x, len_y, sequence, coordinates):
    
    my_matrix=[[max(sequence)*(1+5.0/100)] * (len_x+1) for _ in range(len_y+1)]
    for k in range(len(sequence)):
        my_matrix[coordinates[k][1]][coordinates[k][0]]=sequence[k]
        
    return my_matrix
    
def prepare_coordinates_path(opt_path):
    opt_path=array(opt_path)
    coord_x=array(opt_path[:,0]).reshape(len(opt_path[:,0]),1)
    coord_y=array(opt_path[:,1])
    return (coord_x, coord_y)

def regression(reg,new_X, new_y, y_level_deb, y_level_fin, len_template):

    
    new_y_prime=[y for y in new_y if y_level_fin >= y >=y_level_deb] 
    
    if len(new_y_prime)==0:
        reg.fit(new_X,new_y)
    else:
        new_X=[new_X[i] for i in range(len(new_X)) if new_y[i] in new_y_prime]
        new_y=new_y_prime 
        
        reg.fit(new_X,new_y)
    
        
    R_max=reg.score(new_X,new_y)
    reg_min=reg
    w3=reg.coef_
    if w3==0:
        w2=len_template
    else:
        w2=-reg.intercept_/w3
    # We consider that if w2 is a bit less than 0, that means that it is equal to 0 ie there is no delay.
#     if w2 < 0 and w2 > -5:
#         w2=0
    return (w2,w3, reg.score, new_X, new_y)

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
    
    len_template=len(template)
    
    template=[template[i] for i in range(0, len(template),2)]
    serie=[serie[i] for i in range(0, len(serie),2)]
    reg = linear_model.LinearRegression()
    reg_min = linear_model.LinearRegression()
    reg_select = linear_model.LinearRegression()
    
#     (template_max, template_min)=local_extremums(template)
#     (serie_max, serie_min)=local_extremums(serie)
#         
#     (cost,path,weight)=DTWCumulMat(medoid=template,s=serie)
#     (opt_path,weight_opt_path)=optimal_path(len(template), len(serie),path,weight)
#     dist=cost[-1][-1]

#     (d,coordinates,weight_cost_path)=dtw(template, serie)
    dist, cost, acc, dump_path, weight_opt_path = dtw(template, serie)
     
    opt_path=[]
    for i in range(len(dump_path[0])):
        opt_path.append([dump_path[1][i], dump_path[0][i]])
    
    #########################################################
    # Goal : cocomputehe best linear regression of the path #
    #########################################################    
    
    (X,y)=prepare_coordinates_path(opt_path)
    
    
    ##                                  ##
    ## Intialisation of the regressions ##
    ##                                  ##
    
    # We remove all the peack and all the tray !
    level=len(X)/10 # The peack/tray must have at least level% of the total points
    (new_X,new_y, y_deb, y_fin)=remove_front(X,y,level)
    (new_y,new_X, x_deb, x_fin)=remove_front(new_y,new_X,level)    
    
#     print("x_deb, x_fin=", x_deb,x_fin)
#     print("y_deb, y_fin=", y_deb,y_fin)
       
#     if y_deb=="NaN":
#         if y_fin=="NaN":
#             (w2,w3, R_max, new_X, new_y)=regression(reg,new_X, new_y, 0, len(y))
#         else:
#             if y_fin < median(y):
#                 (w2,w3,R_max, new_X, new_y)=regression(reg,new_X, new_y, y_fin, len(y))
#             else:
#                 (w2,w3,R_max, new_X, new_y)=regression(reg,new_X, new_y, 0, y_fin)
#     elif y_fin=="NaN":
#         if y_deb>median(y):
#             (w2,w3,R_max, new_X, new_y)=regression(reg,new_X, new_y, 0, y_deb)
#         else:
#             (w2,w3,R_max, new_X, new_y)=regression(reg,new_X, new_y, y_deb, len(y))
#     else:
    (w2,w3,R_max, new_X, new_y)=regression(reg,new_X, new_y, y_deb, y_fin, len_template)   
             
            
    (X_min,y_min)=(new_X,new_y)
    
    ##                                                    ##
    ## Search of the best sub-interval for the regression ##    
    ##                                                    ##
     
     #Choose of the sub set
    n=len(new_X)
    # TODO : on peux en enlever autant que l'on veut � condition qu'il y ait suffisament de point % au nombre de points r�el
    minus=4*n/10
    
    reg_min=reg
     
    # Searching of the best sub-intervalle regression.
    for i in reversed(range(1,minus,3)):
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
    X_min=new_X
    y_min=new_y
    R_max=1
        
    
    if plot:
        print("w2,w3,dist=",w2,w3,dist)
        plot_temporal_shift(template, serie, cost, opt_path, weight_opt_path, new_X, new_y, X_min, y_min, reg, reg_min, y_deb, y_fin)
    # w2 is the number of points shift ! Does not have any sense to return a float !
   
#     if w2 < -7:
#         w2 = w2 % (len(template)*w3)
#          
#     if w2 > len(serie)-len(template):
#         w2 = w2 % (len(serie) - len(template))
    from fastdtw import fastdtw as f
    dist=f(serie, template)[0]
    try:
        return (int(w2),w3[0],dist)
    except(ValueError):
        return (len_template,w3[0],dist)

def remove_tray_and_peack(x,y):
    """
    Examples
    --------
    >>> remove_tray_and_peack([0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,8,8],[1,1,1,1,1,1,1,1,2,3,4,5,6,7,8,9,10,10])
    ([], [])
    """
    save_double_list([int(x[i]) for i in range(len(x))],y,"exemple_qui_foire.csv")
    level=len(x)/10
    (X_select,y_select,y_deb, y_fin)=remove_front(x,y,level)
    (y_select, X_select, x_deb, x_fin)=remove_front(y_select,X_select,level)
    return (X_select,y_select, y_deb,y_fin)

def remove_front(x,y,level):
    # Remove tray
    
    cluster=[[x[i] for i in range(len(x)) if y[i]==j] for j in range(len(y))]
    tray=[]
    new_x=x
    new_y=y
    for k in range(len(cluster)):
        c=cluster[k]
        if len(c)>level:
            tray.append(k)
            new_x=[x[i] for i in range(len(x)) if x[i] not in c]
            new_y=[y[i] for i in range(len(y)) if x[i] not in c]
            x=new_x
            y=new_y
    if tray==[]:
        return (new_x,new_y,0, len(y))
    # Si il n'y a qu'un plateau : est-ce un plateau dans la partie supérieur ou dans la partie inférieur.
    elif len(tray)==1:
        if tray[0]>median(y):
            return (new_x,new_y, 0, tray[0])
        return (new_x, new_y, tray[0], len(y))
    else:
        inf=tray[0]
        sup=tray[len(tray)-1]
        # Si les plateau sont très proches : 
        if little_variation(inf/float(len(y)), sup/float(len(y)), 30):
            if sup>median(y):
                return (new_x,new_y, 0, sup)
            return (new_x,new_y, inf, len(y))
    return (new_x,new_y,tray[0],tray[len(tray)-1])
            

def plot_temporal_shift(template, serie, cost, opt_path, weight_opt_path, new_X, new_y, X_min, y_min, reg, reg_min, y_deb, y_fin):
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
    plt.figure()
    # Pour le cout
    ax1= plt.subplot2grid((3,4), (1,0), rowspan=2, colspan=2)
    # Pour le poid
    ax2= plt.subplot2grid((3,4), (1,2), rowspan=2, colspan=2)
    # Pour les courbes
    ax3= plt.subplot2grid((3,4), (0,0), colspan=4)
    
    X, y = prepare_coordinates_path(opt_path)
    
    im1 = ax1.imshow(cost)
    plt.colorbar(im1, ax=ax1)
    ax1.scatter(X,y, color="k")
    ax1.set_xlabel("Series")
    ax1.set_ylabel("Template")
    ax1.set_title("Cost")
    
    mat_weight_opt_path=sequence_to_matrix(len(serie), len(template), weight_opt_path, opt_path)
    
    im2 = ax2.imshow(mat_weight_opt_path)
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlabel("Series")
    ax2.set_ylabel("Template")
    ax2.set_title("Weight of the optimal path")
    
    ax3.plot(serie, label="Series")
    ax3.plot(template, label="Template")
    plt.ylabel("Normalized acceleration")
    plt.xlabel("Times")
    plt.title("Times series comparison")
    ax3.legend(loc="best")
    
    plt.figure()
     
    plt.subplot(131)
    plt.scatter(X,y,color="b")
    plt.scatter(new_X,new_y, color="r")
    plt.plot(new_X,reg.predict(new_X), color="k")
    plt.title("Initial regression")
    plt.xlabel("Series")
    plt.ylabel("Template")
 
    plt.subplot(132)
    plt.scatter(X,y,color="b")
    plt.scatter(new_X,new_y, color="r")
    plt.scatter(X_min,y_min, color="g")
    plt.plot(X_min,reg_min.predict(X_min), color="k",linewidth=2.5)
    plt.plot(X,[y_deb for x in X], color="k",linewidth=2.5)
    plt.plot(X,[y_fin for x in X], color="k",linewidth=2.5)
    plt.title("Regression by pieces")
    plt.xlabel("Series")
    plt.ylabel("Template")
     
     
    # Plot the series
    plt.subplot(133)
    plt.plot(template, label="Template")
    plt.plot(serie, label="Serie")
    plt.legend(loc="best")
    plt.xlabel("Times")
    plt.ylabel("Normalized acceleration")
    plt.title("Times series comparison")
    plt.show()

