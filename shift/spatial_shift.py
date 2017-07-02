# coding: utf-8

"""
@author: Philippenko

Let c a template and s a segment.
Thus, one wants to write : c(t) ~ w1 * s(w3*t + w2) + w0

This module focuses on the spatial shift characterization ie on the segmentation_construction of w0 and w1

"""

from math import sqrt

def compute_spatial_shift_parameters(template, variance, serie):
    """
    Compute distance between a template and a serie.
    
    The distance is computed with regards to the spatial shift. 
    
    The function minimize the distance : 
        (template-w0*serie+w1)^T * 1/variance * (template-w0*serie+w1)
    
    Return the distance optimization parameters w0 and w1.
    
    It is a least square method between the template and the considered sub-serie.
    
    Parameters
    ----------
    template: list-like
    variance: list-like
        the variance of the template
    serie: list-like
    """
    X=[serie[i]/variance[i] for i in range(len(serie))]
    Y=[template[i]/variance[i] for i in range(len(serie))]
    X=serie
    Y=template
    X_moy=sum(X)/len(X)
    Y_moy=sum(Y)/len(Y)
    covXY=sum([X[i]*Y[i] for i in range(len(X))])/len(X)-X_moy*Y_moy
    varX=sum([x*x for x in X])/len(X)-X_moy*X_moy
    w0=covXY/varX
    w1=Y_moy-covXY*X_moy/varX
    
#     from sklearn import linear_model
#     from numpy import array
#     reg = linear_model.LinearRegression()
#     
#     Xx=array(X).reshape(len(X),1)
#     reg.fit(Xx,Y)
#     score=reg.score(Xx, Y)
#     w1=reg.coef_
#     w0=reg.intercept_
#     print("w1,w0=",w1,w0)
#     print("R^2=", score)
#     print("w0,beta0=",w0,w1)
    distance=[int(Y[i] - w0*X[i]-w1)**2 for i in range(len(X))]
    #return sqrt(sum(distance))
    #return most_important_components(distance)
    return (w0, w1,sqrt(sum(distance)))