'''
:author: Philippenko
:date: Juil. 2017

This module is devoted to the plotting of the shift process of a time series.
'''

import matplotlib.pyplot as plt 
from sklearn import linear_model

def plot_shifting(initial_segment, normalized_segment, shifted_segment, template):
    """Plots the shift process of a time series
    
    Parameters
    ----------
    initial_segment: list
    normalized_segment: list
    shifted_segment: list
    template: list
    """
    plt.subplot(131)
    plt.plot(initial_segment, label="Series")
    plt.plot(template, label="Template")
    plt.legend(loc="best")
    plt.subplot(132)
    plt.plot(normalized_segment, label="Normalized Series")
    plt.plot(template, label="Template")
    plt.legend(loc="best")
    plt.subplot(133)
    plt.plot(shifted_segment, label="Shifted Series")
    plt.plot(template, label="Template")
    plt.legend(loc="best")
    plt.show()

def plot_temporal_shift(template, serie, cost, mat_weight_opt_path, 
                        X, y, new_X, new_y, X_min, y_min, y_deb, y_fin):
    """Plots the most useful informations of the temporal shift
    
    Parameters
    ----------
    template, series: list
    cost: list-like with len(template) lines and len(series) columns. 
        the cost matrix computed via the DBA algorithm.
    X: numpy.ndarray
        the abscissa of the optimal path.
    y: list-like
        the ordinates of the optimal path.
    X_min: numpy.ndarray
        the sub-abscissa of the optimal path where the regression is performed.
    y_min: list
        the sub-ordinates of the optimal path where the regression is performed.
    """
    plt.figure()
    # Pour le cout
    ax1= plt.subplot2grid((3,4), (1,0), rowspan=2, colspan=2)
    # Pour le poid
    ax2= plt.subplot2grid((3,4), (1,2), rowspan=2, colspan=2)
    # Pour les courbes
    ax3= plt.subplot2grid((3,4), (0,0), colspan=4)
    
    im1 = ax1.imshow(cost)
    plt.colorbar(im1, ax=ax1)
    ax1.scatter(X,y, color="k")
    ax1.set_xlabel("Series")
    ax1.set_ylabel("Template")
    ax1.set_title("Cost")
    
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
    reg = linear_model.LinearRegression()
    reg.fit(X,y)
    plt.scatter(X,y,color="b")
    plt.scatter(new_X,new_y, color="r")
    plt.plot(new_X,reg.predict(new_X), color="k")
    plt.title("Initial regression")
    plt.xlabel("Series")
    plt.ylabel("Template")
 
    plt.subplot(132)
    reg_min = linear_model.LinearRegression()
    reg_min.fit(new_X,new_y)
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
