# coding: utf-8
'''
Created on 6 juil. 2017

@author: Philippenko
'''      

import numpy as np
import matplotlib.pyplot as plt
from storage import load as ld
from comparison import build_distance_vector
from mpl_toolkits.mplot3d import Axes3D

def plot_dynamic_data_components_i_j(X,y,k,j):
    """
    Plots the data repartition w.r.t the components i and j
    
    Parameters
    -----------
    X: list of list-like
        the data
    y: list-like
        the label
    i,j : int-like
        the plotted components
    x_label, y_label: string-like
        the axis title
    save: boolean-like
        True if the figure must be saved
    """
    import pandas as pd
    import mpld3
    from mpld3 import plugins
    
    # Define some CSS to control our custom labels
    css = """
    table
    {
      border-collapse: collapse;
    }
    th
    {
      color: #ffffff;
      background-color: #000000;
    }
    td
    {
      background-color: #cccccc;
    }
    table, th, td
    {
      font-family:Arial, Helvetica, sans-serif;
      border: 1px solid black;
      text-align: right;
    }
    """
    
    fig, ax = plt.subplots()
    ax.grid(True, alpha=0.3)
    
    
    df = pd.DataFrame(X)
    
    labels = []
    for i in range(len(X)):
        label = df.ix[[i], :].T
        label.columns = ['Row {0}'.format(i)]
        # .to_html() is unicode; so make leading 'u' go away with str()
        labels.append(str(label.to_html()))
    
    points = ax.scatter(df.iloc[:,k], df.iloc[:,j], c=y, alpha=.6, s=[100 for e in range(len(y))])
                     
    x_label=str(k)
    y_label=str(j)
    ax.set_xlabel("Component :"+x_label)
    ax.set_ylabel("Component :"+y_label)
    ax.set_title('HTML tooltips', size=20)
    
    tooltip = plugins.PointHTMLTooltip(points, labels,
                                       voffset=10, hoffset=10, css=css)
    plugins.connect(fig, tooltip)
    
    mpld3.show()
    
def plot_all_components(X,y, save=True):
    """
    Plots the whole data repartition
    
    Parameters
    -----------
    X: list of list-like
        the data
    y: list-like
        the label
    save: boolean-like
        True if the figure must be saved
    """
    n=len(X[0,:])
    for i in range(n):
        plot_data_component_i(X, y, i, save)
  
def plot_data_component_i(X,y,i,save=False):
    """
    Plots the data repartition w.r.t the components i
    
    Parameters
    -----------
    X: list of list-like
        the data
    y: list-like
        the label
    i : int-like
        the plotted component
    save: boolean-like
        True if the figure must be saved
    """
    n=len(X[0,:])
    for j in range(n):
        if i!=j:
            plot_data_components_i_j(X, y, i, j, save)
    
def plot_data_components_i_j(X, y, i, j, save=False, hightlight=None):
    print("Size of the True Classe :", len([e for e in y if e==1]))
    print("Size of the False Classe :", len([e for e in y if e==0]))
    X=np.array([[row[i],row[j]] for row in X])
    plt.figure()
    plt.xlabel(str(i))
    plt.ylabel(str(j))
    if len(set(y))==2:
        X_true=np.array([X[k] for k in range(len(X)) if y[k]==0])
        X_false=np.array([X[k] for k in range(len(X)) if y[k]==1])
        plt.scatter(X_true[:, 0], X_true[:, 1], color="b", alpha=0.6, label="Walking")#,cmap=plt.cm.bone)s
        plt.scatter(X_false[:, 0], X_false[:, 1], color="r", alpha=0.6, label="Other")
        plt.legend()
    else:
        plt.scatter(X[:,0], X[:,1], c=y,alpha=0.6)
    if hightlight!=None:
            plt.scatter(X[hightlight,0],X[hightlight,1], color="k", marker="*")
    if save:
        plt.savefig("data\\DistanceVectorComponents\\"+str(i)+"_"+str(j))
        plt.close()
    else:
        if hightlight==None:
            plt.show()
     
def plot_data_components_i_j_k(X, y, i, j, k , x_label, y_label, save=False):
    print("Size of the True Classe :", len([e for e in y if e==1]))
    print("Size of the False Classe :", len([e for e in y if e==0]))
    X=np.array([[row[i],row[j], row[k]] for row in X])
    X_true=np.array([X[k] for k in range(len(X)) if y[k]==0])
    X_false=np.array([X[k] for k in range(len(X)) if y[k]==1])
    ax = plt.subplot(111, projection='3d')
    ax.scatter(X_true[:, 0], X_true[:, 1], X_true[:,2], color="b", alpha=0.6, label="Walking")#,cmap=plt.cm.bone)s
    ax.scatter(X_false[:, 0], X_false[:, 1], X_false[:,2], color="r", alpha=0.6, label="Other")
    plt.legend()
    if save:
        plt.savefig("data\\DistanceVectorComponents\\"+x_label+"_"+y_label)
        plt.close()
    else:
        plt.show()

def plot_with_marker(serie, marker, fin, movement, clf):
    """
    Here ? Are you really ?
    """    
    fig, ax=plt.subplots()
    ax.plot(serie) 
    x=[ x for x in range(0,len(serie))]
    # Affichage d'une barre verticale � chaque point de rupture
    for i in range(len(marker)):
        p=marker[i]
        f=fin[i]
        ax.axvspan(p, f, alpha=0.1, color='red')
        plt.axvline(x=p, linewidth=0.5, color='m')
        plt.axvline(x=f, linewidth=0.5, color='g')
    plt.title("Recognition of "+movement+" by "+clf)
    plt.xlabel("Times (50 Hz)")
    plt.ylabel("Acceleration (g)")
#     plt.savefig("report_pictures\\continuous_recognition\\recognition_of_"+movement+"_by_"+clf)
#     plt.close()
    plt.show()

def hightlight(X,y,i,j,k,l,z,index):
    plot_data_components_i_j(X, y, i, j, save=False, hightlight=index)
    plot_data_components_i_j(X, y, i, k, save=False, hightlight=index)
    plot_data_components_i_j(X, y, i, l, save=False, hightlight=index)
    plot_data_components_i_j(X, y, i, z, save=False, hightlight=index)
    plt.show()
    
def display_problem(num):
    sgmtt=ld.load_segmentation("forecsys_data\\juncture")
    bp=sgmtt.get_breaking_points()
    if num > 198:
        num-=199
        other_serie=ld.load_serie("forecsys_data\\juncture")
        serie=ld.load_serie("forecsys_data\\other_classe")
        index=[bp[k]*len(serie)/len(other_serie) for k in range(len(bp)-1)]
    else:
        serie=ld.load_serie("forecsys_data\\juncture")
        index=[bp[k] for k in range(len(bp)-1)]
    build_distance_vector(index[num],serie,True)
    
    
def plot_hist_i(X,y,i):
    X_true=np.array([X[k] for k in range(len(X)) if y[k]==0])
    X_false=np.array([X[k] for k in range(len(X)) if y[k]==1])
    plt.subplot(121)
    plt.hist(X_true[:,i], bins=30)
    plt.subplot(122)
    plt.hist(X_false[:,i], bins=30)
    plt.show()