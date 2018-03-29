# coding: utf-8
'''
:author: Philippenko
:date: Juil. 2017

This module is devoted to the plotting of the data's distribution.
'''      

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from storage import load as ld
from exploitation.featurization import build_features
from plotting import save_or_not

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
     
def plot_data_components_i_j_k(X, y, i, j, k, save=False):
    print("Size of the True Classe :", len([e for e in y if e==1]))
    print("Size of the False Classe :", len([e for e in y if e==0]))
    X=np.array([[row[i],row[j], row[k]] for row in X])
    X_true=np.array([X[k] for k in range(len(X)) if y[k]==0])
    X_false=np.array([X[k] for k in range(len(X)) if y[k]==1])
    ax = plt.subplot(111, projection='3d')
    ax.scatter(X_true[:, 0], X_true[:, 1], X_true[:,2], color="b", alpha=0.6, label="Walking")#,cmap=plt.cm.bone)s
    ax.scatter(X_false[:, 0], X_false[:, 1], X_false[:,2], color="r", alpha=0.6, label="Other")
    plt.legend()
    save_or_not(save,"data\\DistanceVectorComponents\\"+str(i)+"_"+str(j)+"_"+str(k))

def hightlight(X,y,i,j,k,l,z,index):
    plot_data_components_i_j(X, y, i, j, save=False, hightlight=index)
    plot_data_components_i_j(X, y, i, k, save=False, hightlight=index)
    plot_data_components_i_j(X, y, i, l, save=False, hightlight=index)
    plot_data_components_i_j(X, y, i, z, save=False, hightlight=index)
    plt.show()
    
def display_problem(num, files_node="\\forecsys_data"):
    sgmtt=ld.load_segmentation(files_node+"\\step")
    bp=sgmtt.get_breaking_points()
    if num > 198:
        num-=199
        other_serie=ld.load_serie(files_node+"\\step")
        serie=ld.load_serie(files_node+"\\other_classe")
        index=[bp[k]*len(serie)/len(other_serie) for k in range(len(bp)-1)]
    else:
        serie=ld.load_serie(files_node+"\\step")
        index=[bp[k] for k in range(len(bp)-1)]
    templates_library=ld.load(files_node+"\\templates_library.txt")
    len_max_template=max([len(t) for t in templates_library[:,0]])
    windows_length = int(len_max_template * (1 + 20.0 / 100))
    build_features(serie[index[num]:index[num]+windows_length], templates_library, True)
    
    
def plot_hist_i(X,y,i):
    X_true=np.array([X[k] for k in range(len(X)) if y[k]==0])
    X_false=np.array([X[k] for k in range(len(X)) if y[k]==1])
    plt.subplot(121)
    plt.hist(X_true[:,i], bins=30)
    plt.subplot(122)
    plt.hist(X_false[:,i], bins=30)
    plt.show()