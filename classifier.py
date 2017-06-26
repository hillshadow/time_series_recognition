# coding: utf-8

"""
@author: Philippenko

This module implements the classification problem of the time series.

In particular it could performs:
    - the data preparation so as to classify them
    - the performance measures
    - the performances plotting
"""

from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import tree
from sklearn import svm

import numpy as np
import matplotlib.pyplot as plt
import load as ld
from comparaison import build_distance_vector
from computation import normalization
from variables import activities
from time import time

classifiers=[svm.SVC(),GaussianNB(),neighbors.KNeighborsClassifier(),tree.DecisionTreeClassifier(min_samples_leaf=5,max_depth=15)]

def compute_data(category="manual"):
    """
    Gathers the series by class, compute their distance vectors and create the associated label vector
    
    Parameter
    ----------
    catergorie: string-like
        where to take the data, for instance : 
    """
    start = time()
    X=[]
    y=[]
    for i in range(0,7):    
        print(activities[i])
        serie_path=ld.get_filename(activities[i], category)
        serie=ld.load_serie(serie_path)
        sgmtt=ld.load_segmentation(serie_path)
        bp=sgmtt.get_breaking_points()
        print("Computing distance characterization of the series ...")
        for k in range(len(bp)-1):
            print("Number : ",k)
            X.append(build_distance_vector(bp[k],serie))
            y.append(i)
#         for k in range(len(bp)-2):
#             X.append(distance_optimum(template, template_variance, normalization(serie[bp[k]+longueur_fenetre/2:bp[k]+3*longueur_fenetre/2])))
#             y.append(0)
    print("X=",X)
    print("Y=",y)
    end = time()
    print("Execution time :", end-start)
    return (np.array(X),np.array(y))
    
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
            plot_data_components_i_j(X, y, i, j, str(i), str(j), save)
        
        
def plot_data_components_i_j(X,y,i,j, x_label, y_label,save=False):
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
    print("Size of the True Classe :", len([e for e in y if e==1]))
    print("Size of the False Classe :", len([e for e in y if e==0]))
    X=np.array([[row[i],row[j]] for row in X])
    plt.figure()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.9)#,cmap=plt.cm.bone)
    if save:
        plt.savefig("data\\DistanceVectorComponents\\"+x_label+"_"+y_label)
        plt.close()
    else:
        plt.show()
    

def performance_classifier(train_data, train_label, test_data, test_label, label_predicted):
    """
    Measures the performance of a prediction
    """
    print("True Test Label : ")
    print(test_label)
    print("Predicted Test Label :")
    print(label_predicted)
    exact=[test_label[i] for i in range(len(test_label)) 
           if test_label[i]==label_predicted[i]] 
    total=[] 
    print("Erreur Globale: ", (1-float(len(exact))/len(test_label))*100,"%")
    # For each class we will look what we have recognized
    for i in range(len(activities[0:7])):
        under_total=[]
        # For each activities, we look how many we have recognized it.
        for j in range(len(activities[0:7])):
            under_total.append(recognition_rate_of_j_when_i(i,j,test_label,label_predicted))
        total.append(under_total)
            
    return np.array(total)

def recognition_rate_of_j_when_i(i,j,test,predicted):
    """
    Measures the rate of element which have been labeled with j when it should have been labeled with i.
    Perfect performances:
        - i == j => return 100
        - i != j => return 0
    """
    n=sum([t==i for t in test])
    return float(sum([test[k]==i and predicted[k]==j for k in range(len(test))])*100)/n

def test_performance_clf(X,x,Y,y,predicted,title):
    """
    Measures the auto-recognition and an hold-out performances and plot it via a matrix.
    
    Parameters
    ----------
    X: training data
    x: training labels
    Y: test data
    y: test labels
    predicted: the label predicted with the classifier
    title: string
        the title of the graph
    """
    rate=performance_classifier(X, x, Y, y,predicted)
    np.savetxt('data\\rate.txt', rate, delimiter=',')
    plt.matshow(rate)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()
    print(rate)

    
def test_performance():
    """
    Performs a test of the classifiers performances.
    For each of our classifier, carries out an auto-recognition test and an hold-out test.
    """
    X,x=compute_data("manual")
    Y,y=compute_data("test")
    for clf in classifiers:
        clf.fit(X, x)
        predicted_labels=clf.predict(X)
        test_performance_clf(X, x, X, x, predicted_labels, "Auto-recognition performance")
        predicted_labels=clf.predict(Y)
        test_performance_clf(X, x, Y, y, predicted_labels, "Hold-Out recognition performance")
        
    
    