# coding: utf-8

"""
:author: Philippenko
:date: June 2017

This module contains some useful data which are often needed.
In particular :
    #. The classifiers
    #. The parameters of the classifiers
    #. The interval of the parameters
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import tree
from sklearn import svm

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import gaussian_mixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

classes=["WalkingForward","WalkingLeft","WalkingRight","WalkingUpstairs",
            "WalkingDownstairs","RunningForward","JumpingUp","Sitting","Standing",
            "Sleeping","ElevatorUp","ElevatorDown"]

intervalles=[2000,2000,2000,5000,5000,1500,1000,4000,2000,4000,9000,9000]

n_act=1

clf_parameters=[["C"], [], ["n_neighbors", "p"],["max_depth", "min_samples_leaf", "max_features", "min_samples_split"],
                ["n_estimators", "max_depth", "min_samples_leaf", "max_features", "min_samples_split"],
                ["C"]]
 
range_clf_parameters=[[np.linspace(0.1,10,50)], [], [np.linspace(1,10,10, dtype=int), np.linspace(1,10,10, dtype=int)],
                      [np.linspace(1,20,20, dtype=int), np.linspace(1,20,20, dtype=int), np.linspace(1,9,9, dtype=int), np.linspace(2,20,19, dtype=int)],
                      [np.linspace(1,20,20, dtype=int), np.linspace(1,20,20, dtype=int), np.linspace(1,20,20, dtype=int), np.linspace(1,9,9, dtype=int), np.linspace(2,20,19, dtype=int)],
                      [np.linspace(0.1,10,50)]]

classifiers=[svm.LinearSVC(dual=False),GaussianNB(),neighbors.KNeighborsClassifier(n_neighbors=2),
             tree.DecisionTreeClassifier(min_samples_leaf=4,max_depth=15),
             RandomForestClassifier(min_samples_leaf=4,max_depth=10,random_state=1), LogisticRegression()]