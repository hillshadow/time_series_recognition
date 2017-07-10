# coding: utf-8
'''
Created on 6 juil. 2017

@author: Philippenko
'''

import numpy as np


from sklearn.model_selection import KFold
from classification.quality_visualisation import plot_confusion_matrix

def recognition_rate_of_j_when_i(i,j,label_test,label_predicted):
    """
    Measures the rate of the elements which have been labeled with j when they should have been labeled with i.
    Perfect performances:
        - i == j => return 100
        - i != j => return 0
        
    Parameter:
    ----------
    i,j: int-like
        the number of the class
    label_test:  numpy.array-like
        the labels of the test data
    label_predicted: numpy.array-like
        the predicted labels
    """
    n=sum([t==i for t in label_test])
    return float(sum([label_test[k]==i and label_predicted[k]==j for k in range(len(label_test))])*100)/n
    
def confusion_matrix(test_label, label_predicted):
    """
    Measures the performance of a prediction
    
    Parameters
    ----------
    train_data, train_label: numpy.array-like
        the train set
    test_data, test_label: numpy.array-like
        the test set
    label_predicted: numpy.array-like
        the predicted label of the test set w.r.t to the training
        
    Return 
    ------
    c_m: matrix of type numpy.array, size 
        The confusion matrix of the classifier
    """
    print("True Test Label : ")
    print(test_label)
    print("Predicted Test Label :")
    print(label_predicted)
    exact=[test_label[i] for i in range(len(test_label)) 
           if test_label[i]==label_predicted[i]] 
    c_m=[] 
    n=len(set(test_label))
    print("Erreur Globale: ", (1-float(len(exact))/len(test_label))*100,"%")
    # For each class we will look what we have recognized
    for i in range(n):
        under_total=[]
        # For each activities, we look how many we have recognized it.
        for j in range(n):
            under_total.append(recognition_rate_of_j_when_i(i,j,test_label,label_predicted))
        c_m.append(under_total)
    return np.array(c_m)

def KFold_validation_confusion_matrix(X,y, clf):
    kf=KFold(n_splits=3, shuffle=True)
    my_cms=[]
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(y_train)
        clf.fit(X_train,y_train)
        my_cms.append(confusion_matrix(y_test, clf.predict(X_test)))
    return np.array( [ [np.mean([d[j,i] for d in my_cms]) for i in range(my_cms[0].shape[0]) ] 
                      for j in range(my_cms[0].shape[1]) ] )
    
        
 