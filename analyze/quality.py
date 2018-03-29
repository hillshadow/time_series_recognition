# coding: utf-8
'''
:author: Philippenko
:date: Juil. 2017
'''

import numpy as np
from time import time
from copy import deepcopy

from exploitation.classifier import continuous_recognition
from preparation.forecsys_data import compute_data_features
from utilities.variables import classifiers
import storage.load as ld

from sklearn.model_selection import KFold
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


def recognition_rate_of_j_when_i(i,j,label_test,label_predicted):
    """Measures the rate of the elements which have been labeled with j when they should have been labeled with i.
    
    Perfect performances :
        - i == j => return 100
        - i != j => return 0
        
    Parameter:
    ----------
    i,j: int
        the numbers of the class
    label_test:  numpy.array
        the labels of the test data
    label_predicted: numpy.array
        the predicted labels
    """
    n=sum([t==i for t in label_test])
    return float(sum([label_test[k]==i and label_predicted[k]==j for k in range(len(label_test))])*100)/n
    
def confusion_matrix(test_label, label_predicted):
    """Measures the performance of a prediction
    
    Parameters
    ----------
    test_label: numpy.array
        the test labels
    label_predicted: numpy.array
        the predicted label of the test set w.r.t to the training
        
    Return 
    ------
    c_m: matrix of type numpy.array, size 
        The confusion matrix of the classifier
    """
    c_m=[] 
    n=len(set(test_label))
    # For each class we will look what we have recognized
    for i in range(n):
        under_total=[]
        # For each activities, we look how many time we have recognized it.
        for j in range(n):
            under_total.append(recognition_rate_of_j_when_i(i,j,test_label,label_predicted))
        c_m.append(under_total)
    return np.array(c_m)

def KFold_validation_confusion_matrix(X,y, clf, n_split=3, index=None, second_kind=False):
    """Computes the k-Fold confusion matrices. 
    
    That is to say, it carry out a k-fold partition of the data and compute k confusion matrices.
    Then it compute the mean of the rates and return it. By this way, the noise is smoothed.
    
    Parameters
    ----------
    X,y:
        the data set
    clf: 
        the classifier
    n_split: int, optional
        Default : 3
        The number of part of the k-fold partition
    index: list of integer, useless
        Added only to match to the definition of a quality function and to be easily useable.
    second_kind: boolean, useless
        Added only to match to the definition of a quality function and to be easily useable.
        
    Returns
    -------
    numpy.array
        the mean of the confusion matrices.
    """
    kf=KFold(n_splits=n_split, shuffle=True)
    my_cms=[]
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train,y_train)
        my_cms.append(confusion_matrix(y_test, clf.predict(X_test)))
    return np.array( [ [np.mean([d[j,i] for d in my_cms]) for i in range(my_cms[0].shape[0]) ] 
                      for j in range(my_cms[0].shape[1]) ] )
    
def KFold_AUC(X,y,clf,n_split=3, index=None, second_kind=False, save=False):
    """Computes the k-Fold AUC. 
    
    That is to say, it carry out a k-fold partition of the data and compute 2*k AUC.
    The first AUC is computed with the training set and the second with the test set.
    Then it compute the mean of the rates and return it. By this way, the noise is smoothed.
    
    Parameters
    ----------
    X,y:
        the data set
    clf: 
        the classifier
    n_split: int, optional
        Default : 3
        The number of part of the k-fold partition
    index: list of integer, useless
        Added only to match to the definition of a quality function and to be easily useable.
    second_kind: boolean, useless
        Added only to match to the definition of a quality function and to be easily useable.
    save: boolean, useless
        Added only to match to the definition of a quality function and to be easily useable.
        
    Returns
    -------
    (AUC_train, AUC_test): a tuple of float.
        the mean of the train/test AUC.
    """
    kf=KFold(n_splits=n_split, shuffle=True)
    roc_auc_train=[]
    roc_auc_test=[]
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train,y_train)
        
        predictions_train=np.array(clf.predict_proba(X_train))[:,1]
        false_positive_rate_train, true_positive_rate_train, thresholds = roc_curve(y_train, predictions_train)
        roc_auc_train.append(auc(false_positive_rate_train, true_positive_rate_train))       
        
        predictions_test=np.array(clf.predict_proba(X_test))[:,1]
        false_positive_rate_test, true_positive_rate_test, thresholds = roc_curve(y_test, predictions_test)
        roc_auc_test.append(auc(false_positive_rate_test, true_positive_rate_test))
        
    return np.mean(roc_auc_train), np.mean(roc_auc_test)

def quality_of_the_prediction(X,y,clf=classifiers[4], iteration=1, second_kind=True, index=None, save=False): 
    """Computes the *official* quality of the prediction. 
    
    Notes
    -----
    It's very hard to compute properly an accurate measure of the quality of a classifier's prediction and
    to define an *official* quality.
        - Which criterion to choose ? 
        - On what data set apply it ? 
        - In which context this measure is relevant ?
    That is why we have defined a criterion of quality and tried to make it as most informative as possible
    This quality criterion is at the present time adapted *only* for a binary recognition, here step or not step.
    It tries to compute the error of the first and second kind.
    The data are splitted in two parts : The training data and the test data. The classifiers is fitted with *only*
    the training data. The data are are built with the time series of references :
        - The time series consisting solely of steps
        - The time series without any steps   
    The training data is constituted of half of the steps of the steps-time-series and of all the no-steps segments.
    
    The error of first kind is defined as the precision, recall and F-measure of the step's recognition.
    
    The error of second kind is defined as the recognition's rate of the non-step series. The lower it is, the better.
    
    Parameters
    ----------
    X,y:
        the data set
    clf: 
        the classifier
    n_split: int, optional
        Default : 3
        The number of part of the k-fold partition
    index: list of integer, optional
        Default : None.
        The list of the features index to be considered by the classifiers. Very useful when one wants to measure the 
        quality of a sub-set of features.
    second_kind: boolean, optional
        Default : True
        Computing the error of second kind is a very long process. Set to false if ones does not want to compute it
        
    Returns
    -------
    numpy.array
        the mean of the confusion matrices.
    """
    print("Features indexing : ", index)
    start=time()
    
    files_node="forecsys_data"
    classes=["step","other_classe"]

    
#     if (X is None) and (y is None):
#         X,y=compute_data_features(files_node, classes)

    # We take the first half part of the steps and all the non-step segments for the train dataset
    number_train_steps=int(0.5*len([y[i] for i in range(len(y)) if y[i]==0]))
    y_train=[y[i] for i in range(len(y)) if y[i]==1 or 
             (y[i]==0 and i < number_train_steps)]
    X_train=[X[i] for i in range(len(y)) if y[i]==1 or 
             (y[i]==0 and i < number_train_steps)]
    
    precisions=[]
    recalls=[]
    path="forecsys_data\\step"
    sgmtt=ld.load_segmentation(path)
    t=len(sgmtt.get_average_segment())
    bp=sgmtt.get_breaking_points()
    deb, fin = bp[number_train_steps], bp[-1]
    true_segments=[ [bp[i], bp[i+1]] for i in range(number_train_steps,len(bp)-1)]
    for i in range(iteration):
        np.random.seed(seed=10+i*100)
        clf.fit(X_train, y_train)
#         clf.fit(X,y)
        recognized_segments = continuous_recognition(sgmtt.get_serie(), deb, fin, files_node, classes, 
                                                     mono_class=classes[0], clf=clf, index=index, save=save)
        marker=recognized_segments_are_true_segments(true_segments,recognized_segments)
        TP=len([marker[i] for i in range(len(marker)) if marker[i]==1])
        precisions.append(TP/float(len(marker)))
        recalls.append(TP/float(len(true_segments)))
        
    if second_kind:
        print("Computing the error of second kind")
        np.random.seed(seed=10)
        clf.fit(X_train, y_train)
        path="forecsys_data\\other_classe"
        sgmtt=ld.load_segmentation(path)
        deb=0
        fin=len(sgmtt.get_serie())
        marker=continuous_recognition(sgmtt.get_serie(), deb, fin, files_node, classes,mono_class=classes[1],
                                       clf=clf, index=index, save=save)
        print(marker.shape[1])
        false_rate=marker.shape[1]*t/float(fin)
    
    precision=np.mean(precisions)
    recall=np.mean(recalls)
    if recall+precisions != 0:
        f_measure=2*recall*precision/(recall+precision)
    else:
        f_measure=0
    end=time()
    if second_kind:
        print("Precision :", precision, " Recall : ", recall, " F-measure :", f_measure, " False rate :", false_rate)
        print("Execution Time :", end-start, " s")
        return precision,recall, f_measure, false_rate
    else:
        print("Precision :", precision, " Recall : ", recall, " F-measure :", f_measure)
        print("Execution Time :", end-start, " s")
        return precision,recall, f_measure
        
def recognized_segments_are_true_segments(true_segments, recognized_segments): 
    """Checks that the recognized segment are true segments.
    
    That is to say, check that the recognized segment should indeed have been recognized.
    We tolerate a high margin of error, the recognized segments must have at least half of its
    points in common with the true segment.
    
    Parameters
    ----------
    true_segments: list of [index_start, index_end]
        the true segments that one wants to recognized
    recognized_segments: list of [index_start, index_end]
        the segments which have recognized

    Notes
    -----
    Given that one a priori does not know to which of the footstep corresponds the recognized 
    segment, one has to carry out a comparison on each of the still not recognized segment.
    """
    copy_true_segments=deepcopy(true_segments)     
    marker=[]
    recognized_segments=recognized_segments[0]
    for r in recognized_segments:
        to_be_continued=True
        i=0
        while to_be_continued and i < len(copy_true_segments):
            if at_least_half_intersection(copy_true_segments[i], r):
                marker.append(True)
                to_be_continued=False
                del copy_true_segments[i]
            i+=1
        if to_be_continued==True:
            marker.append(False)
            
    assert len(marker)==len(recognized_segments), "The number of marks and of recognized segments are not equal !"
    return marker

def at_least_half_intersection(big_segment,sub_segment):
    """
    Return True if the number of points included in the both series is superior 
    to 50% of the series length.
    
    Parameters
    ----------
    big_segment, sub_segment: list-like, same length
    
    Example
    -------
    >>> at_least_half_intersection([0,5],[5,10])
    False
    >>> at_least_half_intersection([0,5],[2,10])
    True
    >>> at_least_half_intersection([2,10],[0,5])
    False
    >>> at_least_half_intersection([0,5],[0,5])
    True
    """
    return min(big_segment[1],sub_segment[1])-max(big_segment[0],sub_segment[0])>(big_segment[1]-big_segment[0])/2.0
        
 