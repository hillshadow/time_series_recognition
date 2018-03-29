# coding: utf-8

"""
:author: Philippenko
:date: Juil. 2017

This module is devoted to the measure of the recognition performances.
"""

import numpy as np

from utilities.variables import clf_parameters, range_clf_parameters
from utilities.variables import classifiers
from storage import load as ld
from time import time
import storage.save as sv
from analyze.quality import confusion_matrix, KFold_validation_confusion_matrix
from plotting.plot_classification_quality import plot_ROC, plot_confusion_matrix, plot_validation_curve
from exploitation.classifier import continuous_recognition
        
def performance_continuous_recognition(files_node, classes, clf):
    """Computes the performance of the continuous recognition.
        
    .. seealso:: That is the old version of :func:`quality_of_the_prediction`
    """
    start = time()
    path="forecsys_data\\step"
    sgmtt=ld.load_segmentation(path)
    m=len(sgmtt.get_serie())
    n=len(sgmtt.get_breaking_points())-2
    recognized_true=[]
    for clf in classifiers:
        print("Classifier :"+str(clf))
        serie=ld.load_list(path+"\\serie.csv")
        recognized_true.append(len(continuous_recognition(serie, 0, m, files_node,classes=classes, clf=clf)))
    for r in recognized_true:
        print("In reality :", n, ", recognized :",r, " ie recognition rate :", float(r)/n*100, "%")
         
    path="forecsys_data\\other_classe"
    m=len(ld.load_serie(path))
    recognized_false=[]
    for clf in classifiers:
        print("Classifier :"+str(clf))
        serie=ld.load_list(path+"\\serie.csv")
        marker_false=continuous_recognition(serie, 0, m, files_node, classes, clf)
        sv.save_list(marker_false, "forecsys_data/bp_false_recognition{0}.txt".format(str(clf)[0:9]))
        recognized_false.append(len(marker_false))
         
    for i in range(len(recognized_false)):
        r_t=recognized_true[i]
        r_f=recognized_false[i]
        print("###########    Classifier :")
        print(str(classifiers[i]))
        print("In reality :", n, ", recognized :",r_t, " ie recognition rate :", float(r_t)/n*100, "%")
        print("In the non-walking series we recognized", r_f, " walks while there was in fact none.")
         
    sv.save_double_list(recognized_true, recognized_false, "forecsys_data/performance_recognition_continue.txt")
    end = time()
    print("Execution time :", end-start)

def performance_clf(y, predicted, title, save=True):
    """Measures the recognition performances of a classifier, save it and plot it.
    
    Parameters
    ----------
    y: numpy.array
        test labels
    predicted: numpy.array-like
        the predicted labels
    title: string
        the title of the graph
    save : boolean, optional
        default : True. If the curves are saved or directly plotted on the screen.
        
    Returns
    -------
    Nothing, plot the confusion matrices.
    """
    rate=confusion_matrix(y,predicted)
    np.savetxt('data\\rate.txt', rate, delimiter=',')
    plot_confusion_matrix(rate, title=title, save=save)

    
def confusion_matrices_performance(X,y,save=True):
    """Computes and plot the confusion matrix of a classifier and the ROC curves if possible.
    
    The confusion matrix is the mean of three confusions matrices built with three sub-set of the inital data.
    That is to say, it is a 3-Fold partition of the data.
    
    Parameters
    ----------
    X, y : np.array
        the training data and labels
    save : boolean, optional
        default : True. If the curves are saved or directly plotted on the screen.
        
    Returns
    -------
    Nothing, plot the confusion matrices and the ROC curves.
    """    
#     if X==None or y==None:
#         X_train,y_train=compute_data("manual")
#         X_test,y_test=compute_data("test")
#         X=np.concatenate([X_train,X_test])
#         y=np.concatenate([y_train,y_test])
#     else:
#         n=len(X)/4
#         X_train=np.concatenate([X[:n],X[2*n:3*n]])
#         y_train=np.concatenate([y[:n],y[2*n:3*n]])
#         X_test=np.concatenate([X[n:2*n],X[3*n:]])
#         y_test=np.concatenate([y[n:2*n],y[3*n:]])
    rates=[]
    for clf in classifiers:
#         clf.fit(X_train, y_train)
#         predicted_labels=clf.predict(X_train)
#         test_performance_clf(X_train, y_train, X_train, y_train, predicted_labels, "Auto-recognition_performance_for_"+str(clf)[0:9])
#         predicted_labels=clf.predict(X_test)
#         test_performance_clf(X_train, y_train, X_test, y_test, predicted_labels, "Hold-Out_recognition_performance_for_"+str(clf)[0:9])
        my_cm=KFold_validation_confusion_matrix(X, y, clf)
    #         my_cm=[]
        rates.append([my_cm[i,i] for i in range(my_cm.shape[0])])
        plot_confusion_matrix(my_cm, title="3-fold_validation_for_"+str(clf)[0:9],save=save)
        if len(set(y))==2:  
            try:
                plot_ROC(X, y, clf,save)
            except(AttributeError):
                None  
    if my_cm.shape[0]==2:
        clf_type="binary"
    else:
        clf_type="multi"
    sv.save(np.array(rates).transpose(), "docs//performances//"+clf_type+"_rates.txt")
    

        
def find_the_best_classifiers_parameters(X,y,save=True):
    """Helps to find the optimal value of each of the parameters for all the classifiers
    
    For each classifier and each of it's parameters compute and plot the training score and the 
    cross-validation score on the defined parameter interval. 
    
    Parameters
    ----------
    X, y : np.array
        the training data and labels
        
    Returns
    -------
    Nothing, plot the validation curves.
    """
    for i in range(len(clf_parameters)):
        print(str(classifiers[i])[0:9])
        parameters=clf_parameters[i]
        parameters_intervalle=range_clf_parameters[i]
        for j in range(len(parameters)):
            print(parameters[j])
            plot_validation_curve(X,y, classifiers[i], parameters[j], parameters_intervalle[j],str(classifiers[i])[0:9]+"_"+str(parameters[j]),save)
        

