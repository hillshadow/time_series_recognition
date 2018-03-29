# coding: utf-8
'''
:author: Philippenko
:date: Juil. 2017

This module is devoted to the features selection. It implements several methods more or less 
automatics.

The features_selection function carries out an automatic selection while the
quality_by_grouped_features and quality_by_bigger_grouped_features functions are manual and 
need pre-registered combinations of features.


'''

import numpy as np
from time import time

from utilities.variables import classifiers
from analyze.quality import quality_of_the_prediction, KFold_validation_confusion_matrix

from plotting.plot_features_selection import plot_grouped_bar, plot_features_elimination


def add_matrices(A,B):
    """Adds a np.array-like and a list of same shape
    
    Parameters
    ----------
    A: np.array-like, possibly None
    B: a list of same shape
    
    Returns
    -------
    A+B: the sum of the two "matrices"
    """
    if A is None:
        return np.array(B)
    else:
        return A+np.array(B)
    
def transform_matrice(A,iteration):
    """Performs the needed transformation on an array matrix to get a mean of list
    
    Parameters
    ----------
    A: np.array-like
        the futur mean of list
    iteration: int-like
        the number of iterations done
    """
    A=A/float(iteration)
    return A.tolist()
    
def select_number_features(rates, combinaisons=None, my_classifiers=classifiers):
    """Given a rates matrix, return the index of the combination which optimize the classifier's
    score. 
    
    .. warning:: does not work for afs.quality_by_bigger_grouped_features(X,y,quality_func= aq.KFold_AUC) ! Must be corrected !
    """
    features_numbers=[]
    for i in range(np.array(rates).shape[0]):
        r=rates[i]
        clf=my_classifiers[i]
        try:
            the_max=max(r[:-1],key=lambda x:x[0]**2+x[1]**2)
            print("######## "+str(clf)[0:9])
            if combinaisons is None:
                print("Features Numbers : ", r.index(the_max))
            else:
                print(combinaisons)
                print(r.index(the_max))
                print("Features Numbers : ", combinaisons[r.index(the_max)])
            print("The associated score : ", the_max)
            features_numbers.append(r.index(the_max)+1)
        except(ValueError):
            None
    return features_numbers
           
def univariate_selection(X,y, save):
    """Statistical tests can be used to select those features that have the strongest 
        relationship with the output variable.The scikit-learn library provides the SelectKBest 
        class that can be used with a suite of different statistical tests to select a specific 
        number of features.
    
    Parameters
    ----------
    X, y : np.array
        the trainning data and the training labels
        
    Returns
    -------
    None, print to the screen the analyze results
    """
    from sklearn.feature_selection import SelectKBest
    start=time()
    mean_rates=None
    iteration=10
    for k in range(iteration):
        rates=[[] for clf in classifiers]
        for n_features_to_select in range(1,X.shape[1]):
            for c in range(len(classifiers)):
                clf=classifiers[c]
                selector = SelectKBest(k=n_features_to_select)
                fit = selector.fit(X, y)
                X_transf= fit.transform(X)
                clf.fit(X_transf,y)
                my_cm=KFold_validation_confusion_matrix(X_transf, y, clf)
                rates[c].append([my_cm[i,i] for i in range(my_cm.shape[0])])
        mean_rates=add_matrices(mean_rates, rates)
    mean_rates=transform_matrice(mean_rates, iteration)
    features_numbers=select_number_features(mean_rates)
    for i in range(len(classifiers)):
        clf = classifiers[i]
        f = features_numbers[i]
        selector = SelectKBest(k=f)
        fit = selector.fit(X, y)
        print("For "+str(clf)[0:9]+" : "+str(selector.get_support(True)))
    end=time()
    print("Execution Time : ", end-start, " s")
    plot_features_elimination(rates,np.linspace(1,X.shape[1]-1, X.shape[1]-1), 
                              "Number of selected features","univariate_selection", classifiers, save)
    
def recursive_features_elimination(X, y, save):
    """Works by recursively removing attributes and building a model on those attributes that 
        remain. It uses the model accuracy to identify which attributes (and combination of 
        attributes) contribute the most to predicting the target attribute.
        
    Parameters
    ----------
    X, y : np.array
        the training data and labels
        
    Returns
    -------
    Nothing, print to the screen the analyze results
    """
    from sklearn.feature_selection import RFE
    start=time()
    mean_rates=None
    iteration=10
    for k in range(iteration):
        rates=[[] for clf in classifiers]
        for n_features_to_select in range(1,X.shape[1]):
            for c in range(len(classifiers)):
                clf=classifiers[c]
                selector = RFE(clf,n_features_to_select)
                try:
                    selector = selector.fit(X, y)
                    my_cm=KFold_validation_confusion_matrix(X, y, selector)
                    rates[c].append([my_cm[i,i] for i in range(my_cm.shape[0])])
                except(RuntimeError):
                    rates[c].append([0,0])
        mean_rates=add_matrices(mean_rates,rates)
    mean_rates=transform_matrice(mean_rates, iteration)
    features_numbers=select_number_features(mean_rates)
    for i in range(len(classifiers)):
        clf = classifiers[i]
        f = features_numbers[i]
        try:
            selector = RFE(clf,f)
            selector.fit(X, y)
            print("For "+str(clf)[0:9]+" : "+str(selector.get_support(True)))
        except(RuntimeError):
            None
    end=time()
    print("Execution Time : ", end-start, " s")
    plot_features_elimination(rates,np.linspace(1,X.shape[1]-1, X.shape[1]-1), 
                              "Number of selected features","recursive_features_elimination", classifiers, save)
   
def principal_component_analyse(X,y, save):
    """Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into 
        a compressed form. Generally this is called a data reduction technique.
        
    Parameters
    ----------
    X, y : np.array
        the trainning data and the training labels
        
    Returns
    -------
    None, print to the screen the analyze results
    """
    start= time()
    from sklearn.decomposition import PCA
    mean_rates=None
    iteration=10
    for k in range(iteration):
        rates=[[] for clf in classifiers]
        for n_components in range(1,X.shape[1]):
            pca = PCA(n_components=n_components)
            pca.fit(X,y)
            X_transf=pca.fit_transform(X, y)
            for c in range(len(classifiers)):
                clf=classifiers[c]
                clf.fit(X_transf,y)
                my_cm=KFold_validation_confusion_matrix(X_transf, y, clf)
                rates[c].append([my_cm[i,i] for i in range(my_cm.shape[0])])
        mean_rates=add_matrices(mean_rates, rates)
    mean_rates=transform_matrice(mean_rates, iteration)
    select_number_features(mean_rates)
    end=time()
    print("Execution time : ", end-start)
    plot_features_elimination(mean_rates, np.linspace(1, X.shape[1]-1, X.shape[1]-1), 
                              "Number of components of the PCA", "principal_component_analyse", classifiers, save)
    
def select_from_model(X,y):
    """Meta-transformer for selecting features based on importance weights.
    
    Parameters
    ----------
    X, y : np.array
        the trainning data and the training labels
        
    Returns
    -------
    None, print to the screen the analyze results
    """
    from sklearn.feature_selection import SelectFromModel
    for clf in classifiers:
        try:
            fitted_model=clf.fit(X,y)
            selector=SelectFromModel(fitted_model, prefit=True)
            print("Features Numbers : ", selector.transform(X).shape[1])
            print("For "+str(clf)[0:9]+" : "+str(selector.get_support(True))) 
        except(ValueError):
            None  
    
def features_selection(X,y, save=True):    
    """Performs several methods of features selection for all the classifiers and plot the results 
    
    1. Univariate Selection :
        Statistical tests can be used to select those features that have the strongest 
        relationship with the output variable.The scikit-learn library provides the SelectKBest 
        class that can be used with a suite of different statistical tests to select a specific 
        number of features.
        
    2. Recursive Feature Elimination :
        Works by recursively removing attributes and building a model on those attributes that 
        remain. It uses the model accuracy to identify which attributes (and combination of 
        attributes) contribute the most to predicting the target attribute.
        
    3. Principal Component Analysis :
        Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into 
        a compressed form.Generally this is called a data reduction technique.
        
    4. Select From Model :
        Meta-transformer for selecting features based on importance weights.
    
    .. seealso:: 
    
        `Feature Selection For Machine Learning in Python <http://machinelearningmastery.com/feature-selection-machine-learning-python/>`_
          Small tutorial for features selection
    
    """
    univariate_selection(X,y, save)
    recursive_features_elimination(X, y, save)
    principal_component_analyse(X,y, save)
    select_from_model(X,y)

def quality_by_grouped_features(X,y, quality_func=quality_of_the_prediction, my_classifiers=[classifiers[4]], 
                                iteration=1, save=False): 
    """Carry out an analyze of a given set of features and display it.
    
    The users need to pre-registered the features combination set, then the score are computed for each classifiers
    and each features combinations. For each classifier the best set of features is selected.
    The results are plotted on a bar graph. 
    
    The scores depend of the chosen quality function.
    
    .. warning:: The users need to pre-registered exactly 7 combinations. The last one must be with all the the features.
    
    .. warning:: The plotting function used here is not very well written and need that the quality function returns a vector of size 2 or 4. An improvement could be done in the plot_grouped_bar function if one want it to accept  every kind of quality results.
    
    Parameters
    ----------
    X,y: np.array
        The training data and labels.
    quality_function: a function, optional
        The function which compute the quality of a classification. Default : quality_of_the_prediction
    my_classifiers: a list of classifier, optional
        The set of classifiers on which the score will be computed. So as to carry out a comparison.
        default : [classifiers[4]] ie RandomForest.
    iteration: int, optional
        default : 1
        Number of iterations of the quality function. At the end the average score is returned.
    save: boolean, optional
        default : True
        If True then the picture is saved, else it is dislpayed on the screen.
    
    Returns
    -------
    rates: np.array of shape (number of combination, number of classifier, size of the vector qulity)
        The rates for each combination and classifier
    features_numbers: list of shape (number of classifier)
        the index of the combination which optimize the score of the classifiers.
    """
    import pandas as pd
    start=time()
    df=pd.DataFrame(X, columns=["w0","w1","d_spatial","w2","w3","d_temporal", 
                                "fft0","Re(fft1)","Im(fft1)","dispersion"])
    combinaisons=[[0,1,3,4],[2,5],[6,7,8,9],[0,1,2,3,4,5],[2,5,6,7,8,9],[0,1,3,4,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]]
    mean_rates=None
    for k in range(iteration):
        print("Iteration "+str(k))
        rates=[[] for clf in my_classifiers]
        for comb in combinaisons:
            subX=df.iloc[:, comb].as_matrix()
            for c in range(len(my_classifiers)):
                clf=my_classifiers[c]
                quality=quality_func(subX, y, clf, index=comb, second_kind=True, save=save)
                rates[c-1].append([q for q in quality])
        mean_rates=add_matrices(mean_rates, rates)
    mean_rates=transform_matrice(mean_rates, iteration)
    features_numbers=select_number_features(mean_rates, my_classifiers=my_classifiers)
    print(features_numbers)
    plot_grouped_bar(combinaisons, features_numbers, my_classifiers, mean_rates)   
    return mean_rates, features_numbers


    
def quality_by_bigger_grouped_features(X,y, quality_func=KFold_validation_confusion_matrix):
    """Carry out an analyze of a given set of features. 
    
    Unlike the quality_by_grouped_features function, this function do not display the result and 
    could take much more combinations.
    
    The users need to pre-registered the features combination set, then the score are computed for each classifiers
    and each features combinations. For each classifier the best set of features is selected.
    The results are plotted on a bar graph. 
    
    The scores depend of the choosen quality function.
    
    Parameters
    ----------
    X,y: np.array
        The training data and labels.
    quality_function: a function, optional
        The function which compute the quality of a classification. Default : quality_of_the_prediction
    my_classifiers: a list of classifier, optional
        The set of classifiers on which the score will be computed. So as to carry out a comparison.
        Default : [classifiers[4]] ie RandomForest.
    
    Returns
    -------
    rates: np.array of shape (number of combination, number of classifier, size of the vector qulity)
        The rates for each combination and classifier
    features_numbers: list of shape (number of classifier)
        the index of the combination which optimize the score of the classifiers.

    """
    import pandas as pd
    start=time()
    df=pd.DataFrame(X, columns=["w0","w1","d_spatial","w2","w3","d_temporal", 
                                "fft0","Re(fft1)","Im(fft1)","dispersion"])
    combinaisons=[[6,9], # fft + dispersion
                [5,6], [5,9], [2,6], [2,9], #distance1/2 + fft0/dispersion
                [2,6,9], [5,6,9], # distance1/2 + fft0 + dispersion
                [2,5,6,9], # distance1+2 + fft0+ dispersion
                [6,7,8], [7,8,9], [2,5,7,8], #Refft+Imfft+distance1/2/dispersion
                [0,1,2,6,9], [3,4,5,6,9], # temp/spatial + fft + dispersion
                [0,1,2,3,4,5], #temp+spatial
                [0,1,2], [3,4,5], #temp/spatial
                [0,1,2,6], [3,4,5,6], [0,1,2,9], [3,4,5,9], #temp/spatial + fft0/dispersion
                [1,2,3,4,5,6,7,8,9]
                ]
    
    mean_rates=None
    iteration=10
    for k in range(iteration):
        print("Iteration "+str(k))
        rates=[[] for clf in classifiers]
        for comb in combinaisons:
            subX=df.iloc[:, comb].as_matrix()
            for c in range(len(classifiers)):
                clf=classifiers[c]
                clf.fit(subX,y)
                try:
                    my_cm=quality_func(subX, y, clf)
                    rates[c].append([my_cm[i,i] for i in range(my_cm.shape[0])])
                except(AttributeError):
                    None
        mean_rates=add_matrices(mean_rates, rates)
    mean_rates=transform_matrice(mean_rates, iteration)
    features_numbers=select_number_features(mean_rates, combinaisons)
    end=time()
    print("Execution time : ", end-start)
    return mean_rates, features_numbers
     

