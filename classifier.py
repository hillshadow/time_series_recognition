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

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import gaussian_mixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import scipy.cluster.hierarchy as hier

cluster=[KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), gaussian_mixture]

import numpy as np
import matplotlib.pyplot as plt
from storage import load as ld
from comparison import build_distance_vector
from utilities.variables import n_act, activities
from time import time
from storage.save import save_double_list, save_list, save_segments

from classification.quality import confusion_matrix, KFold_validation_confusion_matrix
from classification.quality_visualisation import plot_ROC, plot_confusion_matrix, plot_validation_curve
from classification.distribution_visualisation import plot_with_marker

classifiers=[svm.LinearSVC(dual=False),GaussianNB(),neighbors.KNeighborsClassifier(n_neighbors=2),
             tree.DecisionTreeClassifier(min_samples_leaf=4,max_depth=15),
             RandomForestClassifier(min_samples_leaf=4,max_depth=10), LogisticRegression()]

clf_parameters=[["C", "tol"], [], ["n_neighbors", "p"]]
range_clf_parameters=[[np.linspace(0.1,10,50), np.logspace(-6, 0, 15)], [], [np.linspace(1,10,10, dtype=int), np.linspace(1,10,10, dtype=int)]]

def compute_data(category="manual"):
    """
    Gathers the series by class, compute their distance vectors and create the associated label vector
    
    Parameter
    ----------
    categorie: string-like
        where to take the data, for instance : "manual", "test", "automatic", "worker"
        
    Return 
    ------
    (X,y): numpy.array-like
        the data and the associated label
    """
    
    start = time()
    X=[]
    y=[]
    print(n_act)
    for i in range(0,n_act):   
        print(activities[i])
        if category=="worker":
            serie_path="forecsys_data\\juncture"
            other_serie=ld.load_serie("forecsys_data\\other_classe")
        else:
            serie_path=ld.get_filename(activities[i], category)
        serie=ld.load_serie(serie_path)
        sgmtt=ld.load_segmentation(serie_path)
        bp=sgmtt.get_breaking_points()
        print("Computing distance characterization of the series ...")
        for k in range(len(bp)-1):
            print("Number : ",k)
            X.append(build_distance_vector(bp[k],serie))
            y.append(i)
        if category=="worker":
            for k in range(len(bp)-1):
                print("Number : ",k)
                X.append(build_distance_vector(bp[k]*len(other_serie)/len(serie), other_serie))
                y.append(1)
    print("X=",X)
    print("Y=",y)
    end = time()
    print("Execution time :", end-start)
    return (np.array(X),np.array(y))

        


def test_performance_clf(X,x,Y,y,predicted,title):
    """
    Measures the auto-recognition and an hold-out performances and plot it via a matrix.
    
    Parameters
    ----------
    X: matrix of numpy.array-like 
        training data
    x: numpy.array-like
        training labels
    Y: matrix of numpy.array-like 
        test data
    y: numpy.array-like
        test labels
    predicted: numpy.array-like
        the label predicted with the classifier
    title: string
        the title of the graph
    """
    rate=confusion_matrix(y,predicted)
    np.savetxt('data\\rate.txt', rate, delimiter=',')
    
    plot_confusion_matrix(rate, title=title)
    print("Yo",rate)

    
def test_performance(X,y):
    """
    Performs a test of the classifiers performances.
    For each of our classifier, carries out an auto-recognition test and an hold-out test.
    """
    
    if X==None or y==None:
        X_train,y_train=compute_data("manual")
        X_test,y_test=compute_data("test")
        X=np.concatenate([X_train,X_test])
        y=np.concatenate([y_train,y_test])
    else:
        n=len(X)/4
        X_train=np.concatenate([X[:n],X[2*n:3*n]])
        y_train=np.concatenate([y[:n],y[2*n:3*n]])
        X_test=np.concatenate([X[n:2*n],X[3*n:]])
        y_test=np.concatenate([y[n:2*n],y[3*n:]])
    rates=[]
    for clf in classifiers:
        clf.fit(X_train, y_train)
        predicted_labels=clf.predict(X_train)
        test_performance_clf(X_train, y_train, X_train, y_train, predicted_labels, "Auto-recognition_performance_for_"+str(clf)[0:9])
        predicted_labels=clf.predict(X_test)
        test_performance_clf(X_train, y_train, X_test, y_test, predicted_labels, "Hold-Out_recognition_performance_for_"+str(clf)[0:9])
        my_cm=KFold_validation_confusion_matrix(X, y, clf)
        rates.append([my_cm[i,i] for i in range(my_cm.shape[0])])
        plot_confusion_matrix(my_cm, title="3-fold_validation_for_"+str(clf)[0:9])
        if len(set(y))==2:  
            try:
                plot_ROC(X, y, clf)
            except(AttributeError):
                None  
    if my_cm.shape[0]==2:
        clf_type="binary"
    else:
        clf_type="multi"
    save_segments(np.array(rates).transpose(), "docs//performances//"+clf_type+"_rates.txt")
         
    
        
def continuous_recognition(X,y,deb,fin,clf=classifiers[0],path="forecsys_data\\other_classe", movement="step"):
#     serie=ld.load_list("forecsys_data\\forecsys_data0.csv")
    serie=ld.load_list(path+"\\serie.csv")
#     serie=ld.load_serie("forecsys_data\\juncture")
    filename="forecsys_data\\juncture"
    template=ld.load_list(filename+"\\average_segment.csv")
    marker=[]
    intervalle=fin-deb
    clf.fit(X, y)
    t0=deb
    t_current=len(template)*(1+20/100)+deb
    while(t_current < fin):
        vector=build_distance_vector(t0,serie)
        if clf.predict([vector])==0:
            marker.append(t0)
            if vector[4]==0:
                t0=t0+max(len(template),len(template))
                t_current=t_current+max(len(template),len(template))
            else:
                t0=t0+max(int(len(template)/vector[4]),int(len(template)/vector[4]))
                t_current=t_current+max(int(len(template)/vector[4]),int(len(template)/vector[4]))
        else:
            t0+=1
            t_current+=1
    marker=np.array([ [m] for m in marker])
    plot_with_marker(serie, marker, movement, str(clf)[0:9])
#         plot_with_marker(serie, build_hierarchical_clustering(marker,intervalle/50))
    return marker

def features_selection(X,y):
    
    from sklearn.feature_selection import SelectFromModel
    for clf in classifiers[:1]+classifiers[3:]:
        fitted_model=clf.fit(X,y)
        new_model=SelectFromModel(fitted_model, prefit=True)
        print(str(clf))
        print("The new model have selected ", new_model.transform(X).shape[1]," features")
        print(new_model.get_support(True))
        
def optimum_parameters(X,y):
    for i in range(1,len(clf_parameters)):
        print(str(classifiers[i])[0:9])
        parameters=clf_parameters[i]
        parameters_intervalle=range_clf_parameters[i]
        for j in range(len(parameters)):
            print(parameters[j])
            print(parameters_intervalle[j])
            plot_validation_curve(X,y, classifiers[i], parameters[j], parameters_intervalle[j],str(classifiers[i])[0:9]+"_"+str(parameters[j]))
        

def performance_contituous_recognition(X,y):
    path="forecsys_data\\juncture"
    sgmtt=ld.load_segmentation(path)
    m=len(sgmtt.get_serie())
    n=len(sgmtt.get_breaking_points())-2
    recognized_true=[]
    for clf in classifiers:
        print("Classifier :"+str(clf))
        recognized_true.append(len(continuous_recognition(X, y, 0, m, clf, path)))
    for r in recognized_true:
        print("In reality :", n, ", recognized :",r, " ie recognition rate :", float(r)/n*100, "%")
        
    path="forecsys_data\\other_classe"
    m=len(ld.load_serie(path))
    recognized_false=[]
    for clf in classifiers:
        print("Classifier :"+str(clf))
        marker_false=continuous_recognition(X, y, 0, m, clf, path, "not_step")
        save_list(marker_false, "forecsys_data/bp_false_recognition{0}.txt".format(str(clf)[0:9]))
        recognized_false.append(len(marker_false))
        
    for i in range(len(recognized_false)):
        r_t=recognized_true[i]
        r_f=recognized_false[i]
        print("###########    Classifier :")
        print(str(classifiers[i]))
        print("In reality :", n, ", recognized :",r_t, " ie recognition rate :", float(r_t)/n*100, "%")
        print("In the non-walking series we recognized", r_f, " walks while there was in fact none.")
        
    save_double_list(recognized_true, recognized_false, "forecsys_data/performance_recognition_continue.txt")
     
        
def build_hierarchical_clustering(marker, t):
    #t doit correspondre à longueur template/3 (empirique)
    if marker.shape[0]==0:
        return marker
    Z = hier.linkage(marker, 'ward')
    dendo=hier.dendrogram(Z)
    plt.show()
    T=hier.fcluster(Z,max(marker)/t, 'distance')
    n_cluster=len(hier.leaders(Z,T)[1])
    my_clusters=[[marker[i] for i in range(len(marker)) if T[i]==k] for k in range(1,n_cluster+1)]
    moy=[sum(c)/len(c) for c in my_clusters] 
    return moy
     
def test_clusterisation():
    filename="forecsys_data\\juncture"
    template=ld.load_list(filename+"\\average_segment.csv")
    serie=ld.load_serie("forecsys_data\\juncture")
    t0=0
    t_current=len(template)*(1+20/100)
    X=[]
    while(t_current<len(serie)):
        X.append(build_distance_vector(t0, serie))
        t0+=1
        t_current+=1
        
    clu=cluster[0]
    clu.fit(X)
    
    plt.figure(figsize=(15, 4))
    plt.plot(serie)  
    # Affichage d'une barre verticale � chaque point de rupture
    for i in range(len(clu.labels_)):
        if clu.labels_[i]==0:
            plt.axvline(x=i, linewidth=0.5, color='r')
    plt.show()
    return (X,clu)


    