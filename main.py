# coding: utf-8

files_node="forecsys_data"
classes=["step", "other_classe"]
plot=True
save=False

import ipdb
ipdb.set_trace()

# ==============
# I. Preliminary
# ==============

# ---
# 1. Compute the data
# ---

import preparation.forecsys_data as frc
X,y=frc.compute_data_features(files_node, classes)
import storage.save as sv
sv.save(X, files_node+"\\training_data.txt")
sv.save(y, files_node+"\\training_labels.txt")

# ---
# 2. Classifier fitting
# ---
        
import utilities.variables as var
the_clf=var.classifiers[4] # Random Forests
the_clf.fit(X,y)
sv.save(the_clf,files_node+"\\fitted_classifier.txt")

# ==============
# II. Exploitation
# ==============

# ---
# 1. Prediction : example
# ---

print("This is an example of the way to exploit the classifier "
        "and to predict the number of element of a classe in a time series.")
import storage.load as ld
import exploitation.classifier as clf
for i in range(len(classes)):
    print("Prediction of "+classes[i])
    serie=ld.load_list(files_node+"\\"+classes[i]+"\\serie.csv")
    clf.continuous_recognition(serie[:5000],save=False)
    
# ==============
# II. Analyze
# ==============

# ---
# 1. Visualization of the data distribution
# ---

import plotting.plot_data_distribution as pdv
# 2D Dynamique plot of the temporal distance and the fft
# Warning : You must interrupt the kernel to end this command
#pdv.plot_dynamic_data_components_i_j(X,y,5,6)
# 2D Plot of w0 and the dispersion
pdv.plot_data_components_i_j(X, y, 0, 9)
# 3D Plot of w1, the spatial distance and w2
pdv.plot_data_components_i_j_k(X,y,1,2,3)

# If a point have a strange behaviour : we can higlight this point and display 
#the problem
# Here we higlight the 227th points on the temporal distance and w1/spatial 
#distance/fft/dispersion graphes.
pdv.hightlight(X,y,5,1,2,6,9,227)
pdv.display_problem(227, files_node)

# ---
# 2. Visualization of the classification quality
# ---

import analyze.performances as perf
perf.confusion_matrices_performance(X,y,save)
perf.performance_continuous_recognition(files_node, classes, the_clf)

# ---
# 3. Analyze of the features pertinences
# ---

import analyze.features_selection as afs
import analyze.quality as aq
# Below, we try three different methods to determine wich features are the most
#relevants.
# The graphs shows the quality evolution and one can read on the terminal for which
#features set the optimum is found.
afs.features_selection(X,y, save) 
#Measure the AUC for each of the classifiers and for each of the features set.
afs.quality_by_grouped_features(X,y,quality_func= aq.KFold_AUC,
                                my_classifiers=var.classifiers[1:])
# Measure the quality of the Random Forest classification for each of the 
#defined features set. That is very long ! Indeed, it also performs the 
#continuous recognition of the non-step time series.
afs.quality_by_grouped_features(X,y)

# Measure the quality of each classifier for several features set. Do not plot.
# Then display for each classifier the feature set resulting to the best score.
afs.quality_by_bigger_grouped_features(X,y)
afs.quality_by_bigger_grouped_features(X,y,quality_func= aq.KFold_AUC)