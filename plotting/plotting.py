# coding: utf-8
'''
Created on 1 juil. 2017

@author: Philippenko

This module serve to plot the principal information of a segmentation.
'''

import matplotlib.pyplot as plt

# from storage import load as ld/
# 
# def plot_one_breaking_points_graph(j, automatic=True):
#     from segmentation.segmentation import Segmentation
#     print("Activity :",classes[j])
#     filename=ld.get_filename(classes[j], automatic)
#     print(filename)
#     print(os.path.abspath(filename))
#     (serie,order,activity,sd_serie,breaking_points,segments,average_segment,dispersion_segment)=ld.load_segmentation(filename)
#     serie=serie[breaking_points[0]:breaking_points[len(breaking_points)-1]]
#     sgmtt=Segmentation(serie=serie,order=order,activity=activity,sd_serie=sd_serie,breaking_points=breaking_points,segments=segments,average_segment=average_segment,dispersion_segment=dispersion_segment)
#     sgmtt.plot_breaking_points(filename)
#      
# def plot_all_breaking_points_graph(automatic=True):
#     for j in range(12):
#         plot_one_breaking_points_graph(j, automatic)
#         
# def plot_one_serie(j, deb, fin, save=True):
#     print("## \t Activity :",j) 
#     filename="USC-Activities\\{0}\\SSQserie.csv".format(classes[j])
#     serie=ld.load_list(filename)
#     plt.figure(figsize=(25, 10))
#     plt.plot(serie)
#     plt.title(classes[j])
#     if save:
#         plt.savefig("data\\Series"+classes[j])
#     else:
#         plt.show()
#          
# def plot_all_series(deb=0, fin=5000, save=True):
#     for j in range(12):
#         plot_one_serie(j, deb, fin, save)
#          
# def plot_one_average_segment(j, automatic=True, save=True):
#     print("## \t Activity :",classes[j]) 
#     filename=ld.get_filename(classes[j], automatic)
#     average_segment=ld.load_list(filename+"\\average_segment.csv")
#     dispersion_segment=ld.load_list(filename+"\\dispersion_segment.csv")
#     plt.figure(figsize=(2, 4))
#     plt.plot(average_segment)
#     plt.plot([average_segment[i]+3*dispersion_segment[i] 
#               for i in range(len(dispersion_segment))], '--r')
#     plt.plot([average_segment[i]-3*dispersion_segment[i] 
#               for i in range(len(dispersion_segment))], '--r')
#     plt.title(classes[j])
#     if save:
#         plt.savefig("data\\AverageSegments\\"+classes[j])
#     else:
#         plt.show()
#         
# def plot_all_average_segment(automatic=True, save=True):
#     for j in range(12):
#         plot_one_average_segment(j, automatic, save)
#         
def save_or_not(save,filepath):
    if save:
        plt.savefig(filepath)
        plt.close()
    else: 
        plt.show()
        
def plot_series(series, title, filepath, save):
    plt.figure(figsize=(15, 4))
    plt.plot(series)  
    plt.title(title)  
    save_or_not(save,filepath)
    
def plot_series_superposition(series,title,filepath,save, colors=None, figsize=(15,4)):
    plt.figure(figsize=figsize)
    for i in range(len(series)):
        s=series[i]
        if colors is not None:
            plt.plot(s, colors[i])
        else:
            plt.plot(s)
    plt.title(title)
    save_or_not(save, filepath)
    
def plot_series_and_marker(series, marker, title, filepath, save):
    plt.figure(figsize=(15, 4))
    plt.plot(series)  
    plt.title(title)
    # Affichage d'une barre verticale � chaque point de rupture
    for p in marker:
        plt.axvline(x=p, linewidth=0.5, color='r')
    save_or_not(save,filepath)
                
def plot_recognition(serie, start_marker, end_marker, movement, clf, save=True, mono_classe=None): 
    fig, ax=plt.subplots()
    ax.plot(serie) 
    x=[x for x in range(0,len(serie))]
    # Affichage d'une barre verticale � chaque point de rupture
    for i in range(len(start_marker)):
        p=start_marker[i]
        f=end_marker[i]
        ax.axvspan(p, f, alpha=0.1, color='red')
        plt.axvline(x=p, linewidth=0.5, color='m')
        plt.axvline(x=f, linewidth=0.5, color='g')
    plt.title("Recognition of "+movement+" by "+clf)
    plt.xlabel("Times (50 Hz)")
    plt.ylabel("Acceleration (g)")
    if mono_classe is None:
        save_or_not(save,"report_pictures\\continuous_recognition\\recognition_of_"+movement+"_by_"+clf)
    else:
        save_or_not(save,"report_pictures\\continuous_recognition\\recognition_of_"+mono_classe
                    +"_with_"+movement+"_by_"+clf)
    
 

        