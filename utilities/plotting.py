'''
Created on 1 juil. 2017

@author: Philippenko

This module serve to plot the principal information of a segmentation.
'''

from utilities.variables import activities
import matplotlib.pyplot as plt
import os
from storage import load as ld
from segmentation import Segmentation

def plot_one_breaking_points_graph(j, automatic=True):
    print("Activity :",activities[j])
    filename=ld.get_filename(activities[j], automatic)
    print(filename)
    print(os.path.abspath(filename))
    (serie,order,activity,sd_serie,breaking_points,segments,average_segment,dispersion_segment)=ld.load_segmentation(filename)
    serie=serie[breaking_points[0]:breaking_points[len(breaking_points)-1]]
    sgmtt=Segmentation(serie=serie,order=order,activity=activity,sd_serie=sd_serie,breaking_points=breaking_points,segments=segments,average_segment=average_segment,dispersion_segment=dispersion_segment)
    sgmtt.plot_breaking_points(filename)
     
def plot_all_breaking_points_graph(automatic=True):
    for j in range(12):
        plot_one_breaking_points_graph(j, automatic)
        
def plot_one_serie(j, deb, fin, save=True):
    print("## \t Activity :",j) 
    filename="USC-Activities\\{0}\\SSQserie.csv".format(activities[j])
    serie=ld.load_list(filename)
    plt.figure(figsize=(25, 10))
    plt.plot(serie)
    plt.title(activities[j])
    if save:
        plt.savefig("data\\Series"+activities[j])
    else:
        plt.show()
         
def plot_all_series(deb=0, fin=5000, save=True):
    for j in range(12):
        plot_one_serie(j, deb, fin, save)
         
def plot_one_average_segment(j, automatic=True, save=True):
    print("## \t Activity :",activities[j]) 
    filename=ld.get_filename(activities[j], automatic)
    average_segment=ld.load_list(filename+"\\average_segment.csv")
    dispersion_segment=ld.load_list(filename+"\\dispersion_segment.csv")
    plt.figure(figsize=(2, 4))
    plt.plot(average_segment)
    plt.plot([average_segment[i]+3*dispersion_segment[i] 
              for i in range(len(dispersion_segment))], '--r')
    plt.plot([average_segment[i]-3*dispersion_segment[i] 
              for i in range(len(dispersion_segment))], '--r')
    plt.title(activities[j])
    if save:
        plt.savefig("data\\AverageSegments\\"+activities[j])
    else:
        plt.show()
        
def plot_all_average_segment(automatic=True, save=True):
    for j in range(12):
        plot_one_average_segment(j, automatic, save)
        