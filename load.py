# coding: utf-8

"""
@author: Philippenko

This module is devoted to the loading of our data.
In particular :
    1) load a simple list from a csv file
    2) load a list of list of different lengths (for the segments)
"""

import csv
import pickle
import os
from segmentation import Segmentation

def load_list(filename, in_float=True):
    cr = csv.reader(open(filename,"rb"))
    liste=[]
    for row in cr:
        if in_float:
            liste.append(float(row[0]))
        else:
            liste.append(int(row[0]))
    return liste
        
def load_segments(filename):
    with open (filename, 'rb') as fp:
        segments = pickle.load(fp)
    return segments

def load_double_list(filename):
    file = open(filename, "rb") 
    try:
        list1=[]
        list2=[]
        reader = csv.reader(file)
        for row in reader:
            list1.append(float(row[0]))
            list2.append(float(row[1]))
    finally:
        file.close() 
    return (list1, list2)  

def load_serie(filename):
    #TODO : généraliser le test/manuel/auto
    bp=load_list(filename+"\\breaking_points.csv", in_float=False)
    serie=load_list(filename+"\\serie.csv")
    serie=serie[bp[0]:bp[len(bp)-1]]
    return serie

def load_segmentation(filepath):
    print("Loading segmentation ...")
    serie=load_list(filepath+"\\serie.csv")
    breaking_points=load_list(filepath+"\\breaking_points.csv",False)
    segments=load_segments(filepath+"\\segments.txt")
    average_segment=load_list(filepath+"\\average_segment.csv")
    dispersion_segment=load_list(filepath+"\\dispersion_segment.csv")  
    return Segmentation(serie=serie,order="NaN",activity="",sd_serie=[],breaking_points=breaking_points,segments=segments,average_segment=average_segment,dispersion_segment=dispersion_segment)

def get_filename(activity, category):
    path="USC-Activities\\{0}\\".format(activity)
    return path+category
    