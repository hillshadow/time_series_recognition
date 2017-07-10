# coding: utf-8

"""
@author: Philippenko

This module is devoted to the loading of our data.
In particular :
    1) loads a simple list from a csv file
    2) loads a list of list of different lengths (for the segments)
    3) loads a segmentation
    4) loads a serie and its breaking points
    6) gets the filename given an activity and a category (test, manuel, automatic)
"""

import csv
import pickle
import os
from segmentation.segmentation import Segmentation
import datetime

def load_list(filename, in_float=True, temps=False):
    """
    Loads a list saved in csv file. The list could be a list of int, float or datetime.datetime
    
    Parameters
    ----------
    filename: string-like
        the name of the file where the list is saved
    in_float: boolean-like
        if True the list will be load as float
    temps: boolean-like
        if True the list will be load as datetim.datetime 
        
    ::warning temps as the priority on in_float ie if the both are true the result will be a time list.
    
    Return
    ------
    list: list-like
        the wished list
    """
    cr = csv.reader(open(filename,"rb"))
    liste=[]
    for row in cr:
        if temps:
            try:
                liste.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f'))
            except(ValueError):
                liste.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
        elif in_float:
            liste.append(float(row[0]))
        else:
            liste.append(int(row[0]))
    return liste
        
def load_segments(filename):
    """
    Loads a list saved in txt file using the pickle module. This format is not readable by human.
    
    Parameters
    ----------
    filename: string-like
        the name of the file where the list is saved
    
    Return
    ------
    list: list of list-like
        the wished segments
    """
    with open (filename, 'rb') as fp:
        segments = pickle.load(fp)
    return segments

def load_double_list(filename,temp=False):
    """
    Loads a double-list saved in csv file. The second list of the double-list could be a datetime.datetime type.
    
    Parameters
    ----------
    filename: string-like
        the name of the file where the list is saved
    temps: boolean-like
        if True the second part will be load as a datetime.datetime type
    
    Return
    ------
    (list1, list2): tuple of list-like
        the wished double-list
    """
    file = open(filename, "rb") 
    try:
        list1=[]
        list2=[]
        reader = csv.reader(file)
        for row in reader:
            list1.append(float(row[0]))
            if temp:
                try:
                    list2.append(datetime.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S.%f'))
                except(ValueError):
                    list2.append(datetime.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S'))
                    
            else:
                list2.append(float(row[1]))
    finally:
        file.close() 
    return (list1, list2)  

def load_serie(filename):
    """
    Loads a series ie the sub time series defines by its associated breaking points
    
    ::warning we do not have tested this function with a series indexed with datetime.datetime
    
    Parameters
    ----------
    filename: string-like
        the name of the file where the list is saved
        
    Return
    ------
    serie: list-like
        the sub time series
    """
    #TODO : généraliser le test/manuel/auto
    bp=load_list(filename+"\\breaking_points.csv", in_float=False)
    serie=load_list(filename+"\\serie.csv")
    serie=serie[bp[0]:bp[len(bp)-1]]
    return serie

def load_segmentation(filepath):
    """
    Loads a segmentation.
    
    Parameters
    ----------
    filename: string-like
        the name of the file where the list is saved
        
    Return
    ------
    Segmentation(...): Segmentation-like
        the wished segmentation
    """
    print("Loading segmentation ...")
    try:
        absc=load_list(filepath+"\\time.csv", temps=True)
    except(IOError):
        absc=[]
    serie=load_list(filepath+"\\serie.csv")
    breaking_points=load_list(filepath+"\\breaking_points.csv",False)
    segments=load_segments(filepath+"\\segments.txt")
    average_segment=load_list(filepath+"\\average_segment.csv")
    dispersion_segment=load_list(filepath+"\\dispersion_segment.csv")  
    return Segmentation(absc=absc,serie=serie,order="NaN",activity="",sd_serie=[],breaking_points=breaking_points,segments=segments,average_segment=average_segment,dispersion_segment=dispersion_segment)

def get_filename(activity, category):
    """
    Gets the appropriate filename (in fact filepath) corresponding to an activity and a category
    
    Parameters
    ----------
    activity: string-like
        the activity, for instance WalkingForward
    category: string-like
        the category of the segmentation, for instance manual
    
    Return:
    filename: string-like
        the wished path
    """
    path="USC-Activities\\{0}\\".format(activity)
    return path+category
    