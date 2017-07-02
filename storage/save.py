# coding: utf-8

"""
@author: Philippenko

This module is devoted to conveniently save the different manipulated data.
In particular : the list, the double-list or the segments.

Except the segments, the two other types are saved in csv file.
"""

import csv
import pickle

def save_segments(segments, filename):
    """
    Saves the segment list of a segmentation
    
    Parameters
    ----------
    segments: list of list-like
        the segments to be saved
    filename: string-like
        the name of the file where the segments have to be saved.
    """
    with open(filename, 'wb') as fp:
        pickle.dump(segments, fp)
        
def save_list(list,filename):
    """
    Saves a list
    
    Parameters
    ----------
    list: list-like
        the list to be saved
    filename: string-like
        the name of the file where the list have to be saved.
    """
    file = open(filename, "wb")
    try:
        cw = csv.writer(file)
        for p in list:
            cw.writerow([p]) 
    finally:
        file.close()
        
def save_double_list(list1, list2, filename):
    """
    Saves a double-list ie a list constituted of two element for each of its rows.
    
    Parameters
    ----------
    list1: list-like 
        the first part of the double-list
    list2: list-like
        the second part of the double-list
    filename: string-like
        the name of the file where the double-list have to be saved.
    """
    file = open(filename, "wb")
    try:
        writer = csv.writer(file)
        if len(list1)!=len(list2):
            raise Exception("Saving a double list : The list have not the same length !")
        for i in range(len(list1)):
            writer.writerow( (list1[i], list2[i]) ) 
    finally:
        file.close()