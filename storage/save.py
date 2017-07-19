# coding: utf-8

"""
:author: Philippenko
:date: Juil. 2017

This module is devoted to conveniently save the different manipulated data.
In particular : the list, the double-list or the segments.
"""

import csv
import pickle

def save(data, filename):
    """
    Saves any type of data in txt file using the pickle module
    
    Parameters
    ----------
    data: list of list-like
        the data to be saved
    filename: string-like
        the name of the file where the data have to be saved.
    """
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)
        
def save_list(the_list,filename):
    """
    Saves a the_list
    
    Parameters
    ----------
    the_list: the_list-like
        the the_list to be saved
    filename: string-like
        the name of the the_file where the the_list have to be saved.
    """
    the_file = open(filename, "wb")
    try:
        cw = csv.writer(the_file)
        for p in the_list:
            cw.writerow([p]) 
    finally:
        the_file.close()
        
def save_double_list(list1, list2, filename):
    """
    Saves a double-list ie a list constituted of two elements for each of its rows.
    
    Parameters
    ----------
    list1: list-like 
        the first part of the double-list
    list2: list-like
        the second part of the double-list
    filename: string-like
        the name of the the_file where the double-list have to be saved.
    """
    the_file = open(filename, "wb")
    try:
        writer = csv.writer(the_file)
        if len(list1)!=len(list2):
            raise Exception("Saving a double list : The list have not the same length !")
        for i in range(len(list1)):
            writer.writerow( (list1[i], list2[i]) ) 
    finally:
        the_file.close()