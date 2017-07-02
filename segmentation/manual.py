'''
Created on 30 mai 2017

@author: Philippenko
This module is devoted to the manual selection of the breaking points of a time serie.
We create a ClickEventSegmentation class so as to use the interactive plot
'''
#!/usr/bin/python
# coding: utf-8

import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import num2date


class ClickEventSegmentation:
    
    def __init__(self,ax):
        self.ax=ax
        self.points=[]
        
    def __call__(self, event):
        clickX = event.xdata
        if event.button==1:
            print("Zoom from", clickX)
        elif event.button==3:
            print(clickX)
            self.points.append(int(clickX))
        
    def get_point(self):
        return self.points        

    def display_points(self):
        print(self.points)
        
class ClickEventSegmentationWithTime:
    
    def __init__(self,ax, temps):
        self.ax=ax
        self.points=[]
        self.temps=temps
        
    def __call__(self, event):
        string=str(num2date(event.xdata))
        clickX = datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S.%f+00:00')
        if event.button==1:
            print("Zoom from", clickX)
        elif event.button==3:
            true_date=takeClosest(self.temps, clickX)
            print(clickX)
            self.points.append(self.temps.index(true_date))
        
    def get_point(self):
        return self.points        

    def display_points(self):
        print(self.points)
        
class ClickEventSeriesBuilding:
    
    def __init__(self,ax):
        self.ax=ax
        self.points=[]
        
    def __call__(self, event):
        clickX = event.xdata
        print(clickX)
        self.points.append((int(clickX),event.ydata))
        
    def get_point(self):
        return self.points        

    def display_points(self):
        print(self.points)


def manuel_selection_breaking_points(serie): 
    print("Select the breaking points ...")   
    print("Only the right click will be considered ...")
    fig, ax = plt.subplots()
    ax.plot(serie)
    c=ClickEventSegmentation(ax)
    fig.canvas.mpl_connect('button_press_event', c)
    plt.show()
    return c.get_point()    

from bisect import bisect_left

def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def manuel_selection_breaking_points_with_time(absc, serie): 
    print("Select the breaking points ...")   
    print("Only the right click will be considered ...")
    fig, ax = plt.subplots()
    ax.plot(absc, serie)
    c=ClickEventSegmentationWithTime(ax,abs)
    fig.canvas.mpl_connect('button_press_event', c)
    plt.show()
    return c.get_point() 

def manual_time_series_construction(serie):     
    print("Select the points ... ")   
    fig, ax = plt.subplots()
    ax.plot(serie)
    c=ClickEventSeriesBuilding(ax)
    fig.canvas.mpl_connect('button_press_event', c)
    plt.show()
    return c.get_point()   

