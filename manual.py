'''
Created on 30 mai 2017

@author: Philippenko
This module is devoted to the manual selection of the breaking points of a time serie.
We create a ClickEvent class so as to use the interactive plot
'''
#!/usr/bin/python
# coding: utf-8

import matplotlib.pyplot as plt


class ClickEvent:
    
    def __init__(self,ax):
        self.ax=ax
        self.points=[]
        
    def __call__(self, event):
        # !!!!!!!!!!!!!!!!! Remodifier !!!!!!!!!!!!!!!!
        clickX = event.xdata
        print(clickX)
        self.points.append((clickX,event.ydata))
        
    def get_point(self):
        return self.points        

    def display_points(self):
        print(self.points)


def manuel_selection_breaking_points(serie): 
    print("Select the breaking points ... please be aware : the first click is not consider (if you want to zoom)")   
    fig, ax = plt.subplots()
    ax.plot(serie)
    c=ClickEvent(ax)
    fig.canvas.mpl_connect('button_press_event', c)
    plt.show()
    return c.get_point()           

