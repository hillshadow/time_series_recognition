# coding: utf-8

"""
:author: Philippenko
:date: June 2017

This class is devoted to the segmentation of a time serie.
There is also the associated tools for printing the segmentation elements.
"""

from statistics import median

import segmentation_construction as com
import manual as man
import storage.save as sv
from utilities.dba import DBA
from plotting.plotting import plot_series, plot_series_and_marker, plot_series_superposition
        
class Segmentation:
    """
    A segmentation is an object which gathers the major information of a time series constituted by action
    of a a single class.
        #. the points of the time series
        #. the order (optional) : this information is depracated and is useful only when the segmentation is 
        done automatically.
        #. the name of the class (optional)
        #. the time abscisse (optional) : only when the time aspect is important
        #. the breaking points
        #. the segments of the time series
        #. the average segment (optional)
        #. the dispersion segment (optional)
    """
    
    def __init__(self, **keys):
        if 'segments' in keys: self.loading(**keys)
        else: self.initialisation(**keys)
    
    def initialisation(self, serie,order, activity, automatic=True, compute=True, absc=[]):
        self.serie=serie
        self.order=[]
        self.activity=activity
        self.absc=absc
        if automatic:
            self.sd_serie=com.preparation(serie,order)
            self.breaking_points=com.selection_relevant_points(com.compute_breaking_points(serie, self.sd_serie))
        else:
            print("Manuel segmentation")
            self.sd_serie=[]
            if absc==[]:
                self.breaking_points=man.manuel_selection_breaking_points(self.serie)[1:]
            else:
                self.breaking_points=man.manuel_selection_breaking_points_with_time(self.absc, self.serie)
            # The first breaking point mark the beginning of the movement.
            # While the last breaking point mark the end of all mouvements.
            # In this manner, one performs an index shift so as to keep only the significant points.
            print(self.breaking_points)
            self.serie=self.serie[self.breaking_points[0]:self.breaking_points[len(self.breaking_points)-1]]
            self.breaking_points=[b-self.breaking_points[0] for b in self.breaking_points]
        # If there is a duplicate points, the variance segmentation_construction will fail.
        # This case is possible with the manually method.
        self.breaking_points=sorted(list(set(self.breaking_points)))
        self.segments=com.compute_segments(self.breaking_points, serie)
        self.segments=[com.normalization(s) for s in self.segments]
        if compute==True:
            print("Computing the average segment via DBA ...")
            (self.average_segment,self.dispersion_segment)=DBA(self.segments,3)
        else:
            self.average_segment=[]
            self.dispersion_segment=[]
            
    def loading(self,absc,serie,order,activity,sd_serie,breaking_points,segments,average_segment,dispersion_segment):
        self.serie=serie
        self.order=order
        self.absc=[]
        self.activity=activity
        self.sd_serie=sd_serie
        self.breaking_points=breaking_points
        self.segments=segments
        self.average_segment=average_segment
        self.dispersion_segment=dispersion_segment   
        
    def get_absc(self):
        return self.absc
        
    def get_serie(self):
        return self.serie
    
    def get_order(self):
        return self.order
    
    def get_activity(self):
        return self.activity
    
    def get_sd_serie(self):
        return self.sd_serie
    
    def get_breaking_points(self):
        return self.breaking_points
    
    def get_segments(self):
        return self.segments
    
    def get_average_segment(self):
        return self.average_segment
    
    def get_dispersion_segment(self):
        return self.dispersion_segment
    
    def get_all(self):
        return(self.absc, self.serie,self.order,self.sd_serie,self.sd_serie,self.breaking_points,
               self.segments, self.average_segment, self.dispersion_segment)
        
    def set_breaking_points(self,bp):
        self.breaking_points=bp
        
    def recompute_average_segment(self, iteration):
        self.segments=com.compute_segments(self.breaking_points, self.serie)
        self.segments=[com.normalization(s) for s in self.segments]
        (self.average_segment,self.dispersion_segment)=DBA(self.segments,iteration)
        
    def prunning_breaking_points(self):
        self.breaking_points=sorted(list(set(self.breaking_points)))
        
    def recompute_bp(self):
        self.breaking_points=sorted(list(set(self.breaking_points)))
        
    def recompute_segments(self):
        self.recompute_bp()
        self.segments=com.compute_segments(self.breaking_points, self.serie)
        self.segments=[com.normalization(s) for s in self.segments]
        
        
    def store(self,filepath):
        sv.save_list(self.absc, filepath+"\\time.csv")
        sv.save_list(self.serie, filepath+"\\serie.csv")
        sv.save_list(self.breaking_points, filepath+"\\breaking_points.csv")
        sv.save(self.segments, filepath+"\\segments.txt")
        sv.save_list(self.average_segment,filepath+"\\average_segment.csv")
        sv.save_list(self.dispersion_segment,filepath+"\\dispersion_segment.csv")    
    
    # Affichage de la superposition des segments.
    def plot_segments_superposition(self, filepath,save=True):
        plot_series_superposition(self.segments, "Segments", filepath+"\\segments_superposition.png", save)

        
    def plot_serie(self, filepath,save=True):
        plot_series(self.serie,self.activity,filepath+"\\serie.png",save)
        
    def plot_breaking_points(self, filepath,save=True):
        plot_series_and_marker(self.serie, self.breaking_points, self.activity, filepath+"\\breaking_points.png", save)
        
    def plot_smooth_diff(self, filepath, save=True):
        plot_series(self.sd_serie,"Smoothing and Differenciation",filepath+"\\smooth_diff.png", save)

        
    def plot_average_segment(self, filepath,save=True):
        three_series=[self.average_segment,
                      [self.average_segment[i]-3*self.dispersion_segment[i] for i in range(len(self.average_segment))],
                      [self.average_segment[i]+3*self.dispersion_segment[i] for i in range(len(self.average_segment))]]
        plot_series_superposition(three_series, "Average_segment", 
                                 filepath+"\\average_segment.png", 
                                 save, ['-b', '--r', '--r'], figsize=(2,4))
    
    def display_segmentation(self,filepath,save=True):
        self.plot_breaking_points(filepath)
        self.plot_segments_superposition(filepath)
        self.plot_average_segment(filepath)
        
    def check_break_points_index_increasing(self):
        prev=0
        for p in self.breaking_points:
            if prev>p:
                print("There is a problem with the couple ", (prev,p))
            prev=p
            
    def aberant_points(self):
        distance=[self.breaking_points[i]-self.breaking_points[i-1] for i in range(1,len(self.breaking_points))]
        med=median(distance)
        for i in range(len(distance)):
            if not com.little_variation(med, distance[i], 50):
                print("The distance is too big !")
                print("Distance : ", distance[i])
                print("Point :", i)
        
