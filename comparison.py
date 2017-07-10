# coding: utf-8

"""
@author: Philippenko

This module implements the comparison process between a sub-serie and all the templates.
"""


from shift.spatial_shift import compute_spatial_shift_parameters
from shift.temporal_shift import compute_temporel_shift_parameters
from utilities.variables import activities
from segmentation.segmentation_construction import normalization
from segmentation.segmentation_construction import smoothing
from storage.load import get_filename, load_list

from dba import LB_Keogh

import numpy as np

#Indicates the number of class
from utilities.variables import n_act

def build_distance_vector(start,serie, plot=False):
    """
    Build the distance vector of a sub-serie.
    
    At the present time, the sub-serie can not be define. Indeed, the length of the
    template is needed. 
    
    The distance vector is computed regarding all the activities ! And for each activity,
    there is a new template, with a new length.
    
    Parameters
    -----------
    serie: list-like
        the whole serie
    start:
        the start point of the considered sub-serie 
        
    """
    distance=[]
    for act in activities[:n_act]:
#         filename=get_filename(act, "manual")
        filename="forecsys_data\\juncture"
        template=load_list(filename+"\\average_segment.csv")
        template=smoothing(template,1)
        template_variance=load_list(filename+"\\dispersion_segment.csv")
        longueur_fenetre=int(len(template)*(1+20.0/100))
        normalized_serie=normalization(serie[start:start+longueur_fenetre])
        
        # Calcul de w2,w3
        (w2,w3,dist)=compute_temporel_shift_parameters(template, normalized_serie,plot)
        if abs(w2)>len(template)/2:
            (w0,w1,dist2)=compute_spatial_shift_parameters(template, template_variance, normalized_serie[:len(template)])
            half_shifted_serie=normalized_serie[0:len(template)]
        else:
            if w2<0:
                template2=template[-int(w2):]
                template_variance2=template_variance[-int(w2):]
                half_shifted_serie=normalized_serie[0:len(template2)]
            else:
                half_shifted_serie=normalized_serie[int(w2):int(w2)+len(template)]
                template2=template
                template_variance2=template_variance
            try:
                (w0,w1,dist2)=compute_spatial_shift_parameters(template2, template_variance2, half_shifted_serie)
            except(ZeroDivisionError):
                (w0,w1,dist2)=compute_spatial_shift_parameters(template, template_variance, normalized_serie[:len(template)])
        distance.append(w0)
        distance.append(w1)
        distance.append(dist2)
        distance.append(w2)
        distance.append(w3)
        distance.append(dist)
    my_fft=np.fft.fft(half_shifted_serie)
    distance.append(np.real(my_fft[0]))
    distance.append(np.real(my_fft[1]))
    distance.append(np.imag(my_fft[1]))
    from statistics import stdev
    distance.append(stdev(serie[start:start+longueur_fenetre])) 
        
    if plot:
        print("Start point :", start)
        print(serie[start:start+longueur_fenetre])
        import matplotlib.pyplot as plt
        plt.subplot(131)
        plt.plot(serie[start:start+longueur_fenetre], label="Series")
        plt.plot(template, label="Template")
        plt.legend(loc="best")
        plt.subplot(132)
        plt.plot(normalized_serie, label="Normalized Series")
        plt.plot(template, label="Template")
        plt.legend(loc="best")
        plt.subplot(133)
        plt.plot(half_shifted_serie, label="Half Shifted Series")
        plt.plot(template, label="Template")
        plt.legend(loc="best")
        plt.show()

    return distance

def var_distance(template,serie):
    from math import exp
    var_serie=[]
    var_temp=[]
    n=len(serie)/20
    for i in range(len(serie)-n):
        var_serie.append(max(serie[i:i+n])-min(serie[i:i+n]))
        var_temp.append(max(template[i:i+n])-min(serie[i:i+n]))
    return sum([exp(var_temp[i]-var_serie[i]) for i in range(len(var_serie))])
                           
######################################################################

# The following functions are depracated. They have been useful when we was computing
# a scalar distance.

#######################################################################

# def selection_mouvements(mouvements, fenetre):
#     """
#     Deprecated
#     """
#     i=0
#     while i<len(mouvements)-1:
#         if mouvements[i+1][0]-mouvements[i][0]<fenetre:
#             if mouvements[i][2]<mouvements[i+1][2]:
#                 del mouvements[i+1]
#             else:
#                 del mouvements[i]
#         else:
#             i+=1
#     return mouvements   
#     
# def marker_selection(marker,fenetre):
#     """
#     Deprecated
#     """
#     centers=[]
#     center=[]
#     if len(marker)==0:
#         return marker
#     center.append(marker[0])
#     for i in range(1,len(marker)):
#         if marker[i][0]<marker[i-1][0]+15:
#             center.append(marker[i])
#         else:   
#             centers.append(moy(center))
#             center=[marker[i]]
#     return centers
#         
# def moy(center):
#     """
#     Deprecated
#     """
#     return [int(sum([c[0] for c in center])/len(center)), 0]
#     
# def compare(template, template_variance, serie, threshold, densite=True):
#     """
#     Deprecated
#     """
#     marker=[]
#     longueur_fenetre=len(template)
#     mes_dist=[]
#     for i in range (len(serie)-longueur_fenetre):
#         dist=distance_optimum(template, template_variance, normalization(serie[i:i+longueur_fenetre]))
#         mes_dist.append(dist)
#         if threshold==0:
#             marker.append([i, i+longueur_fenetre, dist])
#         elif under_threshold(dist,threshold):
#             marker.append([i, i+longueur_fenetre, dist])
#     if densite==False:
#         marker=selection_mouvements(marker, longueur_fenetre)
#     if len(mes_dist)!=0:
#         print("min des distance=",min(mes_dist, key=lambda colonnes: colonnes[0]))
#         print("max des distance=",max(mes_dist, key=lambda colonnes: colonnes[0]))
#     if threshold==0:
#         return marker
#     else:
#         return marker_selection(marker, longueur_fenetre)
#     
# def most_important_components(vector):
#     weights = np.ones_like(vector)/float(len(vector)) 
#     amplitude, bins = np.histogram(vector, bins=3)
#     #plt.show()
#     component=[(amplitude[i],i) for i in range(len(amplitude))]
#     component=list(reversed(sorted(component, key=lambda colonnes: colonnes[0])))
#     #return sorted([component[i+1][1] for i in range(n)])       
#     if component[1][0]==0:
#         ratio=sys.float_info.max
#     else:
#         ratio=component[0][0]/component[1][0]
# #     return [amplitude[0],bins[1]-bins[0], amplitude[-1]+amplitude[-2], component[1][1],
# #             len([a for a in amplitude if a==0]), variance([a for a in amplitude[0:10]]),
# #             ratio]
#     return [amplitude[0],amplitude[1],amplitude[2],bins[1]-bins[0],variance(amplitude)]
# 
# def under_threshold(d,t):
#     return d==t


