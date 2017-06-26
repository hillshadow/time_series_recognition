# coding: utf-8

"""
@author: Philippenko

This module implements the comparaison process between a sub-serie and all the templates.
"""


from spatial_shift import compute_spatial_shift_parameters
from temporal_shift import compute_temporel_shift_parameters
from variables import activities
from computation import normalization
from load import get_filename, load_list

def build_distance_vector(start,serie):
    """
    Build the distance vector of a sub-serie.
    
    At the present time, the sub-serie can not be define. Indeed, the length of the
    template is needed. 
    
    The distance vector is computed regarding all the activities ! And for each activity,
    there is a new tempalte, with a new length.
    
    Parameters
    -----------
    serie: list-like
        the whole serie
    start:
        the start point of the considered sub-serie 
        
    """
    distance=[]
    for act in activities[:7]:
        filename=get_filename(act, "manual")
        template=load_list(filename+"\\average_segment.csv")
        template_variance=load_list(filename+"\\dispersion_segment.csv")
        longueur_fenetre=160
        normalized_serie=normalization(serie[start:start+longueur_fenetre])
        
        # Calcul de w2,w3
        (w2,w3,R2,rts,ttl)=compute_temporel_shift_parameters(template, normalized_serie,True)
        # If the rising is too strong that means that the pattern have begin before the current looked time.
        # Thus, at the current time, the pattern could not be recongize even it is the good one.
        # As a result, the param1 and param2 are save as "Not a Number".
#         if rts:
#             w0=0
#             w1=0
        
        # If the tray is too long that means that at the current looked time, the pattern has still not begin
        # A shift must be performed. This shift is possible only if the pattern could be contains in the window.
        # So, it depends of the shift and the speed of the serie.
        # TODO : For the moment, the parameters are initialized with "NaN". Indeed, at the current looked time,
        #        the pattern have not started.
        #        However, the pattern could be included in the looking window. An optimization of the algorithm
        #        will consist of looking how to take that into consideration or shift at once.
#         elif ttl:
#             w0="NaN"
#             w1="NaN"
        
        # Or on the contrary : we will try to know if there the template is included is the looking window.
        # However the temporal shift can not be performed ! Indeed, the shift needs a regression of the 
        # time serie. Or this regression is not feasible ! We have too few points, and it will requires
        # a long time ! Furthermore, if other points are taken, a lot of informations will be lost 
        # (all the peak, all the hollows ...)
        
        # We will only carry out the departure shift, not the speed one.
#         else:
        if w2<0:
            half_shifted_serie=normalized_serie[-int(w2):-int(w2)+len(template)]
        else:
            half_shifted_serie=normalized_serie[0:len(template)]
            
        # The R2 parameter indicates the quality of the regression and the fiability of the computed w2,w3.
        # TODO : Maybe, one could find a threshold criterion which will permit to set w0 and w1 at NaN.
        #        For the moment, the R2 indicators is only added at the distance parameters.
        (w0,w1)=compute_spatial_shift_parameters(template, template_variance, half_shifted_serie)

        distance.append(w0)
        distance.append(w1)
        distance.append(w2)
        distance.append(w3)

    return distance

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


