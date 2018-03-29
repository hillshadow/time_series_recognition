# coding: utf-8
'''
:author: Philippenko
:date: Juil. 2017

This module is devoted to the preparation of the forecsys data.
'''

from time import time
import numpy as np
import pandas as pd
from storage.save import save_double_list, save
import datetime
from storage.load import load_double_list, load_segmentation, load
from segmentation.segmentation import Segmentation
from segmentation.segmentation_construction import union

from exploitation.featurization import build_features
from numpy import mean

# 0er = jaune !!! segmented until 15:24:30 !
# 1eme = t-shirt pale !!! segmented until 15:15:05

filenames=['1-right-leg-7_2017-03-09T14.57.14.327_D5ECF7066E47_Accelerometer_50.0Hz.csv','2-right-leg-8_2017-03-09T14.59.09.810_CC7A3221D5C1_Accelerometer_50.0Hz.csv']

def data_preparation():   
    """Transforms the data from the raw form to a more usable form and saves it.
    
    Keeps only the needed data and compute the sum of square.
    """
    start = time()
    print("#### \t Starting ! ")  
    
    for i in range(2):
        data=pd.DataFrame({})
    
        print("Worker ", i)
        filename=filenames[i]
        data = pd.concat([data,pd.DataFrame(pd.read_csv(filename))])
        
        data=data.reset_index(drop=True)
        data = data.drop(['epoch (ms)', 'elapsed (s)'], 1)
        
        print("SSQ segmentation_construction")
        col=[]
        time=[]
        origine=data.iloc[0,0]
        for j in range(len(data)):
                col.append(data["x-axis (g)"][j]**2+data["y-axis (g)"][j]**2+data["z-axis (g)"][j]**2)
                if i==0:
                    time.append(datetime.datetime.strptime(data.iloc[j,0], '%Y-%m-%dT%H:%M:%S.%f')+
                                datetime.timedelta(seconds=3*60+6))
                else:
                    time.append(datetime.datetime.strptime(data.iloc[j,0], '%Y-%m-%dT%H:%M:%S.%f')+
                                datetime.timedelta(seconds=3*60+8))
                    
        data["SSQ"]=col
        data["Time"]=time
     
        serie=list(data["SSQ"])
        save_double_list(serie,list(time),"forecsys_data\\forecsys_data{0}.csv".format(i))
        
    end=time()
    
    print("Running time :", end-start)
        
def segmentation_data_worker_j(files_node,j, started=True):
    """This function is a tool which serve to build the time series of references.
    
    Parameters
    ----------
    files_node: string
        The node of the files, that is to say the path to the file where all the recognition
        information are saved.
    j: int
        The number of the worker, if superior to 2 one automatically use the second worker.
    started: boolean, optional
        default: True
        True if the time series have already been started. If False, start from scratch. Remove every thing
    """
    if j<2:
        filepath=files_node+"\\{0}".format("Worker{0}".format(j))
        filename=files_node+"\\forecsys_data{0}.csv".format(j)
    else:#15:50:56
        filepath=files_node+"\\other_classe"
        filename=files_node+"\\forecsys_data{0}.csv".format(1)
    print("Worker :",j)
    (serie_init,temps_init)=load_double_list(filename, True)
    while True:
        if started==False:
            sgmtt=Segmentation(absc=temps_init, serie=serie_init,order=4,activity="Worker{0}".format(0),automatic=False, compute=False)
            started=True
        else:
            sgmtt=load_segmentation(filepath)
            sgmtt2=Segmentation(absc=temps_init, serie=serie_init,order=4,activity="Worker{0}".format(0),automatic=False, compute=False)
            (absc,serie,order,activity,sd_serie,breaking_points,segments,average_segment,dispersion_segment)=union(sgmtt,sgmtt2)
            sgmtt=Segmentation(absc=absc,serie=serie,order=order,activity=activity,sd_serie=sd_serie,
                               breaking_points=breaking_points,segments=segments,average_segment=average_segment,
                               dispersion_segment=dispersion_segment)
        sgmtt.store(filepath)
        sgmtt.display_segmentation(filepath)
        
def forecsys_data(files_node, classe_name):
    """Builds the time series of references.
    
    Parameters
    ----------
    files_node: string
        The node of the files, that is to say the path to the file where all the recognition
        information are saved.
    classes_name : string
        name of the class  in construction
    """
    segmentation_data_worker_j(files_node,0)
    segmentation_data_worker_j(files_node,1)
    segmentation_data_worker_j(files_node, 2)
    join_the_two_workers_segmentation(files_node, classe_name)
    re_segment_other_classe(files_node)
    save_templates_library()
    
def re_segment_other_classe(files_node):
    """Re segments the time series of the *other classes* class.
    
    Indeed, the *other classes* class has been constructed with small part of dump time series. The breaking points
    are almost random. So as to facilitate the featurization process and then the classification, one re-segments
    the time series
    
    Parameters
    ----------
    files_node: string
        The node of the files, that is to say the path to the file where all the recognition
        information are saved.
    """
    templates_library=load(files_node+"\\templates_library.txt")
    sgmtt=load_segmentation(files_node+"\\"+"other_classe")
    len_serie=len(sgmtt.get_serie())
    average_len=float(mean([len(t) for t in templates_library[0,:]]))
    bp=[int(k*average_len) for k in range(int(len_serie/average_len)+1)]
    if bp[-1]>len_serie:
        raise ValueError("The last breaking points is out of range : it is bigger than the length of the series")
    sgmtt.set_breaking_points(bp)
    sgmtt.recompute_segments()
    sgmtt.store(files_node+"\\other_classe")
    
def join_the_two_workers_segmentation(files_node, classe_name):
    """Join the two performed segmentation (one for each worker) and save it.    
    
    Parameters
    ----------
    files_node: string
        The node of the files, that is to say the path to the file where all the recognition
        information are saved.
    classes_name : string
        name of the class  in construction
    """
    filepath=files_node+"\\{0}".format("Worker{0}".format(1))
    sgmtt1=load_segmentation(filepath)
    filepath=files_node+"\\{0}".format("Worker{0}".format(0))
    sgmtt0=load_segmentation(filepath)
    sgmtt1.recompute_segments()
    sgmtt0.recompute_segments()
    (absc,serie,order,activity,sd_serie,breaking_points,segments,average_segment,dispersion_segment)=union(sgmtt0,sgmtt1)
    sgmtt=Segmentation(absc=absc,serie=serie,order=order,activity=activity,sd_serie=sd_serie,
                        breaking_points=breaking_points,segments=segments,average_segment=average_segment,
                        dispersion_segment=dispersion_segment)
    sgmtt.recompute_segments()
    sgmtt.plot_breaking_points(filepath="", save=False)
    sgmtt.plot_segments_superposition("", False)
    sgmtt.recompute_average_segment(30)
    sgmtt.store(filepath=files_node+"\\"+classe_name)
    
def save_templates_library(files_node,classes):
    """Load the template of each class and save it in the libray.
    
    Parameters
    ----------
    files_node: string
        The node of the files, that is to say the path to the file where all the recognition
        information are saved.
    classes : list of classes
    """
    len_classes=len(classes)
    library=[]
    for i in range(len_classes):
        sgmtt=load_segmentation(files_node+"\\"+classes[i])
        library.append([sgmtt.get_average_segment(), sgmtt.get_dispersion_segment()])
    save(np.array(library),files_node+"\\templates_library.txt")
    
def compute_data_features(files_node, classes):
    """Computes the features of the segments of each class and gather them into a numpy.array structure.
    
    Parameters
    ----------
    files_node: string
        The node of the files, that is to say the path to the file where all the recognition
        information are saved.
    classes : list of classes
    
    Returns
    -------
    (X,y): tuple of numpy.array
        The training data and label
    """
    start = time()
    X=[]
    y=[]
    len_classes=len(classes)
    templates_library=load(files_node+"\\templates_library.txt")
    len_max_template=max([len(t) for t in templates_library[:,0]])
    windows_length = int(len_max_template * (1 + 20.0 / 100))
    for i in range(len_classes):   
        print(classes[i])
        serie_path=files_node+"\\"+classes[i]
        try:
            sgmtt=load_segmentation(serie_path)
        except(IOError):
            raise IOError(serie_path, "is not a valid segmentation and can not be load.")
        serie=sgmtt.get_serie()
        bp=sgmtt.get_breaking_points()
        print("Computing features ...")
        for k in range(len(bp)-1):
            print("Number : ",k)
            X.append(build_features(serie[bp[k]:bp[k]+windows_length], templates_library, False))
            import matplotlib.pyplot as plt
            y.append(i)
    end = time()
    print("Execution time :", end-start)
    return (np.array(X),np.array(y))
    
        