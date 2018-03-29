# coding: utf-8

"""
Hightly depracted !
"""
from segmentation.segmentation import Segmentation
from storage import load as ld
from storage import save as sv
from exploitation.featurization import compare, distance_optimum
from segmentation.segmentation_construction import normalization
import matplotlib.pyplot as plt
import os.path
from statistics import mean, variance


classes=["WalkingForward","WalkingLeft","WalkingRight","WalkingUpstairs",
            "WalkingDownstairs","RunningForward","JumpingUp","Sitting","Standing",
            "Sleeping","ElevatorUp","ElevatorDown"]

def recognize_all_series(deb=0, fin=1000, automatic=True):
    print("Recognize All Series")
    for i in range(12):
        print("#### \t Pattern :",classes[i]) 
        if automatic:
            template_path="USC-Activities\\{0}\\automatic".format(classes[i])
        else:
            template_path="USC-Activities\\{0}\\manual".format(classes[i])
        for j in range(12):
            print("## \t Activity :",classes[j]) 
            serie_path="USC-Activities\\{0}\\SSQserie.csv".format(classes[j])
            (serie,marker)=recognize_two_series(template_path, serie_path)
            save_plot_recognition(marker, serie, classes[i], classes[j])
            
            #performance_recognition(template_path,serie_path,i)
            
def recognize_itself(automatic=True):     
    for i in range(12):
        if automatic:
            template_path="USC-Activities\\{0}\\automatic".format(classes[i])
        else:
            template_path="USC-Activities\\{0}\\manual".format(classes[i])
        serie_path="USC-Activities\\{0}\\SSQserie.csv".format(classes[i])
        (serie,marker)=recognize_two_series(template_path, serie_path)
        save_plot_recognition(marker, serie, classes[i], classes[i])
        
            
def recognize_two_series(template_path, serie_path):
    template=ld.load_list(template_path+"\\average_segment.csv")
    template_variance=ld.load_list(template_path+"\\dispersion_segment.csv")
    serie=ld.load_serie(serie_path)
    # TODO : se serait bien de pouvoir charger SSQserie en entier ou par morceaux !
    marker=compare(template, template_variance, serie, densite=True)
    return (serie, marker)

def save_plot_recognition(marker, serie, act1, act2, save=False):        
    plt.figure(figsize=(15, 4))
    plt.plot(serie)  
    # Affichage d'une barre verticale chaque point de rupture
    for p in marker:
        if len(marker)==len(serie):
            if p[2]<250:
                plt.axvline(x=p[0], linewidth=0.5, color='g')
            elif p[2]<350:
                plt.axvline(x=p[0], linewidth=0.5, color='m')
            elif p[2]<450:
                plt.axvline(x=p[0], linewidth=0.5, color='r')
            elif p[2]<600:
                plt.axvline(x=p[0], linewidth=0.5, color='k')
        else:
            plt.axvline(x=p[0], linewidth=0.5, color='r')  
    if save:
        picture_name="{0}_REC_{1}.png".format(act1,act2)
        plt.savefig("data\\ImagesOfRecognition\\"+picture_name)
    else:
        plt.show()
        
def recognize(template_path, serie_path, num_template, num_serie, threshold=0, automatic=True):   
    (serie,marker)=recognize_two_series(template_path, serie_path)
    save_plot_recognition(marker, serie, classes[num_template], classes[num_serie], save=False)
    return len(marker)

def performance_recognition_one_vs_one(i,j,automatic=True):
    """
    This function serve to compute the recognition statistics for a given template i, 
    and a given time series j.
    The statistics are built with regards to the test\info.txt document which 
    summarize which classes and how many elementary movements of the activity 
    should be detected.
    """
    serie_path="USC-Activities\\{0}\\test".format(classes[j])
    true_rows=ld.load_list(serie_path+"\\info.csv", in_float=False)
    if automatic:
        template_path="USC-Activities\\{0}\\automatic".format(classes[i])
    else:
        template_path="USC-Activities\\{0}\\manual".format(classes[i])
    recognized=recognize(template_path, serie_path, i, j, automatic=False)
    print("Template :", classes[i], ", Recognized serie :", classes[j], "detected : ", recognized, ", in reality : ", true_rows[i])
    return recognized

def individual_performance_recognition(j, automatic=True):
    """
    This function serve to compute the recognition statistics of each template
    for a given activity.
    The statistics are built with regards to the test\info.txt document which 
    summarize which classes and how many elementary movements of the activity 
    should be detected.
    """
    print("### Activity : ", classes[j])
    serie_path="USC-Activities\\{0}\\test".format(classes[j])
    true_rows=ld.load_list(serie_path+"\\info.csv", in_float=False)
    recognized=[0 for act in classes]
    for i in range(len(classes)):
        recognized[i]=performance_recognition_one_vs_one(i, j, automatic)
   
    for i in range(len(classes)): 
        print("Template :", classes[i], ", Recognized serie :", classes[j], "detected : ", recognized[i], ", in reality : ", true_rows[i])
    sv.save_double_list(recognized, true_rows, serie_path+"rec_info.csv")

def global_performance_recognition(automatic=True):
    """
    This function serve to compute the recognition statistics 
    of each template/activity
    The statistics are built with regards to the test\info.txt document which 
    summarize which classes and how many elementary movements of the activity 
    should be detected.
    """
    # For each test series ...
    for j in range(len(classes)):
        individual_performance_recognition(j, automatic)

# TODO : to improve  
def compute_threshold(i,automatic=True):
    path="USC-Activities\\{0}\\manual".format(classes[i])
    sgmtt=ld.load_segmentation(path)
    (serie,marker)=recognize_two_series(path, path, 0)
    marker=sorted(marker)
    bp=sgmtt.get_breaking_points()
    print("bp=",bp)
    true_marker=[marker[bp[j]] for j in range(len(bp)) if bp[j] < len(marker)]
    print("marker=",true_marker)
    threshold=(min([t[2] for t in true_marker])+mean([t[2] for t in true_marker]))/2
    (serie,marker)=recognize_two_series(path, path, threshold)
    while len(marker)<len(bp)-2:
        threshold+=40
        (serie,marker)=recognize_two_series(path, path, threshold)
    print("Threshold = ", threshold)
    recognize(path, path, i, i, threshold, automatic)
    
def print_distance_at_bp(i,automatic=True):
    path="USC-Activities\\{0}\\manual".format(classes[i])
    template=ld.load_list(path+"\\average_segment.csv")
    template_variance=ld.load_list(path+"\\dispersion_segment.csv")
    longueur_fenetre=len(template)
    serie=ld.load_serie(path)
    sgmtt=ld.load_segmentation(path)
    bp=sgmtt.get_breaking_points()
    for i in range(len(bp)-2):
        dist=distance_optimum(template, template_variance, normalization(serie[bp[i]+longueur_fenetre/2:bp[i]+3*longueur_fenetre/2]))
        print("At bp", bp[i], ": ", dist)
        
def auto_recognition(i, threshold, automatic=True):
    path=ld.get_filename(classes[i], automatic)
    recognize(path, path, i, i, threshold, automatic)

    