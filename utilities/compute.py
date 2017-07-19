"""
:author: Philippenko
:date: June 2017

This module serve to recompute the templates.

Most of this function are depracted but have been conserved with the hope to be recycle.
"""

from segmentation.segmentation import Segmentation
from storage import load as ld
import numpy as np
from utilities.variables import intervalles
from segmentation.segmentation_construction import union
from storage.save import save


def compute_all_segmentation(automatic=True):  
    '''Computes the USC data segmentation.
    
    .. warning:: This function is deprecated and is usable only with the USC data.
    .. seealso:: That is the old version of :func:`data_preparation`.
    ''' 
    print("Compute All Segmentation") 
    compute_one_segmentation("ElevatorUp", deb=0, fin=0, automatic=automatic)
    compute_one_segmentation("ElevatorDown", deb=0, fin=0, automatic=automatic)
    compute_one_segmentation("JumpingUp", deb=0, fin=1000, automatic=automatic)
    compute_one_segmentation("RunningForward", deb=0, fin=1500, automatic=automatic)
    compute_one_segmentation("Sitting", deb=0, fin=4000, automatic=automatic)
    compute_one_segmentation("Sleeping", deb=0, fin=4000, automatic=automatic)
    compute_one_segmentation("Standing", deb=4000, fin=6000, automatic=automatic)
    compute_one_segmentation("WalkingDownstairs", deb=0, fin=5000, automatic=automatic)
    compute_one_segmentation("WalkingForward", deb=0, fin=2000, automatic=automatic)
    compute_one_segmentation("WalkingLeft", deb=0, fin=2000, automatic=automatic)
    compute_one_segmentation("WalkingRight", deb=0, fin=2000, automatic=automatic)
    compute_one_segmentation("WalkingUpstairs", deb=0, fin=5000, automatic=automatic)
    
def compute_complete_segmentation_of_i(i,started=True,automatic=False):
    """Computes the complete segmentation of the whole serie : for each trials (5) and each subjects (14)
    
    .. warning:: This function is deprecated and is usable only with the USC data.
    
    This function does not force the user to begin the segmentation from the beginning of the serie.
    The user could stop the segmentation and continue later at the same point ! Very useful if there is a problem.
    
    Parameter:
    ----------
    i: int-like
        the number of the activity
    started: boolean-like 
        True if the segmentation is already started and the user
        does not want to restart from the beginning
    automatic:
        True if the segmentation must be automatic.
    """
    from utilities.variables import classes
    activity=classes[i]
    print("## \t Activity :",activity) 
    filename="USC-Activities\\{0}\\SSQserieTotale.csv".format(activity)
    filepath="USC-Activities\\{0}\\complete".format(activity)
    serie=ld.load_list(filename, True)
    n=len(serie)
    if started==True:
        k_deb=ld.load(filepath+"\\deb.txt")
        deb=k_deb*intervalles[i]
        print(deb)
        fin=intervalles[i]*(k_deb+1)
        sgmtt=ld.load_segmentation(filepath)
    else: 
        deb=0
        fin=intervalles[i]
        sgmtt=''    
    for k in range(int(float(n)/intervalles[i])):
        new_segmentation=Segmentation(serie=ld.load_list(filename, True)[deb:fin],
                           order=4,activity=activity,automatic=automatic, compute=False)
        (serie,order,activity,sd_serie,breaking_points,segments,average_segment,
         dispersion_segment)=union(sgmtt,new_segmentation)
        sgmtt=Segmentation(serie=serie,order=order,activity=activity,sd_serie=sd_serie,
                        breaking_points=breaking_points,segments=segments,
                        average_segment=average_segment,
                        dispersion_segment=dispersion_segment)
        deb+=intervalles[i]
        fin+=intervalles[i]
        sgmtt.store(filepath)
        sgmtt.display_segmentation(filepath)
        save(k+1, filepath+"\\deb.txt")
        
        
    sgmtt.display_segmentation(filepath)
    
    
def compute_one_segmentation(activity, deb, fin, automatic=True):
    """Compute the segmentation of an USC data class.
    
    .. warning:: This function is deprecated and is usable only with the USC data.
    .. seealso:: That is the old version of :func:`segmentation_data_worker_j`.
    """
    filename="USC-Activities\\{0}\\SSQserie.csv".format(activity)
    if fin==0:
        sgmtt=Segmentation(serie=ld.load_list(filename, True),
                           order=4,activity=activity,automatic=automatic)
    else:
        sgmtt=Segmentation(serie=ld.load_list(filename, True)[deb:fin],
                           order=4,activity=activity,automatic=automatic)
    if automatic:
        filepath="USC-Activities\\{0}\\automatic".format(activity)
    else:
        filepath="USC-Activities\\{0}\\manual".format(activity)
    sgmtt.store(filepath)
    sgmtt.display_segmentation(filepath)
    
def compute_one_template(files_node, classes, j, iteration):
    """Re-computes the template and its dispersion
    
    Parameters
    ----------
    files_node: string
        The node of the files, that is to say the path to the file where all the recognition
        information are saved.
    classes_name : string
        name of the class  in construction
    j: int
        the number of the classe to be re-computed.
    iteration: int
        the number of iteration in the DBA algorithm.
    """   
    templates_library=ld.load(files_node+"\\templates_library.txt")
    sgmtt=ld.load_segmentation(files_node+"\\"+classes[j])
    sgmtt.recompute_average_segment(iteration)
    templates_library[j]=[sgmtt.get_average_segment(), sgmtt.get_dispersion_segment()]
    sgmtt.store(files_node+"\\"+classes[j])
    save(np.array(templates_library),files_node+"\\templates_library.txt")
    sgmtt.display_segmentation(files_node+"\\"+classes[j])

        
def compute_all_template(files_node, classes, iteration):
    """Re-computes the template and its dispersion of each class.
    
    Parameters
    ----------
    files_node: string
        The node of the files, that is to say the path to the file where all the recognition
        information are saved.
    classes_name : string
        name of the class  in construction
    iteration: int
        the number of iteration in the DBA algorithm
    """
    for j in range(12):
        compute_one_template(files_node, classes, j, iteration)

