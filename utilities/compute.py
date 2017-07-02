'''
Created on 1 juil. 2017

@author: Philippenko
'''

from segmentation.segmentation import Segmentation
from utilities.variables import activities
from storage import load as ld
import os
from utilities.variables import intervalles
from segmentation.segmentation_construction import union
from storage.save import save_segments


def compute_all_segmentation(automatic=True):  
    '''
    @return: generate segmentations's data and picture
    This function serve to compute automatically or manually the segmentation,
    and by this way the average segment, of all activities given a set of points.
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
    
def compute_one_segmentation(activity, deb, fin, automatic=True):
    """
    @return: generate the data and pictures of one segmentation
    This function serve to compute automatically or manually the segmentation,
    and by this way the average segment, of the specified activity given a set of points.
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
    
def compute_one_template(iteration, j, automatic=True):
    """
    @return: generate the template of one activities
    This function serve to re-compute the template with the 
    manual or automatic segmentation
    """
    activity=activities[j]
    print("Activity :",activity)
    filename=ld.get_filename(activity, automatic)
    print(filename)
    print(os.path.abspath(filename))    
    sgmtt=ld.load_segmentation(filename)
    sgmtt.recompute_average_segment(iteration)
    sgmtt.store(filename)
    sgmtt.display_segmentation(filename)
        
def compute_all_template(iteration, automatic=True):
    """
    @return: generate the template of all the activities
    This function serve to re-compute all the template with the 
    manual or automatic segmentation
    """
    for j in range(12):
        compute_one_template(iteration, j, automatic)
        



def compute_complete_segmentation_of_i(i,started=True,automatic=False):
    """
    Computes the complete segmentation of the whole serie : for each trials (5) and each subjects (14)
    
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
    activity=activities[i]
    print("## \t Activity :",activity) 
    filename="USC-Activities\\{0}\\SSQserieTotale.csv".format(activity)
    filepath="USC-Activities\\{0}\\complete".format(activity)
    serie=ld.load_list(filename, True)
    n=len(serie)
    if started==True:
        k_deb=ld.load_segments(filepath+"\\deb.txt")
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
        save_segments(k+1, filepath+"\\deb.txt")
        
        
    sgmtt.display_segmentation(filepath)
    
