# coding: utf-8

from storage.load import load_list, load_segmentation
from storage.save import save_list, save_segments
from segmentation.segmentation import Segmentation
from reconnaissance.reconnaissance import recognize
import os.path


activities=["WalkingForward","WalkingLeft","WalkingRight","WalkingUpstairs",
            "WalkingDownstairs","RunningForward","JumpingUp","Sitting","Standing",
            "Sleeping","ElevatorUp","ElevatorDown"]

def prepare_test_data():
    for j in range(12):
        prepare_test_data_j(j)
        
def prepare_test_data_j(j):
        print("Activity :",j)
        filename="USC-Activities\\{0}\\SSQserie.csv".format(activities[j])
        print(os.path.abspath(filename))
        serie=load_list(filename, True)
        sgmtt=Segmentation(serie=load_list(filename, True),order=4,activity=activities[j],automatic=False, compute=False)
        filepath="USC-Activities\\{0}\\test".format(activities[j])
        sgmtt.store(filepath)
        sgmtt.display_segmentation(filepath)
        rec=[0 for act in activities]
        rec[j]=len(sgmtt.get_breaking_points())-1
        rows=[[activities[i], rec[i]] for i in range(len(rec))] 
        #TODO : gérer le problème : écriture dans une seule case !
        save_segments(rows, filepath+"\\info.txt")
        
def recompute_stat(j):
    sgmtt=load_segmentation("USC-Activities\\{0}\\test".format(activities[j]))
    rec=[0 for act in activities]
    rec[j]=len(sgmtt.get_breaking_points())-1
    save_list(rec, "USC-Activities\\{0}\\test".format(activities[j])+"\\info.csv")
        
