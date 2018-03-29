# coding: utf-8

from storage.load import load_list, load_segmentation
from storage.save import save_list, save
from segmentation import Segmentation
import os.path


classes=["WalkingForward","WalkingLeft","WalkingRight","WalkingUpstairs",
            "WalkingDownstairs","RunningForward","JumpingUp","Sitting","Standing",
            "Sleeping","ElevatorUp","ElevatorDown"]

def prepare_test_data():
    for j in range(12):
        prepare_test_data_j(j)
        
def prepare_test_data_j(j):
        print("Activity :",j)
        filename="USC-Activities\\{0}\\SSQserie.csv".format(classes[j])
        print(os.path.abspath(filename))
        serie=load_list(filename, True)
        sgmtt=Segmentation(serie=serie,order=4,activity=classes[j],automatic=False, compute=False)
        filepath="USC-Activities\\{0}\\test".format(classes[j])
        sgmtt.store(filepath)
        sgmtt.display_segmentation(filepath)
        rec=[0 for act in classes]
        rec[j]=len(sgmtt.get_breaking_points())-1
        rows=[[classes[i], rec[i]] for i in range(len(rec))] 
        #TODO : gérer le problème : écriture dans une seule case !
        save(rows, filepath+"\\info.txt")
        
def recompute_stat(j):
    sgmtt=load_segmentation("USC-Activities\\{0}\\test".format(classes[j]))
    rec=[0 for act in classes]
    rec[j]=len(sgmtt.get_breaking_points())-1
    save_list(rec, "USC-Activities\\{0}\\test".format(classes[j])+"\\info.csv")
        
