# coding: utf-8

# TODO : regarder l'échelle !
# TODO : 


"""
@author: Philippenko

This script has been written so has to prepare the USC data.
That means : 
    1) gathering the time series by activities
    2) computing the SSQ series
    3) save the just computed SSQ serie in the appropriate file
     under the name USC-Activities\The_Activity\SSQserie.csv
Run time : TODO
"""

from timeit import default_timer as timer
import pandas as pd
import mat4py as mat
from save import save_list  

activities=["WalkingForward","WalkingLeft","WalkingRight","WalkingUpstairs",
            "WalkingDownstairs","RunningForward","JumpingUp","Sitting","Standing",
            "Sleeping","ElevatorUp","ElevatorDown"]

def prepare_data():   
    
    start = timer()
    print("#### \t Starting ! ")  
    
    # For all the activities:
    for j in range(12):
        data=pd.DataFrame({})
        print("###### Activity : ", activities[j])
        # For the 14 subjects
        for i in range(1,15):
            print("### Sujet ", i)
            # For the 5 experiments
            for k in range(1,6):
                filename = "USC-HAD\\Subject{0}\\a{1}t{2}.mat".format(i,j+1,k)
                data = pd.concat([data,pd.DataFrame(mat.loadmat(filename))])
        #Reindex
        data=data.reset_index(drop=True)
        data = data.drop(['activity', 'age', 'date', 'height', 'sensor_location', 'sensor_orientation', 'title', 'trial', 'version', 'weight'], 1)
        
        print("SSQ csegmentation_construction")
        col=[]
        for i in range(len(data)):
                col.append(data["sensor_readings"][i][0]**2+data["sensor_readings"][i][1]**2+data["sensor_readings"][i][2]**2)
        data["SSQ"]=col

        serie=list(data["SSQ"])
        save_list(serie,"USC-Activities\{0}\SSQserieTotale.csv".format(activities[j]))
    
    end=timer()
    print("Run time : ", start-end)
        
if __name__ == '__main__':
    prepare_data()
