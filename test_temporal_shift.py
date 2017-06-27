# coding: utf-8
import temporal_shift as sft

from math import sin,cos,pi
from load import load_list, load_segments
from computation import normalization

from variables import activities

def template(x):
    if 0<x<2*pi:
        return sin(x-1)
    return -0.6 

different=lambda x:cos(x*x)
    
deformation=lambda x,w0,w1,w2,w3: w1*template(w3*x+w2)+w0

from numpy import arange
from matplotlib.pyplot import plot, legend,show, title
    
def deformation_visualisation(w0,w1,w2,w3):
    X=arange(0,5,0.1)
    plot([template(x) for x in X],label="original")
    plot([deformation(x,w0,w1,w2,w3) for x in X], label="deformation")
    title("Visualisation of the deformation")
    legend()
    show()
    sft.compute_temporel_shift_parameters([template(x) for x in X], [deformation(x,w0,w1,w2,w3) for x in X], plot=True)
    
def plot_three_cases():
    X=arange(-1,1,0.1)
    
    print("Delay")
    X_grand=arange(-1.5,0.5,0.1)
    plot([template(x) for x in X],label="original")
    plot([deformation(x,0,1,0,1) for x in X_grand], label="deformation")
    title("Delay")
    legend()
    show()
    sft.compute_temporel_shift_parameters([template(x) for x in X], [deformation(x,0,1,0,1) for x in X_grand], plot=True)
    
    print("Advanced")
    X_grand=arange(-0.5,1.5,0.1)
    plot([template(x) for x in X],label="original")
    plot([deformation(x,0,1,0,1) for x in X_grand], label="deformation")
    title("Advanced")
    legend()
    show()
    sft.compute_temporel_shift_parameters([template(x) for x in X], [deformation(x,0,1,0,1) for x in X_grand], plot=True)
    
    print("Different")
    X_grand=arange(-0.5,1.5,0.1)
    plot([template(x) for x in X],label="original")
    plot([different(x) for x in X_grand], label="deformation")
    title("Different")
    legend()
    show()
    sft.compute_temporel_shift_parameters([template(x) for x in X], [different(x) for x in X_grand], plot=True)
    
def exemple_speed_variations():
    
    X=arange(-1,1,0.1)
    
    (w0,w1,w2,w3)=(0,1,0,0.8)
    print("Premier exemple : pas de décalage, par contre plus lent")
    X_grand=arange(-1,2,0.1)
    plot([template(x) for x in X],label="original")
    plot([deformation(x,w0,w1,w2,w3) for x in X_grand], label="deformation")
    titre="w0="+str(w0)+", w1="+str(w1)+", w2="+str(w2)+", w3="+str(w3)
    title(titre)
    legend()
    show()
    sft.compute_temporel_shift_parameters([template(x) for x in X], [deformation(x,w0,w1,w2,w3) for x in X_grand],plot=True)
    show()
    
    (w0,w1,w2,w3)=(0,1,0,1.1)
    print("Premier exemple : pas de décalage, par contre plus rapide")
    X_grand=arange(-1,2,0.1)
    plot([template(x) for x in X],label="original")
    plot([deformation(x,w0,w1,w2,w3) for x in X_grand], label="deformation")
    titre="w0="+str(w0)+", w1="+str(w1)+", w2="+str(w2)+", w3="+str(w3)
    title(titre)
    legend()
    show()
    sft.compute_temporel_shift_parameters([template(x) for x in X], [deformation(x,w0,w1,w2,w3) for x in X_grand], plot=True)
    
    
def clean_example(w2=1,w3=-1):  
    
    my_deformation=lambda x: template(w2*x+w3)
    
    X_template=arange(0,5,0.1)
    X_serie=arange(0,6,0.1)
    plot([template(x) for x in X_template],label="Template")
    plot([my_deformation(x) for x in X_serie], label="Serie")
    legend()
    show()   
    (w2,w3,score_min,rising_too_strong, tray_too_long)=sft.compute_temporel_shift_parameters(
        [template(x) for x in X_template], [my_deformation(x) for x in X_serie],plot=True)
    
def plot_all_example_path(template):
    from load import load_list
    list_name=["test_serie"+str(i) for i in range(1,7)]
    list_series=[load_list(l) for l in list_name]
    for s in list_series:
        (w2,w3,score_min,rising_too_strong, tray_too_long)=sft.compute_temporel_shift_parameters1(
            template, s)#,plot=True)
        
def compute_w2_w3_for_template_i_serie_j(i,j,start=10):
    template=load_list("USC-Activities\\{0}\\manual\\average_segment.csv".format(activities[i]))
    serie=load_list("USC-Activities\\{0}\\SSQserieTotale.csv".format(activities[j]))[start:len(template)+20+start]

    sft.compute_temporel_shift_parameters(template, normalization(serie), True)    
    
        
    
    
    