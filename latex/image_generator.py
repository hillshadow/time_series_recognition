# coding: utf-8

'''
Created on 7 juil. 2017

@author: Philippenko
'''

from storage.load import load_segments, load_double_list, load_serie, load_list
from classifier import classifiers
from utilities.variables import activities

def write_CV_image_including(clf_type="binary"):
    slide=""
    for i in range(len(classifiers)):
        clf=classifiers[i]
        sub_slide="""
        \onslide<"""+str(2*i+1)+"""> 
        \\begin{figure}[H]
            \\begin{minipage}[c]{.46\linewidth}
                  \includegraphics[scale=0.35,center]{../report_pictures/"""+clf_type+"""_classification/Auto-recognition_performance_for_"""+str(clf)[0:9]+""".png}
            \end{minipage} \hfill
            \\begin{minipage}[c]{.46\linewidth}
                \includegraphics[scale=0.35,center]{../report_pictures/"""+clf_type+"""_classification/Hold-Out_recognition_performance_for_"""+str(clf)[0:9]+""".png}
            \end{minipage}
            \caption{Auto and Hold-Out Validation for """+str(clf)[0:9]+"""}
            \label{CV_"""+str(clf)[0:9]+"""}
        \end{figure}   
   
        \onslide<"""+str(2*(i+1))+"""> 
        \\begin{figure}[H]
            \includegraphics[scale=0.46,center]{../report_pictures/"""+clf_type+"""_classification/3-fold_validation_for_"""+str(clf)[0:9]+""".png}
        \caption{3-Fold Validation for """+str(clf)[0:9]+"""}
        \label{3Fold_"""+str(clf)[0:9]+"""}
        \end{figure} 
        """
        slide=slide+sub_slide
        
    write_full_text(slide,'docs\\sub_tex\\'+clf_type+'_CV_image_including.tex')
    
def write_tables_CV_recap():
    slide=""    
    my_types=["binary", "multi"]
    for k in range(len(my_types)):
        clf_type=my_types[k]
        rates=load_segments("docs//performances//"+clf_type+"_rates.txt")
        
        col_def="|"
        for i in range(rates.shape[1]+1):
            col_def+="c|"
        col_names="Activity & "
        for clf in classifiers:
            col_names=col_names+str(clf)[0:9]+" & "
        col_names=col_names[:len(col_names)-2]
        
        lines=""
        if rates.shape[0]==2:
            the_activities=["Step", "Not Step"]
        else:
            the_activities=activities[:rates.shape[0]]
        for i in range(rates.shape[0]):
            sub_lines=the_activities[i]+" (\%) & "
            for j in range(rates.shape[1]):
                sub_lines=sub_lines+str(int(rates[i,j]*10)/10.0)+" & "
            sub_lines=sub_lines[:len(sub_lines)-2]
            sub_lines+=" \\\\"
            lines=lines+sub_lines+"\n \t\t\hline \n \t\t"
            
        if k==0:    
            sub_slide="""
            \onslide<"""+str(k+1)+""">
            The following tables summarize the 3-fold validation percentages of success recognition for each activity and for each classifier.
    
            \medskip
            
            Clearly, the random forest give the best results : a hight score on the both classes.
            
            \medskip
            
            On the other hand, the k-nearest neighbors classifier shows an huge over-fitting and should be avoid.
            """+write_table(col_def, col_names, lines)
        else:
            sub_slide="""
            \onslide<"""+str(k+1)+""">"""+write_table(col_def, col_names, lines)+"""
            
            \medskip
            
            At the first glance, the results looks like to be very poor for the multi-classes classification with almost all 
            rates below 65\%. 
            
            \medskip
            
            However, if we consider once again the confusion matrix and the four clusters cited above, the rates skyrocket to almost 
            95\% for the SVM, Gaussian, Decision Tree and Random Forest classifiers.

            """
        slide+=sub_slide
    write_full_text(slide,'docs\\sub_tex\\tables_CV_recap.tex')
    
    
def write_full_text(text,filename):
    f = open(filename,'w')
    f.write(text)
    f.close
    
def write_line(liste, activity):
    sub_lines=activity+" & "
    for j in range(len(liste)):
        sub_lines=sub_lines+str(liste[j])+" & "
    sub_lines=sub_lines[:len(sub_lines)-2]
    sub_lines+=" \\\\"
    return sub_lines

def write_table(col_def, col_names, lines):
    table = """
    \\begin{table}[h!]
    \centering
    \\resizebox{\\textwidth}{!}{
    \\begin{tabular}{"""+col_def+"""}
        \hline
        """+col_names+""" \\\\
        \hline
        """+lines+"""
    \end{tabular}}
    \end{table}
    """
    return table
    
def write_recap_continuous_recognition():
    step, no_step =load_double_list("forecsys_data/performance_recognition_continue.txt")
    template=load_list("forecsys_data/juncture/average_segment.csv")
    t=len(template)
    
    col_def="|"
    for i in range(len(step)+1):
        col_def+="c|"
    col_names="Activity & "
    for clf in classifiers:
        col_names=col_names+str(clf)[0:9]+" & "
    col_names=col_names[:len(col_names)-2]
    
    lines=""
    lines=lines+write_line(step, "Step (number)")+"\n \t\t\hline \n \t\t"
    lines=lines+write_line(no_step, "Not Step (number)")+"\n \t\t\hline \n \t\t"
    
    n=len(load_list("forecsys_data\\juncture\\breaking_points.csv"))-1
    m=len(load_serie("forecsys_data\\other_classe"))
    lines_norm=""
    lines_norm=lines_norm+write_line([int(1000.0*s/n)/10.0 for s in step], "Step (\%)")+"\n \t\t\hline \n \t\t"
    lines_norm=lines_norm+write_line([int(1000.0*s/(m/t))/10.0 for s in step],"Not Step (\%)")+"\n \t\t\hline \n \t\t"
        
        
    text=write_table(col_def, col_names, lines)
    text=text+write_table(col_def, col_names, lines_norm)
    
    write_full_text(text, 'docs\\sub_tex\\recap_continuous_recognition.tex')
    


    