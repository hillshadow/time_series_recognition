'''
Created on 17 juil. 2017

@author: Philippenko
'''

import matplotlib.pyplot as plt
import numpy as np

from plotting import save_or_not


def plot_features_elimination(rates, features_range, x_label,title, classifiers, save=True):
    for i in range(len(classifiers)):
        r=rates[i]
        clf=classifiers[i]
        plt.subplot(121)
        plt.plot(features_range, np.array(r)[:,0], label=str(clf)[0:9])
        plt.legend(loc="best")
        plt.ylabel("Recognition score of the steps")
        plt.xlabel(x_label)
        plt.title("Step")
        plt.ylim(0.0, 110)
        plt.subplot(122)
        plt.plot(features_range, np.array(r)[:,1], label=str(clf)[0:9])
        plt.legend(loc="best")
        plt.ylabel("Recognition score of the non-steps")
        plt.xlabel(x_label)
        plt.title("Not Step")
        plt.ylim(0.0, 110)
    plt.suptitle(title)
    save_or_not(save, "report_pictures\\select_features\\"+title+".png")
    
def plot_grouped_bar(combinaisons, features_numbers, my_classifiers, mean_rates):
    ind = np.arange(1,len(my_classifiers)+1)
    fig=plt.figure()
    axes=[plt.subplot2grid((3,3), (0,0), rowspan=1, colspan=1), 
          plt.subplot2grid((3,3), (0,1), rowspan=1, colspan=1),
          plt.subplot2grid((3,3), (0,2), rowspan=1, colspan=1),
          plt.subplot2grid((3,3), (1,0), rowspan=1, colspan=1),
          plt.subplot2grid((3,3), (1,1), rowspan=1, colspan=1),
          plt.subplot2grid((3,3), (1,2), rowspan=1, colspan=1),
          plt.subplot2grid((3,3), (2,0), rowspan=1, colspan=3)]
    
    if np.array(mean_rates).shape[2]>2:
        width = 0.15
        for i in range(np.array(mean_rates).shape[1]):
            r=np.array(mean_rates)[:,i,:]
            ax=axes[i]
            p0 = ax.bar(ind, np.array(r)[:,0], width, color=(1,0,0,0.5))
            p1 = ax.bar(ind+width, np.array(r)[:,1], width, color=(0,1,0,0.5))
            p2 = ax.bar(ind+2*width, np.array(r)[:,2], width, color=(0,0,1,0.5))
            p3 = ax.bar(ind+3*width, np.array(r)[:,3], width, color=(0.7,0.5,0.5,0.5))
            for num_clf in range(len(features_numbers)):
                num_comb=features_numbers[num_clf]-1
                if num_comb==i:
                    p0[num_clf].set_color((1,0,0,1))
                    p1[num_clf].set_color((0,1,0,1))
                    p2[num_clf].set_color((0,0,1,1))
                    p3[num_clf].set_color((0.7,0.5,0.5,1))
            autolabel(p0, ax)
            autolabel(p1, ax)
            autolabel(p2, ax)
            autolabel(p3, ax)
            ax.set_ylabel('AUC')
            ax.set_ylim(0.65,1)
            ax.set_title(str(combinaisons[i]))
            ax.set_xticks(ind)
            ax.set_xticklabels([str(my_classifiers[k])[0:5] for k in range(len(my_classifiers))])
            ax.legend((p0[0], p1[0], p2[0], p3[0]), ('Precision', 'Recall', 'F-Measure', 'Error of 2nd kind'), prop={'size':5}, loc="best")   
    else:
        width = 0.35
        for i in range(np.array(mean_rates).shape[1]):
            r=np.array(mean_rates)[:,i,:]
            print(np.array(r)[0,:])
            print(ind)
            print(np.array(r)[1,:])
            ax=axes[i]
            p0 = ax.bar(ind, np.array(r)[:,0], width, color=(1,0,0,0.5))
            p1 = ax.bar(ind+width, np.array(r)[:,1], width, color=(0,1,0,0.5))
            for num_clf in range(len(features_numbers)):
                num_comb=features_numbers[num_clf]-1
                if num_comb==i:
                    p0[num_clf].set_color((1,0,0,1))
                    p1[num_clf].set_color((0,1,0,1))
            autolabel(p0, ax)
            autolabel(p1, ax)
            ax.set_ylabel('AUC')
            ax.set_ylim(0.65,1)
            ax.set_title(str(combinaisons[i]))
            ax.legend((p0[0], p1[0]), ('Train', 'Test'), prop={'size':10},loc='lower center')     
    ax.set_xticks(ind)
    ax.set_xticklabels([str(my_classifiers[k])[0:9] for k in range(len(my_classifiers))])         
    plt.show()
    
def autolabel(rects,ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 0.97*height,
                '%d' % (int(height*100)),
                ha='center', va='bottom')