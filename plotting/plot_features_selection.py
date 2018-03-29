'''
:author: Philippenko
:date: Juil. 2017

This module is devoted to the plotting of the features's selection process.
'''

import matplotlib.pyplot as plt
import numpy as np

from plotting import save_or_not

colors=[(1,0,0,0.5),(0,1,0,0.5),(0,0,1,0.5),(0.7,0.5,0.5,0.5)]
stronger_color= [(1,0,0,1),(0,1,0,1),(0,0,1,1),(0.7,0.5,0.5,1)]


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
    fig=plt.figure()
    axes=[plt.subplot2grid((3,3), (0,0), rowspan=1, colspan=1), 
          plt.subplot2grid((3,3), (0,1), rowspan=1, colspan=1),
          plt.subplot2grid((3,3), (0,2), rowspan=1, colspan=1),
          plt.subplot2grid((3,3), (1,0), rowspan=1, colspan=1),
          plt.subplot2grid((3,3), (1,1), rowspan=1, colspan=1),
          plt.subplot2grid((3,3), (1,2), rowspan=1, colspan=1),
          plt.subplot2grid((3,3), (2,0), rowspan=1, colspan=3)]
    
    _plot_bars(mean_rates, features_numbers, axes, combinaisons, my_classifiers, legend=None)    
    
    axes=[plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)]
    _plot_bars([ [r] for r in np.array(mean_rates)[:,6,:]] , 
              features_numbers, axes, [combinaisons[len(combinaisons)-1]], my_classifiers)
    
def _autolabel(rects,ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        pos=0.95*height
        ax.text(rect.get_x() + rect.get_width()/2., pos,
                '%d' % (int(height*100)),
                ha='center', va='bottom')

def _plot_bars(mean_rates, features_numbers, axes, combinaisons, my_classifiers, legend=None, y_label=None):
    ind = np.arange(1,len(my_classifiers)+1)
    mean_rates=np.array(mean_rates)
    number_of_bars=mean_rates.shape[2]
    number_of_cases=mean_rates.shape[1]
    print("bars=",number_of_bars)
    print("cases=",number_of_cases)
    if legend is None and number_of_bars==4:
        legend=('Precision', 'Recall', 'F-Measure', 'Error of 2nd kind')
    if legend is None and number_of_bars==3:
        legend=('Precision', 'Recall', 'F-Measure', 'Error of 2nd kind')
    elif legend is None and number_of_bars==2:
        legend=('Train', 'Test')
        
    if y_label is None and number_of_bars==2:
        y_label="AUC"
    if y_label is None and number_of_bars in [3,4]:
        y_label="SCORE"
    for i in range(number_of_cases):
        ax, my_bars=_construct_bars(i,mean_rates, axes,ind,features_numbers, number_of_bars)
        ax.set_ylabel(y_label)
        ax.set_ylim(0,1)
        ax.set_title(str(combinaisons[i]))
        ax.set_xticks(ind)
        ax.set_xticklabels([str(my_classifiers[k])[0:1] for k in range(len(my_classifiers))])
        if legend is not None:
            ax.legend([b[0]for b in my_bars], legend, prop={'size':5}, loc="best")   
    plt.show()
    
def _construct_bars(i,mean_rates, axes,ind,features_numbers, number_of_bars):
    r=mean_rates[:,i,:]
    print(i)
    ax=axes[i]
    width=0.80/number_of_bars
    my_bars=[ax.bar(ind+j*width, np.array(r)[:,j], width, color=colors[j]) for j in range(number_of_bars)]
    for num_clf in range(len(features_numbers)):
        num_comb=features_numbers[num_clf]-1
        if num_comb==i:
            for j in range(len(my_bars)):
                b=my_bars[j]
                b[num_clf].set_color(stronger_color[j])
    for b in my_bars:
        _autolabel(b, ax)
    return ax, my_bars
        