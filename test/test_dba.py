'''
Created on 29 juin 2017

@author: Philippenko
'''

from storage import load as ld
from time import time
from dba import DTWCumulMat, optimal_path, fastdtw, dtw, distanceTo

def test_vitesse_dtw():
    pattern=ld.load_list("test_pattern2")
    series=ld.load_list("test_serie1")
    
    start=time()
    for i in range(1000):
        (cost,path,weight)=DTWCumulMat(medoid=pattern,s=series)
        (opt_path,weight_opt_path)=optimal_path(len(pattern), len(series),path,weight)
    end=time()
    print("Our DTW : ", end-start, "s")
    
    start=time()
    for i in range(1000):
        dist, cost, acc, dump_path, weight_opt_path = fastdtw(pattern, series)
    end=time()
    print("FASTDTW : ", end-start, "s")
    
    start=time()  
    for i in range(1000):
        dist, cost, acc, dump_path, weight_opt_path = dtw(pattern, series)
    end=time()
    print("DTW : ", end-start, "s")
    

def test_dba():
    if 1:
        x=[1,1,2,3,2,0]
        y=[0,1,1,2,3,2,1]
        dist_fun=distanceTo
    elif 0: # 1-D numeric
        from sklearn.metrics.pairwise import manhattan_distances
        x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
        y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
        dist_fun = manhattan_distances
    elif 0: # 2-D numeric
        from sklearn.metrics.pairwise import euclidean_distances
        x = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [4, 3], [2, 3], [1, 1], [2, 2], [0, 1]]
        y = [[1, 0], [1, 1], [1, 1], [2, 1], [4, 3], [4, 3], [2, 3], [3, 1], [1, 2], [1, 0]]
        dist_fun = euclidean_distances
    else: # 1-D list of strings
        from nltk.metrics.distance import edit_distance
        #x = ['we', 'shelled', 'clams', 'for', 'the', 'chowder']
        #y = ['class', 'too']
        x = ['i', 'soon', 'found', 'myself', 'muttering', 'to', 'the', 'walls']
        y = ['see', 'drown', 'himself']
        #x = 'we talked about the situation'.split()
        #y = 'we talked about the situation'.split()
        dist_fun = edit_distance
        
    dist, cost, acc, path, path_weigth = fastdtw(x, y, dist_fun)
    
    
    
    print(cost)
    print(acc)
    print(path)
    print(path_weigth)

    # vizualize
    from matplotlib import pyplot as plt
    plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
    plt.plot(path[0], path[1], '-o') # relation
    plt.xticks(range(len(x)), x)
    plt.yticks(range(len(y)), y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('tight')
    plt.title('Minimum distance: {}'.format(dist))
    plt.show()
        
        
    

