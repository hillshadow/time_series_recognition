# coding: utf-8
import numpy as np
from math import sin
import matplotlib.pyplot as plt


# def test1():
#     X=np.linspace(0,5,500)
#     sin1=[sin(x) for x in X]
#     sin2=[sin(1.1*x)*1.1 for x in X]
#     #sin3=[sin(x)*2 for x in X]
#     sin4=[sin(x)+0.1 for x in X]
#     sin5=[sin(x+0.1) for x in X]
#     mes_sin=[sin1, sin2, sin4, sin5]
#     for s in mes_sin:
#         plt.plot(X,s,'--')
#     medoid=dba.DBA(mes_sin,20)
#     medoid=sgmc.smoothing(medoid,3)
#     plt.plot(X,sgmc.smoothing(medoid,3), '-x')
#     plt.plot(X,sgmc.smoothing(medoid,3), '-x')
#     plt.plot(X,sgmc.smoothing(medoid,3), '-x')
#     plt.show()
#     
# def test2():
#     X=np.linspace(0,5,500)
#     lin1=[x for x in X]
#     lin2=[1.1*x for x in X]
#     lin3=[x+2 for x in X]
#     lin4=[(x+0.1)*1.1 for x in X]
#     lin5=[1.02*x+0.1 for x in X]
#     mes_lin=[lin1, lin2, lin3, lin4, lin5]
#     for s in mes_lin:
#         plt.plot(X,s,'--')
#     medoid=sgmc.smoothing(dba.DBA(mes_lin,20),1)
#     plt.plot(X,medoid, '-x')
#     plt.show()
#     medoid=sgmc.smoothing(dba.DBA(mes_lin,20),2)
#     plt.plot(X,medoid, '-x')
#     plt.show()
#     medoid=sgmc.smoothing(dba.DBA(mes_lin,20),3)
#     plt.plot(X,medoid, '-x')
#     plt.show()

def test_bidon():
    assert 1==2
