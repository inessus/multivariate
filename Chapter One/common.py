import pandas as pd
import numpy as np
from functools import reduce
from operator import *

def theta(P1, P2):
    return 180*np.arccos(np.dot(P1, P2)/(D(P1)*D(P2)))/np.pi

def D(a):
    return np.sqrt(reduce(add, map(lambda x:x*x, a)))

def X(a, n):
    x = np.zeros(n, dtype=np.float)
    l = len(a[0])
    return [reduce(lambda a,b:a+b, a[i])/l for i in range(n)]
    

def Sn(a, n):
    sn = np.zeros((n, n), dtype=np.float)
    x = a.mean(1)
    l = len(a[0])
    for i in range(n):
        for j in range(n):
            sn[i][j] = reduce(lambda a,b:a+b, map(lambda a: a[0]*a[1], zip(map(lambda a:a-x[i], a[i]), map(lambda a:a-x[j], a[j]))))/l
    return sn

def R(a, n):
    sn = np.zeros((n, n), dtype=np.float)
    x = a.mean(1)
    l = len(a[0])
    for i in range(n):
        for j in range(n):
            sn[i][j] = reduce(lambda a,b:a+b, map(lambda a: a[0]*a[1], zip(map(lambda a:a-x[i], a[i]), map(lambda a:a-x[j], a[j]))))/l
    
    r = np.zeros((n, n), dtype=np.float)
    for i in range(n):
        for j in range(n):
            r[i][j] = sn[i][j]/(np.sqrt(sn[i][i])*np.sqrt(sn[j][j]))
    return r