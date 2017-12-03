import numpy as np
import pandas as pd
from pandas import DataFrame, Series
# import matplotlib.pyplot as plt
from numpy.random import randn
import random
d = {'x': np.random.randn(50), 'y': np.random.randn(50)}
data = DataFrame(d)
first_index = np.random.randint(50, size=1)[0]
second_index = np.random.randint(50, size=1)[0]
particle_first = DataFrame(data.loc[first_index].rename('0')).T
particle_second = DataFrame(data.loc[second_index].rename('1')).T
particle = particle_first.append(particle_second)
clusterAssment = np.zeros((50, 2))


def EulerDistance(arrA, arrB):
    return np.math.sqrt(sum(np.power(arrA-arrB, 2)))


for i in range(50):
    minDist = np.inf
    minIndex = -1
    for j in range(2):
        value = data.loc[i]
        particle_dot = particle.iloc[j]
        dist_value = EulerDistance(particle_dot, value)
        if dist_value < minDist:
            minDist = dist_value
            minIndex = j+1
    if minDist == 0:
        print('距离%f' % minDist)
        print(i)
    if clusterAssment[i, 0] != minIndex:
        clusterChanged = True
        clusterAssment[i, :] = minIndex, minDist**2

for i in range(2):
    index_all = clusterAssment[:, 0]
    value = np.nonzero(index_all == i+1)
    sample_dot = data.loc[value]
    print(np.mean(sample_dot, axis=0))
