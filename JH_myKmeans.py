# Jingyi Hui, 09/08/2018
# CSCI573 Homework 1
# Implementation of K-means algorithm

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def generator(k, n):
    '''
    generate k random numbers represent the line numbers in the data file
    :param k: the desired number of clusters
    :param n: the number of lines of program input data file
    :return centroid: a list of integers
    '''

    # error handling
    if k > n:
        print("The number of clusters should be no larger than your data points!")
        sys.exit()
    else:
        centroids = random.sample(range(1, n+1), k)
        return centroids


class Cluster(object):
    def __init__(self):
        self.centroid = 0
        self.pointAssign = [] # a list of integers representing each line number in the dataset
        self.dataPoints = []

    def mean(self):
        '''
        Use index to construct the cluster array and calculate the mean
        :return :the array of the mean of the cluster
        '''
        index = np.array(self.pointAssign) -1
        clusterArray = dataPoints[index]
        mean = np.mean(clusterArray, axis = 0)
        return mean

    def SSE(self):
        '''

        :return: SSE the sum of square error
        '''
        return SSE

    def display(self):
        print('Mean: ' + str(self.mean()))
        print('SSE score: ' + str(self.SSE))
        print('Cluster assignment: ' + str(self.pointAssign))
        print('Size of cluster: ' + str(len(self.pointAssign)))





##


dataFile = sys.args[0] #get the name of the input file
clusterNumber = int(sys.args[1]) #get the number of clusters
dataPointsRaw = np.genfromtxt(dataFile, delimiter=',', dtype=None) #ndarray with label
dataPoints = dataPointsRaw[:][:-2]# dataset w/o label
pointNumber, dimension = dataPoints.shape #get the shape of the dataset
epsilon = 0.0001
iteration = 0
clusterList = []
centroids = generator(clusterNumber, pointNumber)

## initiate teh clusters
for i in range(clusterNumber):



## display the result
print('1. The number of points: ' + str(pointNumber) + ';')
print('   The number of dimensions: ' + str(dimension-1) + ';')
print('   The value of k: ' + str(clusterNumber))
print('2. The number of iterations: ' + str(iteration))
print('3. Cluster data:')
for i in range(clusterNumber): #display each cluster
    print('Cluster ' + str(i+1) + ' :')
    clusterList[i].display()



