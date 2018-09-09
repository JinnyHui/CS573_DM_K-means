# Jingyi Hui, 09/08/2018
# CSCI573 Homework 1
# Implementation of K-means algorithm

import sys
import random
import numpy as np
import matplotlib.pyplot as plt


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



##


dataFile = sys.args[0] #get the name of the input file
clusterNumber = int(sys.args[1]) #get the number of clusters
dataPoints = np.genfromtxt(dataFile, delimiter=',', dtype=None) #ndarray
pointNumber =
epsilon = 0.0001
centroids = generator(clusterNumber, pointNumber)


