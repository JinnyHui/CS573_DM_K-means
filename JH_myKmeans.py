# Jingyi Hui, 09/08/2018
# CSCI573 Homework 1
# Implementation of K-means algorithm

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def generator(k, n):
    """
    generate k random numbers represent the line numbers in the data file
    :param k: the desired number of clusters
    :param n: the number of lines of program input data file
    :return centroid: a list of integers
    """

    # error handling
    if k > n:
        print("The number of clusters should be no larger than your data points!")
        sys.exit()
    else:
        centroids_gen = random.sample(range(1, n+1), k)
        return centroids_gen


def reader(file, k):
    """
    read from the text file if
    :param file: the txt file in which the centroids are read from
    :param k: the desired number of cluster
    :return: a list of integers
    """
    centroids_read = []
    f = open(file, 'r')
    for line in f:
        centroids_read.append(int(line))
    if len(centroids) != k:
        print('The number you input does not equal to the number of centroids in file!')
        sys.exit()
    else:
        return centroids_read


def SSE(cluster, dataset):
    """
    add the square of difference between mean and each point in the cluster
    :param cluster: each final cluster
    :param dataset: the whole dataset
    :return: SSE score
    """

    SSE_cluster = 0
    for point in cluster.pointAssign:
        SSE_cluster += np.linalg.norm(cluster.centroid, dataset[point - 1])**2
    return SSE_cluster


def mean(cluster, dataset):
    """
    get the index of each point in the cluster and calculate the mean
    :param cluster: the cluster object
    :param dataset: the whole dataset
    :return: the mean of the cluster
    """
    index = np.array(cluster.pointAssign) - 1
    cluster_array = dataset[index]
    cluster_mean = np.mean(cluster_array, axis=0)
    return cluster_mean


class Cluster(object):
    def __init__(self):
        self.centroid = None
        self.pointAssign = []  # a list of integers representing each line number in the dataset
        self.pre_mean = None
        self.mean = None
        self.SSE = 0

    def display(self):
        print('Mean: ' + str(self.mean))
        print('Cluster assignment: ' + str(self.pointAssign))
        print('Size of cluster: ' + str(len(self.pointAssign)))


##
dataFile = sys.args[0]  # get the name of the input file
clusterNumber = int(sys.args[1])  # get the number of clusters
centroidFile = sys.args[2]  # get the name of the file listing centroids if any
dataPointsRaw = np.genfromtxt(dataFile, delimiter=',', dtype=None)  # ndarray with label
dataPoints = dataPointsRaw[:][:-2]  # dataset w/o label
pointNumber, dimension = dataPoints.shape  # get the shape of the dataset
SSE_sum = 0  # the sum of SSE of each cluster
epsilon = 0.0001
change = float('inf')  # initiate the change with +inf
iteration = 0
clusterList = []  # store a list of k cluster objects

# get the centroids either from input file or randomly generated
if centroidFile is None:
    centroids = generator(clusterNumber, pointNumber)
else:
    centroids = reader(centroidFile, clusterNumber)


# initiate the clusters with the centroid
for i in centroids:
    c = Cluster()
    c.centroid = dataPoints[int(i)-1]
    clusterList.append()

# iterations of k-means algorithm; terminate when change <= epsilon
while change > epsilon:
    iteration += 1
    change = 0
    for i in range(len(dataPoints)):  # assign all the points to clusters
        dist_min = float('-inf')
        cluster_index = -1
        for j in range(clusterNumber):
            a = dataPoints[i]
            b = clusterList[j].centroid
            if np.linalg.norm(a, b) < dist_min:
                cluster_index = j
        clusterList[cluster_index].pointAssign.append(i+1)

    for i in range(clusterNumber):  # calculate the mean and assign it as the new centroid
        clusterList[i].pre_mean = clusterList[i].centroid
        clusterList[i].mean = mean(clusterList[i], dataPoints)
        clusterList[i].centroid = clusterList[i].mean

    for i in range(clusterNumber):  # calculate the sum of change of centroids
        change_root = np.linalg.norm(clusterList[i].pre_mean, clusterList[i].centroid)
        change += change_root**2


# display the result
print('1. The number of points: ' + str(pointNumber) + ';')
print('   The number of dimensions: ' + str(dimension-1) + ';')
print('   The value of k: ' + str(clusterNumber))
print('2. The number of iterations: ' + str(iteration))
print('3. Cluster data:')
for i in range(clusterNumber):  # display each cluster
    clusterList[i].SSE = SSE(clusterList[i], dataPoints)
    SSE_sum += clusterList[i].SSE
    print('Cluster ' + str(i+1) + ' :')
    clusterList[i].display()
print('4. The SSE score is :' + str(SSE_sum))
