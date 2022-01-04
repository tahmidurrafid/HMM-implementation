import numpy
import math
from copy import deepcopy

from numpy.core.numeric import ones


class GDistribution:
    def __init__(self, mean, stDeviation):
        self.mean = mean
        self.stDeviation = stDeviation
    def print(self):
        print("MEAN: " + str(self.mean) + ", stDeviation: " + str(self.stDeviation))

data = numpy.loadtxt("Input/data.txt")
paramFile = open("Input/parameters.txt.txt", "r")
stateCount = int(paramFile.readline())
transitionMatrix = []
distributions = []

for i in range(0, stateCount):
    line = paramFile.readline()
    arr = list(map(float, line.split()))
    transitionMatrix.append(arr)
transitionMatrix = numpy.array(transitionMatrix)

matrix = [list(map(float, line.split())) for line in paramFile.read().split("\n")]
paramFile.close()
for i in range(0, stateCount):
    distributions.append(GDistribution(matrix[0][i], math.sqrt(matrix[1][i]) ))

def calcStationaryDistribution(tranMatrix):
    A = deepcopy(tranMatrix)
    A = numpy.transpose(A)
    A[0] = numpy.array(ones(len(A)))
    for i in range(1, len(A)):
        A[i][i] -= 1
    # print(A)
    B = numpy.zeros(len(A))
    B[0] = 1
    # print(B)
    X = numpy.linalg.inv(A).dot(B)
    print(X)

calcStationaryDistribution(transitionMatrix)