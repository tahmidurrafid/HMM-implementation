import numpy
import math

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

