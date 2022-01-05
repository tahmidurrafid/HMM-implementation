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
    def pdf(self, x):
        return (1/(self.stDeviation*math.sqrt(2*math.pi)) )*math.exp(-((x-self.mean)**2)/(2*self.stDeviation*self.stDeviation))


class HMM:
    # def __init__(self):

    def readFile(self, dataFile, paramFile):
        data = numpy.loadtxt(dataFile)
        paramFile = open(paramFile, "r")

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
            # distributions.append(GDistribution(matrix[0][i], matrix[1][i] ))
            distributions.append(GDistribution(matrix[0][i], math.sqrt(matrix[1][i]) ))
        self.stateCount = stateCount
        self.transitionMatrix = transitionMatrix
        self.distributions = distributions
        self.stationaryDistribution = self.calcStationaryDistribution()
        self.data = data

    def calcStationaryDistribution(self):
        tranMatrix = self.transitionMatrix
        A = deepcopy(tranMatrix)
        A = numpy.transpose(A)
        A[0] = numpy.array(ones(len(A)))
        for i in range(1, len(A)):
            A[i][i] -= 1
        B = numpy.zeros(len(A))
        B[0] = 1
        X = numpy.linalg.inv(A).dot(B)
        return X

    def runViterbi(self):
        dp = numpy.full((len(self.data)+1, 2), -math.inf)
        parent = numpy.full((len(self.data), 2), -math.inf)
        path = numpy.zeros(len(self.data), dtype=int)
        dp[0] = self.stationaryDistribution

        for i in range(0, self.stateCount):
            dp[0][i] = math.log(dp[0][i]) + math.log(self.distributions[i].pdf(self.data[i]))

        for j in range(1, len(self.data)):
            for i in range(0, self.stateCount):
                for k in range(0, self.stateCount):
                    val = dp[j-1][k] + math.log(self.transitionMatrix[k][i]*self.distributions[i].pdf(self.data[j]))
                    if(val >= dp[j][i]):
                        dp[j][i] = val
                        parent[j][i] = k

        lastValue = -math.inf
        for i in range(0, self.stateCount):
            if(dp[len(self.data)-1][i] > lastValue):
                lastValue = dp[len(self.data)-1][i]
                path[len(self.data)-1] = int(i)

        for i in range(len(self.data)-1, 0, -1):
            path[i-1] = parent[i][path[i]]

        a_file = open("test.txt", "w")
        for row in path:
            if(row == 0):
                a_file.write("\"El Nino\"" + "\n" )        
            else:
                a_file.write("\"La Nina\"" + "\n" )
        a_file.close()

hmm = HMM()
hmm.readFile("Input/data.txt", "Input/parameters.txt.txt")
hmm.runViterbi()