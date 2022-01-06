from os import TMP_MAX
import numpy
import math
from copy import deepcopy

from numpy.core.numeric import ones
from numpy.ma.core import std


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
        self.stationaryDistribution = self.calcStationaryDistribution(transitionMatrix)
        self.data = data

    def calcStationaryDistribution(self, tranMatrix):
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
        dp = numpy.full((len(self.data)+1, self.stateCount), -math.inf)
        parent = numpy.full((len(self.data), self.stateCount), -math.inf)
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

    def normalize(arr):
        sum = numpy.sum(arr)
        arr = arr/sum
        return arr

    def runBaumWelch(self):
        trMat = deepcopy(self.transitionMatrix)
        distrib = deepcopy(self.distributions)

        for step in range(0, 20):
            forward = numpy.full((len(self.data)+1, self.stateCount), 0.0)
            backward = numpy.full((len(self.data)+1, self.stateCount), 0.0)
            pieS = numpy.full((len(self.data)+1, self.stateCount), 0.0)
            pieSS = numpy.full((len(self.data)+1, self.stateCount, self.stateCount), 0.0)

            forward[0] = self.calcStationaryDistribution(trMat)
            for i in range(0, self.stateCount):
                forward[0][i] = forward[0][i]*(distrib[i].pdf(self.data[0]))
            forward[0] = forward[0]/numpy.sum(forward[0])

            for j in range(1, len(self.data)):
                for i in range(0, self.stateCount):
                    for k in range(0, self.stateCount):
                        forward[j][i] += forward[j-1][k]*trMat[k][i]*distrib[i].pdf(self.data[j])
                forward[j] = forward[j]/numpy.sum(forward[j])

            for i in range(0, self.stateCount):
                backward[len(self.data)-1][i] = 1/self.stateCount

            for j in range(len(self.data)-2, -1, -1):
                for i in range(0, self.stateCount):
                    for k in range(0, self.stateCount):
                        backward[j][i] += backward[j+1][k]*trMat[i][k]*distrib[k].pdf(self.data[j+1])
                backward[j] = backward[j]/numpy.sum(backward[j])

            for i in range(0, len(self.data)):
                for k in range(0, self.stateCount):
                    pieS[i][k] = forward[i][k]*backward[i][k]
                pieS[i] = pieS[i]/numpy.sum(pieS[i])
            
            for i in range(0, len(self.data)-1):
                for k in range(0, self.stateCount):
                    for l in range(0, self.stateCount):
                        pieSS[i][k][l] = forward[i][k]*trMat[k][l]*distrib[l].pdf(self.data[i+1])*backward[i+1][l]
                pieSS[i] = pieSS[i]/numpy.sum(pieSS[i])

            for k in range(0, self.stateCount):
                for l in range(0, self.stateCount):
                    trMat[k][l] = 0
                    for i in range(0, len(self.data)-1):
                        trMat[k][l] += pieSS[i][k][l]
                trMat[k] = trMat[k]/numpy.sum(trMat[k])
            
            for i in range(0, self.stateCount):
                distrib[i].mean = 0
                sum = 0
                for j in range(0, len(self.data)):
                    distrib[i].mean += pieS[j][i]*self.data[j]
                    sum += pieS[j][i]
                distrib[i].mean = distrib[i].mean/sum

            for i in range(0, self.stateCount):
                distrib[i].stDeviation = 0
                sum = 0
                for j in range(0, len(self.data)):
                    distrib[i].stDeviation += pieS[j][i]*((self.data[j] - distrib[i].mean)**2)
                    sum += pieS[j][i]
                distrib[i].stDeviation = math.sqrt(distrib[i].stDeviation/sum)

        print(trMat)
        distrib[0].print()
        distrib[1].print()
        # print(forward)
        # print(backward)
        # print(pieS)
        # out = open("out", "w")
        # for row in pieS:
        #     out.write(str(row) + "\n")
        # out.close()

hmm = HMM()
hmm.readFile("Input/data.txt", "Input/parameters.txt.txt")
hmm.runViterbi()
hmm.runBaumWelch()
