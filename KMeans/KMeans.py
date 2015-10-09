__author__ = 'Rohith'
import math
import arff
import numpy as np
import sys
import csv

def main(fileName):
    data, mean, std = loadData(fileName)
    centroidIndex = loadCentroidIndex('centroidIndex')
    preprocessedData = []
    for testRow in data:
        finalResult = normalizationKMean(testRow,mean,std)
        preprocessedData.append(finalResult)
    k = 1
    ErrorMean = []
    ErrorStd = []
    while(k<13):
        errorMean,errorStd = findKMeans(k,preprocessedData,centroidIndex)
        ErrorMean.append(errorMean)
        ErrorStd.append(errorStd)
        k += 1

    writeOutput(ErrorMean,ErrorStd)

def writeOutput(ErrorMean, ErrorStd):
    outputFileName = 'Output-KMean.csv'
    lengthEin = len(ErrorMean)
    with open(outputFileName, 'wb') as f:
        writer = csv.writer(f, delimiter = ',')
        for i in range(0,lengthEin):
            writer.writerow([ErrorMean[i],ErrorStd[i], (ErrorMean[i] - 2*ErrorStd[i]),(ErrorMean[i] + 2*ErrorStd[i])])


def loadData(fileName):
    data = []
    count = 0
    mean = []
    standardDeviation = []

    for row in arff.load(fileName):
        interData = []
        tempData = []
        for i in range(0,len(row)-1):
            interData.insert(i,row[i])
        data.append(interData)
        tempData = interData[:]

        if len(mean) == 0:
            mean.insert(0,tempData)
            mean = mean[0]
        else:
            for j in range(0,len(tempData)):
                mean[j] += tempData[j]
        count += 1

    intermediateArr = np.asarray(data)

    for dataCount in range(0,i+1):
        standardDeviation.insert(dataCount,np.std(intermediateArr[:,dataCount]))

    for val in range(0,len(mean)):
        mean[val] = mean[val]/float(count)
    return data, mean, standardDeviation

def loadCentroidIndex(fileName):
    f = open(fileName)
    centroidIndex = []
    for lines in f.readlines():
        if lines != '\n':
            lineData = lines.strip().split(',')

            centroidIndex = map(int,lineData)

    return centroidIndex

def normalizationKMean(rowData,mean,standardDeviation):
    zScore = []
    for i in range(0,len(rowData)):
        if standardDeviation[i] != 0.0:
            zScore.append((rowData[i] - mean[i])/standardDeviation[i])
        else:
            rowData[i] = 0
            zScore.append(rowData[i])

    return zScore

def findKMeans(k,preprocessedData,centroidIndex):
    Error = 0
    i = 0
    dupCentroidIndex = centroidIndex[:]
    ErrorStd = []
    while(i < 25):
        sse = calculateCluster(k,preprocessedData,dupCentroidIndex)
        ErrorStd.append(sse)
        Error = Error + sse

        i += 1


    # print(Error/i, "Deviation:", np.std(ErrorStd), len(ErrorStd), ErrorStd, Error)
    return Error/i, np.std(ErrorStd)

def calculateCluster(k,preprocessedData,dupCentroidIndex):
    centroid = getRandomCentroids(k,preprocessedData,dupCentroidIndex)
    count = 0
    distVal = 0
    while(count < 50): # 50 iterations here
        # print("Calling find cluster")
        flag = 0
        innerCentroids = []
        cluster = findCluster(preprocessedData,centroid)
        # print("To Calculate")
        clusterLength = len(cluster)


        for keys, values in cluster.iteritems():

            newCentroid = calculateCentroid(values)
            centroidArr = np.asarray(centroid[keys], dtype='float')
            distVal = calculateDistance(centroidArr,newCentroid)

            if distVal == 0.0:
                # print(distVal)
                flag += 1
                innerCentroids.append(centroid[keys])
            else:
                innerCentroids.append(newCentroid.tolist())

        if flag == clusterLength:
            break
        else:
            centroid = innerCentroids[:]

        count += 1

    SSE = 0

    for index in cluster:
        points = cluster[index]
        for point in points:
            centroidPoint = centroid[index]
            centroidPointMatrix = np.asarray(centroidPoint, dtype="float")
            dataPointMatrix = np.asarray(point, dtype="float")
            diffMatrix = centroidPointMatrix - dataPointMatrix
            diffMatrixTranspose = np.transpose(diffMatrix)
            resultMatrix = diffMatrix.dot(diffMatrixTranspose)
            SSE += resultMatrix


    return SSE

def calculateCentroid(clusterValues):
    # print("testCalculate")
    avgArr = np.asarray(clusterValues, dtype='float')
    meanArr = np.mean(avgArr,axis=0)

    return meanArr

def getRandomCentroids(k,preprocessedData,dupCentroidIndex):
    selectedCentroids = []
    for i in range(0,k):
        selectedCentroids.append(preprocessedData[dupCentroidIndex[0]])
        dupCentroidIndex.pop(0)

    return selectedCentroids

def findCluster(preprocessedData, centroid):
    centroidMap = {}
    for i in range(0,len(centroid)):
        centroidMap[i] = []

    for dataPoint in preprocessedData:
        closeCentroid = []
        centroidIndexVal = 0
        min = float(sys.maxint)

        for i, centroidPoint in enumerate(centroid):

            distance = calculateDistance(centroidPoint,dataPoint)
            if distance < min:
                min = distance
                centroidIndexVal = i
        valueList = centroidMap[centroidIndexVal]
        valueList.append(dataPoint)
        centroidMap[centroidIndexVal] = valueList

    return centroidMap

def calculateDistance(centroidPoint,dataPoint):
    centroidPointMatrix = np.asarray(centroidPoint, dtype="float")
    dataPointMatrix = np.asarray(dataPoint, dtype="float")
    diffMatrix = centroidPointMatrix - dataPointMatrix
    diffMatrixTranspose = np.transpose(diffMatrix)
    # print(diffMatrix)
    resultMatrix = diffMatrix.dot(diffMatrixTranspose)

    euclideanDistance = math.sqrt(resultMatrix)

    return euclideanDistance

main('segment.arff')