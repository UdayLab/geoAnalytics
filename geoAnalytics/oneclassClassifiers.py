# import required libraries
import pandas as pd
import time
import os
import psutil
import gc
import sys
import numpy as np
import math


def rasterFuzzyTSC(training, testing,topElements):
    startTime = time.time()

    # To drop the first column which is a class label of each sample
    training.drop(0, axis='columns', inplace=True)

    # separating classes into different dictionary variables and creating mean, min and max curves
    maxData = training.max(axis=0)
    meanData = training.mean(axis=0)
    minData = training.min(axis=0)

    del training
    gc.collect()

    # Classification Phase

    counter = {}
    num_rows, num_columns = testing.shape

    # print(num_rows, num_columns)
    newColumn = list()
    # print(num_columns)
    correct = 0
    for i in range(num_rows):
        counter = 0.0
        for j in range(1, num_columns):
            if testing.iloc[i, j] >= meanData[j]:
                if testing.iloc[i, j] <= maxData[j]:
                    counter = counter + 0.5 * (testing.iloc[i, j] - meanData[j]) / (maxData[j] - meanData[j])
                else:
                    counter = counter + 1
            else:
                if testing.iloc[i, j] >= minData[j]:
                    counter = counter + 0.5 * (meanData[j] - testing.iloc[i, j]) / (meanData[j] - minData[j])
                else:
                    counter = counter + 1

        counter = counter / num_columns

        newColumn.append(counter)

    testing['RD'] = newColumn

    # N = int(input("Enter the how many number of top elements to be retrieved between: 1 to " + str(num_rows) + ":"))
    N = topElements
    sortedDF = testing.sort_values('RD').head(N)
    getStatistics(startTime)
    return testing, sortedDF


def rasterOneNNDTW(training, testing,topElements):
    startTime = time.time()
    training.drop(0, axis='columns', inplace=True)
    num_rows_test, num_columns_test = testing.shape
    num_rows_train, num_columns_train = training.shape
    training_noclass = training  # self.training[:, 1:]
    testing_noclass = testing  # self.testing[:, 1:]

    newColumn = list()
    # predicted_label = None
    correct = 0

    for i in range(num_rows_test):
        # print(i)
        least_distance = float('inf')
        for j in range(num_rows_train):
            dist = dtw(testing_noclass.iloc[i].to_numpy(), training_noclass.iloc[j].to_numpy())
            if dist < least_distance:
                # predicted_label = self.training[j][0]
                least_distance = dist

        newColumn.append(least_distance)
        # if predicted_label == self.testing[i][0]:
        #    correct = correct + 1
    testing['1NNmaxNorm'] = newColumn   # print final test
    N = topElements
    sortedDF = testing.sort_values('1NNmaxNorm').head(N) # top k
    getStatistics(startTime)
    return testing,sortedDF


def rasterOneNNED(training, testing,topElements):
    startTime = time.time()
    training.drop(0, axis='columns', inplace=True)
    num_rows_test, num_columns_test = testing.shape
    num_rows_train, num_columns_train = training.shape
    predicted_label = None
    correct = 0
    newColumn = list()

    for i in range(num_rows_test):
        least_distance = float('inf')
        for j in range(num_rows_train):
            squaring = 0
            for k in range(num_columns_train):
                squaring = squaring + (testing.iloc[i, k] - training.iloc[j, k]) ** 2
            dist = math.sqrt(squaring)

            if dist < least_distance:
                # predicted_label = self.training[j][0]
                least_distance = dist
        newColumn.append(least_distance)

    testing['1NNED'] = newColumn
    N = topElements
    sortedDF = testing.sort_values('1NNED').head(N)
    getStatistics(startTime)
    return testing,sortedDF



def rasterOneNNHausdorff(training, testing,topElements):
    startTime = time.time()
    training.drop(0, axis='columns', inplace=True)
    num_rows_test, num_columns_test = testing.shape
    num_rows_train, num_columns_train = training.shape
    training_noclass = training  # self.training[:, 1:]
    testing_noclass = testing  # self.testing[:, 1:]
    # predicted_label = None
    # correct = 0
    newColumn = list()

    for i in range(num_rows_test):
        least_distance = float('inf')
        for j in range(num_rows_train):
            dist = max(hausdorff(testing_noclass.iloc[i].to_numpy(), training_noclass.iloc[j].to_numpy()),
                       hausdorff(training_noclass.iloc[j].to_numpy(),
                                      testing_noclass.iloc[i].to_numpy()))  # math.sqrt(squaring)
            if dist < least_distance:
                # predicted_label = self.training[j][0]
                least_distance = dist
        # if predicted_label == self.testing[i][0]:
        #    correct = correct + 1
        newColumn.append(least_distance)

    testing['1NNHausdorff'] = newColumn
    N = topElements
    sortedDF = testing.sort_values('1NNHausdorff').head(N)
    getStatistics(startTime)
    return testing,sortedDF

def rasterOneNNMaxNorm(training, testing,topElements):
    startTime = time.time()
    training.drop(0, axis='columns', inplace=True)
    num_rows_test, num_columns_test = testing.shape
    num_rows_train, num_columns_train =     training.shape
    newColumn = list()

    for i in range(num_rows_test):
        least_distance = float('inf')
        for j in range(num_rows_train):
            maximum = float('-inf')
            for k in range(num_columns_train):
                temp = np.abs(testing.iloc[i, k] - training.iloc[j, k])
                if temp > maximum:
                    maximum = temp
            dist = maximum  # math.sqrt(squaring)
            if dist < least_distance:
                # predicted_label = self.training[j][0]
                least_distance = dist
        newColumn.append(least_distance)
        # if predicted_label == self.testing[i][0]:
        #    correct = correct + 1

    testing['1NNmaxNorm'] = newColumn
    N = topElements
    sortedDF = self.testing.sort_values('1NNmaxNorm').head(N)
    getStatistics(startTime)
    return testing,sortedDF

def hausdorff(u, v):
    row, = u.shape
    lea_distance = 0
    for i in range(row):
        distance1 = np.amin(np.absolute((np.diff(u[i] - v))))
        if distance1 > lea_distance:
            lea_distance = distance1
    return lea_distance


def dtw(A, B):
        N = len(A)
        M = len(B)
        d = np.zeros((N, M))
        for n in range(N):
            for m in range(M):
                d[n][m] = (A[n] - B[m]) ** 2
        D = np.zeros(d.shape)
        D[0][0] = d[0][0]
        for n in range(1, N):
            D[n][0] = d[n][0] + D[n - 1][0]
        for m in range(1, M):
            D[0][m] = d[0][m] + D[0][m - 1]
        for n in range(1, N):
            for m in range(1, M):
                D[n][m] = d[n][m] + min(D[n - 1][m], D[n - 1][m - 1], D[n][m - 1])
        return D[N - 1][M - 1]


def showFinalTestSamples(df):
    print("Final test samples after calculating the Ravi Distance (RD):")
    print(df)


def showFinalTopNSamples(df):
    print("Final top- N  testing samples retrieved based on the Ravi Distance:")
    print(df)


def saveFinalTestSamples(df, outputFile):
    df.iloc[:, -1].to_csv(outputFile, header=False)


def saveFinalTopNSamples(df, outputFile):
    df.iloc[:, -1].to_csv(outputFile, header=False)


def getStatistics(startTime):
    # Statistics of the Algorithm after execution
    # print("Dataset name:", "SampleRunTrainFile.txt")
    print("Total Execution time of proposedAlgo", time.time() - startTime)
    process = psutil.Process(os.getpid())
    memory = process.memory_full_info().uss
    memory_in_KB = memory / 1024
    print("Memory of proposedAlgo in KB:", memory_in_KB)  # in bytes
