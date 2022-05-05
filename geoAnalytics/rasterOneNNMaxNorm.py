#import numpy as np
import pandas as pd
#import math
import time
import psutil
import os
import sys

class rasterOneNNMaxNorm:

    def __init__(self, trainingFile, testFile):
        self.startTime = time.time()
        self.training = pd.read_csv(sys.argv[1], sep='\t', header=None)  # np.loadtxt(sys.argv[1], delimiter='\t')
        self.testing = pd.read_csv(sys.argv[2], sep='\t', header=None)  # np.loadtxt(sys.argv[2],  delimiter='\t')

        # To drop the first column which is a class label of each sample
        self.training.drop(0, axis='columns', inplace=True)

    def Run(self):
        num_rows_test, num_columns_test = self.testing.shape
        num_rows_train, num_columns_train = self.training.shape
        newColumn = list()

        self.startTime = time.time()
        for i in range(num_rows_test):
            least_distance = float('inf')
            for j in range(num_rows_train):
                maximum = float('-inf')
                for k in range(num_columns_train):
                    temp = pd.Series.abs(self.testing.iloc[i, k] - self.training.iloc[j, k])
                    #print(temp)
                    if temp > maximum:
                        maximum = temp
                dist = maximum  # math.sqrt(squaring)
                if dist < least_distance:
                    #predicted_label = self.training[j][0]
                    least_distance = dist
            newColumn.append(least_distance)
            # if predicted_label == self.testing[i][0]:
            #    correct = correct + 1

        print(newColumn)
        self.testing['1NNmaxNorm'] = newColumn
        print("Final test samples after calculating the Ravi Distance (RD):")
        print(self.testing)

        N = int(input("Enter the how many number of top elements to be retrieved between: 1 to " + str(num_rows_test) + ":"))
        sortedDF = self.testing.sort_values('1NNmaxNorm').head(N)
        print("Final top- " + str(N) + " testing samples retrieved based on the Ravi Distance:")
        print(sortedDF)
        #accuracy = (correct / num_rows_test) * 100
        #print("Total Accuracy of 1NNMaxNorm is:", accuracy)
        print("Total Execution time 1NNMaxNorm is:", time.time() - self.startTime)
        process = psutil.Process(os.getpid())
        memory = process.memory_full_info().uss
        memory_in_KB = memory / (1024)
        print("Memory of 1NNmaxinorm in KB", memory_in_KB)  # in bytes


if __name__ == '__main__':
    obj = rasterOneNNMaxNorm(sys.argv[1], sys.argv[2])
    obj.Run()