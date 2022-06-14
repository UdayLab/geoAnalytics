import pandas as pd
import time
import os
import psutil
import gc
import sys

class rasterFuzzyTSC:

    def __init__(self, trainingFile, testFile):
        self.startTime = time.time()
        self.training = pd.read_csv(trainingFile, sep='\t', header=None)  # np.loadtxt(sys.argv[1], delimiter='\t')
        self.testing = pd.read_csv(testFile, sep='\t', header=None)  # np.loadtxt(sys.argv[2],  delimiter='\t')
        #self.training = np.loadtxt(trainingFile, delimiter='\t')
        #self.testing = np.loadtxt(testFile,  delimiter='\t')
        # To drop the first column which is a class label of each sample
        self.training.drop(0, axis='columns', inplace=True)

    def Run(self):

        # separating classes into different dictionary variables and creating mean, min and max curves
        maxData = self.training.max(axis=0)
        meanData = self.training.mean(axis=0)
        minData = self.training.min(axis=0)

        del self.training
        gc.collect()

        # Classification Phase

        counter = {}
        num_rows, num_columns = self.testing.shape

        # print(num_rows, num_columns)
        newColumn = list()
        # print(num_columns)
        correct = 0
        for i in range(num_rows):
            counter = 0.0
            for j in range(1, num_columns):
                if self.testing.iloc[i, j] >= meanData[j]:
                    if self.testing.iloc[i, j] <= maxData[j]:
                        counter = counter + 0.5 * (self.testing.iloc[i, j] - meanData[j]) / (maxData[j] - meanData[j])
                    else:
                        counter = counter + 1
                else:
                    if self.testing.iloc[i, j] >= minData[j]:
                        counter = counter + 0.5 * (meanData[j] - self.testing.iloc[i, j]) / (meanData[j] - minData[j])
                    else:
                        counter = counter + 1

            counter = counter / num_columns

            newColumn.append(counter)

        self.testing['RD'] = newColumn
        print("Final test samples after calculating the Ravi Distance (RD):")
        print(self.testing)

        N = int(input("Enter the how many number of top elements to be retrieved between: 1 to " + str(num_rows) + ":"))
        sortedDF = self.testing.sort_values('RD').head(N)
        print("Final top- " + str(N) + " testing samples retrieved based on the Ravi Distance:")
        print(sortedDF)

        # Statistics of the Algorithm after execution
        print("Dataset name:", sys.argv[1])
        print("Total Execution time of proposedAlgo", time.time() - self.startTime)
        process = psutil.Process(os.getpid())
        memory = process.memory_full_info().uss
        memory_in_KB = memory / 1024
        print("Memory of proposedAlgo in KB:", memory_in_KB)  # in bytes


if __name__ == '__main__':
    obj = rasterFuzzyTSC(sys.argv[1], sys.argv[2])
    obj.Run()

