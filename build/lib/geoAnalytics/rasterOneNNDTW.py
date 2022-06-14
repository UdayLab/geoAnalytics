import numpy as np
import pandas as pd
import os
import psutil
import time
import sys

class rasterOneNNDTW:

    def __init__(self, trainingFile, testFile):
        self.training = pd.read_csv(sys.argv[1], sep='\t', header=None)  # np.loadtxt(sys.argv[1], delimiter='\t')
        self.testing = pd.read_csv(sys.argv[2], sep='\t', header=None)  # np.loadtxt(sys.argv[2],  delimiter='\t')

        # To drop the first column which is a class label of each sample
        self.training.drop(0, axis='columns', inplace=True)

    def dtw(self, A, B):
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

    def Run(self):
        num_rows_test, num_columns_test = self.testing.shape
        num_rows_train, num_columns_train = self.training.shape
        training_noclass = self.training#self.training[:, 1:]
        testing_noclass = self.testing#self.testing[:, 1:]

        newColumn = list()
        #predicted_label = None
        correct = 0
        startTime = time.time()
        for i in range(num_rows_test):
            # print(i)
            least_distance = float('inf')
            for j in range(num_rows_train):
                dist = self.dtw(testing_noclass.iloc[i].to_numpy(), training_noclass.iloc[j].to_numpy())
                if dist < least_distance:
                    #predicted_label = self.training[j][0]
                    least_distance = dist

            newColumn.append(least_distance)
            #if predicted_label == self.testing[i][0]:
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
        #print("Total Accuracy of oneNNDTW is:", accuracy)
        print("Total Execution time oneNNDTW is:", time.time() - startTime)
        process = psutil.Process(os.getpid())
        memory = process.memory_full_info().uss
        memory_in_KB = memory / (1024)
        print("Total Memory of oneNNDTW inKB", memory_in_KB)  # in bytes


if __name__ == '__main__':
    obj = rasterOneNNDTW(sys.argv[1], sys.argv[2])
    obj.Run()

