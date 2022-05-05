import numpy as np
import pandas as pd
import time
import math
import sys
import os
import psutil

class rasterOneNNHausdorff:

    def __init__(self, trainingFile, testFile):
        self.training = pd.read_csv(sys.argv[1], sep='\t', header=None)  # np.loadtxt(sys.argv[1], delimiter='\t')
        self.testing = pd.read_csv(sys.argv[2], sep='\t', header=None)  # np.loadtxt(sys.argv[2],  delimiter='\t')

        # To drop the first column which is a class label of each sample
        self.training.drop(0, axis='columns', inplace=True)

    def hausdorff(self, u, v):
        row, = u.shape
        lea_distance = 0
        for i in range(row):
            distance1 = np.amin(np.absolute((np.diff(u[i] - v))))
            if distance1 > lea_distance:
                lea_distance = distance1
        return lea_distance

    def Run(self):
        num_rows_test, num_columns_test = self.testing.shape
        num_rows_train, num_columns_train = self.training.shape
        training_noclass = self.training#self.training[:, 1:]
        testing_noclass = self.testing#self.testing[:, 1:]
        #predicted_label = None
        #correct = 0
        newColumn = list()
        start_time = time.time()
        for i in range(num_rows_test):
            least_distance = float('inf')
            for j in range(num_rows_train):
                dist = max(self.hausdorff(testing_noclass.iloc[i].to_numpy(), training_noclass.iloc[j].to_numpy()),
                           self.hausdorff(training_noclass.iloc[j].to_numpy(), testing_noclass.iloc[i].to_numpy()))  # math.sqrt(squaring)
                if dist < least_distance:
                    #predicted_label = self.training[j][0]
                    least_distance = dist
            #if predicted_label == self.testing[i][0]:
            #    correct = correct + 1
            newColumn.append(least_distance)
        print(newColumn)
        self.testing['1NNmaxNorm'] = newColumn
        print("Final test samples after calculating the Ravi Distance (RD):")
        print(self.testing)

        N = int(input(
            "Enter the how many number of top elements to be retrieved between: 1 to " + str(num_rows_test) + ":"))
        sortedDF = self.testing.sort_values('1NNmaxNorm').head(N)
        print("Final top- " + str(N) + " testing samples retrieved based on the Ravi Distance:")
        print(sortedDF)
        ##accuracy = (correct / num_rows_test) * 100
        #print("Total Accuracy of oneNNHausdorff is:", accuracy)
        print("Total Execution time of oneNNHausdorff", time.time() - start_time)
        process = psutil.Process(os.getpid())
        memory = process.memory_full_info().uss
        memory_in_KB = memory / (1024)
        print("Total Memory of oneNNHausdorff inKB", memory_in_KB)  # in bytes


if __name__ == '__main__':
    obj = rasterOneNNHausdorff(sys.argv[1], sys.argv[2])
    obj.Run()