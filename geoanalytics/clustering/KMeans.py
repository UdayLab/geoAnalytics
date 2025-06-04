# K-means is a popular and efficient clustering algorithm that partitions data into K distinct groups by iteratively minimizing the distance between data points and their assigned cluster centroids.
#
# **Importing this algorithm into a python program**
#
#             import geoanalytics.clustering.kMeans as alg
#
#             dataframe = df # read the dataframe with pandas
#
#             obj = alg.kMeans(dataframe)
#
#             obj.elbowMethod()
#
#             obj.run(6, max_iter=100) # here 6 represents number of clusters
#



__copyright__ = """
Copyright (C)  2022 Rage Uday Kiran

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans as kmeansAlg
import pandas as pd


class KMeans:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])
        self.labelsDF = None

    def getStatistics(self, start_time):
        print("Total Execution time of proposed Algorithm:", time.time() - start_time)
        process = psutil.Process()
        memory_uss_kb = process.memory_full_info().uss / 1024
        print("Memory (USS) of proposed Algorithm in KB:", memory_uss_kb)
        memory_rss_kb = process.memory_full_info().rss / 1024
        print("Memory (RSS) of proposed Algorithm in KB:", memory_rss_kb)

    def elbowMethod(self):
        wcss = []
        k_values = range(1, 11)
        data = self.df.drop(['x', 'y'], axis=1)

        for k in tqdm(k_values):
            kmeans = kmeansAlg(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
        plt.plot(k_values, wcss, marker='o', linestyle='--')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS')
        plt.title('Elbow Method for Optimal k (Ignoring Location Columns)')
        plt.show()

    def clustering(self, k = 4, max_iter=100):
        start_time = time.time()
        data = self.df.drop(['x', 'y'], axis=1)
        data = data.to_numpy()
        kmeans = kmeansAlg(n_clusters=k, max_iter=max_iter).fit(data)
        label = self.df[['x', 'y']]
        self.labelsDF = label.assign(labels=kmeans.labels_)
        self.getStatistics(start_time)
        return self.labelsDF, kmeans.cluster_centers_

    def save(self, outputFile='KMeansLabels.csv'):
        if self.labelsDF is not None:
            try:
                self.labelsDF.to_csv(outputFile, index=False)
                print(f"Labels saved to: {outputFile}")
            except Exception as e:
                print(f"Failed to save labels: {e}")
        else:
            print("No labels to save. Please run clustering first.")