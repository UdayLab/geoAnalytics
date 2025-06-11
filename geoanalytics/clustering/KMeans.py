# K-means is a popular and efficient clustering algorithm that partitions data into K distinct groups by iteratively minimizing the distance between data points and their assigned cluster centroids.

# **Importing this algorithm into a Python program**
#
#           from geoanalytics.clustering import KMeans as alg
#
#           import pandas as pd
#
#           df = pd.read_csv('dataset.csv')
#
#           obj = alg.KMeans(df)
#
#           obj.elbowMethod()
#
#           labels, centers = obj.clustering(k=3)
#
#           print("Clustered Data with Labels:\n", labels)
#
#           print("Cluster Centers:\n", centers)



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
    """
    **About this algorithm**

    :**Description**:   KMeans clusters data into k groups using scikit-learn's algorithm, excluding 'x' and 'y' spatial columns. It supports elbow plot visualization, and tracks runtime and memory usage.

    :**Parameters**:    - **dataframe** (*pandas.DataFrame*) -- *Input dataset with 'x', 'y' spatial columns and other features.*
                        - **k** (*int*) -- *Number of clusters for the algorithm.*
                        - **max_iter** (*int*) -- *Maximum number of iterations for convergence.*
                        - **sep** (*str*) -- *Separator (default is tab-space if reading from file, not used in this context).*

    :**Attributes**:    - **df** (*pandas.DataFrame*) -- *Internal copy of the input DataFrame with column order ['x', 'y', features...].*
                        - **memory_kb** (*float*) -- *Memory used in kilobytes (USS) during clustering.*
                        - **runtime_sec** (*float*) -- *Runtime duration of clustering in seconds.*

    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            from geoanalytics.clustering import KMeans as alg

            import pandas as pd

            df = pd.read_csv('dataset.csv')

            obj = alg.KMeans(df)

            obj.elbowMethod()

            labels, centers = obj.clustering(k=3)

            print("Clustered Data with Labels:\n", labels)

            print("Cluster Centers:\n", centers)

    **Credits**

    The complete program was written by               and revised by              under the supervision of Professor Rage Uday Kiran.
    """
    def __init__(self, dataframe):
        """
        Initializes the KMeans object by standardizing the column order and copying the dataframe.
        """
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])
        self.labelsDF = None
        self.centers = None
        self.startTime = None
        self.endTime = None
        self.memoryUSS = None
        self.memoryRSS = None

    def getRuntime(self):
        """
        Computes and prints the runtime and USS memory used by the process.
        """
        print("Total Execution time of proposed Algorithm:", self.endTime - self.startTime, "seconds")

    def getMemoryUSS(self):
        """
        Prints the memory usage (USS) of the process in kilobytes.
        """
        print("Memory (USS) of proposed Algorithm in KB:", self.memoryUSS)

    def getMemoryRSS(self):
        """
        Prints the memory usage (RSS) of the process in kilobytes.
        """
        print("Memory (RSS) of proposed Algorithm in KB:", self.memoryRSS)

    def elbowMethod(self):
        """
        Plots the elbow graph to help determine the optimal number of clusters (k).
        Uses WCSS (Within-Cluster Sum of Squares) and excludes 'x', 'y' from the computation.
        """
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

    def run(self, k = 4, max_iter=100):
        """
        Applies KMeans clustering to the dataset, excluding 'x' and 'y', and returns labeled data and cluster centers.

        Returns:
        --------
        labels : pandas.DataFrame
            DataFrame containing 'x', 'y', and the predicted cluster labels.
        centers : numpy.ndarray
            Array of shape (k, n_features) representing the coordinates of cluster centers.
        """
        self.startTime = time.time()
        data = self.df.drop(['x', 'y'], axis=1)
        data = data.to_numpy()
        kmeans = kmeansAlg(n_clusters=k, max_iter=max_iter).fit(data)
        label = self.df[['x', 'y']]
        self.labelsDF = label.assign(labels=kmeans.labels_)
        self.centers = kmeans.cluster_centers_

        self.endTime = time.time()

        process = psutil.Process()
        self.memoryUSS = process.memory_full_info().uss / 1024
        self.memoryRSS = process.memory_full_info().rss / 1024

        return self.labelsDF, self.centers

    def save(self, outputFileLabels='KMeansLabels.csv', outputFileCenters='KMeansCenters.csv'):
        """
        Saves the imputed DataFrame to a CSV file.
        """
        if self.labelsDF is not None:
            try:
                self.labelsDF.to_csv(outputFileLabels, index=False)
                print(f"Labels saved to: {outputFileLabels}")
            except Exception as e:
                print(f"Failed to save labels: {e}")
        else:
            print("No labels to save. Please execute run() method first.")

        if self.centers is not None:
            try:
                pd.DataFrame(self.centers).to_csv(outputFileCenters, index=False)
                print(f"Cluster centers saved to: {outputFileCenters}")
            except Exception as e:
                print(f"Failed to save cluster centers: {e}")
        else:
            print("No cluster centers to save. Please execute run() method first.")