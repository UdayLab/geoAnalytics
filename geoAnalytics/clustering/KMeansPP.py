# KMeans++ enhances KMeans by improving initialization for better convergence and cluster quality, with added utilities for memory tracking, runtime reporting, and Elbow Method plotting on 2D-location DataFrame inputs.
#
# **Importing and Using this KMeans++ Wrapper in a Python Program**
#
#             import pandas as pd
#
#             from goeAnalytics.clustering import KMeansPP
#
#             df = pd.read_csv('data.csv')
#
#             obj = KMeansPP(df)
#
#             obj.elbowMethod()
#
#             obj.clustering(k=3)
#
#             obj.save(outputFile='KMeansPPLabels.csv')
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


class KMeansPP:
    """
        **About this algorithm**

        :**Description**:KMeans++ improves K-Means clustering by using smarter centroid initialization for better stability and faster convergence, applied here to high-dimensional data excluding x, y coordinates.

        :**Parameters**:    - **dataframe** (*pd.DataFrame*) -- A Pandas DataFrame that contains the input dataset.
                            - The first two columns must be spatial or positional features (e.g., 'x' and 'y').
                            - All other columns are treated as feature vectors for clustering.

        :**Attributes**:    - **df** (*pd.DataFrame*) -- Stores the copy of the input dataset, renaming first two columns to 'x' and 'y'.
                            - **start_time** (*float*) -- Records the clustering start time for runtime analysis.
                            - **memory_kb** (*float*) -- Measures memory usage in kilobytes after execution.
                            - **labels** (*pd.DataFrame*) -- Final dataframe containing 'x', 'y', and cluster label for each instance.
                            - **cluster_centers_** (*np.ndarray*) -- Coordinates of the final cluster centroids after fitting.

        **Execution methods**

        **Calling from a Python program**

        .. code-block:: python

                import pandas as pd

                from goeAnalytics.clustering import KMeansPP

                df = pd.read_csv('data.csv')

                obj = KMeansPP(df)

                obj.elbowMethod()

                obj.clustering(k=3)

                obj.save(outputFile='KMeansPPLabels.csv')

        **Credits**

        The complete program was written by Raashika and revised by M.Charan Teja under the supervision of Professor Rage Uday Kiran.

    """
    def __init__(self, dataframe):
        """
        Constructor to initialize the KMeans++ object with the given dataframe.
        """
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])
        self.labelsDF = None

    def getStatistics(self, start_time):
        """
        Prints memory usage and execution time after clustering.
        """
        print("Total Execution time of proposed Algorithm:", time.time() - start_time)
        process = psutil.Process()
        memory_uss_kb = process.memory_full_info().uss / 1024
        print("Memory (USS) of proposed Algorithm in KB:", memory_uss_kb)
        memory_rss_kb = process.memory_full_info().rss / 1024
        print("Memory (RSS) of proposed Algorithm in KB:", memory_rss_kb)

    def elbowMethod(self):
        """
        Applies the elbow method to help decide the optimal number of clusters (k).
        It plots WCSS (within-cluster sum of squares) for k in range 1 to 10.
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

    def clustering(self, k = 4, max_iter=300):
        """
        Runs KMeans++ clustering on the input dataset using scikit-learn.

        :param k: Number of clusters to form.
        :param max_iter: Maximum number of iterations for a single run.
        :return: A DataFrame with original x, y and cluster labels, and the cluster centers.
        """
        start_time = time.time()
        data = self.df.drop(['x', 'y'], axis=1)
        data = data.to_numpy()
        kmeans = kmeansAlg(n_clusters=k, max_iter=max_iter, init='k-means++').fit(data)
        label = self.df[['x', 'y']]
        self.labelsDF = label.assign(labels=kmeans.labels_)
        self.getStatistics(start_time)
        return self.labelsDF, kmeans.cluster_centers_

    def save(self, outputFile='KMeansPPLabels.csv'):
        """
        Save the outputFile in CSV
        """
        if self.labelsDF is not None:
            try:
                self.labelsDF.to_csv(outputFile, index=False)
                print(f"Labels saved to: {outputFile}")
            except Exception as e:
                print(f"Failed to save labels: {e}")
        else:
            print("No labels to save. Please run clustering first.")