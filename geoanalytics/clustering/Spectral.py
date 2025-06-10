# Spectral Clustering performs graph-based partitioning using eigen decomposition of similarity matrices. It is especially suitable for identifying clusters that are not necessarily spherical or linearly separable.
#
# **Importing and Using Spectral Clustering in a Python Program**
#
#         import pandas as pd
#
#         from goeAnalytics.clustering import Spectral
#
#         df = pd.read_csv('data.csv')
#
#         obj = Spectral(df)
#
#         obj.clustering(n_clusters=4)
#
#         labels_df = output[0]
#
#         obj.save(outputFile='SpectralLabels.csv')
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
from sklearn.cluster import SpectralClustering
import pandas as pd


class Spectral:
    """
    **About this algorithm**

    :**Description**: Spectral Clustering uses graph-based methods to cluster points based on the eigenvalues
                      of a similarity matrix. It's particularly effective for complex cluster shapes that
                      traditional methods like KMeans may not detect.

    :**Parameters**:   - **dataframe** (*pd.DataFrame*) -- Input dataset where the first two columns are assumed to be spatial ('x', 'y') and all remaining columns are used for clustering.

    :**Attributes**:   - **df** (*pd.DataFrame*) -- Cleaned and formatted input data.
                       - **labelsDF** (*pd.DataFrame*) -- Output containing 'x', 'y', and assigned cluster labels.

    **Execution methods**

    .. code-block:: python

            import pandas as pd

            from goeAnalytics.clustering import Spectral

            df = pd.read_csv("data.csv")

            obj = Spectral(df)

            obj.clustering(n_clusters=4)

            labels_df = output[0]

            obj.save(outputFile="SpectralLabels.csv")

    **Credits**

    This implementation was created and revised under the guidance of Professor Rage Uday Kiran.

    """
    def __init__(self, dataframe):
        """
        Constructor to initialize the Spectral Clustering object with input DataFrame.

        :param dataframe: A pandas DataFrame with at least two spatial columns and feature columns.
        """
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])
        self.labelsDF = None
        self.startTime = None
        self.endTime = None
        self.memoryUSS = None
        self.memoryRSS = None

    def getRuntime(self):
        """
        Prints the total runtime of the clustering algorithm.
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

    def run(self, n_clusters=8, assign_labels='discretize'):
        """
        Applies Spectral Clustering to the dataset.

        :param n_clusters: Number of clusters to form.
        :param assign_labels: Method for assigning labels after clustering ('kmeans' or 'discretize').
        :return: A DataFrame containing x, y, and assigned labels.
        """
        self.startTime = time.time()
        data = np.ascontiguousarray(np.array(self.df.drop(['x', 'y'], axis=1)))
        spectralClustering = SpectralClustering(assign_labels=assign_labels, n_clusters=n_clusters, random_state=0).fit(data)
        label = self.df[['x', 'y']]
        self.labelsDF = label.assign(labels=spectralClustering.labels_)

        self.endTime = time.time()

        process = psutil.Process()
        self.memoryUSS = process.memory_full_info().uss / 1024
        self.memoryRSS = process.memory_full_info().rss / 1024

        return self.labelsDF

    def save(self, outputFile='SpectralLabels.csv'):
        """
        Saves the clustering results to a CSV file.

        :param outputFile: Filename to save the resulting DataFrame.
        """
        if self.labelsDF is not None:
            try:
                self.labelsDF.to_csv(outputFile, index=False)
                print(f"Labels saved to: {outputFile}")
            except Exception as e:
                print(f"Failed to save labels: {e}")
        else:
            print("No labels to save. Please execute run() method first.")