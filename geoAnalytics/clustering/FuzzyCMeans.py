# Fuzzy C-Means (FCM) is a soft clustering algorithm where each data point can belong to multiple clusters with varying degrees of membership.
#
# **Importing this algorithm into a Python program**
#
#             from geoAnalytics.clustering import FuzzyCMeans
#
#             import pandas as pd
#
#             df = pd.read_csv("data.csv")
#
#             obj = FuzzyCMeans(df)
#
#             output = obj.clustering(n_clusters=3)
#
#             labels = output[0]
#
#             fcm.centers = output[1]
#
#             obj.save('FuzzyCMeansLabels.csv')
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
from fcmeans import FCM
import pandas as pd


class FuzzyCMeans:
    """
    **About this algorithm**

    :**Description**: Fuzzy C-Means (FCM) is a soft clustering technique where each data point is assigned a membership score to all clusters,
                      allowing partial membership instead of a hard assignment to a single cluster.

    :**Parameters**: - **dataframe** (*pd.DataFrame*) -- * where:
        - The first two columns are coordinates.
        - The rest of the columns contain the features used for clustering.*

    :**Attributes**: - **df** (*pd.DataFrame*) -- *A copy of your original data, with the first two columns renamed to `'x'` and `'y'` to make things easy and clear during clustering*


    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            from geoAnalytics.clustering import FuzzyCMeans

            import pandas as pd

            df = pd.read_csv("data.csv")

            obj = FuzzyCMeans(df)

            output = obj.clustering(n_clusters=3)

            labels = output[0]

            fcm.centers = output[1]

            obj.save('FuzzyCMeansLabels.csv')


     **Credits**

     The complete program was written by M. Charan Teja under the supervision of Professor Rage Uday Kiran.

     """

    def __init__(self, dataframe):
        """
        Constructor that initializes the FuzzyCMeans object with input DataFrame.

        :param dataframe:  - A Pandas DataFrame that contains the data to be clustered.
        """
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])
        self.labelsDF = None

    def getStatistics(self, start_time):
        """
        Prints memory usage and runtime of the clustering function.

        :param start_time: Time when the clustering function started execution.
        """
        print("Total Execution time of proposed Algorithm:", time.time() - start_time)
        process = psutil.Process()
        memory_uss_kb = process.memory_full_info().uss / 1024
        print("Memory (USS) of proposed Algorithm in KB:", memory_uss_kb)
        memory_rss_kb = process.memory_full_info().rss / 1024
        print("Memory (RSS) of proposed Algorithm in KB:", memory_rss_kb)

    def clustering(self, n_clusters=3):
        """
        Performs fuzzy clustering on data.
        -It excludes the 'x' and 'y' columns during clustering, and uses the remaining columns.
        -After clustering, it adds the cluster label (based on the highest membership score) back to the original coordinates
        :param n_clusters: Number of clusters to form (default is 3).
        :return: A tuple containing:
                 - A DataFrame with 'x', 'y', and cluster labels for each data point (based on maximum membership).
                 - A NumPy array with coordinates of the cluster centers.
        """
        start_time = time.time()
        data = self.df.drop(['x', 'y'], axis=1)
        data = data.to_numpy()
        fcm = FCM(n_clusters=n_clusters)
        fcm.fit(data)
        fcmResult = fcm.predict(data)
        label = self.df[['x', 'y']]
        self.labelsDF = label.assign(labels=fcmResult)
        self.getStatistics(start_time)
        return self.labelsDF, fcm.centers

    def save(self, outputFile='FuzzyCMeansLabels.csv'):
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