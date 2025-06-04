# GaussianMixture-based clustering algorithm using scikit-learn to assign cluster labels to multidimensional data with runtime and memory tracking, and support for saving results.
#
# **Importing and Using the Gaussianmixture Class in a Python Program**
#
#             import pandas as pd
#
#             from geoAnalytics.clustering import Gaussianmixture
#
#             df = pd.read_csv('input.csv')
#
#             gm = Gaussianmixture(df)
#
#             output = gm.clustering(n_components=3)
#
#             labels_df = output[0]
#
#             weights = output[1]
#
#             centers = output[2]
#
#             gm.save('GaussianMixtureLabels.csv')

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
from sklearn.mixture import GaussianMixture
import pandas as pd


class Gaussianmixture:
    """
    **About this algorithm**

    :**Description**: Gaussian Mixture Model (GMM) is a probabilistic clustering algorithm that clusters feature-rich data by modeling it as a mixture of Gaussians, with runtime and memory tracking and exportable label results.

    :**Parameters**:    - Dataset (pandas DataFrame) must be provided during object initialization.
                        - No other parameters are required during instantiation.

    :**Attributes**:    - **df** (*pd.DataFrame*) -- The input data with 'x', 'y' coordinates and features.
                        - **labelsDF** (*pd.DataFrame*) -- DataFrame containing 'x', 'y', and assigned cluster labels.
                        - **model** (*GaussianMixture*) -- The trained scikit-learn GaussianMixture model instance for reuse or further analysis.


    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            import pandas as pd

            from geoAnalytics.clustering import Gaussianmixture

            df = pd.read_csv("input.csv")

            gm = Gaussianmixture(df)

            output = gm.clustering(n_components=3)

            labels_df = output[0]

            weights = output[1]

            centers = output[2]

            gm.save('GaussianMixtureLabels.csv')

    **Credits**

    This implementation was created by Raashika and revised by M.Charan Teja under the guidance of Professor Rage Uday Kiran.
    """
    def __init__(self, dataframe):
        """
        Constructor to initialize the Gaussianmixture object with the input DataFrame.

        :param dataframe: pandas DataFrame containing at least columns ['x', 'y'] and feature columns.
        """
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])
        self.labelsDF = None

    def getStatistics(self, start_time):
        """
        Prints execution time and memory usage statistics of the clustering operation.

        :param start_time: Time.time() reference from before clustering.

        This method prints:
        - Total execution time in seconds.
        - Memory usage (USS - Unique Set Size) in kilobytes (KB), representing the memory
          uniquely used by the process.
        - Memory usage (RSS - Resident Set Size) in kilobytes (KB), representing the total
          physical memory used by the process.
        """
        print("Total Execution time of proposed Algorithm:", time.time() - start_time)
        process = psutil.Process()
        memory_uss_kb = process.memory_full_info().uss / 1024
        print("Memory (USS) of proposed Algorithm in KB:", memory_uss_kb)
        memory_rss_kb = process.memory_full_info().rss / 1024
        print("Memory (RSS) of proposed Algorithm in KB:", memory_rss_kb)

    def clustering(self, n_components=1, max_iters=100, covariance_type="full", init_params='kmeans', random_state=0):
        """
        Performs Gaussian Mixture Model clustering on the feature columns.

        :param n_components: Number of Gaussian components (clusters) to use.
        :param max_iters: Maximum number of iterations allowed during EM algorithm.
        :param covariance_type: Type of covariance parameters ('full', 'tied', 'diag', 'spherical').
        :param init_params: Initialization method ('kmeans' or 'random').
        :param random_state: Random seed to ensure reproducibility.
        :return: Tuple of (DataFrame with labels, array of component weights, array of component means).
        """
        start_time = time.time()
        data = self.df.drop(['x', 'y'], axis=1)
        data = data.to_numpy()
        gaussianMixture = GaussianMixture(n_components=n_components, max_iter=max_iters,
                                          covariance_type=covariance_type,
                                          init_params=init_params, random_state=random_state).fit(data)
        gaussianResult = gaussianMixture.predict(data)
        label = self.df[['x', 'y']]
        self.labelsDF = label.assign(labels=gaussianResult)
        self.getStatistics(start_time)
        return self.labelsDF, gaussianMixture.weights_, gaussianMixture.means_

    def save(self, outputFile='GaussianMixtureLabels.csv'):
        """
        Saves the clustering result (labels) into a CSV file.

        :param outputFile: Path to the output CSV file.
        """
        if self.labelsDF is not None:
            try:
                self.labelsDF.to_csv(outputFile, index=False)
                print(f"Labels saved to: {outputFile}")
            except Exception as e:
                print(f"Failed to save labels: {e}")
        else:
            print("No labels to save. Please run clustering first.")