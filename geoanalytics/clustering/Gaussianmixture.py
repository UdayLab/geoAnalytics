# GaussianMixture-based clustering algorithm using scikit-learn to assign cluster labels to multidimensional data with runtime and memory tracking, and support for saving results.
#
# **Importing and Using the Gaussianmixture Class in a Python Program**
#
#             import pandas as pd
#
#             from geoanalytics.clustering import Gaussianmixture
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

            from geoanalytics.clustering import Gaussianmixture

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
        self.weights_ = None
        self.means_ = None
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

    def run(self, n_components=4, max_iters=100, covariance_type="full", init_params='kmeans', random_state=0):
        """
        Performs Gaussian Mixture Model clustering on the feature columns.

        :param n_components: Number of Gaussian components (clusters) to use.
        :param max_iters: Maximum number of iterations allowed during EM algorithm.
        :param covariance_type: Type of covariance parameters ('full', 'tied', 'diag', 'spherical').
        :param init_params: Initialization method ('kmeans' or 'random').
        :param random_state: Random seed to ensure reproducibility.
        :return: Tuple of (DataFrame with labels, array of component weights, array of component means).
        """
        self.startTime = time.time()
        data = self.df.drop(['x', 'y'], axis=1)
        data = data.to_numpy()
        gaussianMixture = GaussianMixture(n_components=n_components, max_iter=max_iters,
                                          covariance_type=covariance_type,
                                          init_params=init_params, random_state=random_state).fit(data)
        gaussianResult = gaussianMixture.predict(data)
        label = self.df[['x', 'y']]
        self.labelsDF = label.assign(labels=gaussianResult)
        self.weights_ = gaussianMixture.weights_
        self.means_ = gaussianMixture.means_

        self.endTime = time.time()

        process = psutil.Process()
        self.memoryUSS = process.memory_full_info().uss / 1024
        self.memoryRSS = process.memory_full_info().rss / 1024

        return self.labelsDF, self.weights_, gaussianMixture.means_


    def save(self,
             outputFileLabels='GaussianMixtureLabels.csv',
             outputFileWeights='GaussianMixtureWeights.csv',
             outputFileMeans='GaussianMixtureMeans.csv'):
        """
        Saves labels, weights, and means to separate CSV files.
        """
        if self.labelsDF is not None:
            try:
                self.labelsDF.to_csv(outputFileLabels, index=False)
                print(f"Labels saved to: {outputFileLabels}")
            except Exception as e:
                print(f"Failed to save labels: {e}")
        else:
            print("No labels to save. Please execute run() method first.")

        if self.weights_ is not None:
            try:
                pd.DataFrame(self.weights_).to_csv(outputFileWeights, index=False, header=["weights"])
                print(f"Weights saved to: {outputFileWeights}")
            except Exception as e:
                print(f"Failed to save weights: {e}")
        else:
            print("No weights to save. Please execute run() method first.")

        if self.means_ is not None:
            try:
                pd.DataFrame(self.means_).to_csv(outputFileMeans, index=False)
                print(f"Means saved to: {outputFileMeans}")
            except Exception as e:
                print(f"Failed to save means: {e}")
        else:
            print("No means to save. Please execute run() method first.")