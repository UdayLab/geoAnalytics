# MultipleImputation performs iterative multivariate imputation using chained equations with linear regression to estimate missing values based on relationships among features.
#
# **Importing and Using the MultipleImputation Class in a Python Program**
#
#             import pandas as pd
#
#             from geoanalytics.imputation import MultipleImputation
#
#             df = pd.read_csv('input.csv')
#
#             mi = MultipleImputation(df)
#
#             output = mi.run()
#
#             mi.save('MultipleImputation.csv')
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
from tqdm import tqdm
import pandas as pd
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression


class MultipleImputation:
    """
    **About this algorithm**

    :**Description**: MultipleImputation performs iterative multivariate imputation using chained equations with linear regression to estimate missing values based on relationships among features.

    :**Parameters**:    - Dataset (pandas DataFrame) must be provided during object initialization.
                        - Additional tuning parameters can be provided during the `run()` call.

    :**Attributes**:    - **df** (*pd.DataFrame*) -- The input data with 'x', 'y' coordinates and feature columns.
                        - **imputedDF** (*pd.DataFrame*) -- DataFrame containing 'x', 'y', and imputed values.
    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            import pandas as pd

            from geoanalytics.imputation import MultipleImputation

            df = pd.read_csv("input.csv")

            mi = MultipleImputation(df)

            output = mi.run()

            mi.save('MultipleImputaion.csv')


    **Credits**

    This implementation was created by      and revised by    under the guidance of Professor Rage Uday Kiran.
    """

    def __init__(self, dataframe):
        """
        Constructor to initialize the MultipleImputation object with the input DataFrame.

        :param dataframe: pandas DataFrame containing at least columns ['x', 'y'] and feature columns.
        """
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])
        self.imputedDF = None
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


    def run(self, n_nearest_features=None, max_iter=10, random_state=0):
        """
        Executes iterative multivariate imputation using linear regression.

        :param n_nearest_features: int or None, optional
            Number of features to use when estimating missing values. If None, all features are used.
        :param max_iter: int, default=10
            Maximum number of imputation iterations.
        :param random_state: int, default=0
            Seed for reproducibility.

        :return: pandas DataFrame with imputed values and original 'x', 'y' columns.
        """
        self.startTime = time.time()
        xy = self.df[['x', 'y']].reset_index(drop=True)
        data = self.df.drop(['x', 'y'], axis=1).reset_index(drop=True)
        imputedArray = IterativeImputer(estimator=LinearRegression(), n_nearest_features=n_nearest_features, max_iter=max_iter, random_state=random_state).fit_transform(data)
        imputedData = pd.DataFrame(imputedArray, columns=data.columns)
        self.imputedDF = pd.concat([xy, imputedData], axis=1)

        self.endTime = time.time()

        process = psutil.Process()
        self.memoryUSS = process.memory_full_info().uss / 1024
        self.memoryRSS = process.memory_full_info().rss / 1024

        return self.imputedDF


    def save(self, outputFile='MultipleImputation.csv'):
        """
        Saves the imputed DataFrame to a CSV file.

        :param outputFile: Filename to save the resulting DataFrame.
        """
        if self.imputedDF is not None:
            try:
                self.imputedDF.to_csv(outputFile, index=False)
                print(f"Imputed data saved to: {outputFile}")
            except Exception as e:
                print(f"Failed to save labels: {e}")
        else:
            print("No imputed data to save. Run impute() first")