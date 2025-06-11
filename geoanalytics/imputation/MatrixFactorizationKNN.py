#MatrixFactorizationKNN uses `fancyimpute.KNN` to perform missing value imputation based on row-wise similarity, excluding 'x' and 'y' columns and restoring them after imputation.
#
# **Importing this algorithm into a Python program**
#
#           from geoanalytics.imputation import MatrixFactorizationKNN as alg
#
#           import pandas as pd
#
#           df = pd.read_csv('dataset.csv')
#
#           obj = alg.MatrixFactorizationKNN(df)
#
#           imputed_df = obj.run(k=5)
#
#           obj.save('MatrixFactorizationKNN.csv')
#
#           obj.getRuntime()
#
#           obj.getMemoryUSS()
#
#           obj.getMemoryRSS()
#
#           print("Data after KNN Imputation:", imputed_df)
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
from fancyimpute import KNN

class MatrixFactorizationKNN:
    """
    **About this algorithm**

    :**Description**:   MatrixFactorizationKNN applies KNN-based imputation using `fancyimpute.KNN` to estimate missing values based on similarity between rows. It excludes 'x' and 'y' during imputation and reports runtime and memory usage.

    :**Parameters**:    - **dataframe** (*pandas.DataFrame*) -- *Input dataset with 'x', 'y' spatial columns followed by features with possible missing values.*
                        - **k** (*int*) -- *Number of nearest neighbors used in the imputation (default is 5).*

    :**Attributes**:    - **df** (*pandas.DataFrame*) -- *Internal copy of the input DataFrame with reordered columns.*
                        - **imputedDF** (*pandas.DataFrame*) -- *Final DataFrame after applying KNN Imputation.*
                        - **startTime** (*float*) -- *Start time of the imputation.*
                        - **endTime** (*float*) -- *End time of the imputation.*
                        - **memoryUSS** (*float*) -- *Memory usage (USS in KB) during the run.*
                        - **memoryRSS** (*float*) -- *Memory usage (RSS in KB) during the run.*

    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            from geoanalytics.imputation import MatrixFactorizationKNN as alg

            import pandas as pd

            df = pd.read_csv('dataset.csv')

            obj = alg.MatrixFactorizationKNN(df)

            imputed_df = obj.run(k=5)

            obj.save('MatrixFactorizationKNN.csv')

            obj.getRuntime()

            obj.getMemoryUSS()

            obj.getMemoryRSS()

            print("Data after KNN Imputation:", imputed_df)


    **Credits**

    The complete program was written by               and revised by              under the supervision of Professor Rage Uday Kiran.
    """
    def __init__(self, dataframe):
        """
        Initializes the MatrixFactorizationKNN object with a copy of the dataframe.
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
        Prints the total runtime of the algorithm.
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


    def run(self, k = 5):
        """
        Executes KNN Imputation on the dataset (excluding 'x' and 'y' columns),and returns the imputed DataFrame with original coordinates.

        Parameters:
        -----------
        k : int
            Number of neighbors to use (default: 5)

        Returns:
        --------
        imputedDF : pandas.DataFrame
            The DataFrame with missing values imputed.
        """
        self.startTime = time.time()
        xy = self.df[['x', 'y']].reset_index(drop=True)
        data = self.df.drop(['x', 'y'], axis=1).reset_index(drop=True)
        imputedArray = KNN(k=k).fit_transform(data)
        imputedData = pd.DataFrame(imputedArray, columns=data.columns)
        self.imputedDF = pd.concat([xy, imputedData], axis=1)

        self.endTime = time.time()

        process = psutil.Process()
        self.memoryUSS = process.memory_full_info().uss / 1024
        self.memoryRSS = process.memory_full_info().rss / 1024

        return self.imputedDF


    def save(self, outputFile='MatrixFactorizationKNN.csv'):
        """
        Saves the imputed DataFrame to a CSV file.
        """
        if self.imputedDF is not None:
            try:
                self.imputedDF.to_csv(outputFile, index=False)
                print(f"Imputed data saved to: {outputFile}")
            except Exception as e:
                print(f"Failed to save labels: {e}")
        else:
            print("No imputed data to save. Run impute() first")