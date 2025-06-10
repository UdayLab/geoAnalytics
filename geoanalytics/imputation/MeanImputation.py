# MeanImputation fills missing values in a DataFrame using column-wise mean substitution, with performance tracking and optional CSV export.
#
# **Importing and Using the MeanImputation Class in a Python Program**
#
#             import pandas as pd
#
#             from geoanalytics.imputation import MeanImputation
#
#             df = pd.read_csv('data_with_nans.csv')
#
#             obj = MeanImputation(df)
#
#             imputed_df = obj.impute()
#
#             obj.save('MeanImputation.csv')
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

class MeanImputation:
    """
    **About this algorithm**

    :**Description**: MeanImputation fills missing values in a dataset by replacing them with the mean of their respective columns.

    :**Parameters**:    - **dataframe** (*pd.DataFrame*) -- A Pandas DataFrame containing missing values.
                        - The first two columns must represent spatial/positional attributes, typically 'x' and 'y'.

    :**Attributes**:    - **df** (*pd.DataFrame*) -- Original dataframe with renamed first two columns ('x', 'y') and copied features.
                        - **imputedDF** (*pd.DataFrame*) -- Stores the resulting dataframe after mean imputation.

    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            import pandas as pd

            from geoanalytics.imputation import MeanImputation

            df = pd.read_csv('data_with_nans.csv')

            obj = MeanImputation(df)

            imputed_df = obj.impute()

            obj.save('MeanImputation.csv')

    **Credits**

    The complete program was written by     and revised by     under the supervision of Professor Rage Uday Kiran.

    """
    def __init__(self, dataframe):
        """
        Constructor to initialize the MeanImputation object.

        :param dataframe: Input dataframe where missing values need to be imputed.
        :type dataframe: pd.DataFrame
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


    def run(self):
        """
        Performs mean imputation on all feature columns (excluding x and y).

        :return: DataFrame with 'x', 'y', and imputed features.
        :rtype: pd.DataFrame
        """
        self.startTime = time.time()
        xy = self.df[['x', 'y']].reset_index(drop=True)
        data = self.df.drop(['x', 'y'], axis=1).reset_index(drop=True)
        imputedData = data.fillna(data.mean())
        self.imputedDF = pd.concat([xy, imputedData], axis=1)

        self.endTime = time.time()

        process = psutil.Process()
        self.memoryUSS = process.memory_full_info().uss / 1024
        self.memoryRSS = process.memory_full_info().rss / 1024

        return self.imputedDF


    def save(self, outputFile='MeanImputation.csv'):
        """
        Saves the imputed DataFrame to a CSV file.

        :param outputFile: File path to save the output. Defaults to 'MeanImputation.csv'.
        :type outputFile: str
        """
        if self.imputedDF is not None:
            try:
                self.imputedDF.to_csv(outputFile, index=False)
                print(f"Imputed data saved to: {outputFile}")
            except Exception as e:
                print(f"Failed to save labels: {e}")
        else:
            print("No imputed data to save. Run impute() first")