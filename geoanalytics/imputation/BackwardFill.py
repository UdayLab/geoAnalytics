# BackwardFill imputes missing values using the next valid entry in each column, with forward fill as a fallback for initial NaNs.
#
# **Importing and Using the BackwardFill Class in a Python Program**
#
#         import pandas as pd
#
#         from geoanalytics.imputation import BackwardFill
#
#         df = pd.read_csv('input.csv')
#
#         imputer = BackwardFill(df)
#
#         output = imputer.impute()
#
#         imputer.save('BackwardFilled.csv')
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

class BackwardFill:
    """
    **About this algorithm**

    :**Description**: Backward Fill imputes missing values using the next valid observation, with forward fill as a fallback for leading NaNs.

    :**Parameters**:    - **dataframe** (*pd.DataFrame*) -- Input dataset containing spatial columns ('x', 'y') followed by features with potential missing values.

    :**Attributes**:    - **df** (*pd.DataFrame*) -- Cleaned copy of input data with 'x', 'y' as first two columns.
                        - **imputedDF** (*pd.DataFrame*) -- Resulting DataFrame after imputation, preserving spatial columns.

    **Execution methods**

    .. code-block:: python

            import pandas as pd

            from geoanalytics.imputation import BackwardFill

            df = pd.read_csv("input.csv")

            imputer = BackwardFill(df)

            output = imputer.impute()

            imputer.save("BackwardFilled.csv")

    **Credits**

    This implementation was created and revised under the guidance of Professor Rage Uday Kiran.
    """
    def __init__(self, dataframe):
        """
        Constructor to initialize the BackwardFill object with input DataFrame.

        :param dataframe: A pandas DataFrame with at least two spatial columns and feature columns.
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
        Applies backward fill followed by forward fill to impute missing values in the feature columns.

        :return: A DataFrame containing original 'x', 'y' columns and imputed feature columns.
        """
        self.startTime = time.time()
        xy = self.df[['x', 'y']].reset_index(drop=True)
        data = self.df.drop(['x', 'y'], axis=1)
        imputed = data.fillna(method='bfill').fillna(method='ffill')
        self.imputedDF = pd.concat([xy, imputed.reset_index(drop=True)], axis=1)

        self.endTime = time.time()

        process = psutil.Process()
        self.memoryUSS = process.memory_full_info().uss / 1024
        self.memoryRSS = process.memory_full_info().rss / 1024

        return self.imputedDF

    def save(self, outputFile='BackwardFilled.csv'):
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