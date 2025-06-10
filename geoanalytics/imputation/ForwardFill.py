# Forward Fill is an imputation method that fills NaNs using the previous valid value, with backward fill as a fallback for trailing NaNs.
#
# **Importing and Using the ForwardFill Class in a Python Program**
#
#         import pandas as pd
#
#         from geoanalytics.imputation import ForwardFill
#
#         df = pd.read_csv('input.csv')
#
#         ff = ForwardFill(df)
#
#         imputed_df = ff.impute()
#
#         ff.save('ForwardFilled.csv')
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

class ForwardFill:
    """
    **About this algorithm**

    :**Description**: Forward Fill imputes missing values using the previous valid entry, with backward fill as a fallback for trailing NaNs.

    :**Parameters**:    - **dataframe** (*pd.DataFrame*): Input dataset where the first two columns are assumed to be spatial ('x', 'y') and all remaining columns are treated as features to be imputed.

    :**Attributes**:    - **df** (*pd.DataFrame*): Copy of the input data, with the first two columns renamed to 'x', 'y'.
                        - **imputedDF** (*pd.DataFrame*): Output DataFrame after missing value imputation.

    **Execution methods**

    .. code-block:: python

            import pandas as pd

            from geoanalytics.imputation import ForwardFill

            df = pd.read_csv("input.csv")

            ff = ForwardFill(df)

            imputed_df = ff.impute()

            ff.save("ForwardFilled.csv")

    **Credits**

    This implementation was created and revised under the guidance of Professor Rage Uday Kiran.
    """

    def __init__(self, dataframe):
        """
        Constructor to initialize the ForwardFill object with input DataFrame.

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
        Applies forward fill imputation followed by backward fill as fallback.

        :return: A pandas DataFrame with the original spatial coordinates and imputed feature values.
        """
        self.startTime = time.time()
        xy = self.df[['x', 'y']].reset_index(drop=True)
        data = self.df.drop(['x', 'y'], axis=1)
        imputed = data.fillna(method='ffill').fillna(method='bfill')
        self.imputedDF = pd.concat([xy, imputed.reset_index(drop=True)], axis=1)

        self.endTime = time.time()

        process = psutil.Process()
        self.memoryUSS = process.memory_full_info().uss / 1024
        self.memoryRSS = process.memory_full_info().rss / 1024

        return self.imputedDF


    def save(self, outputFile='ForwardFilled.csv'):
        """
        Saves the imputed DataFrame to a CSV file.

        :param outputFile: The filename for saving the DataFrame. Defaults to 'ForwardFilled.csv'.
        """
        if self.imputedDF is not None:
            try:
                self.imputedDF.to_csv(outputFile, index=False)
                print(f"Imputed data saved to: {outputFile}")
            except Exception as e:
                print(f"Failed to save labels: {e}")
        else:
            print("No imputed data to save. Run impute() first")