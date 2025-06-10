# Interpolation-based missing value imputation using linear estimation strategy.
#
# **Importing and Using the Interpolation Class in a Python Program**
#
#             import pandas as pd
#
#             from geoanalytics.imputation import Interpolation
#
#             df = pd.read_csv('input.csv')
#
#             ip = Interpolation(df)
#
#             output = ip.impute()
#
#             ip.save('Interpolation.csv')
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

class Interpolation:
    """
    **About this algorithm**

    :**Description**: Interpolation is a missing data imputation technique that estimates NaN values using linear interpolation between known values. It fills values in both forward and backward directions to ensure completeness.

    :**Parameters**:    - Dataset (pandas DataFrame) must be provided during object initialization.
                        - No other parameters are required during instantiation.

    :**Attributes**:    - **df** (*pd.DataFrame*) -- The input data with 'x', 'y' coordinates and feature columns.
                        - **imputedDF** (*pd.DataFrame*) -- DataFrame containing 'x', 'y', and imputed values.

    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            import pandas as pd

            from geoanalytics.imputation import Interpolation

            df = pd.read_csv("input.csv")

            ip = Interpolation(df)

            output = ip.impute()

            ip.save('Interpolation.csv')

    **Credits**

    This implementation was created by     and revised by   under the guidance of Professor Rage Uday Kiran.
    """
    def __init__(self, dataframe):
        """
        Constructor to initialize the Interpolation object with the input DataFrame.

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
        Performs linear interpolation to fill missing values in the dataset.

        The method:
        - Separates 'x' and 'y' columns from the rest of the data.
        - Applies linear interpolation along each column.
        - Uses both forward and backward filling to handle edge NaNs.
        - Reconstructs the complete DataFrame with imputed values.

        :return: Imputed DataFrame with original 'x', 'y' columns and interpolated features.
        """
        self.startTime = time.time()
        xy = self.df[['x', 'y']].reset_index(drop=True)
        data = self.df.drop(['x', 'y'], axis=1).reset_index(drop=True)
        interpolatedData = data.interpolate(method='linear', limit_direction='both')
        self.imputedDF = pd.concat([xy, interpolatedData], axis=1)

        self.endTime = time.time()

        process = psutil.Process()
        self.memoryUSS = process.memory_full_info().uss / 1024
        self.memoryRSS = process.memory_full_info().rss / 1024

        return self.imputedDF


    def save(self, outputFile='Interpolation.csv'):
        """
        Saves the imputed DataFrame to a CSV file.

        :param outputFile: The filename for saving the DataFrame. Defaults to 'Interpolation.csv'.
        """
        if self.imputedDF is not None:
            try:
                self.imputedDF.to_csv(outputFile, index=False)
                print(f"Imputed data saved to: {outputFile}")
            except Exception as e:
                print(f"Failed to save labels: {e}")
        else:
            print("No imputed data to save. Run impute() first")