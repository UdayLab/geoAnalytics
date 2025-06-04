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
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])
        self.imputedDF = None

    def getStatistics(self, start_time):
        print("Total Execution time of proposed Algorithm:", time.time() - start_time)
        process = psutil.Process()
        memory_uss_kb = process.memory_full_info().uss / 1024
        print("Memory (USS) of proposed Algorithm in KB:", memory_uss_kb)
        memory_rss_kb = process.memory_full_info().rss / 1024
        print("Memory (RSS) of proposed Algorithm in KB:", memory_rss_kb)


    def impute(self):
        start_time = time.time()
        xy = self.df[['x', 'y']].reset_index(drop=True)
        data = self.df.drop(['x', 'y'], axis=1).reset_index(drop=True)
        imputedData = data.fillna(data.mean())
        self.imputedDF = pd.concat([xy, imputedData], axis=1)
        self.getStatistics(start_time)
        return self.imputedDF


    def save(self, outputFile='MeanImputation.csv'):
        if self.imputedDF is not None:
            try:
                self.imputedDF.to_csv(outputFile, index=False)
                print(f"Imputed data saved to: {outputFile}")
            except Exception as e:
                print(f"Failed to save labels: {e}")
        else:
            print("No imputed data to save. Run impute() first")