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
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer


class Normalization:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])
        self.normalizedDF = None
        self.startTime = None
        self.endTime = None
        self.memoryUSS = None
        self.memoryRSS = None

    def recordResources(self):
        process = psutil.Process()
        self.memoryUSS = process.memory_full_info().uss / 1024
        self.memoryRSS = process.memory_full_info().rss / 1024
        self.endTime = time.time()

    def prepareData(self):
        xy = self.df[['x', 'y']].reset_index(drop=True)
        data = self.df.drop(['x', 'y'], axis=1).reset_index(drop=True)
        return xy, data

    def finalize(self, xy, data):
        self.normalizedDF = pd.concat([xy, data], axis=1)
        self.recordResources()
        return self.normalizedDF

    def getRuntime(self):
        print("Total Execution Time:", self.endTime - self.startTime, "seconds")

    def getMemoryUSS(self):
        print("Memory (USS) in KB:", self.memoryUSS)

    def getMemoryRSS(self):
        print("Memory (RSS) in KB:", self.memoryRSS)

    def save(self, outputFile='NormalizedOutput.csv'):
        if self.normalizedDF is not None:
            self.normalizedDF.to_csv(outputFile, index=False)
            print(f"Saved to: {outputFile}")
        else:
            print("No normalized data found. Run a transformation method first.")

    def DecimalScaling(self):
        self.startTime = time.time()
        xy, data = self.prepareData()
        maxAbs = data.abs().max().max()
        divisor = 10 ** int(np.ceil(np.log10(maxAbs)))
        return self.finalize(xy, data / divisor)

    def LogTransformation(self):
        self.startTime = time.time()
        xy, data = self.prepareData()
        return self.finalize(xy, np.log1p(data))

    def RootTransformation(self, root=2):
        self.startTime = time.time()
        xy, data = self.prepareData()
        return self.finalize(xy, data ** (1 / root))

    def ZScore(self):
        self.startTime = time.time()
        xy, data = self.prepareData()
        scaler = StandardScaler()
        return self.finalize(xy, pd.DataFrame(scaler.fit_transform(data), columns=data.columns))

    def MinMax(self):
        self.startTime = time.time()
        xy, data = self.prepareData()
        scaler = MinMaxScaler()
        return self.finalize(xy, pd.DataFrame(scaler.fit_transform(data), columns=data.columns))

    def MaxAbs(self):
        self.startTime = time.time()
        xy, data = self.prepareData()
        scaler = MaxAbsScaler()
        return self.finalize(xy, pd.DataFrame(scaler.fit_transform(data), columns=data.columns))

    def RobustScaling(self):
        self.startTime = time.time()
        xy, data = self.prepareData()
        scaler = RobustScaler()
        return self.finalize(xy, pd.DataFrame(scaler.fit_transform(data), columns=data.columns))

    def PowerTransform(self):
        self.startTime = time.time()
        xy, data = self.prepareData()
        scaler = PowerTransformer(method='yeo-johnson')
        return self.finalize(xy, pd.DataFrame(scaler.fit_transform(data), columns=data.columns))

    def QuantileTransform(self):
        self.startTime = time.time()
        xy, data = self.prepareData()
        scaler = QuantileTransformer(output_distribution='normal')
        return self.finalize(xy, pd.DataFrame(scaler.fit_transform(data), columns=data.columns))

    def UnitVector(self):
        self.startTime = time.time()
        xy, data = self.prepareData()
        scaler = Normalizer(norm='l2')
        return self.finalize(xy, pd.DataFrame(scaler.fit_transform(data), columns=data.columns))