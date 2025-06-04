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
from sklearn.cluster import OPTICS
import pandas as pd


class OpticsClustering:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])
        self.labelsDF = None

    def getStatistics(self, start_time):
        print("Total Execution time of proposed Algorithm:", time.time() - start_time)
        process = psutil.Process()
        memory_uss_kb = process.memory_full_info().uss / 1024
        print("Memory (USS) of proposed Algorithm in KB:", memory_uss_kb)
        memory_rss_kb = process.memory_full_info().rss / 1024
        print("Memory (RSS) of proposed Algorithm in KB:", memory_rss_kb)

    def clustering(self, min_samples=5, eps=None):
        start_time = time.time()
        data = np.ascontiguousarray(np.array(self.df.drop(['x', 'y'], axis=1)))
        OPTICS_Clustering = OPTICS(min_samples=min_samples, eps=eps).fit(data)
        label = self.df[['x', 'y']]
        self.labelsDF = label.assign(labels=OPTICS_Clustering.labels_)
        self.getStatistics(start_time)
        return self.labelsDF

    def save(self, outputFile='MEANshiftLabels.csv'):
        if self.labelsDF is not None:
            try:
                self.labelsDF.to_csv(outputFile, index=False)
                print(f"Labels saved to: {outputFile}")
            except Exception as e:
                print(f"Failed to save labels: {e}")
        else:
            print("No labels to save. Please run clustering first.")