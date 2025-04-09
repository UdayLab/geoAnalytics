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

    def getStatistics(self, start_time):
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process()
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    def clustering(self, min_samples=5, eps=None):
        start_time = time.time()
        data = np.ascontiguousarray(np.array(self.df.drop(['x', 'y'], axis=1)))
        OPTICS_Clustering = OPTICS(min_samples=min_samples, eps=eps).fit(data)
        label = self.df[['x', 'y']]
        labels = label.assign(labels=OPTICS_Clustering.labels_)
        self.getStatistics(start_time)
        return labels