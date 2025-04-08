import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import DBSCAN as DBSCAN
import pandas as pd


class DBScan:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])

    def getStatistics(self, start_time):
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process()
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    def clustering(self, ep, min_sample):
        start_time = time.time()
        data = self.df.drop(['x', 'y'], axis=1)
        data = data.to_numpy()
        dbs = DBSCAN(eps=ep, min_samples=min_sample).fit(data)
        label = self.df[['x', 'y']]
        labels = label.assign(labels=dbs.labels_)
        self.getStatistics(start_time)
        return labels