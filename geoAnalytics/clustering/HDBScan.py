import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import hdbscan as hdbscan
import pandas as pd


class HDBScan:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])

    def getStatistics(self, start_time):
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process()
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    def clustering(self, min_sample, min_cluster_size):
        start_time = time.time()
        data = self.df.drop(['x', 'y'], axis=1)
        data = data.to_numpy()
        hdbs = hdbscan.HDBSCAN(min_samples=min_sample, min_cluster_size=min_cluster_size, core_dist_n_jobs=1).fit(data)
        label = self.df[['x', 'y']]
        labels = label.assign(labels=hdbs.labels_)
        self.getStatistics(start_time)
        return labels