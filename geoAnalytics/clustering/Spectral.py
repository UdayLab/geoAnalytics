import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
import pandas as pd


class Spectral:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])

    def getStatistics(self, start_time):
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process()
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    def clustering(self, n_clusters=8, assign_labels='discretize'):
        start_time = time.time()
        data = np.ascontiguousarray(np.array(self.df.drop(['x', 'y'], axis=1)))
        spectralClustering = SpectralClustering(assign_labels=assign_labels, n_clusters=n_clusters, random_state=0).fit(data)
        label = self.df[['x', 'y']]
        labels = label.assign(labels=spectralClustering.labels_)
        self.getStatistics(start_time)
        return labels