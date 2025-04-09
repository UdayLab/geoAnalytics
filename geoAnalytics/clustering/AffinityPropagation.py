import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
import pandas as pd


class AffinityPropagationWrapper:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])

    def getStatistics(self, start_time):
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process()
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    def clustering(self, damping, max_iter=300, convergence_iter=15, affinity='euclidean', random_state=None, preference=None):
        start_time = time.time()
        data = self.df.drop(['x', 'y'], axis=1)
        data = data.to_numpy()
        X = StandardScaler().fit_transform(data)
        affinityProp = AffinityPropagation(damping=float(damping), max_iter=int(max_iter),
                                           convergence_iter=int(convergence_iter), affinity=affinity,
                                           preference=preference, random_state=random_state).fit(X)
        label = self.df[['x', 'y']]
        labels = label.assign(labels=affinityProp.labels_)
        self.getStatistics(start_time)
        return labels, affinityProp.cluster_centers_