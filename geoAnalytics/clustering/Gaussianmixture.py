import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import pandas as pd


class Gaussianmixture:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])

    def getStatistics(self, start_time):
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process()
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    def clustering(self, n_components=1, max_iters=100, covariance_type="full", init_params='kmeans', random_state=0):
        start_time = time.time()
        data = self.df.drop(['x', 'y'], axis=1)
        data = data.to_numpy()
        gaussianMixture = GaussianMixture(n_components=n_components, max_iter=max_iters,
                                          covariance_type=covariance_type,
                                          init_params=init_params, random_state=random_state).fit(data)
        gaussianResult = gaussianMixture.predict(data)
        label = self.df[['x', 'y']]
        labels = label.assign(labels=gaussianResult)
        self.getStatistics(start_time)
        return labels, gaussianMixture.weights_, gaussianMixture.means_