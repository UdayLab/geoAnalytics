import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans as kmeansAlg
import pandas as pd


class KMeansPP:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])

    def getStatistics(self, start_time):
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process()
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    def elbowMethod(self):
        wcss = []
        k_values = range(1, 11)
        data = self.df.drop(['x', 'y'], axis=1)

        for k in tqdm(k_values):
            kmeans = kmeansAlg(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
        plt.plot(k_values, wcss, marker='o', linestyle='--')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS')
        plt.title('Elbow Method for Optimal k (Ignoring Location Columns)')
        plt.show()

    def clustering(self, k, max_iter=300):
        start_time = time.time()
        data = self.df.drop(['x', 'y'], axis=1)
        data = data.to_numpy()
        kmeans = kmeansAlg(n_clusters=k, max_iter=max_iter, init='k-means++').fit(data)
        label = self.df[['x', 'y']]
        labels = label.assign(labels=kmeans.labels_)
        self.getStatistics(start_time)
        return labels, kmeans.cluster_centers_