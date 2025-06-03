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
        self.labelsDF = None

    def getStatistics(self, start_time):
        print("Total Execution time of proposed Algorithm:", time.time() - start_time)
        process = psutil.Process()
        memory_uss_kb = process.memory_full_info().uss / 1024
        print("Memory (USS) of proposed Algorithm in KB:", memory_uss_kb)
        memory_rss_kb = process.memory_full_info().rss / 1024
        print("Memory (RSS) of proposed Algorithm in KB:", memory_rss_kb)

    def clustering(self, damping, max_iter=300, convergence_iter=15, affinity='euclidean', random_state=None, preference=None):
        start_time = time.time()
        data = self.df.drop(['x', 'y'], axis=1)
        data = data.to_numpy()
        X = StandardScaler().fit_transform(data)
        affinityProp = AffinityPropagation(damping=float(damping), max_iter=int(max_iter),
                                           convergence_iter=int(convergence_iter), affinity=affinity,
                                           preference=preference, random_state=random_state).fit(X)
        label = self.df[['x', 'y']]
        self.labelsDF = label.assign(labels=affinityProp.labels_)
        self.getStatistics(start_time)
        return self.labelsDF, affinityProp.cluster_centers_

    def save(self, outputFile='AffinityLabels.csv'):
        if self.labelsDF is not None:
            try:
                self.labelsDF.to_csv(outputFile, index=False)
                print(f"Labels saved to: {outputFile}")
            except Exception as e:
                print(f"Failed to save labels: {e}")
        else:
            print("No labels to save. Please run clustering first.")