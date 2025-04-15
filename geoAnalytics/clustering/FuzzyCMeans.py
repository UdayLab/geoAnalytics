import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fcmeans import FCM
import pandas as pd


class FuzzyCMeans:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])

    def getStatistics(self, start_time):
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process()
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    def clustering(self, n_clusters=3):
        start_time = time.time()
        data = self.df.drop(['x', 'y'], axis=1)
        data = data.to_numpy()
        fcm = FCM(n_clusters=n_clusters)
        fcm.fit(data)
        fcmResult = fcm.predict(data)
        label = self.df[['x', 'y']]
        labels = label.assign(labels=fcmResult)
        self.getStatistics(start_time)
        return labels, fcm.centers