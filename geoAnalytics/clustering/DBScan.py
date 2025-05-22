import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import DBSCAN as DBSCAN
import pandas as pd


class DBScan:
     """
    A wrapper class for performing DBSCAN clustering on a given DataFrame.
    
    This class uses the first two columns of the DataFrame as 'x' and 'y' coordinates,
    and applies DBSCAN on the remaining features. It also logs execution time and memory usage.
    """
    def __init__(self, dataframe):
         """
        Initialize the DBScan class with the input DataFrame.

        Args:
            dataframe (pd.DataFrame): A pandas DataFrame with at least two columns.
                                      The first two columns are treated as 'x' and 'y' coordinates.
        """
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + list(self.df.columns[2:])

    def getStatistics(self, start_time):
        """
        Print the execution time and memory usage of the clustering process.

        Args:
            start_time (float): The start time recorded before running the clustering method.
        """
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process()
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    def clustering(self, ep, min_sample):
        """
        Perform DBSCAN clustering on the DataFrame (excluding 'x' and 'y' columns).

        Args:
            ep (float): The maximum distance between two samples for them to be considered as neighbors.
            min_sample (int): The number of samples in a neighborhood for a point to be a core point.

        Returns:
            pd.DataFrame: A DataFrame with 'x', 'y', and cluster labels for each point.
        """
        start_time = time.time()
        data = self.df.drop(['x', 'y'], axis=1)
        data = data.to_numpy()
        dbs = DBSCAN(eps=ep, min_samples=min_sample).fit(data)
        label = self.df[['x', 'y']]
        labels = label.assign(labels=dbs.labels_)
        self.getStatistics(start_time)
        return labels
