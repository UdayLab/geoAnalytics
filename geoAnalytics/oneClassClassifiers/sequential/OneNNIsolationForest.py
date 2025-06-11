import pandas as pd
import numpy as np
import time
import psutil
from tqdm import tqdm
from sklearn.ensemble import IsolationForest


class OneNNIsolationForest:

    def __init__(self):
        pass

    # ---------------------- Stats Logger ----------------------
    def getStatistics(self, start_time):
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process()
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    # ---------------------- Isolation Forest Modes ----------------------
    def compute_iforest_sequential(self, training, testing):
        print("Training Isolation Forest (sequential)...")
        model = IsolationForest(n_estimators=100, contamination='auto', n_jobs=1, random_state=42)
        model.fit(training)

        print("Scoring test samples...")
        scores = model.decision_function(testing)
        return scores

    def compute_iforest_parallel(self, training, testing):
        print("Training Isolation Forest (parallel)...")
        model = IsolationForest(n_estimators=100, contamination='auto', n_jobs=-1, random_state=42)
        model.fit(training)

        print("Scoring test samples in parallel...")
        scores = model.decision_function(testing)
        return scores

    # ---------------------- Main Function ----------------------
    def run(self, training, testing, topK=-1, mode="sequential", algorithm='rasterIsolationForest'):
        start_time = time.time()

        if algorithm == "difrasterIsolationForest":
            training = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]

        if mode == "sequential":
            scores = self.compute_iforest_sequential(training, testing)
        elif mode == "parallel":
            scores = self.compute_iforest_parallel(training, testing)
        else:
            raise ValueError("Invalid mode. Choose 'sequential', or 'parallel'")

        testing['IForest_Score'] = scores
        sorted_df = testing.sort_values('IForest_Score', ascending=False).head(topK)
        self.getStatistics(start_time)
        return testing, sorted_df