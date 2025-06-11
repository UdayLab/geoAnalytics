import numpy as np
import time
import psutil
from tqdm import tqdm
from sklearn.svm import OneClassSVM


class OneNNSVM:
    def __init__(self):
        pass

    # ---------------------- Stats Logger ----------------------
    def getStatistics(self, start_time):
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process()
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    # ---------------------- One-Class SVM (sequential Sample Loop) ----------------------
    def compute_ocsvm_sequential(self, training_np, testing_np):
        print("Training One-Class SVM (sequential)...")
        model = OneClassSVM(kernel='rbf', gamma='scale')
        model.fit(training_np)

        print("Scoring test samples (loop)...")
        scores = np.array([model.decision_function([x])[0] for x in tqdm(testing_np, desc="Scoring (sequential)")])
        return scores

    # ---------------------- One-Class SVM (Vectorized Batch) ----------------------
    def compute_ocsvm_parallel(self, training_np, testing_np):
        print("Training One-Class SVM (parallel)...")
        model = OneClassSVM(kernel='rbf', gamma='scale')
        model.fit(training_np)

        print("Scoring test samples (batch)...")
        scores = model.decision_function(testing_np)
        return scores

    # ---------------------- Main Entry Point ----------------------
    def run(self, training, testing, topK=-1, mode="sequential", algorithm='OneClassSVM'):
        start_time = time.time()

        # Optional preprocessing
        if algorithm == "difOneClassSVM":
            training_df = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]

        # Convert to NumPy for faster processing
        training_np = training.to_numpy()
        testing_np = testing.to_numpy()

        # Scoring
        if mode == "sequential":
            scores = self.compute_ocsvm_sequential(training_np, testing_np)
        elif mode == "parallel":
            scores = self.compute_ocsvm_parallel(training_np, testing_np)
        else:
            raise ValueError("Invalid mode. Choose 'sequential' or 'parallel'")

        # Return DataFrame with scores and sorted top elements
        testing = testing.copy()
        testing['OCSVM_Score'] = scores
        sorted_df = testing.sort_values('OCSVM_Score', ascending=False).head(topK)

        self.getStatistics(start_time)
        return testing, sorted_df
