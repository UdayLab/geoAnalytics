import pandas as pd
import numpy as np
import time
import psutil
from tqdm import tqdm
from numba import njit, prange, cuda

class OneNNMaxNorm:
    def __init__(self):
        pass

    def getStatistics(self, start_time):
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process()
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    # ----------- Sequential Mode (Optimized) -----------
    def compute_maxnorm_sequential(self, testing, training):
        test_np = testing.to_numpy()
        train_np = training.to_numpy()

        if test_np.shape[1] != train_np.shape[1]:
            min_features = min(test_np.shape[1], train_np.shape[1])
            test_np = test_np[:, :min_features]
            train_np = train_np[:, :min_features]

        # Broadcasting: shape (num_test, 1, num_features) - (1, num_train, num_features)
        diff = np.abs(test_np[:, None, :] - train_np)
        max_diff = np.max(diff, axis=2)  # shape: (num_test, num_train)
        distances = np.min(max_diff, axis=1)
        return distances.tolist()

    # ----------- Parallel Mode -----------
    @staticmethod
    @njit(parallel=True)
    def compute_maxnorm_parallel(test_np, train_np):
        num_test = test_np.shape[0]
        num_train = train_np.shape[0]
        num_features = test_np.shape[1]
        distances = np.empty(num_test)

        for i in prange(num_test):
            min_dist = 1e10
            for j in range(num_train):
                max_diff = 0.0
                for k in range(num_features):
                    diff = abs(test_np[i, k] - train_np[j, k])
                    if diff > max_diff:
                        max_diff = diff
                if max_diff < min_dist:
                    min_dist = max_diff
            distances[i] = min_dist

        return distances

    # ----------- CUDA Mode -----------
    @staticmethod
    @cuda.jit
    def compute_maxnorm_cuda_kernel(testing, training, result):
        i = cuda.grid(1)
        if i >= testing.shape[0]:
            return

        num_features = testing.shape[1]
        min_dist = 1e10

        for j in range(training.shape[0]):
            max_diff = 0.0
            for k in range(num_features):
                diff = abs(testing[i][k] - training[j][k])
                if diff > max_diff:
                    max_diff = diff
            if max_diff < min_dist:
                min_dist = max_diff

        result[i] = min_dist

    def compute_maxnorm_cuda(self, testing, training):
        test_np = testing.to_numpy().astype(np.float32)
        train_np = training.to_numpy().astype(np.float32)

        if test_np.shape[1] != train_np.shape[1]:
            min_features = min(test_np.shape[1], train_np.shape[1])
            test_np = test_np[:, :min_features]
            train_np = train_np[:, :min_features]

        result = cuda.device_array(test_np.shape[0], dtype=np.float32)
        threads_per_block = 128
        blocks_per_grid = (test_np.shape[0] + threads_per_block - 1) // threads_per_block
        self.compute_maxnorm_cuda_kernel[blocks_per_grid, threads_per_block](test_np, train_np, result)

        return result.copy_to_host()

    # ----------- Entry Point -----------
    def run(self, training, testing, topK=-1, mode="sequential", algorithm="MaxNorm"):
        start_time = time.time()

        if algorithm == "difMaxNorm":
            training = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]

        if mode == "sequential":
            distances = self.compute_maxnorm_sequential(testing, training)
        elif mode == "parallel":
            distances = self.compute_maxnorm_parallel(testing.to_numpy(), training.to_numpy())
        elif mode == "cuda":
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine.")
            distances = self.compute_maxnorm_cuda(testing, training)
        else:
            raise ValueError("Invalid mode. Choose 'sequential', 'parallel', or 'cuda'")

        testing = testing.copy()
        testing["1NNmaxNorm"] = distances
        sorted_df = testing.sort_values("1NNmaxNorm").head(topK)
        self.getStatistics(start_time)
        return testing, sorted_df