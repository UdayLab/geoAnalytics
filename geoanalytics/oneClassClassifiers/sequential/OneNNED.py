import pandas as pd
import numpy as np
import time
import math
import os
import psutil
from tqdm import tqdm
from numba import njit, prange, cuda


class OneNNED:
    def __init__(self):
        pass

    @staticmethod
    def get_statistics(start_time):
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process(os.getpid())
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    # ---------------------- Optimized sequential Thread (NumPy Vectorization) ----------------------
    @staticmethod
    def compute_ed_sequential(testing, training):
        test_np = testing.to_numpy()
        train_np = training.to_numpy()

        # Ensure both have the same number of columns (features)
        if test_np.shape[1] != train_np.shape[1]:
            min_features = min(test_np.shape[1], train_np.shape[1])
            test_np = test_np[:, :min_features]
            train_np = train_np[:, :min_features]

        # Compute pairwise squared differences
        dists = np.sqrt(((test_np[:, np.newaxis, :] - train_np[np.newaxis, :, :]) ** 2).sum(axis=2))

        # Take the minimum distance for each test sample
        return np.min(dists, axis=1).tolist()

    # ---------------------- Parallel Version with Numba ----------------------
    @staticmethod
    @njit(parallel=True)
    def compute_ed_parallel(test_np, train_np):
        num_test = test_np.shape[0]
        num_train = train_np.shape[0]
        num_features = train_np.shape[1]
        distances = np.empty(num_test)

        for i in prange(num_test):
            min_dist = 1e10
            for j in range(num_train):
                sq = 0.0
                for k in range(num_features):
                    diff = test_np[i, k] - train_np[j, k]
                    sq += diff * diff
                dist = math.sqrt(sq)
                if dist < min_dist:
                    min_dist = dist
            distances[i] = min_dist

        return distances

    # ---------------------- CUDA Kernel ----------------------
    @staticmethod
    @cuda.jit
    def compute_ed_cuda_kernel(test_data, train_data, result):
        i = cuda.grid(1)
        if i < test_data.shape[0]:
            min_dist = 1e10
            for j in range(train_data.shape[0]):
                sq = 0.0
                for k in range(test_data.shape[1]):
                    diff = test_data[i][k] - train_data[j][k]
                    sq += diff * diff
                dist = math.sqrt(sq)
                if dist < min_dist:
                    min_dist = dist
            result[i] = min_dist

    @staticmethod
    def compute_ed_cuda(testing, training):
        test_np = testing.to_numpy().astype(np.float32)
        train_np = training.to_numpy().astype(np.float32)
        result = cuda.device_array(test_np.shape[0], dtype=np.float32)

        threads_per_block = 128
        blocks_per_grid = (test_np.shape[0] + threads_per_block - 1) // threads_per_block

        OneNNED.compute_ed_cuda_kernel[blocks_per_grid, threads_per_block](test_np, train_np, result)
        return result.copy_to_host()

    # ---------------------- Main Function ----------------------
    def run(self, training, testing, topK=-1, mode="sequential", algorithm='ED'):
        start_time = time.time()

        if algorithm == "difED":
            training = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]

        if mode == "sequential":
            distances = self.compute_ed_sequential(testing, training)
        elif mode == "parallel":
            distances = self.compute_ed_parallel(testing.to_numpy(), training.to_numpy())
        elif mode == "cuda":
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine.")
            distances = self.compute_ed_cuda(testing, training)
        else:
            raise ValueError("Invalid mode. Choose 'sequential', 'parallel', or 'cuda'")

        testing['1NNED'] = distances
        sorted_df = testing.sort_values('1NNED').head(topK)
        self.get_statistics(start_time)

        return testing, sorted_df