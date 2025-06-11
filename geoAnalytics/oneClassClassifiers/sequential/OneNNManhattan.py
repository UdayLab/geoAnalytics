import pandas as pd
import numpy as np
import time
import psutil
from tqdm import tqdm
from numba import njit, prange, cuda


class OneNNManhattan:
    def __init__(self):
        pass

    @staticmethod
    def get_statistics(start_time):
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process()
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    @staticmethod
    def compute_manhattan_sequential(testing, training):
        test_np = testing.to_numpy()
        train_np = training.to_numpy()

        # Fix feature size mismatch
        if test_np.shape[1] != train_np.shape[1]:
            min_features = min(test_np.shape[1], train_np.shape[1])
            test_np = test_np[:, :min_features]
            train_np = train_np[:, :min_features]

        distances = np.min(np.sum(np.abs(test_np[:, None] - train_np), axis=2), axis=1)
        return distances.tolist()

    @staticmethod
    @njit(parallel=True)
    def compute_manhattan_parallel(test_np, train_np):
        num_test = test_np.shape[0]
        num_train = train_np.shape[0]
        distances = np.empty(num_test)

        for i in prange(num_test):
            min_dist = 1e10
            for j in range(num_train):
                dist = np.sum(np.abs(test_np[i] - train_np[j]))
                if dist < min_dist:
                    min_dist = dist
            distances[i] = min_dist
        return distances

    @staticmethod
    @cuda.jit
    def compute_manhattan_cuda_kernel(testing, training, result):
        i = cuda.grid(1)
        if i >= testing.shape[0]:
            return

        num_features = testing.shape[1]
        min_dist = 1e10
        for j in range(training.shape[0]):
            dist = 0.0
            for k in range(num_features):
                dist += abs(testing[i][k] - training[j][k])
            if dist < min_dist:
                min_dist = dist
        result[i] = min_dist

    @staticmethod
    def compute_manhattan_cuda(testing, training):
        test_np = testing.to_numpy().astype(np.float32)
        train_np = training.to_numpy().astype(np.float32)

        # Fix feature size mismatch
        if test_np.shape[1] != train_np.shape[1]:
            min_features = min(test_np.shape[1], train_np.shape[1])
            test_np = test_np[:, :min_features]
            train_np = train_np[:, :min_features]

        result = cuda.device_array(test_np.shape[0], dtype=np.float32)
        threads_per_block = 128
        blocks_per_grid = (test_np.shape[0] + threads_per_block - 1) // threads_per_block
        OneNNManhattan.compute_manhattan_cuda_kernel[blocks_per_grid, threads_per_block](test_np, train_np,
                                                                                               result)
        return result.copy_to_host()

    def run(self, training, testing, topK=-1, mode="sequential", algorithm="Manhattan"):
        start_time = time.time()

        if algorithm == "difManhattan":
            training = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]

        if mode == "sequential":
            distances = self.compute_manhattan_sequential(testing, training)
        elif mode == "parallel":
            distances = self.compute_manhattan_parallel(testing.to_numpy(), training.to_numpy())
        elif mode == "cuda":
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine.")
            distances = self.compute_manhattan_cuda(testing, training)
        else:
            raise ValueError("Invalid mode. Choose 'sequential', 'parallel', or 'cuda'")

        testing['1NNManhattan'] = distances
        sorted_df = testing.sort_values('1NNManhattan').head(topK)
        self.get_statistics(start_time)
        return testing, sorted_df
