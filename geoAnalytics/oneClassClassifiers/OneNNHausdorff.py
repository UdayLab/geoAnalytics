import pandas as pd
import numpy as np
import time
import psutil
from tqdm import tqdm
from numba import njit, prange, cuda

# ---------------------- Numba Hausdorff (Optimized) ----------------------
@njit
def hausdorff_numba(u, v):
    row = u.shape[0]
    lea_distance = 0.0
    for i in range(row):
        diffs = np.empty(row - 1)
        for k in range(row - 1):
            diffs[k] = abs((u[i] - v[k + 1]) - (u[i] - v[k]))
        distance1 = np.min(diffs)
        if distance1 > lea_distance:
            lea_distance = distance1
    return lea_distance

@njit
def hausdorff_distance(u, v):
    return max(hausdorff_numba(u, v), hausdorff_numba(v, u))

@njit
def compute_hausdorff_sequential(test_np, train_np):
    num_test = test_np.shape[0]
    num_train = train_np.shape[0]
    distances = np.empty(num_test)
    for i in range(num_test):
        min_dist = 1e10
        for j in range(num_train):
            dist = hausdorff_distance(test_np[i], train_np[j])
            if dist < min_dist:
                min_dist = dist
        distances[i] = min_dist
    return distances

@njit(parallel=True)
def compute_hausdorff_parallel(test_np, train_np):
    num_test = test_np.shape[0]
    num_train = train_np.shape[0]
    distances = np.empty(num_test)
    for i in prange(num_test):
        min_dist = 1e10
        for j in range(num_train):
            dist = hausdorff_distance(test_np[i], train_np[j])
            if dist < min_dist:
                min_dist = dist
        distances[i] = min_dist
    return distances

# ---------------------- Class Wrapper ----------------------
class OneNNHausdorff:
    def get_statistics(self, start_time):
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process()
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    @staticmethod
    @cuda.jit
    def custom_hausdorff_kernel(testing, training, result):
        test_idx = cuda.grid(1)
        if test_idx >= testing.shape[0]:
            return

        num_features = testing.shape[1]
        min_total_dist = 1e10

        for train_idx in range(training.shape[0]):
            max_min1 = 0.0
            for i in range(num_features):
                min_diff = 1e10
                for k in range(num_features - 1):
                    diff = abs((testing[test_idx][i] - training[train_idx][k + 1]) -
                               (testing[test_idx][i] - training[train_idx][k]))
                    if diff < min_diff:
                        min_diff = diff
                if min_diff > max_min1:
                    max_min1 = min_diff

            max_min2 = 0.0
            for i in range(num_features):
                min_diff = 1e10
                for k in range(num_features - 1):
                    diff = abs((training[train_idx][i] - testing[test_idx][k + 1]) -
                               (training[train_idx][i] - testing[test_idx][k]))
                    if diff < min_diff:
                        min_diff = diff
                if min_diff > max_min2:
                    max_min2 = min_diff

            final_dist = max(max_min1, max_min2)
            if final_dist < min_total_dist:
                min_total_dist = final_dist

        result[test_idx] = min_total_dist

    def compute_hausdorff_cuda(self, testing, training):
        test_np = testing.to_numpy().astype(np.float64)
        train_np = training.to_numpy().astype(np.float64)
        result = cuda.device_array(test_np.shape[0], dtype=np.float64)

        threads_per_block = 128
        blocks_per_grid = (test_np.shape[0] + threads_per_block - 1) // threads_per_block
        self.custom_hausdorff_kernel[blocks_per_grid, threads_per_block](test_np, train_np, result)

        return result.copy_to_host()

    def run(self, training, testing, topK=-1, mode="sequential", algorithm='Hausdorff'):
        start_time = time.time()

        if algorithm == "difHausdorff":
            training = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]

        test_np = testing.to_numpy()
        train_np = training.to_numpy()

        if mode == "sequential":
            distances = compute_hausdorff_sequential(test_np, train_np)
        elif mode == "parallel":
            distances = compute_hausdorff_parallel(test_np, train_np)
        elif mode == "cuda":
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine.")
            distances = self.compute_hausdorff_cuda(testing, training)
        else:
            raise ValueError("Invalid mode. Choose 'sequential', 'parallel', or 'cuda'")

        testing['1NNHausdorff'] = distances
        sorted_df = testing.sort_values('1NNHausdorff').head(topK)
        self.get_statistics(start_time)
        return testing, sorted_df