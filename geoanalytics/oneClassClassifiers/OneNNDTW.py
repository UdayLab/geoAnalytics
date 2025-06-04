import pandas as pd
import numpy as np
import time
import os
import psutil
from tqdm import tqdm
from numba import njit, prange, cuda

# =================== Numba-compatible Functions OUTSIDE the class ===================

@njit
def dtw_numba(A, B):
    N, M = len(A), len(B)
    d = np.zeros((N, M))
    for n in range(N):
        for m in range(M):
            d[n][m] = (A[n] - B[m]) ** 2
    D = np.zeros((N, M))
    D[0][0] = d[0][0]
    for n in range(1, N):
        D[n][0] = d[n][0] + D[n - 1][0]
    for m in range(1, M):
        D[0][m] = d[0][m] + D[0][m - 1]
    for n in range(1, N):
        for m in range(1, M):
            D[n][m] = d[n][m] + min(D[n - 1][m], D[n - 1][m - 1], D[n][m - 1])
    return D[N - 1][M - 1]

@njit(parallel=True)
def compute_dtw_parallel(testing_data, training_data):
    num_test = testing_data.shape[0]
    num_train = training_data.shape[0]
    distances = np.empty(num_test)
    for i in prange(num_test):
        min_dist = 1e10
        for j in range(num_train):
            dist = dtw_numba(testing_data[i], training_data[j])
            if dist < min_dist:
                min_dist = dist
        distances[i] = min_dist
    return distances


@njit
def dtw_manual(A, B):
    """Manually computes DTW distance using explicit formula."""
    N, M = len(A), len(B)
    d = np.zeros((N, M))
    for n in range(N):
        for m in range(M):
            d[n, m] = (A[n] - B[m]) ** 2
    D = np.zeros((N, M))
    D[0, 0] = d[0, 0]
    for n in range(1, N):
        D[n, 0] = d[n, 0] + D[n - 1, 0]
    for m in range(1, M):
        D[0, m] = d[0, m] + D[0, m - 1]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = d[n, m] + min(D[n - 1, m], D[n - 1, m - 1], D[n, m - 1])
    return D[N - 1, M - 1]

@njit(parallel=True)
def compute_dtw_sequential_numba(testing_np, training_np):
    """Optimized sequential-threaded DTW computation using explicit formula."""
    num_test = testing_np.shape[0]
    num_train = training_np.shape[0]
    distances = np.full(num_test, np.inf)  # Preallocate with large values

    for i in prange(num_test):  # Parallel loop with Numba
        for j in range(num_train):
            dist = dtw_manual(testing_np[i], training_np[j])  # Using optimized DTW
            if dist < distances[i]:
                distances[i] = dist

    return distances

# =================== Main Class ===================

class OneNNDTW:
    @staticmethod
    def get_statistics(start_time):
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process(os.getpid())
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    @staticmethod
    @cuda.jit
    def dtw_cuda_kernel(testing_data, training_data, result):
        i = cuda.grid(1)
        if i < testing_data.shape[0]:
            min_dist = 1e10
            for j in range(training_data.shape[0]):
                dist = 0.0
                for k in range(testing_data.shape[1]):
                    val = testing_data[i][k] - training_data[j][k]
                    dist += val * val
                if dist < min_dist:
                    min_dist = dist
            result[i] = min_dist

    @staticmethod
    def compute_dtw_cuda(testing, training):
        test_np = testing.to_numpy().astype(np.float32)
        train_np = training.to_numpy().astype(np.float32)
        result = cuda.device_array(test_np.shape[0], dtype=np.float32)
        threads_per_block = 128
        blocks_per_grid = (test_np.shape[0] + threads_per_block - 1) // threads_per_block
        OneNNDTW.dtw_cuda_kernel[blocks_per_grid, threads_per_block](test_np, train_np, result)
        return result.copy_to_host()

    def compute_dtw_sequential(self, testing, training):
        """Wrapper to convert DataFrame to NumPy and apply optimized function."""
        testing_np = testing.to_numpy()
        training_np = training.to_numpy()
        distances = compute_dtw_sequential_numba(testing_np, training_np)
        return distances.tolist()

    def run(self, training, testing, topK=-1, mode="sequential", algorithm="DTW"):
        start_time = time.time()

        if algorithm == "difDTW":
            training = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]

        if mode == "sequential":
            distances = self.compute_dtw_sequential(testing, training)
        elif mode == "parallel":
            distances = compute_dtw_parallel(testing.to_numpy(), training.to_numpy())
        elif mode == "cuda":
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine.")
            distances = self.compute_dtw_cuda(testing, training)
        else:
            raise ValueError("Invalid mode. Choose 'sequential', 'parallel', or 'cuda'")

        testing['1NNDTW'] = distances
        sorted_df = testing.sort_values('1NNDTW').head(topK)
        self.get_statistics(start_time)
        return testing, sorted_df
