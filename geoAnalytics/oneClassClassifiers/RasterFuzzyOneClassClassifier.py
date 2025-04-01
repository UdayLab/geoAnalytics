import pandas as pd
from tqdm import tqdm
from numba import njit, prange, cuda
import numpy as np
import time
import gc
import os
import psutil
import sys


class RasterFuzzyOneClassClassifier:
    def __init__(self):
        pass

    def getStatistics(self, startTime):
        """Statistics of the Algorithm after execution"""
        print("Total Execution time of proposedAlgo:", time.time() - startTime)
        process = psutil.Process(os.getpid())
        memory = process.memory_full_info().uss
        memory_in_KB = memory / 1024
        print("Memory of proposedAlgo in KB:", memory_in_KB)

    def rasterFuzzyTSC(self, training, testing, top_elements=-1, mode="single", algorithm="rasterMiner"):
        start_time = time.time()

        if algorithm == "difRasterMiner":
            training = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]

        # Calculate max, mean, and min values for each feature in the training dataset
        max_data = training.max(axis=0).values
        mean_data = training.mean(axis=0).values
        min_data = training.min(axis=0).values

        # Clean up training data
        del training
        gc.collect()

        # Convert testing dataset to numpy array for compatibility with Numba and CUDA
        testing_array = testing.to_numpy()
        num_rows, num_columns = testing_array.shape

        if mode == "single":
            rd_scores = self.compute_rd_single(testing_array, mean_data, min_data, max_data, num_columns)
        elif mode == "parallel":
            rd_scores = self.compute_rd_parallel(testing_array, mean_data, min_data, max_data, num_columns)
        elif mode == "cuda":
            rd_scores = self.compute_rd_cuda(testing_array, mean_data, min_data, max_data, num_columns)
        else:
            raise ValueError("Invalid mode. Choose 'single', 'parallel', or 'cuda'.")

        # Add RD scores to the testing DataFrame
        testing['RD'] = rd_scores

        # Retrieve top elements based on RD scores
        sorted_df = testing.sort_values('RD').head(top_elements)

        # Log statistics
        self.getStatistics(start_time)

        return testing, sorted_df

    @staticmethod
    def compute_rd_single(testing_array, mean_data, min_data, max_data, num_columns):
        """Compute RD scores for single-threaded execution."""
        num_rows = testing_array.shape[0]
        rd_scores = np.zeros(num_rows)

        for i in tqdm(range(num_rows)):
            counter = 0.0
            for j in range(1, num_columns):  # Skip the first column
                value = testing_array[i, j]
                if value >= mean_data[j]:
                    if value <= max_data[j]:
                        counter += 0.5 * (value - mean_data[j]) / (max_data[j] - mean_data[j])
                    else:
                        counter += 1
                else:
                    if value >= min_data[j]:
                        counter += 0.5 * (mean_data[j] - value) / (mean_data[j] - min_data[j])
                    else:
                        counter += 1
            rd_scores[i] = counter / num_columns

        return rd_scores

    @staticmethod
    @njit(parallel=True)
    def compute_rd_parallel(testing_array, mean_data, min_data, max_data, num_columns):
        """Compute RD scores using multi-threaded parallel processing with Numba."""
        num_rows = testing_array.shape[0]
        rd_scores = np.zeros(num_rows)

        for i in prange(num_rows):
            counter = 0.0
            for j in range(1, num_columns):  # Skip the first column
                value = testing_array[i, j]
                if value >= mean_data[j]:
                    if value <= max_data[j]:
                        counter += 0.5 * (value - mean_data[j]) / (max_data[j] - mean_data[j])
                    else:
                        counter += 1
                else:
                    if value >= min_data[j]:
                        counter += 0.5 * (mean_data[j] - value) / (mean_data[j] - min_data[j])
                    else:
                        counter += 1
            rd_scores[i] = counter / num_columns

        return rd_scores

    @staticmethod
    @cuda.jit
    def compute_rd_kernel(testing_array, mean_data, min_data, max_data, rd_scores, num_columns):
        """CUDA kernel for RD score computation."""
        i = cuda.grid(1)
        if i < testing_array.shape[0]:  # Ensure index is within bounds
            counter = 0.0
            for j in range(1, num_columns):  # Skip the first column
                value = testing_array[i, j]
                if value >= mean_data[j]:
                    if value <= max_data[j]:
                        counter += 0.5 * (value - mean_data[j]) / (max_data[j] - mean_data[j])
                    else:
                        counter += 1
                else:
                    if value >= min_data[j]:
                        counter += 0.5 * (mean_data[j] - value) / (mean_data[j] - min_data[j])
                    else:
                        counter += 1
            rd_scores[i] = counter / num_columns

    def compute_rd_cuda(self, testing_array, mean_data, min_data, max_data, num_columns):
        """Compute RD scores using CUDA for GPU acceleration."""
        num_rows = testing_array.shape[0]

        # Allocate GPU memory
        d_testing_array = cuda.to_device(testing_array)
        d_mean_data = cuda.to_device(mean_data)
        d_min_data = cuda.to_device(min_data)
        d_max_data = cuda.to_device(max_data)
        d_rd_scores = cuda.device_array(num_rows, dtype=np.float32)

        # Define thread and block configuration
        threads_per_block = 256
        blocks_per_grid = (num_rows + (threads_per_block - 1)) // threads_per_block

        # Launch CUDA kernel
        self.compute_rd_kernel[blocks_per_grid, threads_per_block](
            d_testing_array, d_mean_data, d_min_data, d_max_data, d_rd_scores, num_columns
        )

        # Copy RD scores back to host
        return d_rd_scores.copy_to_host()
