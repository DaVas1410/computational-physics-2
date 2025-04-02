# Import libraries
import numpy as np
import time
from joblib import Parallel, delayed
import multiprocessing as mp

def matrix_mult(seed):
    """Matrix multiplication."""
    np.random.seed(seed)
    n_size = 100
    matrix1 = np.random.rand(n_size, n_size)
    matrix2 = np.random.rand(n_size, n_size)
    result = np.matmul(matrix1, matrix2)
    return np.sum(result)  # Return a scalar to avoid large result transfer

def run_serial(n):
    start_time = time.time()
    results = [matrix_mult(i) for i in range(n)]
    end_time = time.time()
    return end_time - start_time

def run_multiprocessing(n, n_processes):
    start_time = time.time()
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(matrix_mult, range(n))
    end_time = time.time()
    return end_time - start_time

def run_joblib(n, n_processes):
    start_time = time.time()
    results = Parallel(n_jobs=n_processes)(delayed(matrix_mult)(i) for i in range(n))
    end_time = time.time()
    return end_time - start_time   
    
if __name__ == '__main__':
    
    # Test parameters
    n_instances = 10000
    n_processes = 4 
    
    # Run and time the functions
    serial_time = run_serial(n_instances)
    joblib_time = run_joblib(n_instances, n_processes)
    multiprocessing_time = run_multiprocessing(n_instances, n_processes)
    
    # Print results
    print(f"Serial time: {serial_time:.4f} seconds")
    print(f"Joblib time: {joblib_time:.4f} seconds")
    print(f"Multiprocessing time: {multiprocessing_time:.4f} seconds")
