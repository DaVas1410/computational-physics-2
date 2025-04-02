"""
Script to compute execution times.
"""
# Import libraries
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import random_square

# Serial function
def serial(n):
    """
    Function for serial execution.
    """
    # Time stamp
    start = time.time()
    results = []

    # Loop
    for i in range(n):
        results.append(random_square.random_square(i))

    # Time stamp
    end = time.time()
    exec_time = end - start

    return exec_time

# Parallel function
def parallel(n, n_cpu):
    """
    Function for parallel execution.
    Inputs: n -> # instances
    Outputs: n_cpu -> # of cores
    """
    # Time stamp
    start = time.time()

    # Call joblib
    Parallel(n_jobs = n_cpu)(delayed(random_square.random_square)(i) for i in range(n))

    # Time stamp
    end = time.time()
    exec_time = end - start

    return exec_time

# Generate numbers in log-space:
n_run = np.logspace(1, 7, num = 7)

# Call functions for each n_run[i]

t_serial = np.array([serial(int(n)) for n in n_run])

t_parallel_n02 = np.array([parallel(int(n), 2) for n in n_run])
t_parallel_n04 = np.array([parallel(int(n), 4) for n in n_run])
t_parallel_n08 = np.array([parallel(int(n), 8) for n in n_run])
t_parallel_n16 = np.array([parallel(int(n), 16) for n in n_run])

# Plotting
plt.figure(figsize = (9, 5))

plt.plot(n_run, t_serial, '-o', color = "red", label = 'serial')
plt.plot(n_run, t_parallel_n02, '-o', color = "green", label = 'parallel: n=2')
plt.plot(n_run, t_parallel_n04, '-o', color = "blue", label = 'parallel: n=4')
plt.plot(n_run, t_parallel_n08, '-o', color = "magenta", label = 'parallel: n=8')
plt.plot(n_run, t_parallel_n16, '-o', color = "brown", label = 'parallel: n=16')

plt.loglog()
plt.legend()

plt.ylabel('Execution time (s)')
plt.xlabel('Number of random points')

plt.savefig("test_jl.png")
plt.close()

# Saving execution data in a CSV file
with open("test_jl.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Header row
    writer.writerow(["n_run", "serial", "parallel_2", "parallel_4", "parallel_8", "parallel_16"])
    # Data row
    for j in range(len(n_run)):
        writer.writerow([n_run[j], t_serial[j], t_parallel_n02[j], t_parallel_n04[j],\
                       t_parallel_n08[j], t_parallel_n16[j]])
