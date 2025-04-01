# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import random_square
import multiprocessing as mp
# Serial function

def serial(n):
    
    start = time.time()
    results = []
    
    for i in range(n): 
        results.append(random_square.random_square(i))
    
    end = time.time()
    exec_time = end - start
    
    return exec_time

# Parallel function

def parallel(n, n_cpu):
    
    start = time.time()
    #n_cpu = 8

    pool = mp.Pool(processes=n_cpu)
    
    results = [pool.map(random_square.random_square, range(n))]
    
    end = time.time()
    exec_time = end - start
    
    return exec_time


# Generate numbers in log-space:
n_run = np.logspace(1, 7, num = 7)

#print(n_run)

# Call functions for each n_run[i]

t_serial = np.array([serial(int(n)) for n in n_run])

t_parallel_n02 = np.array([parallel(int(n), 2) for n in n_run])
t_parallel_n04 = np.array([parallel(int(n), 4) for n in n_run])
t_parallel_n08 = np.array([parallel(int(n), 8) for n in n_run])
t_parallel_n16 = np.array([parallel(int(n), 16) for n in n_run])

# PLotting
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

plt.savefig("test_1.png")
