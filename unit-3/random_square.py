#!/opt/anaconda3/envs/py39/bin/python
"""
Script to generate 10 million random numbers.
"""
# Import modules
import numpy as np

# Random number function
def random_square(seed):
    """
    Square a random number generated witn an input seed.
    """
    # Allocate seed
    np.random.seed(seed)

    # Generate random number
    random_num = np.random.randint(0, 10)

    # Return squared values
    return random_num**2
