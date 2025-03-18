# Import modules
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Create main class
class MagneticField2D:
   """
   TBA
   """
   
   # Init function -> Grid information
   def __init__(self, x_l = -10., x_r = +10, y_l = -10, y_r = +10, n = 50):
       """
       """
       self.x_l = x_l
       self.x_r = x_r
       self.y_l = y_l
       self.y_r = y_r
       self.n   = n

   # First method for grid generation
   def grid_generator(self):
       """
       """
       # 1D vectors
       x = np.linspace(self.x_l, self.x_r, self.n)
       y = np.linspace(self.y_l, self.y_r, self.n) 
       # Create the 2D grid
       x_2d, y_2d = np.meshgrid(x, y)
       # Return
       return x_2d, y_2d

   @staticmethod 
   def uniform_bfield(x_2d, y_2d):
       """
       """
       # FField components
       bx_2d = np.full_like(x_2d, 1.)
       by_2d = np.full_like(y_2d, 1.)
       return bx_2d, by_2d

# Call the class
if __name__ == "__main__":

    # Instance of the class
    mag = MagneticField2D()
    
    # Access methods
    xx, yy = mag.grid_generator()
    print("We should see the x shapes: ", xx.shape) 
