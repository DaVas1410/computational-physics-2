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
    def __init__(self, x_l=-10., x_r=+10, y_l=-10, y_r=+10, n=50):
        """
        """
        self.x_l = x_l
        self.x_r = x_r
        self.y_l = y_l
        self.y_r = y_r
        self.n = n

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

    # Second method: static method for the uniform field
    @staticmethod
    def uniform_bfield(x_2d, y_2d):
        """
        """
        # FField components
        bx_2d = np.full_like(x_2d, 1.)
        by_2d = np.full_like(y_2d, 1.)
        return bx_2d, by_2d

    # Dipolar field
    @staticmethod
    def dipolar_bfield(x_2d, y_2d):
        """
        """
        # Constants
        b0 = 1.e-5
        R = 5.

        # Radial coordinate
        r = np.sqrt(x_2d**2 + y_2d**2)

        # Define
        theta = np.arctan2(y_2d, x_2d)

        # Field components in spherical coordinates
        b_r = -2.*b0*(R/r**3)*np.cos(theta)
        b_theta = -b0*(R/r**3)*np.sin(theta)

        # Conversion to Cartesian coordinates
        bx_2d = -b_theta*np.sin(theta) + b_r*np.cos(theta)
        by_2d = b_theta*np.cos(theta) + b_r*np.sin(theta)

        # temporary check
        # return b_r, b_theta
        return bx_2d, by_2d

    # Plotting method

    def plot_2dfield(self, field_type="uniform"):
        """
        """
        # Call the grid information
        x_2d, y_2d = self.grid_generator()

        # Call the field generators
        if field_type == "uniform":
            # Call the uniform field generator
            bx_2d, by_2d = self.uniform_bfield(x_2d, y_2d)
        elif field_type == "dipolar":
            # Call the dipolar field generator
            bx_2d, by_2d = self.dipolar_bfield(x_2d, y_2d)
        else:
            raise ValueError(
                "The argument field_type only accepts: 'uniform' or 'dipolar'.")

        # Compute B
        b_mod = np.sqrt(bx_2d**2 + by_2d**2)

        # Create a figure environment
        plt.figure(figsize=(8, 8))
        # Reference for streamplot: https://scipython.com/blog/visualizing-the-earths-magnetic-field/
        plt.streamplot(x_2d, y_2d, bx_2d, by_2d, color=np.log10(
            b_mod), linewidth=1, cmap=plt.cm.inferno, density=2, arrowstyle='->', arrowsize=1.5)
        # plt.quiver(x_2d, y_2d, bx_2d, by_2d, np.log10(b_mod))

        plt.savefig(field_type+".png")


# Call the class
if __name__ == "__main__":

    # Parsing the code
    parser = argparse.ArgumentParser(
        description="Generate and make a map of 2D B fields")
    # Add argument
    parser.add_argument(
        "--btype", choices=["uniform", "dipolar"], required=True, help="Type of B field.")
    args = parser.parse_args()

    # Instance of the class
    mag = MagneticField2D()

    # Access methods
    # xx, yy = mag.grid_generator()

    # Testing
    # b_r, b_theta = mag.dipolar_bfield(xx, yy)

    # print("We should see the shapes: ", b_r.shape)

    # Plot the uniform field
    mag.plot_2dfield(field_type=args.btype)
    # mag.plot_2dfield(field_type = "dipolar")
