import numpy as np
import matplotlib.pyplot as plt
import argparse

class MagneticField2D:
    """
    Generates 2D vector fields representing three types of magnetic fields.
    """

    def generate_grid(self, x_start, x_end, x_points, y_start, y_end, y_points):
        """
        Generates a 2D grid based on user-provided values.
        """
        x = np.linspace(x_start, x_end, x_points)
        y = np.linspace(y_start, y_end, y_points)
        X, Y = np.meshgrid(x, y)
        return X, Y

    @staticmethod
    def uniform_field(x, y, bx=1.0, by=0.0):
        """
        Generates a uniform 2D magnetic field.
        """
        Bx = np.full_like(x, bx)
        By = np.full_like(y, by)
        return Bx, By

    @staticmethod
    def dipole_field(x, y, mx=1.0, my=0.0, x0=0.0, y0=0.0):
        """
        Generates a 2D magnetic field from a dipole.
        """
        dx = x - x0
        dy = y - y0
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        r5 = r2**(5/2)

        Bx = (3 * dx * (mx * dx + my * dy) - mx * r2) / r5
        By = (3 * dy * (mx * dx + my * dy) - my * r2) / r5
        Bx = np.where(r==0,0,Bx)
        By = np.where(r==0,0,By)

        return Bx, By

    @staticmethod
    def current_wire_field(x, y, I=1.0, x0=0.0, y0=0.0):
        """
        Generates a 2D magnetic field from a current-carrying wire.
        """
        dx = x - x0
        dy = y - y0
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)

        Bx = -I * dy / r2
        By = I * dx / r2
        Bx = np.where(r==0,0,Bx)
        By = np.where(r==0,0,By)

        return Bx, By

def normalize_and_plot(X, Y, Bx, By, title, scale=10.0):
    """
    Normalizes the quiver vectors and plots them with a uniform scale.
    """
    magnitude = np.sqrt(Bx**2 + By**2)
    magnitude = np.where(magnitude == 0, 1, magnitude)
    Bx_normalized = Bx / magnitude
    By_normalized = By / magnitude

    plt.quiver(X, Y, Bx_normalized, By_normalized, scale=scale)
    plt.title(title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and plot 2D magnetic fields.")
    parser.add_argument("--type", choices=["uniform", "dipole", "current_wire"], required=True, help="Type of magnetic field to generate.")
    args = parser.parse_args()

    magnetic_field_generator = MagneticField2D()

    x_start = float(input("Enter x start: "))
    x_end = float(input("Enter x end: "))
    x_points = int(input("Enter number of x points: "))
    y_start = float(input("Enter y start: "))
    y_end = float(input("Enter y end: "))
    y_points = int(input("Enter number of y points: "))

    X, Y = magnetic_field_generator.generate_grid(x_start, x_end, x_points, y_start, y_end, y_points)

    if args.type == "uniform":
        Bx, By = MagneticField2D.uniform_field(X, Y, bx=1.0, by=0.5)
        title = "Uniform Magnetic Field"
    elif args.type == "dipole":
        Bx, By = MagneticField2D.dipole_field(X, Y, mx=1.0, my=0.0)
        title = "Dipole Magnetic Field"
    elif args.type == "current_wire":
        Bx, By = MagneticField2D.current_wire_field(X, Y, I=1.0)
        title = "Current Wire Magnetic Field"

    plt.figure()
    normalize_and_plot(X, Y, Bx, By, title)
    plt.show()
