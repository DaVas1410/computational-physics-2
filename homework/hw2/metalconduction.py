import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
import os
import argparse

class MetalConduction:
    """
    Class for simulating heat conduction in metal wires using the Crank-Nicolson method
    """
    def __init__(self, metal_name='Copper', length=20.0, dx=0.1, dt=0.01,
                 ic_type='smooth', bc_type='fixed', noise_amplitude=0.0):
        """
        Initialize the simulation parameters
        Parameters:
        -----------
        metal_name : str
            Name of the metal for the simulation
        length : float
            Length of the metal wire in cm
        dx : float
            Spatial step size in cm
        dt : float
            Time step size in seconds
        ic_type : str
            Type of initial condition ('smooth' or 'noisy')
        bc_type : str
            Type of boundary condition ('fixed' or 'varying')
        noise_amplitude : float
            Amplitude of noise for noisy initial condition
        """
        # Metal thermal diffusivity dictionary (in mm²/s)
        self.metals = {
            'Copper': 111,
            'Iron': 23,
            'Aluminum': 97,
            'Brass': 34,
            'Steel': 18,
            'Zinc': 63,
            'Lead': 22,
            'Titanium': 9.8
        }
        # Convert metal diffusivity from mm²/s to cm²/s
        self.metal_name = metal_name
        self.diffusivity = self.metals[metal_name] / 100.0  # Convert from mm²/s to cm²/s
        # Domain setup
        self.length = length  # Length in cm
        self.dx = dx  # Spatial step in cm
        self.dt = dt  # Time step in seconds
        # Calculate grid parameters
        self.nx = int(length / dx) + 1
        self.x = np.linspace(-length/2, length/2, self.nx)  # Domain from -10cm to +10cm
        # Initial and boundary condition types
        self.ic_type = ic_type
        self.bc_type = bc_type
        self.noise_amplitude = noise_amplitude
        # Calculated r factor for Crank-Nicolson
        self.r_factor = self.diffusivity * dt / (dx**2)
        # Initialize temperature matrix
        self.T = None
        self.time_array = None
        # Equilibrium parameters
        self.equilibrium_threshold = 0.001  # Temperature change threshold
        self.equilibrium_time = None
        # Print initialization info
        print(f"Initialized {metal_name} wire simulation")
        print(f"Diffusivity: {self.diffusivity:.2f} cm²/s")
        print(f"r factor: {self.r_factor:.4f}")

    def initialize_simulation(self, simulation_time=60):
        """
        Initialize the simulation with the chosen initial and boundary conditions
        Parameters:
        -----------
        simulation_time : float
            Maximum simulation time in seconds
        """
        # Create time array
        self.nt = int(simulation_time / self.dt) + 1
        self.time_array = np.linspace(0, simulation_time, self.nt)
        # Initialize temperature matrix T(x,t)
        self.T = np.zeros((self.nx, self.nt))
        # Apply initial conditions
        self._set_initial_conditions()
        # Apply initial boundary conditions
        self._set_boundary_conditions()
        # Create matrices for Crank-Nicolson
        self._setup_crank_nicolson_matrices()
        print(f"Simulation initialized for {simulation_time} seconds with {self.nt} time steps")

    def _set_initial_conditions(self):
        """Set initial temperature profile based on the IC type"""
        # Base temperature profile: T(x, 0) = 175 - 50cos(πx/5) - x²
        base_profile = 175 - 50*np.cos(np.pi*self.x/5) - self.x**2
        if self.ic_type == 'smooth':
            self.T[:, 0] = base_profile
        elif self.ic_type == 'noisy':
            # Generate noise and apply apodisation
            noise = self.noise_amplitude * np.random.randn(self.nx)
            # Apodisation function that goes to zero at boundaries
            edge_dist = np.minimum(self.x - min(self.x), max(self.x) - self.x)
            apodisation = np.sin(np.pi * edge_dist / self.length)
            self.T[:, 0] = base_profile + noise * apodisation
            # Ensure boundary conditions are exactly 25°C
            self.T[0, 0] = 25
            self.T[-1, 0] = 25
        else:
            raise ValueError(f"Unknown initial condition type: {self.ic_type}")

    def _set_boundary_conditions(self):
        """Set boundary conditions for all time steps"""
        if self.bc_type == 'fixed':
            # Fixed temperature of 25°C at both ends
            self.T[0, :] = 25
            self.T[-1, :] = 25
        elif self.bc_type == 'varying':
            # Time-varying boundary conditions
            # T(-10cm, t) = 25 + 0.27t
            # T(+10cm, t) = 25 + 0.12t
            self.T[0, :] = 25 + 0.27 * self.time_array
            self.T[-1, :] = 25 + 0.12 * self.time_array
        else:
            raise ValueError(f"Unknown boundary condition type: {self.bc_type}")

    def _setup_crank_nicolson_matrices(self):
        """Setup matrices for the Crank-Nicolson method"""
        n = self.nx
        r = self.r_factor
        # Matrix D1 for the implicit part (left side of equation)
        self.D1_matrix = np.diag([2 + 2*r]*(n-2)) + np.diag([-r]*(n-3), -1) + np.diag([-r]*(n-3), 1)
        # Matrix D2 for the explicit part (right side of equation)
        self.D2_matrix = np.diag([2 - 2*r]*(n-2)) + np.diag([r]*(n-3), -1) + np.diag([r]*(n-3), 1)

    def run_simulation(self):
        """Run the Crank-Nicolson simulation"""
        start_time = time.time()
        # Make sure we have initialized the simulation
        if self.T is None:
            raise ValueError("Simulation not initialized. Call initialize_simulation() first.")

        # Record if equilibrium was reached
        equilibrium_reached = False
        consecutive_stable_steps = 0
        required_stable_steps = 5  # Number of consecutive steps below threshold

        # Iterate through time steps
        for j in range(self.nt-1):
            # Apply Crank-Nicolson method to update interior points
            # Create right-hand side vector b from current temperature
            b = np.dot(self.D2_matrix, self.T[1:-1, j])
            # Add boundary terms
            b[0] += self.r_factor * (self.T[0, j+1] + self.T[0, j])
            b[-1] += self.r_factor * (self.T[-1, j+1] + self.T[-1, j])
            # Solve system of equations: D1 * T_new = b
            self.T[1:-1, j+1] = np.linalg.solve(self.D1_matrix, b)

            # Check for thermal equilibrium (max temperature change less than threshold)
            if j > 0 and not equilibrium_reached:
                max_change = np.max(np.abs(self.T[:, j+1] - self.T[:, j]))
                if max_change < self.equilibrium_threshold:
                    consecutive_stable_steps += 1
                    if consecutive_stable_steps >= required_stable_steps:
                        self.equilibrium_time = (j+1) * self.dt
                        equilibrium_reached = True
                        print(f"Thermal equilibrium reached at t = {self.equilibrium_time:.2f} seconds")
                else:
                    consecutive_stable_steps = 0

        # If equilibrium was not reached, set a default value (max simulation time)
        if not equilibrium_reached:
            print(f"Warning: Thermal equilibrium not reached within simulation time")
            self.equilibrium_time = self.time_array[-1]  # Use max time instead of None

        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
        return self.equilibrium_time

    def plot_temperature_profile(self, time_indices=None, ax=None, title=None):
        """
        Plot temperature profiles at specified time indices
        Parameters:
        -----------
        time_indices : list
            List of time indices to plot
        ax : matplotlib.axes.Axes
            Axes to plot on. If None, a new figure is created
        title : str
            Title for the plot
        """
        if time_indices is None:
            # If no time indices specified, choose evenly spaced times
            time_indices = np.linspace(0, self.nt-1, 5, dtype=int)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        for idx in time_indices:
            t_value = self.time_array[idx]
            ax.plot(self.x, self.T[:, idx], label=f't = {t_value:.2f}s')

        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Temperature (°C)')
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{self.metal_name} Wire - {self.ic_type.capitalize()} IC, {self.bc_type.capitalize()} BC')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        return ax

    def plot_heatmap(self, ax=None, title=None):
        """
        Plot a heatmap of temperature evolution over time and space
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to plot on. If None, a new figure is created
        title : str
            Title for the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Create meshgrid for x and t
        X, T = np.meshgrid(self.x, self.time_array)
        # Plot heatmap
        im = ax.pcolormesh(X, T, self.T.T, cmap='inferno', shading='auto')
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Temperature (°C)')
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Time (s)')
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{self.metal_name} Wire - Temperature Evolution')
        return ax

    def save_data(self, filename):
        """
        Save simulation data to log file
        Parameters:
        -----------
        filename : str
            Filename to save data to (should end with .log)
        """
        # Change extension if needed
        if not filename.endswith('.log'):
            filename = filename.replace('.npz', '.log')
        if not filename.endswith('.log'):
            filename += '.log'

        with open(filename, 'w') as f:
            # Write header and simulation parameters
            f.write("=================================================\n")
            f.write(f"Metal Conduction Simulation - {self.metal_name}\n")
            f.write("=================================================\n\n")

            # Simulation parameters
            f.write("SIMULATION PARAMETERS:\n")
            f.write("-------------------------------------------------\n")
            f.write(f"Metal: {self.metal_name}\n")
            f.write(f"Thermal Diffusivity: {self.diffusivity:.4f} cm²/s\n")
            f.write(f"Initial Condition: {self.ic_type}\n")
            if self.ic_type == 'noisy':
                f.write(f"Noise Amplitude: {self.noise_amplitude}\n")
            f.write(f"Boundary Condition: {self.bc_type}\n")
            f.write(f"Length: {self.length} cm\n")
            f.write(f"Spatial Step (dx): {self.dx} cm\n")
            f.write(f"Time Step (dt): {self.dt} s\n")
            f.write(f"Total Simulation Time: {self.time_array[-1]} s\n")
            f.write(f"r factor: {self.r_factor:.4f}\n\n")

            # Results
            f.write("SIMULATION RESULTS:\n")
            f.write("-------------------------------------------------\n")
            if self.equilibrium_time is not None:
                f.write(f"Equilibrium Time: {self.equilibrium_time:.4f} seconds\n")
            else:
                f.write("Equilibrium Time: Not reached\n")

            # Summary of temperature
            f.write("\nTemperature Summary:\n")
            f.write("-------------------------------------------------\n")
            f.write(f"Initial Max Temperature: {np.max(self.T[:, 0]):.2f}°C\n")
            f.write(f"Initial Min Temperature: {np.min(self.T[:, 0]):.2f}°C\n")
            f.write(f"Final Max Temperature: {np.max(self.T[:, -1]):.2f}°C\n")
            f.write(f"Final Min Temperature: {np.min(self.T[:, -1]):.2f}°C\n\n")

            # Sample temperature at key points
            f.write("Temperature samples at t=0:\n")
            samples = 5
            sample_indices = np.linspace(0, len(self.x)-1, samples, dtype=int)
            for idx in sample_indices:
                f.write(f" x={self.x[idx]:.2f} cm: {self.T[idx, 0]:.2f}°C\n")

            f.write("\nTemperature samples at final time:\n")
            for idx in sample_indices:
                f.write(f" x={self.x[idx]:.2f} cm: {self.T[idx, -1]:.2f}°C\n")

            # Timestamp
            f.write("\n-------------------------------------------------\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"Data saved to {filename}")

def run_metal_simulation(metal_name, ic_type='smooth', bc_type='fixed',
                         noise_amplitude=0.0, dx=0.1, dt=0.01, sim_time=60,
                         output_dir="output_results"):
    """Run simulation for a specific metal and return the simulation results"""
    start_time = time.time()

    # Create and initialize simulation
    sim = MetalConduction(
        metal_name=metal_name,
        ic_type=ic_type,
        bc_type=bc_type,
        noise_amplitude=noise_amplitude,
        dx=dx,
        dt=dt
    )

    # Initialize and run simulation
    sim.initialize_simulation(simulation_time=sim_time)
    equilibrium_time = sim.run_simulation()

    # Calculate simulation duration
    sim_duration = time.time() - start_time

    # Create a unique filename base
    base_filename = f"{metal_name}_{ic_type}_{bc_type}"
    if ic_type == 'noisy':
        base_filename += f"_noise{noise_amplitude}"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save simulation data
    sim.save_data(f"{output_dir}/{base_filename}.log")

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sim.plot_temperature_profile(ax=ax1)
    sim.plot_heatmap(ax=ax2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{base_filename}.png", dpi=150)
    plt.close()

    # Return simulation results
    return {
        'metal': metal_name,
        'equilibrium_time': equilibrium_time,
        'diffusivity': sim.diffusivity,
        'ic_type': ic_type,
        'bc_type': bc_type,
        'noise_amplitude': noise_amplitude,
        'sim_duration': sim_duration
    }

def run_parallel_simulations(metal_list, n_jobs=1, **kwargs):
    """Run simulations for multiple metals in parallel using joblib"""
    start_time = time.time()

    # Get output directory from kwargs or use default
    output_dir = kwargs.get('output_dir', 'output_results')
    os.makedirs(output_dir, exist_ok=True)

    # Run simulations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_metal_simulation)(metal, **kwargs)
        for metal in metal_list
    )

    # Calculate total execution time
    total_time = time.time() - start_time

    # Log results
    with open(f"{output_dir}/parallel_results_n{n_jobs}.txt", 'w') as f:
        f.write(f"Simulation with {n_jobs} cores completed in {total_time:.4f} seconds\n\n")
        for result in results:
            f.write(f"Metal: {result['metal']}\n")
            f.write(f"Diffusivity: {result['diffusivity']:.2f} cm²/s\n")
            f.write(f"IC Type: {result['ic_type']}\n")
            f.write(f"BC Type: {result['bc_type']}\n")
            if result['ic_type'] == 'noisy':
                f.write(f"Noise Amplitude: {result['noise_amplitude']}\n")
            f.write(f"Equilibrium Time: {result['equilibrium_time']:.2f} seconds\n")
            f.write(f"Simulation Duration: {result['sim_duration']:.2f} seconds\n")
            f.write("\n")

    print(f"All simulations completed in {total_time:.2f} seconds using {n_jobs} cores")
    return results, total_time

def run_multiple_core_tests():
    """Run simulations with varying numbers of cores and record the results"""
    # List of metals to simulate
    metals = ['Copper', 'Iron', 'Aluminum', 'Brass', 'Steel', 'Zinc', 'Lead', 'Titanium']
    
    # Numbers of cores to test
    core_counts = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    
    # Base simulation parameters
    sim_params = {
        'ic_type': 'smooth',
        'bc_type': 'fixed',
        'noise_amplitude': 0.0,
        'dx': 0.1,
        'dt': 0.2,
        'sim_time': 550,
        'output_dir': 'output_results'
    }
    
    # Create main output directory
    os.makedirs(sim_params['output_dir'], exist_ok=True)
    
    # Dictionary to store execution times
    exec_times = {}
    
    # Run simulations for each core count
    for n_cores in core_counts:
        print(f"\n----- Running simulation with {n_cores} cores -----")
        # Create specific output directory for this run
        run_output_dir = f"{sim_params['output_dir']}/run_cores_{n_cores}"
        os.makedirs(run_output_dir, exist_ok=True)
        
        # Update output directory in parameters
        run_params = sim_params.copy()
        run_params['output_dir'] = run_output_dir
        
        # Run the simulation
        results, total_time = run_parallel_simulations(metals, n_jobs=n_cores, **run_params)
        
        # Store execution time
        exec_times[n_cores] = total_time
    
    # Write summary of all runs
    with open(f"{sim_params['output_dir']}/performance_summary.txt", 'w') as f:
        f.write("Metal Conduction Simulation - Performance Summary\n")
        f.write("================================================\n\n")
        f.write("Execution Times with Different Core Counts:\n")
        f.write("------------------------------------------\n")
        for cores, exec_time in exec_times.items():
            f.write(f"{cores} cores: {exec_time:.4f} seconds\n")
        
        # Calculate speedup relative to single core
        single_core_time = exec_times[1]
        f.write("\nSpeedup Relative to Single Core:\n")
        f.write("-------------------------------\n")
        for cores, exec_time in exec_times.items():
            speedup = single_core_time / exec_time
            efficiency = speedup / cores * 100
            f.write(f"{cores} cores: {speedup:.2f}x speedup ({efficiency:.1f}% efficiency)\n")
        
        # Add timestamp
        f.write("\n------------------------------------------\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Print summary to console
    print("\n----- Performance Summary -----")
    for cores, exec_time in exec_times.items():
        speedup = single_core_time / exec_time
        print(f"{cores} cores: {exec_time:.2f} seconds ({speedup:.2f}x speedup)")
    
    return exec_times

if __name__ == "__main__":
    # Run tests with multiple core counts
    print("Starting metal conduction simulations with multiple core counts...")
    execution_times = run_multiple_core_tests()
    print("All simulations completed!")
    
    # Plot performance results
    core_counts = list(execution_times.keys())
    times = list(execution_times.values())
    
    # Create output directory if it doesn't exist
    os.makedirs('output_results', exist_ok=True)
    
    # Plot execution time vs cores
    plt.figure(figsize=(10, 6))
    plt.plot(core_counts, times, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Cores')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Metal Conduction Simulation - Parallel Performance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('output_results/performance_time.png', dpi=150)
    
    # Plot speedup vs cores
    single_core_time = execution_times[1]
    speedups = [single_core_time / time for time in times]
    
    plt.figure(figsize=(10, 6))
    plt.plot(core_counts, speedups, 'o-', linewidth=2, markersize=8)
    plt.plot(core_counts, core_counts, '--', label='Ideal Speedup', alpha=0.7)
    plt.xlabel('Number of Cores')
    plt.ylabel('Speedup (relative to 1 core)')
    plt.title('Metal Conduction Simulation - Parallel Speedup')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('output_results/performance_speedup.png', dpi=150)
    
    print("Performance plots saved to 'output_results' directory")