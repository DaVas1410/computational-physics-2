# CPU test
import time
import os
import psutil

print("Running cpu_script.py")
print(f"Hostname: {os.uname().nodename}")

# Obtain the CPU core count
cpu_n = os.cpu_count()
print(f"# Cores: {cpu_n}")

# Obtain the RAM info
ram_i = psutil.virtual_memory()
ram_i = ram_i.total / (1024 ** 3)
print(f"RAM: {ram_i}")

print("Starting CPU task...")
time.sleep(20)  # Sample code for testing
print("CPU task complete.")
