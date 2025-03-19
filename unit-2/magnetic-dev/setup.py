# For installation
from setuptools import setup, find_packages

# Call setup

setup(name = "magnetic", description = "2D B vector field generator", author = "WEBB", license = "GNU", author_email = "wbanda@yachaytech.edu.ec", packages = find_packages(), install_requires = ["numpy", "matplotlib"])
