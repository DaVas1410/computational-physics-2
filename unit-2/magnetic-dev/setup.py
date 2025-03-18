from setuptools import setup, find_packages

# Call setup

setup(
    name='magnetic',
    description='2D B vector field generator',
    author='JDVV',
    license='GNU',
    author_email='juan.vasconez@yachaytech.edu.ec',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib']
)