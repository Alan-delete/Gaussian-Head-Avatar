from setuptools import setup, find_packages

print(find_packages())

setup(
    name='perm',              # Replace with your project name
    version='0.1.0',                  # Initial version number
    packages=find_packages(),         # Automatically find package directories
)