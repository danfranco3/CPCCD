from setuptools import setup, find_packages

setup(
    name="code_clone_pkg",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
