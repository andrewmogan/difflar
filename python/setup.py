#!/usr/bin/env python

from setuptools import setup, find_packages

VER = "1.0.0"

setup(
    name="lardiff",
    version=VER,
    author="Andrew J. Mogan, Michael Mooney, Sebastian Ruterbories",
    author_email="andrew.mogan@colostate.edu, michael.ryan.mooney@colostate.edu, sebastian.ruterbories@colostate.edu",
    description="A package for measuring the longitudinal and transverse diffusion coefficient in the ICARUS detector.",
    url="https://github.com/andrewmogan/Icarus_Diffusion.git",
    packages=find_packages(where="src"), 
    package_dir={"":"src"},
    install_requires=['numba', 'numpy', 'scipy', 'uproot', 'matplotlib'],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
