# Icarus_Diffusion

This repository contains code for measuring the ionization electron diffusion coefficients, $D_L$ and $D_T$, in the ICARUS LArTPC. The major components included are
- C++ source code for extracting waveform information from calibration ntuples
- Python modules for extracting the $D_L$ and $D_T$ measurements and making plots
- `project.py` configuration xml files for job submission to Fermigrid

The `project.py` configuration files are provided for reference, and should not be expected to run for a generic user without modification. We include them here to document icaruscode versions, fcl file configurations, tar balls used, etc.

# Method Overview
To-do. Include link to technical note.

# Installation

The C++ (waveform analyzer) and Python (measurement and plotting) portions of this code can be installed and used independently. An easy place to do so is on the ICARUS gpvms, but installation should work in any environment that has access to ROOT (for the C++ code) and/or Python v3.6 or greater. The steps listed below have been tested with icaruscode v09_56_00_01, which includes ROOT and Python v3.9.2. From an ICARUS gpvm, this can be setup as follows:

```
source /cvmfs/icarus.opensciencegrid.org/products/icarus/setup_icarus.sh
# Setup any version of icaruscode with python version >= 3.6
setup icaruscode v09_56_00_01 -q e20:prof 
```

## C++ Installation
Assuming you have access to ROOT and have setup the appropriate environment variables, the C++ code is simple to build. From the top directory, run the following commands:
```
cd cpp/src/
make
```
If you encounter build errors, there is likely an issue with your ROOT installation.

## Python Installation

The Python code includes some external dependencies which by default cannot be installed by users on the gpvms (and other servers with similar security measures). The easiest way to manage these installations without causing conflicts is to create a virtual environment using Python's built-in `venv` module. Virtual environments are stored in a directory where you have write access (e.g., in your `/app` space) and activated when necessary.
```
cd /path/to/virtual/environments # e.g., in your /app space
python3 -m venv my_environment
```
Then, to activate the virtual environment and install the packages necessary for the diffusion measurement code,
```
source /path/to/virtual/environments/my_environment/bin/activate
cd /path/to/Icarus_Diffusion/python/icarus_diffusion/
python3 -m pip install .
```

You should then be able to run the Python scripts using the output files from the waveform analyzer as input.

# Usage
To-do
