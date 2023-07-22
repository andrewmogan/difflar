# lardiff

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

The Python code includes some external dependencies which by default cannot be installed by users on the gpvms (and other servers with similar security measures). The easiest way to manage these installations without causing conflicts is to create a virtual environment using Python's built-in `venv` module. Virtual environments are stored in a directory where you have write access (e.g., in your `/app` space) and can be activated when necessary.

```
cd /path/to/virtual/environments # e.g., in your /app space
python3 -m venv my_environment
```

where `my_environment` can be named whatever you want. To activate the virtual environment and install the packages necessary for the diffusion measurement code,
```
source /path/to/virtual/environments/my_environment/bin/activate
cd /path/to/lardiff/python/
python3 -m pip install .
```

This will install the lardiff package and its dependencies in your virtual environment. To test your installation, open up an interactive Python session and attempt to import this package:

```
$ python3
>>> import lardiff
```

If there are no errors, you should then be able to run the Python scripts using the output files from the waveform analyzer as input. 

# Usage
The diffusion analysis code can be run through a command-line interface. The executable located in `/path/to/lardiff/python/bin/` can be run as:
```
python run_diffusion_analysis.py --input_filename <input_file> --config <config.yaml>
```
where `input_file` should be a ROOT-format file output from the C++ `WaveformStudy.cpp` module and `config.yaml` is a config file containing the necessary configuration parameters described below. 

## yaml Configuration 

Due to the large number of configurable parameters, and the possibility of adding more parameters in the future, the diffusion analysis code called from `run_diffusion_analysis.py` is configured through a yaml file. You can find example configuration files in `/path/to/lardiff/python/config`. The config file should contain the following parameters:

- DL_min: minimum $D_L$ value in $cm^2/s$ in the grid scan
- DL_max: maximum $D_L$ value in $cm^2/s$ in in the grid scan
- DL_step: $D_L$ step size in the $\chi^2$ grid scan
- DT_min: minimum $D_T$ value in in $cm^2/s$ in the grid scan
- DT_max: maximum $D_T$ value in in $cm^2/s$ in the grid scan
- DT_step: $D_T$ step size in the grid scan
- angle_min: minimum angular bin value in degrees
- angle_max: maximum angular bin value in degrees
- angle_step: angular bin step size. This should be set to 2.
- test_statistic: which test statistic to use in the grid scan
- interpolation: interpolation method to use when interpolating signal waveforms, e.g. `'scipy'` or `'numpy'`
- isdata: boolean flag stating whether the input file was processed from simulation or data
- save_waveform_data: boolean flag that controls whether a sample set of wire waveforms is saved to a `.pkl` file for offline use

## Outputs

To-do

# Contributing

To-do

# License

Distributed under the MIT License. See LICENSE for more information.











