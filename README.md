# PowerFit

## About PowerFit

PowerFit is a Python package and simple command-line program to automatically fit high-resolution atomic structures in cryo-EM densities.
To this end it performs a full-exhaustive 6-dimensional cross-correlation search between the atomic structure and the density.
It takes as input an atomic structure in PDB-format and a cryo-EM density with its resolution;
and outputs positions and rotations of the atomic structure corresponding to high correlation values.
PowerFit uses the Local Cross-Correlation functions as its base score. 
The score can optionally be enhanced by a Laplace pre-filter and/or a core-weighted version to minimize overlapping densities from neighboring subunits.
It can further be hardware-accelerated by leveraging multi-core CPU machines out of the box or by GPU via the OpenCL framework.
PowerFit is Free Software and has been succesfully installed and used on Linux, MacOSX and Windows machines.

## Requirements

Minimal requirements for the CPU version:

* Python2.7
* NumPy
* Cython
* SciPy

Optional requirement for faster CPU version:

* FFTW3
* pyFFTW

To offload computations to the GPU the following is also required

* OpenCL1.1+
* pyopencl
* clFFT
* gpyfft

## Installation

If you already have fulfilled the requirements, 
the installation should be as easy as opening up a shell and typing

    git clone https://github.com/haddocking/powerfit.git
    cd powerfit
    (sudo) python setup.py install

or if *git* is not available to you, 
download *powerfit-master.zip* from the right side of the page, 
open up a shell, go to the location where you downloaded *powerfit-master.zip* and type

    unzip powerfit-master.zip
    cd powerfit-master/
    (sudo) python setup.py install

If you are starting from a clean system, the following instructions should get you up and running in no time.

### Unix (Linux/MacOSX)

Unix systems usually include already a Python distribution.
To easily install the required Python packages, 
first install the Python package manager [pip](https://pip.pypa.io/en/latest/installing.html).
Download *get-pip.py*, open up a terminal and navigate to the location of *get-pip.py*. Type

    (sudo) python get-pip.py

This installs *pip*, making Python package management easy.
To install NumPy, Cython and SciPy type

    (sudo) pip install numpy cython scipy

Sit back and wait till the compilation and installation is finished.
You system is now prepared to install PowerFit. 
Follow the general instructions above to see how.

### Windows

To run PowerFit on Windows, install a Python distribution with NumPy, Cython and Scipy included such as [Anaconda](http://continuum.io/downloads).
The general instructions above can then be followed.

## Usage

After installing PowerFit the commandline tool *powerfit* should be at your disposal.
The general pattern to invoke *powerfit* is

    powerfit <pdb> <map> <resolution>

where \<pdb\> is an atomic model in the PDB-format, 
\<map\> is a density map in CCP4 or MRC-format, 
and \<resolution\> is the resolution of the map in &aring;ngstrom.
This performs a 10&deg; rotational search using the Local Cross-Correlation score on a single CPU-core.
During the search, *powerfit* will update you about the progress of the search if you are using it interactively in the shell.
When the search is finished, several output files are created

* 10 best scoring structures (fit_*n*.pdb)
* A Cross-correlation map, showing at each voxel the highest LCC-value found (*lcc.mrc*)
* All the non-redundant solutions found ordered by the LCC-score together with their xyz-postions and rotation matrix (*solutions.out*)
* A log file, showing the input parameters and what was happening when (*powerfit.log*)

### Options

First, to see all options and their descriptions type

    powerfit --help

The information should explain all options decently. 
In addtion, here are some examples for common operations.

To perform a search with an approximate 24&deg; rotational sampling interval

    powerfit <pdb> <map> <resolution> -a 24

To use multiple CPU cores with laplace pre-filter and 5&deg; rotational interval

    powerfit <pdb> <map> <resolution> -p 4 -l -a 5

To off-load computations to the GPU and use the core-weighted scoring function and write out the top 15 solutions

    powerfit <pdb> <map> <resolution> -g -cw -n 15

Note that all options can be combined except for the `-g` and `-p` flag:
calculations are either performed on the CPU or GPU.
If both are given, *powerfit* will first try to run on the GPU.

## Licensing

If this software was useful to your research, please cite us

**G.C.P. van Zundert and A.M.J.J. Bonvin**. *Fast and sensitive rigid body fitting in cryo-EM densities with PowerFit.* AIMS Biophysics (submitted).

MIT licence
