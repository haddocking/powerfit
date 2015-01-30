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

If you already have fulfilled the requirements, the installation should be as easy as

    git clone https://github.com/haddocking/powerfit.git
    cd powerfit
    (sudo) python setup.py install

or if git is not available to you, download powerfit-master.zip and

    unzip powerfit-master.zip
    cd powerfit-master/
    (sudo) python setup.py install

If you are starting from a clean system, the following instructrions should get you up and running in no time.

### Unix (Linux/MacOSX)

Unix systems usually include already a Python distribution.
To easily install the required Python packages, first install the Python package manager [pip](https://pip.pypa.io/en/latest/installing.html).
Open up a terminal and go to the location where get-pip.py was downloaded. Type

    (sudo) python get-pip.py

This installs pip, making package management easy.
To install NumPy, Cython and SciPy type

    (sudo) pip install numpy cython scipy

Sit back and wait till the compilation and installing is done.
You system is now prepared to install PowerFit. 
See the general instructions above to see how.

### Windows

To run PowerFit on Windows, install a Python distribution with NumPy, Cython and Scipy included such as [Anaconda](http://continuum.io/downloads).
The general instructions above can then be followed.

## Examples

## Licensing

If this software was useful to your research, please cite us

**G.C.P. van Zundert and A.M.J.J. Bonvin**. *Fast and sensitive rigid body fitting in cryo-EM densities with PowerFit.* AIMS Biophysics (submitted).

MIT licence
