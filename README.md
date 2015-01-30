# PowerFit

## About PowerFit
PowerFit is a Python package and simple command-line program to automatically fit high-resolution atomic structures in cryo-EM densities.
To this end it performs a full-exhaustive 6D cross-correlation search between the atomic structure and the density.
It takes as input an atomic structure in PDB-format and a cryo-EM density with its resolution;
and outputs positions and rotations of the atomic structure corresponding to high correlation values.
PowerFit uses the Local Cross-Correlation functions as its base score. 
The score can optionally be enhanced by a Laplace pre-filter and/or a core-weighted version to minimize overlapping densities from neighboring subunits.
PowerFit has been succesfully installed and used on Linux, MacOSX and Windows. 
It can further be hardware-accelerated by leveraging multi-core CPU machines out of the box or by GPU via the OpenCL framework.
PowerFit is Free Software.

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

* OpenCL1.1
* pyopencl
* clFFT
* gpyfft

## Installation

PowerFit has been succesfully installed on Unix and Windows machines.
If you already have fulfilled the requirements, the installation should be as easy as

    git clone https://github.com/haddocking/powerfit.git
    cd powerfit
    (sudo) python setup.py install

or if git is not available to you, download powerfit.tar.gz and

    tar xvfz powerfit.tar.gz
    cd powerfit/
    (sudo) python setup.py install

The following installation instructions assume you are starting with a clean system.

### Unix (Linux/MacOSX)



### Windows



## Examples

## Licensing

If this software was useful to your research, please cite us

**G.C.P. van Zundert and A.M.J.J. Bonvin**. *Fast and sensitive rigid body fitting in cryo-EM densities with PowerFit.* AIMS Biophysics (submitted).

MIT licence
