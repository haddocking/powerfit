# PowerFit


## About PowerFit

PowerFit is a Python package and simple command-line program to automatically
fit high-resolution atomic structures in cryo-EM densities. To this end it
performs a full-exhaustive 6-dimensional cross-correlation search between the
atomic structure and the density. It takes as input an atomic structure in
PDB-format and a cryo-EM density with its resolution; and outputs positions and
rotations of the atomic structure corresponding to high correlation values.
PowerFit uses the Local Cross-Correlation functions as its base score. The
score can optionally be enhanced by a Laplace pre-filter and/or a core-weighted
version to minimize overlapping densities from neighboring subunits. It can
further be hardware-accelerated by leveraging multi-core CPU machines out of
the box or by GPU via the OpenCL framework. PowerFit is Free Software and has
been succesfully installed and used on Linux, MacOSX and Windows machines.


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

Recommended for installation

* git
* pip


## Installation

If you already have fulfilled the requirements, the installation should be as
easy as opening up a shell and typing

    git clone https://github.com/haddocking/powerfit.git
    cd powerfit
    sudo python setup.py install

If you are starting from a clean system, follow the instructions for your
particular operating system as described below, they should get you up and
running in no time.


### Linux 

Linux systems usually already include a Python2.7 distribution. First make
sure the Python header files, NumPy, SciPy, *git*  and *pip* are available by
opening up a terminal and typing for Debian and Ubuntu systems

    sudo apt-get install python-dev python-pip python-numpy python-scipy git

If you are working on Fedora, this should be replaced by 

    sudo yum install python-devel python-pip numpy scipy git

To install Cython, type

    sudo pip install numpy cython scipy

Sit back and wait till the compilation and installation is finished. You
system is now prepared, follow the general instructions above to install
**PowerFit**.


### MacOSX

First install [*git*](https://git-scm.com/download) by following the
instructions on their website, or using a package manager, e.g. *brew*

    brew install git

Next install [*pip*](https://pip.pypa.io/en/latest/installing.html), the
Python package manager, by following the installation instructions on the
website or open a terminal and type 

    sudo easy_install pip

Next, install NumPy, Cython and SciPy by typing

    sudo pip install numpy cython scipy

Wait for the installation to finish (this might take awhile).

Follow the general instructions above to install **PowerFit**.

Installing pyFFTW for faster CPU version can be done as follows using *brew*

    brew install fftw
    sudo pip install pyfftw


### Windows

First install *git* for Windows, as it comes with a handy bash shell. Go to
[git-scm](https://git-scm.com/download/), download *git* and install it. Next,
install a Python distribution with NumPy, Cython and Scipy included such as
[Anaconda](http://continuum.io/downloads). After installation, open up the
bash shell shipped with *git* and follow the general instructions written
above.


## Usage

After installing PowerFit the command line tool *powerfit* should be at your
disposal. The general pattern to invoke *powerfit* is

    powerfit <pdb> <map> <resolution>

where \<pdb\> is an atomic model in the PDB-format, \<map\> is a density map in
CCP4 or MRC-format, and \<resolution\> is the resolution of the map in
&aring;ngstrom. This performs a 10&deg; rotational search using the Local
Cross-Correlation score on a single CPU-core. During the search, *powerfit*
will update you about the progress of the search if you are using it
interactively in the shell.


### Options

First, to see all options and their descriptions type

    powerfit --help

The information should explain all options decently. 
In addtion, here are some examples for common operations.

To perform a search with an approximate 24&deg; rotational sampling interval

    powerfit <pdb> <map> <resolution> -a 24

To use multiple CPU cores with laplace pre-filter and 5&deg; rotational
interval

    powerfit <pdb> <map> <resolution> -p 4 -l -a 5

To off-load computations to the GPU and use the core-weighted scoring function
and write out the top 15 solutions

    powerfit <pdb> <map> <resolution> -g -cw -n 15

Note that all options can be combined except for the `-g` and `-p` flag:
calculations are either performed on the CPU or GPU. If both are given,
*powerfit* will first try to run on the GPU.


### Output

When the search is finished, several output files are created

* *fit_n.pdb*: the top *n* best fits.
* *solutions.out*: all the non-redundant solutions found, ordered by their
correlation score.
* *lcc.mrc*: a cross-correlation map, showing at each grid position the highest
correlation score found.
* *powerfit.log*: a log file, including the input parameters with date and timing information.


## Licensing

If this software was useful to your research, please cite us

**G.C.P. van Zundert and A.M.J.J. Bonvin**. Fast and sensitive rigid-body
fitting into cryo-EM density maps with PowerFit. *AIMS Biophysics*, 2, 73-87
(2015).

MIT licence


## Tested platforms

| Operating System| CPU single | CPU multi | GPU |
| --------------- | ---------- | --------- | --- |
|Linux            | Yes        | Yes       | Yes |
|MacOSX           | Yes        | Yes       | Yes |
|Windows          | Yes        | Fail      | No  |

The GPU version has been tested on:
* NVIDIA GeForce GTX 680 and AMD Radeon HD 7730M for Linux
* NVIDIA GeForce GTX 775M for MacOSX 10.10
