# PowerFit

[![PyPI - Version](https://img.shields.io/pypi/v/powerfit-em)](https://pypi.org/project/powerfit-em/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14185749.svg)](https://doi.org/10.5281/zenodo.14185749)
[![Research Software Directory Badge](https://img.shields.io/badge/rsd-powerfit-00a3e3.svg)](https://www.research-software.nl/software/powerfit)

## About PowerFit

PowerFit is a Python package and simple command-line program to automatically
fit high-resolution atomic structures in cryo-EM densities. To this end it
performs a full-exhaustive 6-dimensional cross-correlation search between the
atomic structure and the density. It takes as input an atomic structure in
PDB-format and a cryo-EM density with its resolution; and outputs positions and
rotations of the atomic structure corresponding to high correlation values.
PowerFit uses the local cross-correlation function as its base score. The
score can optionally be enhanced by a Laplace pre-filter and/or a core-weighted
version to minimize overlapping densities from neighboring subunits. It can
further be hardware-accelerated by leveraging multi-core CPU machines out of
the box or by GPU via the OpenCL framework. PowerFit is Free Software and has
been succesfully installed and used on Linux and MacOSX machines.

## Requirements

Minimal requirements for the CPU version:

* Python3.10 or greater
* NumPy 1.8+
* SciPy
* GCC (or another C-compiler)
* FFTW3
* pyFFTW 0.10+

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

```shell
# To run on CPU
pip install powerfit-em
# To run on GPU
pip install powerfit-em[opencl]
```

If you are starting from a clean system, follow the instructions for your
particular operating system as described below, they should get you up and
running in no time.

### Docker

Powerfit can be run in a Docker container. 

Install [docker](https://docs.docker.com/engine/install/) by following the
instructions.

### Linux

Linux systems usually already include a Python3.10 or greater distribution. First make
sure the Python header files, pip and *git* are available by
opening up a terminal and typing for Debian and Ubuntu systems

```shell
sudo apt update
sudo apt install python3-dev python3-pip git build-essential
```

If you are working on Fedora, this should be replaced by

```shell
sudo yum install python3-devel python3-pip git development-c development-tools
```

<details>
<summary>Steps for running on GPU</summary>

If you want to use the GPU version of PowerFit, you need to install the
drivers for your GPU. 

After installing
the drivers, you need to install the OpenCL development libraries and [OpenCL fft library](https://github.com/clMathLibraries/clFFT). For
Debian/Ubuntu, this can be done by running

```shell
sudo apt install opencl-headers ocl-icd-opencl-dev libclfft-dev
```
For Fedora, this can be done by running

```shell
sudo dnf install opencl-headers ocl-icd-devel
# Manually install clFFT from https://github.com/clMathLibraries/clFFT
```

Install gpyfft, a Python wrapper for OpenCL fft library, using

```shell
pip install cython
pip install --no-use-pep517 gpyfft@git+https://github.com/geggo/gpyfft@v0.8.0
```

Check that the OpenCL installation is working by running

```shell
python -c 'import pyopencl as cl;from gpyfft import GpyFFT; ps=cl.get_platforms();print(ps);print(ps[0].get_devices())'
# Should print the name of your GPU
```
</details>

Your system is now prepared, follow the general instructions above to install
**PowerFit**.

### MacOSX

First install [*git*](https://git-scm.com/download) by following the
instructions on their website, or using a package manager such as *brew*

```shell
brew install git
```

Next install [*pip*](https://pip.pypa.io/en/latest/installation/), the
Python package manager, by following the installation instructions on the
website or open a terminal and type

```shell
python -m ensurepip --upgrade
```

To get faster score calculation, install the pyFTTW Python package in your conda environment
with `conda install -c conda-forge pyfftw`.

Follow the general instructions above to
install **PowerFit**.

### Windows

First install *git* for Windows, as it comes with a handy bash shell. Go to
[git-scm](https://git-scm.com/download/), download *git* and install it. Next,
install a Python distribution such as
[Anaconda](http://continuum.io/downloads). After installation, open up the
bash shell shipped with *git* and follow the general instructions written
above.

## Usage

After installing PowerFit the command line tool *powerfit* should be at your
disposal. The general pattern to invoke *powerfit* is

```shell
powerfit <map> <resolution> <pdb>
```

where `<map>` is a density map in CCP4 or MRC-format, `<resolution>`  is the
resolution of the map in &aring;ngstrom, and `<pdb>` is an atomic model in the
PDB-format. This performs a 10&deg; rotational search using the local
cross-correlation score on a single CPU-core. During the search, *powerfit*
will update you about the progress of the search if you are using it
interactively in the shell.

<details>
<summary>Usage in Docker</summary>

The Docker images of powerfit are available in the [GitHub Container Registry](https://github.com/haddocking/powerfit/pkgs/container/powerfit).

Running PowerFit in a Docker container with data located at
a hypothetical `/path/to/data` on your machine can be done as follows

```shell
docker run --rm -ti --user $(id -u):$(id -g) \
    -v /path/to/data:/data ghcr.io/haddocking/powerfit:v3.0.4 \
    /data/<map> <resolution> /data/<pdb> \
    -d /data/<results-dir>
```
For `<map>`, `<pdb>`, `<results-dir>` use paths relative to `/path/to/data`.

To run tutorial example use
```shell
# cd into powerfit-tutorial repo
docker run --rm -ti --user $(id -u):$(id -g) \
    -v $PWD:/data ghcr.io/haddocking/powerfit:v3.0.4 \
    /data/ribosome-KsgA.map 13 /data/KsgA.pdb \
    -a 20 -p 2 -l -d /data/run-KsgA-docker
```

To run on NVIDIA GPU using [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) use
```shell
docker run --rm -ti \
    --runtime=nvidia --gpus all -v /etc/OpenCL:/etc/OpenCL \
    -v $PWD:/data ghcr.io/haddocking/powerfit:v3.0.4 \
    /data/ribosome-KsgA.map 13 /data/KsgA.pdb \
    -a 20 -l -d /data/run-KsgA-docker-nv --gpu
```

To run on [AMD GPU](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html) use

```shell
sudo docker run --rm -ti \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video --ipc=host \
    -v $PWD:/data ghcr.io/haddocking/powerfit-rocm:v3.0.4 \
    /data/ribosome-KsgA.map 13 /data/KsgA.pdb \
    -a 20 -l -d /data/run-KsgA-docker-amd --gpu
```

</details>

### Options

First, to see all options and their descriptions type

```shell
powerfit --help
```

The information should explain all options decently. In addtion, here are some
examples for common operations.

To perform a search with an approximate 24&deg; rotational sampling interval

```shell
powerfit <map> <resolution> <pdb> -a 24
```

To use multiple CPU cores with laplace pre-filter and 5&deg; rotational
interval

```shell
powerfit <map> <resolution> <pdb> -p 4 -l -a 5
```

To off-load computations to the GPU and use the core-weighted scoring function
and write out the top 15 solutions

```shell
powerfit <map> <resolution> <pdb> -g -cw -n 15
```

Note that all options can be combined except for the `-g` and `-p` flag:
calculations are either performed on the CPU or GPU.

To run on GPU

```shell
powerfit <map> <resolution> <pdb> --gpu
...
Using GPU-accelerated search.
...
```

### Output

When the search is finished, several output files are created

* *fit_N.pdb*: the top *N* best fits.
* *solutions.out*: all the non-redundant solutions found, ordered by their
correlation score. The first column shows the rank, column 2 the correlation
score, column 3 and 4 the Fisher z-score and the number of standard deviations
(see N. Volkmann 2009, and Van Zundert and Bonvin 2016); column 5 to 7 are the
x, y and z coordinate of the center of the chain; column 8 to 17 are the
rotation matrix values.
* *lcc.mrc*: a cross-correlation map, showing at each grid position the highest
correlation score found during the rotational search.
* *powerfit.log*: a log file, including the input parameters with date and
timing information.

## Creating an image-pyramid

The use of multi-scale image pyramids can signicantly increase the speed of
fitting. PowerFit comes with a script to quickly build a pyramid called
`image-pyramid`. The calling signature of the script is

```shell
image-pyramid <map> <resolution> <target-resolutions ...>
```

where `<map` is the original cryo-EM data, `<resolution` is the original
resolution, and `<target-resolutions>` is a sequence of resolutions for the
resulting maps. The following example will create an image-pyramid with
resolutions of 12, 13 and 20 angstrom

```shell
image-pyramid EMD-1884/1884.map 9.8 12 13 20
```

To see the other options type

```shell
image-pyramid --help
```

## Licensing

If this software was useful to your research, please cite us

**G.C.P. van Zundert and A.M.J.J. Bonvin**.
Fast and sensitive rigid-body fitting into cryo-EM density maps with PowerFit.
*AIMS Biophysics* 2, 73-87 (2015) [https://doi.org/10.3934/biophy.2015.2.73](https://doi.org/10.3934/biophy.2015.2.73).

For the use of image-pyramids and reliability measures for fitting, please cite

**G.C.P van Zundert and A.M.J.J. Bonvin**.
Defining the limits and reliability of rigid-body fitting in cryo-EM maps using
multi-scale image pyramids.
*J. Struct. Biol.* 195, 252-258 (2016) [https://doi.org/10.1016/j.jsb.2016.06.011](https://doi.org/10.1016/j.jsb.2016.06.011).

If you used PowerFit v1, please cite software with [https://doi.org/10.5281/zenodo.1037227](https://doi.org/10.5281/zenodo.1037227).
For version 2 or higher, please cite software with [https://doi.org/10.5281/zenodo.14185749](https://doi.org/10.5281/zenodo.14185749).

Apache License Version 2.0

The elements.py module is licensed under MIT License (see header).
Copyright (c) 2005-2015, Christoph Gohlke

## Tested platforms

| Operating System| CPU single | CPU multi | GPU |
| --------------- | ---------- | --------- | --- |
|Linux            | Yes        | Yes       | Yes |
|MacOSX           | Yes        | Yes       | No  |
|Windows          | Yes        | Fail      | No  |

The GPU version has been tested on:

* NVIDIA GeForce GTX 1050 Ti, GeForce RTX 4070 and AMD Radeon RX 7900 XTX on Linux 
* NVIDIA GeForce GTX 1050 Ti, AMD Radeon RX 7800 XT and AMD Radeon RX 7900 XTX in Docker container

## Development

To develop PowerFit, you need to install the development version of it using.

```shell
pip install -e .[dev]
```

Tests can be run using

```shell
pytest
```

To run OpenCL on **C**PU install use `pip install -e .[pocl]` and make sure no other OpenCL platforms, like 'AMD Accelerated Parallel Processing' or 'NVIDIA CUDA', are installed .

The Docker container, that works for cpu and NVIDIA gpus, can be build with

```shell
docker build -t ghcr.io/haddocking/powerfit:v3.0.4 .
```
The Docker container, that works for AMD gpus, can be build with

```shell
docker build -t ghcr.io/haddocking/powerfit-rocm:v3.0.4 -f Dockerfile.rocm .
```

The binary wheels can be build for all supported platforms by running the
https://github.com/haddocking/powerfit/actions/workflows/pypi-publish.yml GitHub action and downloading the artifacts.
The workflow is triggered by a push to the main branch, a release or can be manually triggered.
