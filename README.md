# PowerFit

[![PyPI - Version](https://img.shields.io/pypi/v/powerfit-em)](https://pypi.org/project/powerfit-em/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14185749.svg)](https://doi.org/10.5281/zenodo.14185749)
[![Research Software Directory Badge](https://img.shields.io/badge/rsd-powerfit-00a3e3.svg)](https://www.research-software.nl/software/powerfit)
[![SBGrid Badge](https://img.shields.io/badge/PowerFit-blue?label=SBGrid&labelColor=%3D)](https://sbgrid.org/software/titles/powerfit)

## About PowerFit

PowerFit is a Python package and simple command-line program to automatically
fit high-resolution atomic structures in cryo-EM densities. To this end it
performs a full-exhaustive 6-dimensional cross-correlation search between the
atomic structure and the density. It takes as input an atomic structure in
PDB-format and a cryo-EM density with its resolution; and outputs positions and
rotations of the atomic structure corresponding to high correlation values.
PowerFit uses the local cross-correlation function as its base score. The
score is enhanced by a Laplace pre-filter and a core-weighted score to 
minimize overlapping densities from neighboring subunits. It can be
hardware-accelerated by leveraging multi-core CPU machines out of the box
or by GPU via the OpenCL framework. PowerFit is Free Software and has
been succesfully installed and used on Linux and MacOSX machines.

## Requirements

Minimal requirements for the CPU version:

* Python3.10 or greater
* NumPy 1.8+
* SciPy
* GCC (or another C-compiler)
* FFTW3
* pyFFTW 0.10+

To offload computations to a discrete or integrated\* GPU the following is also required

* OpenCL1.1+
* pyopencl
* pyvkfft

Recommended for installation

* git
* pip

\* _Integrated graphics on CPUs are able to signficantly outperform the native CPU implementation in some cases. This is mostly applicable to Intel devices, see the section [tested platfoms](installation.md#tested-platforms)_.

## Installation

If you want to run PowerFit on just the CPU, the installation should be as easy as opening up a shell and typing

```shell
# To run on CPU
pip install powerfit-em
```

If you want to offload the calculation to a GPU, please follow the instructions 
for your particular operating system described [here](https://bonvinlab.org/powerfit/installation.html), that should get you up and running in no time.

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

Please refer to the [PowerFit tutorial](https://www.bonvinlab.org/education/Others/powerfit/) to learn how to use PowerFit in the command line.

Please refer to the [PowerFit webserver tutorial](https://www.bonvinlab.org/education/Others/powerfit-webserver/) to learn how to use PowerFit on the webserver.

For more information and details please look at the [general manual](https://bonvinlab.org/powerfit/manual.html).

<details>
<summary>Usage in Docker</summary>

The Docker images of PowerFit are available in the [GitHub Container Registry](https://github.com/haddocking/powerfit/pkgs/container/powerfit).

Running PowerFit in a Docker container with data located at
a hypothetical `/path/to/data` on your machine can be done as follows

```shell
docker run --rm -ti --user $(id -u):$(id -g) \
    -v /path/to/data:/data ghcr.io/haddocking/powerfit:v3.1.0 \
    /data/<map> <resolution> /data/<pdb> \
    -d /data/<results-dir>
```
For `<map>`, `<pdb>`, `<results-dir>` use paths relative to `/path/to/data`.

To run tutorial example use
```shell
# cd into powerfit-tutorial repo
docker run --rm -ti --user $(id -u):$(id -g) \
    -v $PWD:/data ghcr.io/haddocking/powerfit:v3.1.0 \
    /data/ribosome-KsgA.map 13 /data/KsgA.pdb \
    -a 20 -p 2 -l -d /data/run-KsgA-docker
```

To run on NVIDIA GPU using [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) use
```shell
docker run --rm -ti \
    --runtime=nvidia --gpus all -v /etc/OpenCL:/etc/OpenCL \
    -v $PWD:/data ghcr.io/haddocking/powerfit:v3.1.0 \
    /data/ribosome-KsgA.map 13 /data/KsgA.pdb \
    -a 20 -l -d /data/run-KsgA-docker-nv --gpu
```

To run on Intel integrated graphics use

```shell
docker run --rm -ti \
    --device=/dev/dri \
    -v $PWD:/data ghcr.io/haddocking/powerfit:v3.1.0 \
    /data/ribosome-KsgA.map 13 /data/KsgA.pdb \
    -a 20 -l -d /data/run-KsgA-docker-nv --gpu
```

To run on [AMD GPU](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html) use

```shell
sudo docker run --rm -ti \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video --ipc=host \
    -v $PWD:/data ghcr.io/haddocking/powerfit-rocm:v3.1.0 \
    /data/ribosome-KsgA.map 13 /data/KsgA.pdb \
    -a 20 -l -d /data/run-KsgA-docker-amd --gpu
```

</details>

## Output

When the search is finished, several output files are created:

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
* *report.html* and *state.mvsj*: an HTML report and its [MolViewSpec](https://molstar.org/mol-view-spec/) with interactive 3D visualization of the best fits.
  Only written if the `--report --delimiter ,` arguments are passed. For an explanation
  of the HTML report please look [here](https://bonvinlab.org/powerfit/manual.html).

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

## Contributing

To contribute to PowerFit, see our [Contribution guidelines](CONTRIBUTING.md).
