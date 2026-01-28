# Installation 

If you already have fulfilled the [requirements](https://github.com/haddocking/powerfit?tab=readme-ov-file#requirements) 
for offloading to the GPU, the installation should be as easy as opening up a shell and typing

```shell
# To run on GPU
pip install powerfit-em[opencl]
```
If you are starting from a clean system, follow the instructions for your
particular operating system as described below, they should get you up and
running in no time.

### Conda

If you do not have system admin rights, you likely cannot compile `pyvkfft` locally.
However, by installing powerfit in a conda environment, you can still do computations
on GPU. If you are on a Linux system and have Conda or Mamba available, follow
these instructions;

<details><summary>Steps for running on GPU with Conda</summary>

For AMD or NVIDIA GPUs you can run the following command. Note that this relies
on OpenCL drivers being available system wide (under `/etc/OpenCL/vendors/`).

```shell
conda create -n powerfit -c conda-forge python=3.12 ocl-icd ocl-icd-system pyopencl pyvkfft
conda activate powerfit
pip install powerfit-em[opencl]
```

On Intel integrated graphics you can use the following command. This includes
the OpenCL runtime and does not rely on your system setup:

```shell
conda create -n powerfit -c conda-forge python=3.12 ocl-icd intel-compute-runtime pyopencl pyvkfft
conda activate powerfit
pip install powerfit-em[opencl]
```
Some older Intel processors might need to use `intel-opencl-rt` instead of `intel-compute-runtime`.

After installation, you can check that the OpenCL installation is working by running

```shell
python -c 'import pyopencl as cl;from pyvkfft.fft import rfftn; ps=cl.get_platforms();print(ps);print(ps[0].get_devices())'
```

</details>

### Usage in Docker

Powerfit can be run in a Docker container. 

Install [docker](https://docs.docker.com/engine/install/) by following the
instructions.

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

After installing the drivers, you need to install the OpenCL development libraries.
For Debian/Ubuntu, this can be done by running

```shell
sudo apt install ocl-icd-opencl-dev ocl-icd-libopencl1
```
For Fedora, this can be done by running

```shell
sudo dnf install opencl-headers ocl-icd-devel
```

Install pyvkfft, a Python wrapper for the VkFFT library, using

```shell
pip install pyvkfft
```

Check that the OpenCL installation is working by running

```shell
python -c 'import pyopencl as cl;from pyvkfft.fft import rfftn; ps=cl.get_platforms();print(ps);print(ps[0].get_devices())'
# Should print the name of your GPU
```
</details>

Your system is now prepared, follow the general instructions [here](README.md#installation) to install
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

Follow the general instructions [here](README.md#installation) to install
**PowerFit**.

### Windows

First install *git* for Windows, as it comes with a handy bash shell. Go to
[git-scm](https://git-scm.com/download/), download *git* and install it. Next,
install a Python distribution such as
[Anaconda](http://continuum.io/downloads). After installation, open up the
bash shell shipped with *git* and follow the general instructions written
above.

## Tested platforms

| Operating System| CPU single | CPU multi | GPU |
| --------------- | ---------- | --------- | --- |
|Linux            | Yes        | Yes       | Yes |
|MacOSX           | Yes        | Yes       | No  |
|Windows          | Yes        | Fail      | No  |

The GPU version has been successfully tested on Linux and with a Docker container for the following devices;

* NVIDIA GeForce GTX 1050 Ti
* NVIDIA GeForce RTX 4070
* AMD Radeon RX 7800 XT
* AMD Radeon RX 7900 XTX
* Intel Iris Xe Graphics (on a Core i7-1185G7)

The integrated graphics of AMD Ryzen CPUs do not officially support OpenCL.
If they do seem available in PyOpenCL be aware that this [may lead to incorrect results](https://github.com/haddocking/powerfit/issues/76).