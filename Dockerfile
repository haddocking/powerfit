FROM ubuntu:18.04

ENV LD_LIBRARY_PATH=/usr/local/lib64
ENV CLFFT_DIR=/src/clFFT
ENV CLFFT_LIB_DIRS=/usr/local/lib64
ENV CLFFT_INCL_DIRS=/src/clFFT/src/include
ENV CL_INCL_DIRS=/usr/local/include
ENV LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib

RUN apt-get update --fix-missing && \
    apt-get install -y \
        python2.7 \
        python2.7-dev \
        git \
        wget \
        gcc \
        g++ \
        libfftw3-dev \
        libfftw3-doc \
        libfreetype6-dev \
        pkg-config \
        libopenblas-dev \
        gfortran \
        python-scipy \
        python-numpy \
        time \
        cmake \
        make \
        opencl-headers \
        ocl-icd-opencl-dev \
        libboost-all-dev \
        pocl-opencl-icd


RUN mkdir src && \
    cd /src && \
    wget https://bootstrap.pypa.io/pip/2.7/get-pip.py && \
    python2 get-pip.py && \
    python2.7 -mpip install Cython==0.29.33 pyfftw==0.12.0 pybind11 setuptools && \
    python2.7 -mpip install numpy==1.16.6 pyopencl==2020.2 install backports.weakref==1.0.post1

RUN cd /src && \
    git clone https://github.com/clMathLibraries/clFFT.git && \
    cd clFFT/src && \
    cmake . && \
    make && \
    make install

RUN cd /src && \
    git clone https://github.com/haddocking/powerfit && \
    cd powerfit && \
    python2.7 setup.py install

RUN cd /src && \
    git clone https://github.com/geggo/gpyfft.git && \
    export LD_LIBRARY_PATH=/usr/local/lib64 && \
    export CLFFT_DIR=/src/clFFT && \
    export CLFFT_LIB_DIRS=/usr/local/lib64 && \
    export CLFFT_INCL_DIRS=/src/clFFT/src/include && \
    export CL_INCL_DIRS=/usr/local/include && \
    export LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib  && \
    cd gpyfft && \
    sed -i \
        -e "s|CLFFT_DIR = r'/home/gregor/devel/clFFT'|CLFFT_DIR = '/src/clFFT'|" \
        -e "s|CL_INCL_DIRS = \['/opt/AMDAPPSDK-3.0/include'\]|CL_INCL_DIRS = ['/usr/local/include']|" \
    setup.py && \
    python2.7 setup.py install

