"""Module for GPU availability checks and setup."""

from functools import cache
from typing import TYPE_CHECKING

from powerfit_em.helpers import logger

if TYPE_CHECKING:
    # To have imports available run:
    # `uv sync --extra dev --extra opencl --extra pocl --extra cuda13x`
    from cupy.cuda.stream import Stream
    from pyopencl import CommandQueue


@cache
def opencl_available() -> bool:
    """Check if OpenCL is available by trying to import pyopencl and checking for devices."""
    try:
        from importlib import import_module

        cl = import_module("pyopencl")
        try:
            if platforms := cl.get_platforms():
                for platform in platforms:
                    if platform.get_devices():
                        return True
            return False
        except cl.LogicError:
            return False
    except ModuleNotFoundError:
        return False


@cache
def cuda_available() -> bool:
    """Check if CUDA is available by trying to import cupy and pyvkfft.cuda and checking for devices.

    pyvkfft can be installed without CUDA support, will warn if that is the case.
    """
    try:
        from importlib import import_module

        cp = import_module("cupy")
        import_module("pyvkfft.cuda")
        return cp.cuda.runtime.getDeviceCount() > 0
    except ModuleNotFoundError:
        return False
    except OSError:
        logger.warning(
            "pyvkfft CUDA backend is not available. Please make sure pyvkfft is installed with CUDA support."
        )
        return False


def setup_gpu_backend(gpu: str | None) -> "tuple[CommandQueue | None, Stream | None]":
    """Setup GPU backend and return OpenCL queue and/or CUDA stream.

    Resolves the GPU backend from the gpu argument string and initializes
    the corresponding GPU resources (queue for OpenCL, stream for CUDA).

    Args:
        gpu: GPU backend specification string. Use "auto" for automatic selection, "cuda:N"
            for CUDA device N, or "P:D" for OpenCL platform P and device D. If None,
            no GPU is used.

    Returns:
        A tuple of (queue, cuda_stream), where exactly one resource is initialized
        when a GPU backend is selected. For CPU mode, both values are None.
    """
    if gpu is None:
        return None, None

    if gpu == "auto":
        if cuda_available():
            return None, get_cuda_stream(0)
        if opencl_available():
            return get_opencl_queue("0:0"), None
        raise ValueError("Running on GPU requires either the cuda or opencl extra to be installed.")

    if gpu.startswith("cuda:"):
        device_idx = int(gpu.split(":", maxsplit=1)[1])
        if device_idx < 0:
            raise ValueError("Invalid CUDA device index. Use a non-negative integer.")
        return None, get_cuda_stream(device_idx)

    if gpu.count(":") == 1:
        platform_idx, device_idx = map(int, gpu.split(":"))
        return get_opencl_queue(f"{platform_idx}:{device_idx}"), None

    raise ValueError("Invalid --gpu value. Use --gpu, --gpu cuda:N, or --gpu P:D.")


def get_opencl_queue(gpu: str) -> "CommandQueue":
    """Request an OpenCL Queue."""
    if not opencl_available():
        msg = "Running on GPU requires the pyopencl package, however importing pyopencl failed."
        raise ValueError(msg)
    else:
        import pyopencl as cl  # pyright: ignore[reportMissingImports]

    # TODO allow to omit platform, so gpu='4' runs 5th device on first platform
    if ":" in gpu:
        platform_idx, device_idx = map(int, gpu.split(":"))
    else:
        platform_idx, device_idx = 0, 0
    platforms = cl.get_platforms()
    if platform_idx >= len(platforms):
        raise RuntimeError(f"Requested OpenCL platform {platform_idx} not found.")
    platform = platforms[platform_idx]
    devices = platform.get_devices()
    if device_idx >= len(devices):
        raise RuntimeError(f"Requested OpenCL device {device_idx} not found on platform {platform_idx}.")
    context = cl.Context(devices=[devices[device_idx]])
    return cl.CommandQueue(context, device=devices[device_idx])


def get_cuda_stream(device_idx: int) -> "Stream":
    """Request a CUDA stream for a specific device."""
    if not cuda_available():
        msg = "Running on CUDA requires the cupy and pyvkfft.cuda packages, however importing them failed."
        raise ValueError(msg)

    import cupy as cp  # pyright: ignore[reportMissingImports]

    device_count = cp.cuda.runtime.getDeviceCount()
    if device_idx >= device_count:
        msg = f"Requested CUDA device {device_idx} not found. Available device indices: 0-{device_count - 1}."
        raise RuntimeError(msg)

    device = cp.cuda.Device(device_idx)
    device.use()
    return cp.cuda.Stream()
