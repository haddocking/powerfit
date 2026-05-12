from tempfile import NamedTemporaryFile

import pytest

from powerfit_em import powerfit
from powerfit_em.gpu import cuda_available, get_cuda_stream, opencl_available, setup_gpu_backend

CUDA_AVAILABLE = cuda_available()
OPENCL_AVAILABLE = opencl_available()


class TestSetupGpuModuleParsing:
    """Tests for setup_gpu_backend GPU string parsing - no hardware needed."""

    def test_invalid_gpu_value_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid --gpu value"):
            setup_gpu_backend("cuda")

    def test_negative_cuda_device_index_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid CUDA device index"):
            setup_gpu_backend("cuda:-1")

    def test_none_returns_no_resources(self):
        queue, cuda_stream = setup_gpu_backend(None)
        assert queue is None
        assert cuda_stream is None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestSetupGpuModuleCUDA:
    """Tests for setup_gpu_backend with CUDA backend."""

    def test_explicit_cuda_backend_is_selected(self):
        queue, cuda_stream = setup_gpu_backend("cuda:0")
        assert cuda_stream is not None
        assert queue is None

    def test_cuda_device_index_passed_to_stream(self):
        import cupy as cp  # pyright: ignore[reportMissingImports]

        queue, cuda_stream = setup_gpu_backend("cuda:0")
        assert queue is None
        assert isinstance(cuda_stream, cp.cuda.Stream)


@pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
class TestSetupGpuModuleOpenCL:
    """Tests for setup_gpu_backend with OpenCL backend."""

    def test_explicit_opencl_backend_is_selected(self):
        import pyopencl as cl

        queue, cuda_stream = setup_gpu_backend("0:0")
        assert isinstance(queue, cl.CommandQueue)
        assert cuda_stream is None


@pytest.mark.skipif(
    not CUDA_AVAILABLE or not OPENCL_AVAILABLE,
    reason="Requires both CUDA and OpenCL to be installed",
)
class TestSetupGpuModuleBothAvailable:
    def test_auto_gpu_prefers_cuda_when_both_backends_are_available(self):
        queue, cuda_stream = setup_gpu_backend("auto")
        assert cuda_stream is not None
        assert queue is None


@pytest.mark.skipif(
    not OPENCL_AVAILABLE or CUDA_AVAILABLE,
    reason="Requires OpenCL but not CUDA",
)
class TestSetupGpuModuleOpenCLOnly:
    def test_auto_gpu_uses_opencl_when_cuda_is_unavailable(self):
        queue, cuda_stream = setup_gpu_backend("auto")
        assert queue is not None
        assert cuda_stream is None


class TestPowerfitCliGpuFlag:
    def test_bare_gpu_flag_uses_auto_backend_resolution(self):
        with NamedTemporaryFile() as target, NamedTemporaryFile() as template:
            args = powerfit.make_parser().parse_args([target.name, "10", template.name, "--gpu"])
        assert args.gpu == "auto"

    def test_explicit_cuda_gpu_flag_is_preserved(self):
        with NamedTemporaryFile() as target, NamedTemporaryFile() as template:
            args = powerfit.make_parser().parse_args([target.name, "10", template.name, "--gpu", "cuda:0"])
        assert args.gpu == "cuda:0"


@pytest.mark.requires_cuda
@pytest.mark.gpu_integration
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGetCudaStream:
    def test_returns_stream_for_valid_cuda_device(self):
        import cupy as cp  # pyright: ignore[reportMissingImports]

        stream = get_cuda_stream(0)
        assert isinstance(stream, cp.cuda.Stream)

    def test_rejects_out_of_range_cuda_device(self):
        import cupy as cp  # pyright: ignore[reportMissingImports]

        device_count = cp.cuda.runtime.getDeviceCount()
        with pytest.raises(RuntimeError, match="Requested CUDA device"):
            get_cuda_stream(device_count)
