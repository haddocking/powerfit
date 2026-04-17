import unittest
from tempfile import NamedTemporaryFile

import pytest

from powerfit_em import powerfit
from powerfit_em.helpers import cuda_available, opencl_available

CUDA_AVAILABLE = cuda_available()
OPENCL_AVAILABLE = opencl_available()


class TestResolveGpuBackendParsing(unittest.TestCase):
    """Tests for resolve_gpu_backend that require no hardware — pure string/value logic."""

    def test_explicit_cuda_backend_is_selected(self):
        backend, device = powerfit.resolve_gpu_backend("cuda:3")
        self.assertEqual(backend, "cuda")
        self.assertEqual(device, 3)

    def test_explicit_opencl_backend_is_selected(self):
        backend, device = powerfit.resolve_gpu_backend("2:5")
        self.assertEqual(backend, "opencl")
        self.assertEqual(device, (2, 5))

    def test_invalid_gpu_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid --gpu value"):
            powerfit.resolve_gpu_backend("cuda")

    def test_negative_cuda_device_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid CUDA device index"):
            powerfit.resolve_gpu_backend("cuda:-1")

    def test_none_returns_none_backend(self):
        backend, device = powerfit.resolve_gpu_backend(None)
        self.assertIsNone(backend)
        self.assertIsNone(device)


@pytest.mark.skipif(
    not CUDA_AVAILABLE or not OPENCL_AVAILABLE,
    reason="Requires both CUDA and OpenCL to be installed",
)
class TestResolveGpuBackendBothAvailable(unittest.TestCase):
    def test_auto_gpu_prefers_cuda_when_both_backends_are_available(self):
        backend, device = powerfit.resolve_gpu_backend("auto")
        self.assertEqual(backend, "cuda")
        self.assertEqual(device, 0)


@pytest.mark.skipif(
    not OPENCL_AVAILABLE or CUDA_AVAILABLE,
    reason="Requires OpenCL but not CUDA",
)
class TestResolveGpuBackendOpenCLOnly(unittest.TestCase):
    def test_auto_gpu_uses_opencl_when_cuda_is_unavailable(self):
        backend, device = powerfit.resolve_gpu_backend("auto")
        self.assertEqual(backend, "opencl")
        self.assertEqual(device, (0, 0))


class TestPowerfitCli(unittest.TestCase):
    def test_bare_gpu_flag_uses_auto_backend_resolution(self):
        with NamedTemporaryFile() as target, NamedTemporaryFile() as template:
            args = powerfit.make_parser().parse_args([target.name, "10", template.name, "--gpu"])
        self.assertEqual(args.gpu, "auto")

    def test_explicit_cuda_gpu_flag_is_preserved(self):
        with NamedTemporaryFile() as target, NamedTemporaryFile() as template:
            args = powerfit.make_parser().parse_args([target.name, "10", template.name, "--gpu", "cuda:0"])
        self.assertEqual(args.gpu, "cuda:0")


@pytest.mark.requires_cuda
@pytest.mark.gpu_integration
class TestGetCudaStream(unittest.TestCase):
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def setUp(self):
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA not available")

    def test_returns_stream_for_valid_cuda_device(self):
        import cupy as cp  # pyright: ignore[reportMissingImports]

        stream = powerfit.get_cuda_stream(0)
        self.assertIsInstance(stream, cp.cuda.Stream)

    def test_rejects_out_of_range_cuda_device(self):
        import cupy as cp  # pyright: ignore[reportMissingImports]

        device_count = cp.cuda.runtime.getDeviceCount()
        with self.assertRaisesRegex(RuntimeError, "Requested CUDA device"):
            powerfit.get_cuda_stream(device_count)


if __name__ == "__main__":
    unittest.main()
