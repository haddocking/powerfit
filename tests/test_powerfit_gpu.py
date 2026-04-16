import unittest
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any, BinaryIO, cast
from unittest.mock import patch

import numpy as np

from powerfit_em import powerfit


class TestResolveGpuBackend(unittest.TestCase):
    def test_auto_gpu_prefers_cuda_when_both_backends_are_available(self):
        with (
            patch.object(powerfit, "cuda_available", return_value=True),
            patch.object(powerfit, "opencl_available", return_value=True),
        ):
            backend, device = powerfit.resolve_gpu_backend("auto")

        self.assertEqual(backend, "cuda")
        self.assertEqual(device, 0)

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

    def test_auto_gpu_uses_opencl_when_cuda_is_unavailable(self):
        with (
            patch.object(powerfit, "cuda_available", return_value=False),
            patch.object(powerfit, "opencl_available", return_value=True),
        ):
            backend, device = powerfit.resolve_gpu_backend("auto")

        self.assertEqual(backend, "opencl")
        self.assertEqual(device, (0, 0))

    def test_auto_gpu_without_available_backends_fails(self):
        with (
            patch.object(powerfit, "cuda_available", return_value=False),
            patch.object(powerfit, "opencl_available", return_value=False),
            self.assertRaisesRegex(ValueError, "requires either the cuda or opencl extra"),
        ):
            powerfit.resolve_gpu_backend("auto")

    def test_negative_cuda_device_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid CUDA device index"):
            powerfit.resolve_gpu_backend("cuda:-1")


class TestPowerfitCli(unittest.TestCase):
    def test_bare_gpu_flag_uses_auto_backend_resolution(self):
        with NamedTemporaryFile() as target, NamedTemporaryFile() as template:
            args = powerfit.make_parser().parse_args([target.name, "10", template.name, "--gpu"])

        self.assertEqual(args.gpu, "auto")

    def test_explicit_cuda_gpu_flag_is_preserved(self):
        with NamedTemporaryFile() as target, NamedTemporaryFile() as template:
            args = powerfit.make_parser().parse_args([target.name, "10", template.name, "--gpu", "cuda:0"])

        self.assertEqual(args.gpu, "cuda:0")


class TestGetCudaStream(unittest.TestCase):
    def test_returns_stream_for_valid_cuda_device(self):
        stream = object()

        class FakeDevice:
            def __init__(self, idx):
                self.idx = idx

            def use(self):
                return None

        fake_cupy: Any = ModuleType("cupy")
        fake_cupy.cuda = SimpleNamespace(
            runtime=SimpleNamespace(getDeviceCount=lambda: 2),
            Device=FakeDevice,
            Stream=lambda: stream,
        )

        with patch.dict("sys.modules", {"cupy": fake_cupy}):
            self.assertIs(powerfit.get_cuda_stream(1), stream)

    def test_rejects_out_of_range_cuda_device(self):
        fake_cupy: Any = ModuleType("cupy")
        fake_cupy.cuda = SimpleNamespace(
            runtime=SimpleNamespace(getDeviceCount=lambda: 1),
            Device=lambda idx: None,
            Stream=lambda: object(),
        )

        with (
            patch.dict("sys.modules", {"cupy": fake_cupy}),
            self.assertRaisesRegex(RuntimeError, "Requested CUDA device 1 not found"),
        ):
            powerfit.get_cuda_stream(1)

    def test_rejects_missing_cupy(self):
        with (
            patch.object(powerfit, "cuda_available", return_value=False),
            self.assertRaisesRegex(ValueError, "requires the cupy-cuda13x package"),
        ):
            powerfit.get_cuda_stream(0)


class TestPowerfitBackendWiring(unittest.TestCase):
    def test_powerfit_uses_cuda_stream_for_cuda_backend(self):
        target_volume = powerfit.Volume(np.ones((2, 2, 2), dtype=np.float32))
        template = powerfit.Volume(np.ones((2, 2, 2), dtype=np.float32))
        mask = powerfit.Volume(np.ones((2, 2, 2), dtype=np.float32))
        rotations = np.asarray([np.eye(3, dtype=np.float32)])
        analyzer = SimpleNamespace(solutions=[], tofile=lambda *args, **kwargs: None)
        structure = object()
        stream = object()

        with (
            NamedTemporaryFile() as target,
            NamedTemporaryFile() as template_file,
            TemporaryDirectory() as directory,
            patch.object(powerfit, "resolve_gpu_backend", return_value=("cuda", 0)),
            patch.object(powerfit, "get_cuda_stream", return_value=stream) as get_cuda_stream,
            patch.object(powerfit, "setup_target", return_value=target_volume),
            patch.object(powerfit, "setup_template_structure", return_value=(structure, template, mask, 1.0)),
            patch.object(powerfit, "setup_rotational_matrix", return_value=rotations),
            patch.object(powerfit, "PowerFitter") as powerfitter_cls,
            patch.object(powerfit, "Analyzer", return_value=analyzer),
            patch.object(powerfit, "write_fits_to_pdb"),
            patch.object(powerfit.Volume, "tofile"),
        ):
            powerfitter_cls.return_value.lcc = np.zeros((2, 2, 2), dtype=np.float32)
            powerfitter_cls.return_value.rot = np.zeros((2, 2, 2), dtype=np.int32)

            powerfit.powerfit(
                cast(BinaryIO, target.file),
                10,
                cast(BinaryIO, template_file.file),
                directory=directory,
                gpu="cuda:0",
                progress=None,
            )

        get_cuda_stream.assert_called_once_with(0)
        self.assertIsNone(powerfitter_cls.call_args.args[4])
        self.assertIs(powerfitter_cls.call_args.kwargs["cuda_stream"], stream)

    def test_powerfit_uses_opencl_queue_for_opencl_backend(self):
        target_volume = powerfit.Volume(np.ones((2, 2, 2), dtype=np.float32))
        template = powerfit.Volume(np.ones((2, 2, 2), dtype=np.float32))
        mask = powerfit.Volume(np.ones((2, 2, 2), dtype=np.float32))
        rotations = np.asarray([np.eye(3, dtype=np.float32)])
        analyzer = SimpleNamespace(solutions=[], tofile=lambda *args, **kwargs: None)
        structure = object()
        queue = object()

        with (
            NamedTemporaryFile() as target,
            NamedTemporaryFile() as template_file,
            TemporaryDirectory() as directory,
            patch.object(powerfit, "resolve_gpu_backend", return_value=("opencl", (0, 0))),
            patch.object(powerfit, "get_opencl_queue", return_value=queue) as get_opencl_queue,
            patch.object(powerfit, "setup_target", return_value=target_volume),
            patch.object(powerfit, "setup_template_structure", return_value=(structure, template, mask, 1.0)),
            patch.object(powerfit, "setup_rotational_matrix", return_value=rotations),
            patch.object(powerfit, "PowerFitter") as powerfitter_cls,
            patch.object(powerfit, "Analyzer", return_value=analyzer),
            patch.object(powerfit, "write_fits_to_pdb"),
            patch.object(powerfit.Volume, "tofile"),
        ):
            powerfitter_cls.return_value.lcc = np.zeros((2, 2, 2), dtype=np.float32)
            powerfitter_cls.return_value.rot = np.zeros((2, 2, 2), dtype=np.int32)

            powerfit.powerfit(
                cast(BinaryIO, target.file),
                10,
                cast(BinaryIO, template_file.file),
                directory=directory,
                gpu="0:0",
                progress=None,
            )

        get_opencl_queue.assert_called_once_with("0:0")
        self.assertIs(powerfitter_cls.call_args.args[4], queue)
        self.assertIsNone(powerfitter_cls.call_args.kwargs["cuda_stream"])


if __name__ == "__main__":
    unittest.main()
