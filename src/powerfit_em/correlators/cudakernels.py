import importlib.resources
from string import Template

import cupy as cp

import powerfit_em.correlators


class CUDAKernels:
    def __init__(self, shape: tuple[int, int, int]):
        values = {
            "shape_x": shape[2],
            "shape_y": shape[1],
            "shape_z": shape[0],
            "llength": min(shape) // 2,
        }

        t = importlib.resources.read_text(powerfit_em.correlators, "kernels.cu")
        t = Template(t).substitute(**values)

        self._module = cp.RawModule(code=t)
        self._rotate_image3d_linear = self._module.get_function("rotate_image3d_linear")
        self._rotate_image3d_nearest = self._module.get_function("rotate_image3d_nearest")

        # Matches the OpenCL global work size footprint in the xy plane.
        self._block = (8, 8, 4)
        size = 2 * values["llength"] + 1
        self._grid = (
            (size + self._block[0] - 1) // self._block[0],
            (size + self._block[1] - 1) // self._block[1],
            (size + self._block[2] - 1) // self._block[2],
        )

    def rotate_image3d(self, image: cp.ndarray, rotmat, out: cp.ndarray, nearest: bool = False):
        rot = cp.asarray(rotmat, dtype=cp.float32).ravel()
        if nearest:
            self._rotate_image3d_nearest(self._grid, self._block, (image, rot, out))
        else:
            self._rotate_image3d_linear(self._grid, self._block, (image, rot, out))
