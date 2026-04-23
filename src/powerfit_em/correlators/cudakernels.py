import importlib.resources
from string import Template

import cupy as cp
import numpy as np

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
        self._rotate_image3d_linear_batch = self._module.get_function("rotate_image3d_linear_batch")
        self._rotate_image3d_nearest_batch = self._module.get_function("rotate_image3d_nearest_batch")
        self._batch_lcc_kernel = self._module.get_function("powerfit_batch_lcc_and_take_best")

        self._shape = shape
        # Cover the full output grid; kernels zero-out voxels outside the
        # valid spherical region.
        self._block = (8, 8, 4)
        self._grid = (
            (shape[2] + self._block[0] - 1) // self._block[0],
            (shape[1] + self._block[1] - 1) // self._block[1],
            (shape[0] + self._block[2] - 1) // self._block[2],
        )

    def rotate_image3d(self, image: cp.ndarray, rotmat, out: cp.ndarray, nearest: bool = False):
        if isinstance(rotmat, cp.ndarray) and rotmat.dtype == cp.float32:
            rot = rotmat.ravel()
        else:
            rot = cp.asarray(rotmat, dtype=cp.float32).ravel()
        if nearest:
            self._rotate_image3d_nearest(self._grid, self._block, (image, rot, out))
        else:
            self._rotate_image3d_linear(self._grid, self._block, (image, rot, out))

    def rotate_image3d_batch(
        self,
        image: cp.ndarray,
        rotmats: cp.ndarray,
        out: cp.ndarray,
        batch_size: int,
        nearest: bool = False,
    ):
        """Rotate *image* for a batch of rotation matrices in one kernel launch.

        Args:
            image: source volume, shape (Z, Y, X).
            rotmats: flat rotation matrices, shape (batch_size, 9) or (batch_size * 9,).
            out: destination buffer, shape (batch_size, Z, Y, X).
            batch_size: number of rotations to process.
            nearest: use nearest-neighbour interpolation when True, else trilinear.
        """
        if not (isinstance(rotmats, cp.ndarray) and rotmats.dtype == cp.float32):
            rotmats = cp.asarray(rotmats, dtype=cp.float32)
        rot_flat = rotmats.ravel()
        n_batch = np.int32(batch_size)
        total_z = self._shape[0] * batch_size
        grid = (
            self._grid[0],
            self._grid[1],
            (total_z + self._block[2] - 1) // self._block[2],
        )
        if nearest:
            self._rotate_image3d_nearest_batch(grid, self._block, (image, rot_flat, out, n_batch))
        else:
            self._rotate_image3d_linear_batch(grid, self._block, (image, rot_flat, out, n_batch))

    @property
    def batch_lcc_kernel(self):
        return self._batch_lcc_kernel
