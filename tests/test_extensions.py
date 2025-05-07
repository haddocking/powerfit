from unittest import TestCase, main

import numpy as np

from powerfit_em._extensions import rotate_grid3d


class TestExtensions(TestCase):

    def test_rotate_grid3d(self):

        grid = np.zeros((4, 5, 6), dtype=np.float64)
        grid[0, 0, 0] = 1
        grid[0, 0, 1] = 1
        grid[0, 1, 1] = 1
        grid[0, 0, 2] = 1
        grid[0, 0, -1] = 1
        grid[-1, 0, 0] = 1
        # Identity rotation
        rotmat = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        out = np.zeros_like(grid)
        rotate_grid3d(grid, rotmat, 2, out, True)
        self.assertTrue(np.allclose(out, grid))

        # 90 degree rotation around Z-axis
        out.fill(0)
        rotmat = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        rotate_grid3d(grid, rotmat, 2, out, False)
        # Build answer
        answer = np.zeros_like(out)
        answer[0, 0, 0] = 1
        answer[0, 1, 0] = 1
        answer[0, 1, -1] = 1
        answer[0, 2, 0] = 1
        answer[0, -1, 0] = 1
        answer[-1, 0, 0] = 1
        self.assertTrue(np.allclose(answer, out))


if __name__ == "__main__":
    main()
