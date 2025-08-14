import numpy as np
import pyopencl as cl
from pyopencl.array import Array as ClArray
from pyopencl.elementwise import ElementwiseKernel


from string import Template
import importlib.resources

import powerfit_em.correlators


class CLKernels(object):
    def __init__(self, ctx, values):
        self.sampler_nearest = cl.Sampler(ctx, True,
                cl.addressing_mode.REPEAT, cl.filter_mode.NEAREST)
        self.sampler_linear = cl.Sampler(ctx, True,
                cl.addressing_mode.REPEAT, cl.filter_mode.LINEAR)
        self.multiply = ElementwiseKernel(ctx,
                "float *x, float *y, float *z",
                "z[i] = x[i] * y[i];"
                )
        self.conj_multiply = ElementwiseKernel(ctx,
                "cfloat_t *x, cfloat_t *y, cfloat_t *z",
                "z[i] = cfloat_mul(cfloat_conj(x[i]), y[i]);"
                )
        self.calc_lcc_and_take_best = ElementwiseKernel(ctx,
                """float *gcc, float *ave, float *ave2, int *mask,
                    float norm_factor, int nrot, float *lcc, int *grot""",
                """float _lcc;
                    if (mask[i] > 0) {
                        _lcc = gcc[i] / sqrt(ave2[i] * norm_factor - ave[i] * ave[i]);
                        if (_lcc > lcc[i]) {
                            lcc[i] = _lcc;
                            grot[i] = nrot;
                        };
                    };
                """
                )

        t = importlib.resources.read_text(powerfit_em.correlators, "kernels.cl")
        t = Template(t).substitute(**values)

        self._program = cl.Program(ctx, t).build()
        self._rotate_image3d = self._program.rotate_image3d
        self._gws_rotate_grid3d = (96, 64, 1)

    def rotate_grid3d(
        self,
        queue: cl.CommandQueue,
        grid:ClArray,
        rotmat: np.ndarray,
        out: ClArray,
        nearest: bool = False
    ):
        args = (grid.data, rotmat, out.data, np.int32(nearest))
        self._program.rotate_grid3d(queue, self._gws_rotate_grid3d, None, *args)

    def rotate_image3d(self, queue: cl.CommandQueue, image, rotmat, out, nearest=False):
        if nearest:
            args = (image, self.sampler_nearest, rotmat, out.data)
        else:
            args = (image, self.sampler_linear, rotmat, out.data)
        self._rotate_image3d(queue, self._gws_rotate_grid3d, None, *args)