import importlib.resources
from string import Template

import numpy as np
import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel

import powerfit_em.correlators


class CLKernels:
    def __init__(self, ctx, values):
        self.sampler_nearest = cl.Sampler(ctx, True, cl.addressing_mode.REPEAT, cl.filter_mode.NEAREST)
        self.sampler_linear = cl.Sampler(ctx, True, cl.addressing_mode.REPEAT, cl.filter_mode.LINEAR)
        self.multiply = ElementwiseKernel(ctx, "float *x, float *y, float *z", "z[i] = x[i] * y[i];")
        self.conj_multiply = ElementwiseKernel(
            ctx, "cfloat_t *x, cfloat_t *y, cfloat_t *z", "z[i] = cfloat_mul(cfloat_conj(x[i]), y[i]);"
        )
        self.calc_lcc_and_take_best = ElementwiseKernel(
            ctx,
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
                """,
        )

        t = importlib.resources.read_text(powerfit_em.correlators, "kernels.cl")
        t = Template(t).substitute(**values)

        self._program = cl.Program(ctx, t).build()
        self._rotate_image3d = self._program.rotate_image3d
        self._rotate_image3d_batch = self._program.rotate_image3d_batch
        self._batch_lcc_and_take_best = self._program.powerfit_batch_lcc_and_take_best
        self._batch_conj_multiply = self._program.powerfit_batch_conj_multiply
        self._gws_rotate_grid3d = (96, 64, 1)
        self._gws_rotate_grid3d_batch = None

    def rotate_image3d(self, queue: cl.CommandQueue, image, rotmat, out, nearest=False):
        if nearest:
            args = (image, self.sampler_nearest, rotmat, out.data)
        else:
            args = (image, self.sampler_linear, rotmat, out.data)
        self._rotate_image3d(queue, self._gws_rotate_grid3d, None, *args)

    def rotate_image3d_batch(
        self,
        queue: cl.CommandQueue,
        image,
        rotmats,
        rot_offset: int,
        batch_size: int,
        out,
        nearest: bool = False,
    ):
        rot_offset_i32 = np.int32(rot_offset)
        batch_size_i32 = np.int32(batch_size)
        if nearest:
            args = (image, self.sampler_nearest, rotmats.data, rot_offset_i32, batch_size_i32, out.data)
        else:
            args = (image, self.sampler_linear, rotmats.data, rot_offset_i32, batch_size_i32, out.data)
        gws = (batch_size, 1, 1)
        self._rotate_image3d_batch(queue, gws, None, *args)

    def batch_lcc_and_take_best(
        self,
        queue: cl.CommandQueue,
        gcc,
        ave,
        ave2,
        mask,
        lcc,
        rot,
        norm_factor,
        batch_start: int,
        batch_size: int,
        volume_size: int,
    ):
        block = 256
        gws = ((volume_size + block - 1) // block * block,)
        self._batch_lcc_and_take_best(
            queue,
            gws,
            None,
            gcc.data,
            ave.data,
            ave2.data,
            mask.data,
            lcc.data,
            rot.data,
            norm_factor,
            np.int32(batch_start),
            np.int32(batch_size),
            np.int32(volume_size),
        )

    def batch_conj_multiply(
        self,
        queue: cl.CommandQueue,
        batch_a,
        broadcast_b,
        out,
        batch_size: int,
        ft_vol_size: int,
    ):
        """Compute out[b][i] = conj(batch_a[b][i]) * broadcast_b[i] for all b, i."""
        total_size = batch_size * ft_vol_size
        block = 256
        gws = ((total_size + block - 1) // block * block,)
        self._batch_conj_multiply(
            queue,
            gws,
            None,
            batch_a.data,
            broadcast_b.data,
            out.data,
            np.int32(ft_vol_size),
            np.int32(total_size),
        )
