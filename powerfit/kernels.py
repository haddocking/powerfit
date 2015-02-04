from __future__ import print_function
import numpy as np
import os.path
import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel

class Kernels():

    def __init__(self, ctx):
        self.context = ctx

        self.kernel_file = os.path.join(os.path.dirname(__file__), 'kernels', 'kernels.cl')
        self.kernels = cl.Program(ctx, open(self.kernel_file).read()).build()

        self.kernels.multiply_f32 = ElementwiseKernel(ctx,
                     "float *x, float *y, float *z",
                     "z[i] = x[i] * y[i]",
                     )

        self.kernels.multiply_int32 = ElementwiseKernel(ctx,
                     "int *x, int *y, int *z",
                     "z[i] = x[i] * y[i]",
                     )

        self.kernels.c_conj_multiply = ElementwiseKernel(ctx,
                     "cfloat_t *x, cfloat_t *y, cfloat_t *z",
                     "z[i] = cfloat_mul(cfloat_conj(x[i]),y[i]);",
                     )

        self.kernels.set_to_f = ElementwiseKernel(ctx,
            """float* array, float value""",
            """array[i] = value;""",)

        self.kernels.set_to_i = ElementwiseKernel(ctx,
            """int* array, int value""",
            """array[i] = value;""",)

        self.kernels.lcc = ElementwiseKernel(ctx,
            """float *gcc, float *map_ave, float *map2_ave,
               float norm_factor, float *lcc, float varlimit""",
            """float var = norm_factor*map2_ave[i] - pown(map_ave[i], 2);
               if (var > varlimit)
                   lcc[i] = gcc[i]/sqrt(var);
               else
                   lcc[i] = 0.0f;""",
            )

        self.kernels.take_best = ElementwiseKernel(ctx,
            """float *lcc, float *best_lcc, int *rotmat_ind, int ind""",
            """if (lcc[i] > best_lcc[i]) {
                   best_lcc[i] = lcc[i];
                   rotmat_ind[i] = ind;
               }""",
            )
            

        
    def c_conj_multiply(self, queue, array1, array2, out):
        if (array1.dtype == array2.dtype == out.dtype == np.complex64):
            status = self.kernels.c_conj_multiply(array1, array2, out)
        else:
            raise TypeError("Datatype of arrays is not supported")

        return status

    def multiply(self, queue, array1, array2, out):
        if array1.dtype == array2.dtype == out.dtype == np.float32:
            status = self.kernels.multiply_f32(array1, array2, out)
        elif array1.dtype == array2.dtype == out.dtype == np.int32:
            status = self.kernels.multiply_int32(array1, array2, out)
        else:
            raise TypeError("Array type is not supported")
        return status

    def blur_points(self, queue, points, weights, sigma, out):

        kernel = self.kernels.blur_points

        shape = np.zeros(4, dtype=np.int32)
        shape[:3] = out.shape

        kernel.set_args(points.data, weights.data, np.float32(sigma),
                out.data, shape, np.int32(points.shape[0]))

        compute_units = queue.device.max_compute_units
        max_wg = compute_units*16*8*8
        zwg = int(max(1, min(max_wg, out.shape[0])))
        ywg = int(max(1, min(max_wg - zwg, out.shape[1])))
        xwg = int(max(1, min(max_wg - zwg - ywg, out.shape[2])))
        wg = (zwg, ywg, xwg)
        print(wg)

        status = cl.enqueue_nd_range_kernel(queue, kernel, wg, None)

        return status

    def rotate_image3d(self, queue, sampler, image3d,
            rotmat, array_buffer, center):

        kernel = self.kernels.rotate_image3d
        compute_units = queue.device.max_compute_units

        work_groups = (compute_units*16*8, 1, 1)

        shape = np.asarray(list(array_buffer.shape) + [np.product(array_buffer.shape)], dtype=np.int32)

        inv_rotmat = np.linalg.inv(rotmat)
        inv_rotmat16 = np.zeros(16, dtype=np.float32)
        inv_rotmat16[:9] = inv_rotmat.flatten()[:]

        _center = np.zeros(4, dtype=np.float32)
        _center[:3] = center[:]

        kernel.set_args(sampler, image3d, inv_rotmat16, array_buffer.data, _center, shape)
        status = cl.enqueue_nd_range_kernel(queue, kernel, work_groups, None)

        return status

    def lcc(self, queue, gcc, map_ave, map2_ave, norm_factor, lcc, varlimit):

        status = self.kernels.lcc(gcc, map_ave, map2_ave, 
                np.float32(norm_factor), lcc, np.float32(varlimit))

        return status
        
    def take_best(self, queue, lcc, best_lcc, rotmat_ind, n):
        status = self.kernels.take_best(lcc, best_lcc, rotmat_ind, np.int32(n))
        return status

    def fill(self, queue, array, value):
        if array.dtype == np.float32:
            status = self.kernels.set_to_f(array, np.float32(value))
        elif array.dtype == np.int32:
            status = self.kernels.set_to_i(array, np.int32(value))
        else:
            raise TypeError("Array type ({:s}) is not supported.".format(array.dtype))
        return status
