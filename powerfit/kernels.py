import os.path
import numpy as np
import pyopencl as cl

OPENCL_KERNELS = os.path.join(os.path.dirname(__file__), 'kernels', 'powerfit.cl')

class Kernels(object):

    def __init__(self, ctx):

        self.k = cl.Program(ctx, open(OPENCL_KERNELS).read()).build()

    def rmultiply(self, queue, x, y, out):

        compute_units = queue.device.max_compute_units
        work_groups = compute_units*64*8

        status = self.k.rmultiply(queue, (work_groups,), None,
                x.data, y.data, out.data,
                np.int32(x.size), np.int32(y.size))

        return status

    def cconjmultiply(self, queue, x, y, out):

        compute_units = queue.device.max_compute_units
        work_groups = compute_units*64*8

        self.k.cconjmultiply.set_args(x.data, y.data, out.data,
            np.int32(x.size), np.int32(y.size))

        status = cl.enqueue_nd_range_kernel(queue, self.k.cconjmultiply, (work_groups,), None)

        return status

    def rotate_im(self, queue, image, rotmat, out):

        compute_units = queue.device.max_compute_units
        work_groups = compute_units*64*8
        work_groups = [int((compute_units*64*8)**(1/3))]*3

        #status = self.k.rotate_im(queue, work_groups, None,
        #                          image, rotmat.data, out.data)
                 


