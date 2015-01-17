import os.path
import pyopencl as cl

OPENCL_KERNELS = os.path.join(os.path.dirname(__file__), 'kernels', 'powerfit.cl')

class Kernels(object):

    def __init__(self, ctx):

        self.k = cl.Program(ctx, open(OPENCL_KERNELS).read().build())

    def rmultiply(self, queue, x, y, out):

        compute_units = queue.device.max_compute_units
        work_groups = compute_units*64*8

        status = self.k.rmultiply(queue, (work_groups,), None,
                x.data, y.data, z.data,
                )

        return status

