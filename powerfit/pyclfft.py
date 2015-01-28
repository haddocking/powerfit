from gpyfft import GpyFFT

G = GpyFFT()

IMAGINARY = 3
REAL = 5

def builder(context, shape, direction_forward=True):
    pass
    
class RFFTn:
    def __init__(self, context, shape):
        # The maximum number of elements of the transform is 2^24 
	# in clFFT (single precision)
        elements = 1
        for i in shape:
	    elements *= i
        if (elements > (2.0**24.0)):
            from math import log
            power = log(elements, 2)
            raise ValueError('The maximum number of elements for clFFT is 2^24, currently you want 2^{:.2f}'.format(power))
	    

        ndim = len(shape)
        if ndim > 3:
            raise ValueError('clFFT can only work up-to 3 dimensions')

        self.context = context
        self.shape = tuple(shape)
        self.ft_shape = (shape[0]//2 + 1, shape[1], shape[2])
        self.ndim = ndim

        self.plan = G.create_plan(context, self.shape)
        self.plan.inplace = False
        self.plan.layouts = (REAL, IMAGINARY)

        if ndim == 3:
            self.plan.strides_in = (shape[1]*shape[2], shape[2], 1)
        elif ndim == 2:
            self.plan.strides_in = (shape[1], 1)
        elif ndim == 1:
            self.plan.strides_in = (1,) 

        if ndim == 3:
            self.plan.strides_out = (shape[1]*shape[2], shape[2], 1)
        elif ndim == 2:
            self.plan.strides_out = (shape[1], 1) 
        elif ndim == 1:
            self.plan.strides_out = (1,)

        self.baked = False

    def bake(self, queues):
        if not self.baked:
            self.plan.bake(queues)
            self.baked = True

    def __call__(self, queue, inarray, outarray):
        self.plan.enqueue_transform(queue, inarray.data, outarray.data)
        self.baked = True


class iRFFTn:
    def __init__(self, context, shape):

        # The maximum number of elements of the transform is 2^24 
	# in clFFT (single precision)
        elements = 1
	for i in shape:
	    elements *= i
	if elements > 2**24:
	    from math import log
	    power = log(elements, 2)
	    raise ValueError('The maximum number of elements for clFFT is 2^24, currently you want 2^{:.2f}'.format(power))

        ndim = len(shape)
        if ndim > 3:
            raise ValueError('clFFT can only work up-to 3 dimensions')

        self.context = context
        self.shape = tuple(shape)
        self.ndim = ndim
        self.ft_shape = (shape[0]//2 + 1, shape[1], shape[2])

        self.plan = G.create_plan(context, self.shape)
        self.plan.inplace = False
        self.plan.layouts = (IMAGINARY, REAL)

        if ndim == 3:
            self.plan.strides_out = (shape[1]*shape[2], shape[2], 1)
        elif ndim == 2:
            self.plan.strides_out = (shape[1], 1)
        elif ndim == 1:
            self.plan.strides_out = (1,)

        if ndim == 3:
            self.plan.strides_in = (shape[1]*shape[2], shape[2], 1)
        elif ndim == 2:
            self.plan.strides_in = (shape[1], 1)
        elif ndim == 1:
            self.plan.strides_in = (1,)

        self.baked = False

    def bake(self, queues):
        if not self.baked:
            self.plan.bake(queues)
            self.baked = True

    def __call__(self, queue, inarray, outarray):
        self.plan.enqueue_transform(queue, inarray.data, outarray.data,
                                    direction_forward=False)
        self.baked = True


