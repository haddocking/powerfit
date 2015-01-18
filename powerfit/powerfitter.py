from __future__ import absolute_import, division
import numpy as np
from numpy.fft import rfftn, irfftn

from powerfit import volume

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
except ImportError:
    pass



class PowerFitter(object):

    def __init__(self):
        # required parameters
        self._map = None
        self._model = None
        self._rotations = None
        self._resolution = None
        
        # optional
        self._queue = None
        self._laplace = False
        self._coreweighted = False

        # data container
        self._data = {}

    @property
    def map(self):
        return self._map
    @map.setter
    def map(self, em):
        self._map = em

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, model):
        self._model = model

    @property
    def resolution(self):
        return self._resolution
    @resolution.setter
    def resolution(self, resolution):
        self._resolution = resolution

    @property
    def rotations(self):
        return self._rotations
    @rotations.setter
    def rotations(self, rotations):
        self._rotations = rotations

    @property
    def queue(self):
        return self._queue
    @queue.setter
    def queue(self, queue):
        self._queue = queue

    def initialize(self):
        if any(x is None for x in [self.map, self.model, 
            self.rotations, self.resolution]):
            raise ValueError('Not all parameters are set to perform a search')

        d = self._data

        d['map'] = self.map.data
        d['rotations'] = self.rotations
        d['nrot'] = self.rotations.shape[0]

        d['modelmaskvol'] = volume.zeros_like(self.map)
        radius = self.resolution/2.0
        dilate_points(self.model.coor - self.map.origin, radius, d['modelmask'])

        d['modelmapvol'] = volume.zeros_like(self.map)
        blur_points(self.model.coor - self.map.origin, 
                sigma, self.model.atomnumber, d['modelmap'])

        d['modelmap'] = d['modelmapvol'].data
        d['modelmask'] = d['modelmaskvol'].data

        d['modelmap'] *= d['modelmask']

        # normalize map
        ind = d['modelmap'] > 0
        d['modelmap'][ind] /= d['modelmap'][ind].std()
        d['modelmap'][ind] -= d['modelmap'][ind].mean()

        if self.queue is None:
            self._cpu_init()
        else:
            self._gpu_init()

    def search(self):
        self.initialize()

        if self.queue is None:
            self._cpu_search()
        else:
            self._gpu_search()
        pass

    def _cpu_init(self):

        self._cpu_data = {}
        c = self._cpu_data

        c['ft_map'] = np.zeros(self.map.data.shape, dtype=np.complex128)
        c['ft_map2'] = np.zeros_like(c['ft_map'])

        c['ft_map'][:] = rfftn(d['map'])
        c['ft_map2'][:] = rfftn(d['map']**2)
        pass

    def _gpu_init(self):

        self._gpu_data = {}
        g = self._gpu_data
        q = self.queue

        # bring data to gpu
        g['map'] = cl_array.to_device(q, float32array(d['map']))
        g['im_modelmap'] = cl.image_from_array(q.context, float32array(d['modelmap']))
        g['im_modelmask'] = cl.image_from_array(q.context, float32array(d['modelmask']))
        g['rotmat'] = cl_array.to_device(q, float32array(d['rotations']))
        g['sampler'] = cl.Sampler(q.context, False, cl.addressing_mode.CLAMP,
                cl.filter_mode.LINEAR)

        # determine the number of rotations that we can determine in one go
        # for computational efficiency
        g['n'] = d['map'].size//256**3

        # work arrays
        # real arrays
        g['shape'] = list(g['map'].shape)
        g['modelmap'] = cl_array.zeros(q, [n] + g['shape'], dtype=np.float32)
        g['modelmask'] = cl_array.zeros_like(g['modelmap'])
        g['modelmask2'] = cl_array.zeros_like(g['modelmap'])
        g['mapave'] = cl_array.zeros_like(g['modelmap'])
        g['map2ave'] = cl_array.zeros_like(g['modelmap'])
        g['gcc'] = cl_array.zeros_like(g['modelmap'])

        # complex arrays
        g['ft_shape'] = [g['shape'][0]//2 + 1, g['shape'][1], g['shape'][2]]
        # only need one of these
        g['ft_map'] = cl_array.zeros(q, g['ft_shape'], dtype=np.complex64)
        g['ft_map2'] = cl_array.zeros(q, g['ft_shape'], dtype=np.complex64)

        # need multiple of these
        g['ft_modelmap'] = cl_array.zeros(q, [n] + g['ft_shape'], dtype=np.complex64)
        g['ft_modelmask'] = cl_array.zeros_like(g['ft_modelmap'])
        g['ft_modelmask2'] = cl_array.zeros_like(g['ft_modelmap'])
        g['ft_mapave'] = cl_array.zeros_like(g['ft_modelmap'])
        g['ft_map2ave'] = cl_array.zeros_like(g['ft_modelmap'])

        # set kernels
        g['kernels'] = Kernels(q.context)

    def _gpu_search(self):

        q = self.queue
        g = self._gpu_data
        k = g['kernels']

        k.rfftn(q, g['map'], g['ft_map'])
        k.r_multiply(g['map'], g['map'], g['map2ave'])
        k.rfftn(q, g['map2ave'], g['ft_map2'])

        for n in xrange(d['nrot']//g['n'] + 1):

            k.rotate_map_and_mask(q, g['sampler'], 
                    g['im_modelmap'], g['im_mask'], 
                    g['rotmat'],
                    g['modelmap'], g['modelmask'], n)

            k.rfftn(q, g['modelmap'], g['ft_modelmap'])
            k.c_conj_multiply(g['ft_modelmap'], g['ft_map'], g['ft_gcc'])
            k.irfftn(q, g['ft_gcc'], g['gcc'])

            k.rfftn(q, g['modelmask'], g['ft_model_mask'])
            k.c_conj_multiply(g['ft_modelmask'], g['ft_map'])
            k.irfftn(q, g['ft_mapave'], g['ft_mapave'])

            k.c_conj_multiply(g['ft_modelmask'], g['ft_map2'], g['ft_map2ave'])
            k.irfftn(q, g['ft_map2ave'], g['map2ave'])

            k.lcc(g['gcc'], g['mapave'], g['map2ave'], d['varlimit'])

