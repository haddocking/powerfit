from __future__ import absolute_import, division
from time import time as _time
from sys import stdout as _stdout
import warnings

import numpy as np
from numpy.fft import rfftn, irfftn

from powerfit import volume
from powerfit import libpowerfit
from powerfit.solutions import Solutions
from scipy.ndimage import laplace

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
        self._core_weighted = False

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
        self._rotations = np.asarray(rotations, dtype=np.float64)

    @property
    def queue(self):
        return self._queue
    @queue.setter
    def queue(self, queue):
        self._queue = queue

    @property
    def core_weighted(self):
        return self._core_weighted

    @core_weighted.setter
    def core_weighted(self, core_weighted):
        self._core_weighted = core_weighted

    @property
    def laplace(self):
        return self._laplace

    @laplace.setter
    def laplace(self, value):
        if not isinstance(value, bool):
            raise TypeError("Value should be either True or False.")
        self._laplace = value

    def initialize(self):
        if any(x is None for x in [self.map, self.model, 
            self.rotations, self.resolution]):
            raise ValueError('Not all parameters are set to perform a search')

        d = self._data

        d['map'] = self.map.array.astype(np.float64)
        if self.laplace:
            laplace(d['map'], d['map'], mode='constant')
        d['shape'] = d['map'].shape
        d['voxelspacing'] = self.map.voxelspacing
        d['origin'] = self.map.origin
        d['map_center'] = array_center(d['map'])

        d['rotations'] = np.asarray(self.rotations, dtype=np.float64)
        d['nrot'] = self.rotations.shape[0]

        # coordinates are set to the middle of the array
        grid_coor = (self.model.coor + (-self.model.center + d['voxelspacing']*d['map_center']))/d['voxelspacing']

        # make mask
        mask = np.zeros_like(d['map'])
        radius = 0.5*self.resolution/d['voxelspacing']
        libpowerfit.dilate_points(grid_coor, radius, mask)

        # make density
        modelmap = np.zeros_like(d['map'])
        sigma = resolution2sigma(self.resolution)/d['voxelspacing']
        libpowerfit.blur_points(grid_coor, self.model.atomnumber.astype(np.float64), sigma, modelmap)

        if self.laplace:
            laplace(modelmap, output=modelmap, mode='constant')

        if self.core_weighted:
            core_indices(mask, out=mask)

        d['mask'] = mask
        d['norm_factor'] = d['mask'].sum()

        normalize(modelmap, mask)

        if self.core_weighted:
            modelmap *= d['mask']

        d['modelmap'] = modelmap

    def search(self):
        d = self._data
        self.initialize()

        self._cpu_init()
        best_lcc, rot_ind = self._cpu_search()

        best_lcc = volume.Volume(best_lcc, d['voxelspacing'], d['origin'])
        return Solutions(best_lcc, self.rotations, rot_ind)

    def _cpu_init(self):

        d = self._data
        self._cpu_data = {}
        c = self._cpu_data

        c['map'] = d['map']
            
        c['map_center'] = d['map_center']

        c['im_modelmap'] = d['modelmap']
        c['modelmap'] = np.zeros_like(d['modelmap'])

        c['im_mask'] = d['mask']
        c['mask'] = np.zeros_like(d['mask'])
        c['norm_factor'] = d['norm_factor']

        c['rotations'] = d['rotations']
        c['nrot'] = d['nrot']
        c['vlength'] = int(np.linalg.norm(self.model.coor - self.model.center, axis=1).max()/d['voxelspacing'] +\
                0.5*self.resolution/d['voxelspacing'] + 1)


        c['best_lcc'] = np.zeros_like(d['map'])
        c['rot_ind'] = np.zeros(d['shape'], dtype=np.int32)

    def _cpu_search(self):

        c = self._cpu_data
        
        # initial calculations
        c['ft_map'] = rfftn(c['map'])
        c['ft_map2'] = rfftn(c['map']**2)

        time0 = _time()
        for n in xrange(c['nrot']):
            libpowerfit.rotate_image3d(c['im_modelmap'], np.linalg.inv(c['rotations'][n]), c['map_center'], c['vlength'], c['modelmap'])
            libpowerfit.rotate_image3d(c['im_mask'], np.linalg.inv(c['rotations'][n]), c['map_center'], c['vlength'], c['mask'])

            c['gcc'] = irfftn(rfftn(c['modelmap']).conj() * c['ft_map'])

            if self.core_weighted:
                c['map_ave'] = irfftn(rfftn(c['mask']).conj() * c['ft_map'])
                c['map2_ave'] = irfftn(rfftn(c['mask']**2).conj() * c['ft_map2'])
            else:
                # saves a FFT calculation
                c['ft_mask'] = rfftn(c['mask']).conj()
                c['map_ave'] = irfftn(c['ft_mask'] * c['ft_map'])
                c['map2_ave'] = irfftn(c['ft_mask'] * c['ft_map2'])

            c['lcc'] = c['gcc']/np.sqrt((c['map2_ave']*c['norm_factor'] - c['map_ave']**2).clip(0.001))

            ind = c['lcc'] > c['best_lcc']
            c['best_lcc'][ind] = c['lcc'][ind]
            c['rot_ind'][ind] = n

            if _stdout.isatty():
                self._print_progress(n, c['nrot'], time0)

        return c['best_lcc'], c['rot_ind']

    def _print_progress(self, n, total, time0):
        pdone = n/total
        t = _time() - time0
        if n > 0:
            _stdout.write('\r{:d}/{:d} ({:.2%}, ETA: {:d}s)'\
                    .format(n, total, pdone, 
                            int(t/pdone - t)))
            _stdout.flush()

def core_indices(mask, out=None):

    if out is None:
        out = mask.copy()

    tmp = mask.copy()
    tmp2 = np.zeros_like(mask)
    while tmp.sum() > 0:
        libpowerfit.binary_erosion(tmp, tmp2)
        tmp = tmp2.copy()
        out += tmp2

    return out

def array_center(array):
    """Return center of array in xyz coor"""
    return (np.asarray(array.shape, dtype=np.float64)/2.0)[::-1]

def resolution2sigma(resolution):
    return resolution/(np.sqrt(2.0) * np.pi)

def normalize(modelmap, mask):
    modelmap *= mask
    ind = mask > 0
    modelmap[ind] /= modelmap[ind].std()
    modelmap[ind] -= modelmap[ind].mean()
    return modelmap
