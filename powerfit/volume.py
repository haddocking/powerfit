from __future__ import division, print_function
import numpy as np
from scipy.ndimage import zoom
from .libpowerfit import binary_erosion
from .IO.mrc import to_mrc, parse_mrc

class Volume(object):

    @classmethod
    def fromfile(cls, fid):
        array, voxelspacing, origin = parse_mrc(fid)
        return cls(array, voxelspacing, origin)

    def __init__(self, array, voxelspacing=1.0, origin=(0, 0, 0)):

        self.array = array
        self._voxelspacing = voxelspacing
        self._origin = origin

    #@property
    def array(self):
        return self.array

    @property
    def voxelspacing(self):
        return self._voxelspacing
    @voxelspacing.setter
    def voxelspacing(self, voxelspacing):
        self._voxelspacing = voxelspacing

    @property
    def origin(self):
        return np.asarray(self._origin, dtype=np.float64)
    @origin.setter
    def origin(self, origin):
        self._origin = origin

    @property
    def shape(self):
        return self.array.shape

    @property
    def dimensions(self):
        return [x*self.voxelspacing for x in self.shape][::-1]

    @property
    def start(self):
        return [x/self.voxelspacing for x in self.origin]
    @start.setter
    def start(self, start):
        self._origin = [x*self.voxelspacing for x in start]

    def duplicate(self):
        return Volume(self.array.copy(), voxelspacing=self.voxelspacing,
                      origin=self.origin)
    def tofile(self, fid):
        to_mrc(fid, self)

# builders
def zeros(shape, voxelspacing, origin):
    return Volume(np.zeros(shape, dtype=np.float64), voxelspacing, origin)

def zeros_like(volume):
    return Volume(np.zeros_like(volume.array), volume.voxelspacing, volume.origin)

# functions
def erode(volume, iterations, out=None):

    if out is None:
        out = zeros_like(volume)

    tmp = volume.array.copy()
    for i in range(iterations):
        binary_erosion(tmp, out.array)
        tmp[:] = out.array[:]

    return out

def radix235(ninit):
    while True:
        n = ninit
        divided = True
        while divided:
            divided = False
            for radix in (2, 3, 5):
                quot, rem = divmod(n, radix)
                if not rem:
                    n = quot
                    divided = True
        if n != 1:
            ninit += 1
        else:
            return ninit

def resize_radix235(volume):
    
    radix235_shape = [radix235(x) for x in volume.shape]
    array = np.zeros(radix235_shape, dtype=np.float64)
    array[:volume.shape[0], :volume.shape[1], :volume.shape[2]] = volume.array

    return Volume(array, volume.voxelspacing, volume.origin)

def resample(volume, factor, order=1):
    
    resampled_array = zoom(volume.array, factor, order=order)
    resampled_voxelspacing = volume.voxelspacing / factor

    return Volume(resampled_array, resampled_voxelspacing, volume.origin)

def trim(volume, threshold, margin=4):
    
    array = volume.array.astype(np.float64)
    extend = []
    tmp1 = np.copy(volume.array)
    for axis in range(volume.array.ndim):
        array = np.swapaxes(array, 0, axis)
        slices = array.shape[0]
        for s in range(slices):
            if array[s + 1].max() > threshold:
                low = max(0, s - margin)
                break
        for s in range(slices):
            if array[slices - (s + 1)].max() > threshold:
                high = min(slices, slices - s + margin) + 1
                break

        extend.append((low, high))
        array = np.swapaxes(array, axis, 0)

    print(extend)
    (zmin, zmax), (ymin, ymax), (xmin, xmax) = extend
    sub_array = np.ascontiguousarray(array[xmin:xmax, ymin:ymax, zmin:zmax].astype(np.float64))

    origin = []
    origin.append(volume.origin[0] + volume.voxelspacing*xmin)
    origin.append(volume.origin[1] + volume.voxelspacing*ymin)
    origin.append(volume.origin[2] + volume.voxelspacing*zmin)

    return Volume(sub_array, volume.voxelspacing, origin)
