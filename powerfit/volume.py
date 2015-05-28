from __future__ import division, print_function
import numpy as np
from scipy.ndimage import zoom
# zoom gives an annoying userwarning, so turn it off
import warnings
warnings.simplefilter("ignore", UserWarning)
from .libpowerfit import binary_erosion
from .IO import parse_vol, to_vol

class Volume(object):

    @classmethod
    def fromfile(cls, fid):
        array, voxelspacing, origin = parse_vol(fid)
        return cls(array, voxelspacing, origin)

    def __init__(self, array, voxelspacing=1.0, origin=(0, 0, 0)):

        self.array = array
        self._voxelspacing = voxelspacing
        self._origin = origin

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
        to_vol(fid, self)

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


def resize(volume, shape):

    resized_volume = zeros(shape, volume.voxelspacing, volume.origin)
    resized_volume.array[:volume.shape[0], :volume.shape[1], :volume.shape[2]] = volume.array

    return resized_volume

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
    
    extend = []
    for axis in range(volume.array.ndim):
        volume.array = np.swapaxes(volume.array, 0, axis)
        slices = volume.array.shape[0]
        for s in range(slices - 1):
            if volume.array[s + 1].max() > threshold:
                low = max(0, s - margin)
                break
        for s in range(slices):
            if volume.array[slices - (s + 1)].max() > threshold:
                high = min(slices, slices - s + margin)
                break

        try:
            extend.append((low, high))
        except UnboundLocalError:
            raise ValueError("The cutoff value is too high. Reduce the value.")

        volume.array = np.swapaxes(volume.array, axis, 0)

    (zmin, zmax), (ymin, ymax), (xmin, xmax) = extend
    sub_array = volume.array[zmin: zmax, ymin: ymax, xmin: xmax]

    origin = []
    origin.append(volume.origin[0] + volume.voxelspacing*xmin)
    origin.append(volume.origin[1] + volume.voxelspacing*ymin)
    origin.append(volume.origin[2] + volume.voxelspacing*zmin)

    return Volume(sub_array, volume.voxelspacing, origin)


def zone(volume, pdb, radius):

    from powerfit.points import dilate_points

    mask = zeros_like(volume)
    
    dilate_points(pdb.coor, radius, mask)

    zone = zeros_like(volume)
    zone.array = mask.array * volume.array

    return zone
