from __future__ import division
import numpy as np

class Volume(object):

    @classmethod
    def fromfile(cls, fname, fileformat=None):
        pass

    def __init__(self, data, voxelspacing=1.0, origin=(0, 0, 0)):

        self._data = data
        self._voxelspacing = voxelspacing
        self._origin = origin

    @property
    def data(self):
        return self._data

    @property
    def voxelspacing(self):
        return self._voxelspacing

    @voxelspacing.setter
    def voxelspacing(self, voxelspacing):
        self._voxelspacing = voxelspacing

    @property
    def origin(self):
        return self._origin
    @origin.setter
    def origin(self, origin):
        self._origin = origin

    @property
    def shape(self):
        return self.data.shape[::-1]

    @property
    def dimensions(self):
        return [x*self.voxelspacing for x in self.shape]

    @property
    def start(self):
        return [x/self.voxelspacing for x in self.origin]
    @start.setter
    def start(self, start):
        self._origin = [x*self.voxelspacing for x in start]

    def duplicate(self):
        return Volume(self.data.copy(), voxelspacing=self.voxelspacing,
                      origin=self.origin)

# helpers
def zeros_like(volume):
    return Volume(np.zeros_like(volume.data), volume.voxelspacing, volume.origin)
