from .volume import Volume

class EM(Volume):

    def __init__(self, data, voxelspacing=1.0, origin=(0,0,0), resolution=None):
        super(EM, self).__init__(data, voxelspacing, origin)
        self._resolution = resolution

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, resolution):
        self._resolution = resolution



