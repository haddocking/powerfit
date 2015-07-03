from __future__ import print_function, division, absolute_import
from .volume import Volume
from .IO import parse_vol
from scipy.ndimage import gaussian_filter, zoom
from .helpers import resolution2sigma
from math import sqrt

class EM(Volume):

    @classmethod
    def fromfile(cls, fid, resolution):
        array, voxelspacing, origin = parse_vol(fid)
        return cls(array, resolution, voxelspacing, origin)


    def __init__(self, data, resolution, voxelspacing, origin):
        super(EM, self).__init__(data, voxelspacing, origin)
        self._resolution = resolution


    @property
    def resolution(self):
        return self._resolution


    @resolution.setter
    def resolution(self, resolution):
        self._resolution = resolution


    def duplicate(self):
        return EM(self.array.copy(), self.resolution, self.voxelspacing, self.origin)


    def lower_resolution(self, value):
        return lower_resolution(self, value)


    def resample(self, rate=2):
        return resample(self, rate)


def resample(em, rate=2):

    new_voxelspacing = em.resolution / (2 * rate)
    factor = em.voxelspacing / new_voxelspacing
    array = zoom(em.array, factor, order=1)

    return EM(array, em.resolution, new_voxelspacing, em.origin)


def lower_resolution(em, value):
    """Lowers the resolution of the EM-data

    Parameters
    ----------
    em : EM object
        The density and resolution of this object is lowered
    value : float
        The value that the resolution of the map is lowered in angstrom

    Returns
    -------
    lower_resolution : EM object
        Returned EM object with lowered resolution
    """

    # the sigma of the Gaussian kernel that will lower the density with the 
    # specified amount is given Sk = sqrt(Sn^2 - Sc^2)
    # where Sk is the sigma of the kernel, Sn is the new sigma, 
    # and Sc is the current sigma
    # See http://mathworld.wolfram.com/Convolution.html
    res_c = em.resolution
    sigma_c = resolution2sigma(res_c)

    res_n = em.resolution + value
    sigma_n = resolution2sigma(res_n)

    sigma_k = sqrt(sigma_n**2 - sigma_c**2)/em.voxelspacing
    
    blurred_array = gaussian_filter(em.array, sigma_k, mode='constant')

    return EM(blurred_array, res_n, em.voxelspacing, em.origin)
