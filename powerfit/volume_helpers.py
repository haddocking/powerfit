# Original repsositry by haddock labs,
# licensed under the Apache License, Version 2.0.

# Modified by Luc Elliott, 24/04/2023, with the following modifications: 
#   Updated the code to be compatible with Python 3.7.
#   Made Volume class compatible with the TEMPy Map object.
#   Added different ways to initialise the Volume class.

# For more information about the original code, please see https://github.com/haddocking/powerfit. 

from __future__ import division, absolute_import

import warnings
import numpy as np

from scipy.ndimage import zoom, gaussian_filter
from ._powerfit import blur_points, dilate_points
from six.moves import range
from six.moves import zip
from struvolpy import Volume

"""Builders"""
def zeros(shape, voxelspacing, origin):
    return Volume.from_data(np.zeros(shape), voxelspacing, origin)


def zeros_like(volume):
    return Volume.from_data(np.zeros_like(volume.grid), volume.voxelspacing, volume.origin)


def resample(volume, factor, order=1):
    # suppress zoom UserWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        resampled_array = zoom(volume.grid, factor, order=order)
    new_voxelspacing = volume.voxelspacing / factor
    return Volume.from_data(resampled_array, new_voxelspacing, volume.origin)


def trim(volume, cutoff, margin=2):
    if volume.grid.max() <= cutoff:
        raise ValueError('Cutoff value should be lower than density max.')

    extent = []
    for axis in range(volume.grid.ndim):
        tmp = np.swapaxes(volume.grid, 0, axis)
        for n, s in enumerate(tmp):
            if s.max() > cutoff:
                low = max(0, n - margin)
                break
        for n, s in enumerate(tmp[::-1]):
            if s.max() > cutoff:
                high = min(tmp.shape[0], tmp.shape[0] - n + margin)
                break
        extent.append(slice(low, high))
    sub_array = volume.grid[tuple(extent)]
    origin = [coor_origin + volume.voxelspacing * ext.start
            for coor_origin, ext in zip(volume.origin, extent[::-1])]
    return Volume.from_data(sub_array, volume.voxelspacing, origin)


def extend(volume, shape):
    new_volume = zeros(shape, volume.voxelspacing, volume.origin)
    ind = [slice(x) for x in volume.shape]
    new_volume.grid[tuple(ind)] = volume.grid
    return new_volume


def nearest_multiple2357(num):
    nearest = num
    while not is_multiple2357(nearest):
        nearest += 1
    return nearest


def is_multiple2357(num):
    """Returns the nearest larger number that is a multiple of 2, 3, and 5"""

    MULTIPLES = (2, 3, 5, 7)
    for multiple in (MULTIPLES):
        while divmod(num, multiple)[1] == 0:
            num /= multiple
    return num == 1


def res_to_sigma(resolution):
    return resolution / (np.sqrt(2.0) * np.pi)


def sigma_to_res(sigma):
    return sigma * (np.sqrt(2.0) * np.pi)


def lower_resolution(vol, res_high, res_low):
    """Lowers the resolution of the volume"""

    # the sigma of the Gaussian kernel that will lower the density with the
    # specified amount is given Sk = sqrt(Sn^2 - Sc^2)
    # where Sk is the sigma of the kernel, Sn is the new sigma,
    # and Sc is the current sigma
    # See http://mathworld.wolfram.com/Convolution.html
    sigma_high = res_to_sigma(res_high)
    sigma_low = res_to_sigma(res_low)
    sigma_k = np.sqrt(sigma_low**2 - sigma_high**2) / vol.voxelspacing
    blurred_array = gaussian_filter(vol.grid, sigma_k, mode='constant')
    return Volume.from_data(blurred_array, vol.voxelspacing, vol.origin)


def structure_to_shape(
      xyz, resolution, out=None, voxelspacing=None, radii=None, weights=None, shape='vol'
      ):

    if shape not in ('vol', 'mask'):
        raise ValueError("shape should either be 'vol' or 'mask'")

    if out is None and voxelspacing is None:
        voxelspacing = resolution / 4.0
    else:
        voxelspacing = out.voxelspacing

    if shape == 'vol':
        if weights is None:
            weights = np.ones(xyz.shape[1])
        elif weights.size != xyz.shape[1]:
            raise ValueError("weights array is of incorrect size")

    if shape == 'mask':
        if radii is None:
            radii = np.empty(xyz.shape[1], dtype=np.float64)
            radii.fill(5)
        elif radii.size != xyz.shape[1]:
            raise ValueError("weights array is of incorrect size")
        radii /= voxelspacing

    sigma = res_to_sigma(resolution)
    if out is None:
        extend = 4 * sigma
        imin = xyz.min(axis=1) - extend
        imax = xyz.max(axis=1) + extend
        center = (imin + imax) / 2.0
        ni = (np.ceil((imax - imin) / voxelspacing)).astype(np.int32)
        origin = center - (ni * voxelspacing) / 2.0
        grid = np.zeros(ni[::-1])
        xyz_grid = xyz + (ni * voxelspacing / 2.0 - center).reshape(-1, 1)
        out = Volume.from_data(grid, voxelspacing, origin)

    else:
        xyz_grid = xyz - out.origin.reshape(-1, 1)
    xyz_grid /= voxelspacing

    if shape == 'vol':
        blur_points(xyz_grid, weights, sigma / voxelspacing, out.grid, True)
    elif shape == 'mask':
        dilate_points(xyz_grid, radii, out.grid, True)
    return out



def structure_to_shape_like(vol, xyz, resolution=None, weights=None,
        radii=None, shape='vol'):
        # shape like closer to 

    if resolution is None:
        resolution = vol.resolution

    if shape == ' vol':
        if weights is None:
            weights = np.ones(xyz.shape[1])
        elif weights.size != xyz.shape[1]:
            raise ValueError("weights array is of incorrect size")

    if shape == 'mask':
        if radii is None:
            radii = np.empty(xyz.shape[1], dtype=np.float64)
            #radii.fill(0.5 * resolution)
            radii.fill(5)
        elif radii.size != xyz.shape[1]:
            raise ValueError("weights array is of incorrect size")
        radii /= vol.voxelspacing

    sigma = (resolution / (np.sqrt(2.0) * np.pi)) / vol.voxelspacing
    # move the coordinates to the origin of the grid
    xyz_grid = xyz - np.asarray(vol.origin, dtype=np.float64).reshape(3, 1)
    xyz_grid /= vol.voxelspacing

    out = zeros_like(vol)
    if shape == 'vol':
        blur_points(xyz_grid, weights, sigma, out.grid, True)
    elif shape == 'mask':
        dilate_points(xyz_grid, radii, out.grid, True)
    return out




