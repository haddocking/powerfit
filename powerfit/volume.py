# Original repsositry by haddock labs,
# licensed under the Apache License, Version 2.0.

# Modified by Luc Elliott, 24/04/2023, with the following modifications: 
#   Updated the code to be compatible with Python 3.7.
#   Made Volume class compatible with the TEMPy Map object.
#   Added different ways to initialise the Volume class.

# For more information about the original code, please see https://github.com/haddocking/powerfit. 

# Your modified code follows...


from __future__ import division, absolute_import
import os.path
from sys import byteorder as _BYTEORDER
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom, gaussian_filter

from ._powerfit import blur_points, dilate_points
from six.moves import range
from six.moves import zip

# Map parsers
from TEMPy.maps.map_parser import MapParser
from TEMPy.maps.em_map import Map
from TEMPy.protein.structure_blurrer import StructureBlurrer
from TEMPy.protein.scoring_functions import ScoringFunctions 
# Cragnolini T, Sahota H, Joseph AP, Sweeney A, Malhotra S, Vasishtan D, Topf M (2021a) TEMPy2: A Python library with improved 3D electron microscopy density-fitting and validation workflows. Acta Crystallogr Sect D Struct Biol 77:41–47. doi:10.1107/S2059798320014928

from copy import copy


class Volume(object):

    @classmethod
    def fromfile(cls, fid, fmt=None):
        try:
            fname = fid.name
        except AttributeError:
            fname = fid
        if fmt is None:
            fmt = os.path.splitext(fname)[-1][1:]
        if fmt in ('ccp4', 'map' , 'mrc'):
            p = MapParser.readMRC(fname)
        elif fmt in ('xplor', 'cns'):
            p = MapParser._readXPLOR(fname)
        else:
            raise ValueError('Extension of file is not supported.')
        return cls(p)

    @classmethod
    def fromdata(cls, grid, voxelspacing, origin, fout = 'volout.mrc'):
        # TODO: add some error handling
        vx = vy = vz = voxelspacing

        return cls(
            Map(
                grid,
                origin,
                [vx, vy, vz],
                fout
            )
        )

    @classmethod
    def fromMap(cls, map: Map):
        if not isinstance(map, Map):
            raise TypeError('Not a Map object')
        return cls(map)


    def __init__(self, vol):
        self.__vol = vol
        self.__vol.update_header()
        self.__resolution = None


        # Add ID generator e.g. <filename> from <structure> fitted into <target>
        self.__metadata__ = {
            'ID': None,
            'Filename': self.filename,
            'Simulated': False,
            'Mask': False,
            # Anything else
        }
    
    # TODO: Need to fix this
    """
    def __repr__(self) -> str:
        return '\n'.join([f'{k:10}: {v:10}' for (k, v) in \
                        self.__metadata__.items()])
    """

    def calc_threshold(self, simulated = False):
        # Need to come up with a better way to do this
        if simulated:
            if not self.__resolution:
                raise ValueError('No resolution specified')

            #Taken from scores_process.py in CCPEM Scores_process.py used for
            # TEMPy Global scores in GUI
            if self.__resolution > 10.0: t = 2.5
            elif self.__resolution > 6.0: t = 2.0
            else: t = 1.5
            self.__threshold =  t*self.__vol.fullMap.std()#0.0
        else:
            self.__threshold = ScoringFunctions().calculate_map_threshold(self.__vol)

    @property
    def vol(self):
        return self.__vol

    @vol.setter
    def vol(self, vol):
        self.__vol = vol

    @property
    def threshold(self):
        return self.__threshold

    @property
    def box(self):
        return self.__vol.box_size()

    @threshold.setter
    def threshold(self, threshold):
        self.__threshold = threshold

    @property
    def resolution(self):
        return self.__resolution

    @resolution.setter
    def resolution(self, resolution:float):
        self.__resolution = resolution
    
    @property
    def grid(self):
        return self.__vol.fullMap
    
    @grid.setter
    def grid(self, grid):
        self.__vol.fullMap = grid

    @property
    def origin(self):
        return self.__vol.origin

    @property
    def voxelspacing(self):
        return self.__vol.apix[0]

    @property
    def voxelsize(self):
        return self.__vol.apix

    @property
    def shape(self):
        return self.__vol.fullMap.shape

    @property
    def dimensions(self):
        return np.asarray([x * self.voxelspacing for x in self.shape][::-1])

    @property
    def start(self):
        return np.asarray([x/self.voxelspacing for x in self.origin])

    @start.setter
    def start(self, start):
        self.__vol.change_origin(np.asarray([x * self.voxelspacing for x in start]))

    def duplicate(self):
        voldupe = self.__vol.copy()
        volume = copy(self)
        volume.vol = voldupe
        return volume

    @property
    def filename(self):
        return str(Path(self.__vol.filename).resolve())
    
    @filename.setter
    def filename(self, fname):
        self.__vol.filename = fname
        self['Filename'] = fname


    def tofile(self, fid=None, fmt=None):
        if fid is None:
            fid = self.filename
        if fmt is None:
            fmt = os.path.splitext(fid)[-1][1:]
        
        self.__vol.update_header()
        
        if fmt in ('ccp4', 'map', 'mrc'):

            self.__vol.write_to_MRC_file(fid)
        elif fmt in ('xplor', 'cns'):
            self.__vol._write_to_xplor_file(fid)
        else:
            raise RuntimeError("Format is not supported.")

    
    def __setitem__(self, key, item):
        if key not in self.__metadata__: raise KeyError
        self.__metadata__[key] = item

    def __getitem__(self, key):
        if key not in self.__metadata__: raise KeyError

        return self.__metadata__[key]
    



class StructureBlurrerbfac(StructureBlurrer):
    def __init__(self, outname:str, with_vc=False):
        self.outname = outname
        super().__init__(with_vc=with_vc)


    def _gaussian_blur_real_space_vc_bfac(
            self,
            struct,
            resolution,
            exp_map,
            SIGMA_COEFF=0.356,
            cutoff=4.0,
    ):

        if not self.use_vc: return None 

        import voxcov as vc

        blur_vc = vc.BlurMap(
            exp_map.apix,
            exp_map.origin,
            [exp_map.x_size(), exp_map.y_size(), exp_map.z_size()],
            cutoff,
        )

        # Constant for the b-factor sigma conversion 
        SIGMA_CONV = 3 / (8 * (np.pi**2))
        
        for a in struct.atomList:

            sigma = SIGMA_CONV * a.temp_fac * resolution * SIGMA_COEFF
            height = 0.4/sigma
      
            blur_vc.add_gaussian(
                    [a.x, a.y, a.z],
                    a.get_mass() * height, # height
                    SIGMA_COEFF * resolution # width
            )
        full_map = blur_vc.to_numpy()
        
        return Map(
                full_map,
                exp_map.origin,
                exp_map.apix,
                self.outname,
        )


# builders
def zeros(shape, voxelspacing, origin):
    return Volume.fromdata(np.zeros(shape), voxelspacing, origin)


def zeros_like(volume):
    return Volume.fromdata(np.zeros_like(volume.grid), volume.voxelspacing, volume.origin)


def resample(volume, factor, order=1):
    # suppress zoom UserWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        resampled_array = zoom(volume.grid, factor, order=order)
    new_voxelspacing = volume.voxelspacing / factor
    return Volume.fromdata(resampled_array, new_voxelspacing, volume.origin)


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
    return Volume.fromdata(sub_array, volume.voxelspacing, origin)


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
    return Volume.fromdata(blurred_array, vol.voxelspacing, vol.origin)


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
        out = Volume.fromdata(grid, voxelspacing, origin)

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




