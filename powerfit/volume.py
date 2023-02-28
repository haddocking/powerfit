from __future__ import division, absolute_import
from struct import unpack as _unpack, pack as _pack
import os.path
from sys import byteorder as _BYTEORDER
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom, gaussian_filter

from ._powerfit import blur_points, dilate_points
from six.moves import range
from six.moves import zip
import io

# For XYZ_fixed
from scipy.ndimage import binary_dilation, gaussian_filter

# Map parsers
from TEMPy.maps.map_parser import MapParser
from TEMPy.maps.em_map import Map
from TEMPy.protein.structure_blurrer import StructureBlurrer
from TEMPy.protein.scoring_functions import ScoringFunctions 

from copy import copy
# Completly change voume class to reflect 
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

    def maskMap(self):
        # TODO: Takes a Map object and returns a Mask of that Map
        # Temporary, look into how TEMPy does this, incorporate a radaii
        maskmap = self.__vol.copy()
        maskmap.update_header()

        zeros = np.zeros(maskmap.fullMap.shape)
        zeros[maskmap.fullMap >= self.__threshold] = 1

        maskmap.fullMap = zeros

        maskmapVolume = Volume.fromMap(maskmap)

        maskmapVolume['Mask'] = True
        return maskmapVolume

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
    # TEMPy 
    # Adaptation of TEMPy structure blurer which takes into account
    # The b-factor of the molecule
    # TODO: reference TEMPy
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


def structure_to_shape_TEMPy(
                vol,
                structure, 
                resolution=None, 
                bfac=False,
                normalise=False,
                            ):

    if resolution is None:
        resolution = vol.resolution
    
    if resolution is None:
        raise AssertionError ("resolution must be specified\
either in the Volume class or as a kwarg")

    sb = StructureBlurrerbfac('', with_vc=True)

    if bfac:
        method = sb._gaussian_blur_real_space_vc_bfac
        
    else:
        method = sb._gaussian_blur_real_space_vc

    simmap = method(
            structure.prot,
            resolution,
            vol.vol,
        )

    if normalise:
        simmap.normalise()
    
    simmap_vol = Volume.fromMap(simmap)
    simmap_vol.resolution = resolution
    simmap_vol.calc_threshold(simulated=True)
    simmap_vol['Simulated'] = True
    return simmap_vol

    # vol = Volume.fromMap(simmap)
    # vol.tofile(sys.argv[3])

def xyz_fixed_transform(
    target: Volume,
    vol: Volume,
    vol_mask: Volume,
    ) -> Volume:
    
    # Needs to be the same shape to work
    assert target.shape == vol.shape

    # scale the simulated map to the real map
    scaled_grid = (vol.grid- np.min(vol.grid) )/( np.max(
                vol.grid)-np.min(vol.grid))*np.max(target.grid)

    # Duplicate the volume object for cut volume
    reduced_vol = target.duplicate()
    reduced_vol.grid -= scaled_grid # Removes a chunk of the volume
    reduced_vol.grid[reduced_vol.grid < 0] = 0

    reduced_vol.calc_threshold()
    reduced_vol_mask = reduced_vol.maskMap()

    # Create softmask logic here
    dilation_mask = binary_dilation(reduced_vol_mask.grid, iterations=1)
    dilated_points_only = dilation_mask - reduced_vol_mask.grid
    gaussian_points = gaussian_filter(dilated_points_only*4, sigma=2)
    gaussian_points = gaussian_points[vol_mask == 1]
    
    # Hopefully this outputs a lovely cut volume with some gaussian blurring
    reduced_vol.grid += gaussian_points[0]

    # Remove negative info

    return reduced_vol


class ChainRestrictions(object):
    def __init__(self, **kwargs):
        self.volin = kwargs.get('volin') # Volume
        self.volout = kwargs.get('volout', None) # Might not need
        self.pdbin = kwargs.get('pdbin', None)
        self.xyzfixed = kwargs.get('xyzfixed', None)
        self.view = kwargs.get('view', False)
        self.verbose = kwargs.get('verbose', False)

    
    def run(self):
        from powerfit import Structure

        """
        Run the program
        :return: Volume object of the new volume with the chain restrictions
        """
        xyz_fixed = Structure.fromfile(self.xyzfixed)
        pdbin = Structure.fromfile(self.pdbin)
        center, xyz_fixed_radius = tuple(xyz_fixed.centre_of_mass), self.find_largest_side(xyz_fixed) /2
        radius = self.find_largest_side(pdbin) + xyz_fixed_radius

        if self.verbose:
            print(f'Center of the fixed xyz file is: {center} and the radius is: {xyz_fixed_radius}')
            print(f'The search radius is: {radius}')


        if not self.check_file_exists(self.volin):
            raise ValueError('Volume file does not exist')
        
        if self.verbose: print("Voxel size is ",self.vol.voxelspacing)
        radius /= self.vol.voxelspacing
        center /= self.vol.voxelspacing
        center -= self.vol.start
        center = tuple(center)
        
        self.vol.grid = self.sphere_radius(radius, self.vol, center)

        if self.view:
            self.check_gaussian_visually(self.vol.grid)

        filtered_volume = self.add_gasussian_filter(self.vol, 5, center, radius)

        if self.view:
            self.check_gaussian_visually(filtered_volume)

        self.vol.grid = filtered_volume

        return self.vol

    def sphere_radius(self, radius: float, volin: Volume, center: tuple):
    
        """
        This function takes a volume and a radius and returns a volume with the same shape as the input volume, but with all values outside the sphere of radius r set to zero.
        :param radius: radius of the sphere
        :param volin: input volume
        :param center: center of the sphere
        :return: volume with all values outside the sphere of radius r set to zero
        """
        
        if not self.check_radius():
            raise ValueError('Radius cannot be negative and/or must be a float')
        
        if not self.check_center():
            raise ValueError('Center must be a tuple and cannot be negative')
        
        if not self.check_within_volume(center, radius, volin):
            raise ValueError('Center cannot be larger than the volume and/or radius cannot be larger than the volume')
        
        sx, sy, sz = volin.grid.shape

        x, y, z = np.ogrid[:sx, :sy, :sz]

        # Calculate distance from center of grid to each point
        distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

        mask = distances <= radius

        points_within_radius = np.zeros_like(volin.grid)
        points_within_radius[mask] = volin.grid[mask]

        return points_within_radius

    def check_radius(self):
        """
        Check if the radius is negative and/or a float
        :return: True if the radius is negative and/or a float
        """
        if self.radius < 0:
            raise ValueError('Radius cannot be negative')
        if not isinstance(self.radius, float):
            raise ValueError('Radius must be a float')
        return True


    def check_center(self):
        """
        Check if the center is negative and/or a tuple
        :return: True if the center is negative and/or a tuple"""
        if not isinstance(self.center, tuple):
            raise ValueError('Center must be a tuple')
        if self.center[0] < 0 or self.center[1] < 0 or self.center[2] < 0:
            raise ValueError('Center cannot be negative')
        return True


    def check_within_volume(self):
        """
        Check if the center is within the volume and if the radius is within the volume
        :return: True if the center is within the volume and if the radius is within the volume

        """
        sx, sy, sz = self.vol.shape
        if self.center[0] > sx or self.center[1] > sy or self.center[2] > sz:
            raise ValueError('Center cannot be larger than the volume')
        if self.radius > sx or self.radius > sy or self.radius > sz:
            raise ValueError('Radius cannot be larger than the volume')
        return True


    def add_gasussian_filter(vol: Volume, sigma: float, center: tuple, radius: float):
        """
        This function takes a volume and a radius and returns a volume with the same shape as the input volume, but with all values outside the sphere of radius r set to zero.
        :param radius: radius of the sphere
        :param volin: input volume
        :param center: center of the sphere
        :return: Volume with gaussian filter

        """
        from scipy.ndimage import gaussian_filter

        # Create a 3D Boolean mask that selects points outside the sphere
        sx, sy, sz = vol.grid.shape
        x, y, z = np.ogrid[:sx, :sy, :sz]
        distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

        # create a zeros array of the same shape as the volume
        # zeros_array[cx][cy][cz]= np.max(vol.grid) 

        filtered_volume_data = gaussian_filter(vol.grid, sigma=sigma, mode='nearest')

        # Select the points outside the sphere by using the inverse of the boolean mask
        outside_sphere = distances > radius
        filtered_volume_data[~outside_sphere] = vol.grid[~outside_sphere]

        return filtered_volume_data


    # function to read in a using Structure and and find the longest difference between corners of box
    def find_largest_side(self, pdb):
        """
        This function takes a pdb file and returns the longest side of the box
        :param pdb: Structure
        :return: longest side of the box
        """
        coordinates = pdb.coor 
        x = coordinates[:,0]
        y = coordinates[:,1]
        z = coordinates[:,2]
        x_max = max(x)
        x_min = min(x)
        y_max = max(y)
        y_min = min(y)
        z_max = max(z)
        z_min = min(z)
        x_side = x_max - x_min
        y_side = y_max - y_min
        z_side = z_max - z_min
        
        x_y_hypo = np.sqrt(x_side**2 + y_side**2)
        if self.verbose: print("X side is: ", x_side, "Y side is: ", y_side, "Z side is: ", z_side)
        cuboidline = np.sqrt(x_y_hypo**2 + z_side**2)

        return cuboidline