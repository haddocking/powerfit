
from io import BufferedReader
from struct import unpack as _unpack, pack as _pack
import os.path
from sys import byteorder as _BYTEORDER
import warnings

import numpy as np
from scipy.ndimage import zoom, gaussian_filter

from ._powerfit import blur_points, dilate_points

class Volume(object):

    @classmethod
    def fromfile(cls, fid, fmt=None):
        array, voxelspacing, origin = parse_volume(fid, fmt)
        return cls(array, voxelspacing, origin)

    def __init__(self, array, voxelspacing=1.0, origin=(0, 0, 0)):

        self.array = array
        self.voxelspacing = voxelspacing
        self.origin = origin

    @property
    def shape(self):
        return self.array.shape

    @property
    def dimensions(self):
        return np.asarray([x * self.voxelspacing for x in self.array.shape][::-1])

    @property
    def start(self):
        return np.asarray([x/self.voxelspacing for x in self.origin])

    @start.setter
    def start(self, start):
        self._origin = np.asarray([x * self.voxelspacing for x in start])

    def duplicate(self):
        return Volume(self.array.copy(), voxelspacing=self.voxelspacing,
                      origin=self.origin)

    def tofile(self, fid, fmt=None):
        if fmt is None:
            fmt = os.path.splitext(fid)[-1][1:]
        if fmt in ('ccp4', 'map', 'mrc'):
            to_mrc(fid, self)
        elif fmt in ('xplor', 'cns'):
            to_xplor(fid, self)
        else:
            raise RuntimeError("Format is not supported.")


# builders
def zeros(shape, voxelspacing, origin):
    return Volume(np.zeros(shape), voxelspacing, origin)


def zeros_like(volume):
    return Volume(np.zeros_like(volume.array), volume.voxelspacing, volume.origin)


def resample(volume, factor, order=1):
    # suppress zoom UserWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        resampled_array = zoom(volume.array, factor, order=order)
    new_voxelspacing = volume.voxelspacing / factor
    return Volume(resampled_array, new_voxelspacing, volume.origin)


def trim(volume, cutoff, margin=2):
    if volume.array.max() <= cutoff:
        raise ValueError('Cutoff value should be lower than density max.')

    extent = []
    for axis in range(volume.array.ndim):
        tmp = np.swapaxes(volume.array, 0, axis)
        for n, s in enumerate(tmp):
            if s.max() > cutoff:
                low = max(0, n - margin)
                break
        for n, s in enumerate(tmp[::-1]):
            if s.max() > cutoff:
                high = min(tmp.shape[0], tmp.shape[0] - n + margin)
                break
        extent.append(slice(low, high))
    sub_array = volume.array[tuple(extent)]
    origin = [coor_origin + volume.voxelspacing * ext.start
            for coor_origin, ext in zip(volume.origin, extent[::-1])]
    return Volume(sub_array, volume.voxelspacing, origin)


def extend(volume, shape):
    new_volume = zeros(shape, volume.voxelspacing, volume.origin)
    ind = [slice(x) for x in volume.shape]
    new_volume.array[tuple(ind)] = volume.array
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
    blurred_array = gaussian_filter(vol.array, sigma_k, mode='constant')
    return Volume(blurred_array, vol.voxelspacing, vol.origin)


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
        out = Volume(grid, voxelspacing, origin)
    else:
        xyz_grid = xyz - out.origin.reshape(-1, 1)
    xyz_grid /= voxelspacing

    if shape == 'vol':
        blur_points(xyz_grid, weights, sigma / voxelspacing, out.array, True)
    elif shape == 'mask':
        dilate_points(xyz_grid, radii, out.array, True)
    return out


def structure_to_shape_like(vol, xyz, resolution=None, weights=None,
        radii=None, shape='vol'):

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
        blur_points(xyz_grid, weights, sigma, out.array, True)
    elif shape == 'mask':
        dilate_points(xyz_grid, radii, out.array, True)
    return out


# Volume parsers
def parse_volume(fid, fmt=None):
    try:
        fname = fid.name
    except AttributeError:
        fname = fid

    if fmt is None:
        fmt = os.path.splitext(fname)[-1][1:]
    if fmt in ('ccp4', 'map'):
        p = CCP4Parser(fid)
    elif fmt == 'mrc':
        p = MRCParser(fname)
    elif fmt in ('xplor', 'cns'):
        p = XPLORParser(fname)
    else:
        raise ValueError('Extension of file is not supported.')
    return p.density, p.voxelspacing, p.origin


class CCP4Parser(object):

    HEADER_SIZE = 1024
    HEADER_TYPE = ('i' * 10 + 'f' * 6 + 'i' * 3 + 'f' * 3 + 'i' * 3 +
                   'f' * 27 + 'c' * 8 + 'f' * 1 + 'i' * 1 + 'c' * 800)
    HEADER_FIELDS = (
          'nc nr ns mode ncstart nrstart nsstart nx ny nz xlength ylength '
          'zlength alpha beta gamma mapc mapr maps amin amax amean ispg '
          'nsymbt lskflg skwmat skwtrn extra xstart ystart zstart map '
          'machst rms nlabel label'
          ).split()
    HEADER_CHUNKS = [1] * 25 + [9, 3, 12] + [1] * 3 + [4, 4, 1, 1, 800]

    def __init__(self, fid):

        if isinstance(fid, str):
            fhandle = open(fid, 'rb')
        elif isinstance(fid, BufferedReader):
            fhandle = fid
        else:
            raise ValueError("Input should either be a file or filename.")

        self.fhandle = fhandle
        self.fname = fhandle.name

        # first determine the endiannes of the file
        self._get_endiannes()
        # get the header
        self._get_header()
        # Symmetry and non-rectangular boxes are not supported.
        is_orthogonal = True
        for angle_name in ['alpha', 'beta', 'gamma']:
            angle = self.header[angle_name]
            if abs(angle - 90) > 1e-3:
                is_orthogonal = False
                break
        if not is_orthogonal:
            msg = "Only densities in rectangular boxes are supported."
            raise RuntimeError(msg)

        # check the order of axis in the file
        self._get_order()
        # determine the voxelspacing and origin
        spacings = []
        for axis_name in 'xyz':
            length = self.header[axis_name + 'length']
            nvoxels = self.header['n' + axis_name]
            spacing = length / float(nvoxels)
            spacings.append(spacing)

        equal_spacing = True
        average = sum(spacings) / float(len(spacings))
        for spacing in spacings:
            if abs(spacing - average) > 1e-4:
                equal_spacing = False
        if not equal_spacing:
            msg = "Voxel spacing is not equal in all directions."
            raise RuntimeError(msg)

        self.voxelspacing = spacings[0]
        self.origin = self._get_origin()
        # generate the density
        shape_fields = 'nz ny nx'.split()
        self.shape = [self.header[field] for field in shape_fields]
        self._get_density()

    def _get_endiannes(self):
        self.fhandle.seek(212)
        m_stamp = hex(ord(self.fhandle.read(1)))
        if m_stamp == '0x44':
            endian = '<'
        elif m_stamp == '0x11':
            endian = '>'
        else:
            raise RuntimeError('Endiannes is not properly set in file. Check the file format.')
        self._endian = endian
        self.fhandle.seek(0)

    def _get_header(self):
        header = _unpack(self._endian + self.HEADER_TYPE,
                         self.fhandle.read(self.HEADER_SIZE))
        self.header = {}
        index = 0
        for field, nchunks in zip(self.HEADER_FIELDS, self.HEADER_CHUNKS):
            end = index + nchunks
            if nchunks > 1:
                self.header[field] = header[index: end]
            else:
                self.header[field] = header[index]
            index = end
        self.header['label'] = ''.join(
            [c.decode('ascii') for c in self.header['label']]
        )
        self.header['map'] = ''.join(
            [c.decode('ascii') for c in self.header['map']]
        )
        self.header['machst'] = ''.join(
            [c.decode('ascii') for c in self.header['machst']]
        )

    def _get_origin(self):
        start_fields = 'nsstart nrstart ncstart'.split()
        start = [self.header[field] for field in start_fields]
        # Take care of axis order
        start = [start[x - 1] for x in self.order]
        return np.asarray([x * self.voxelspacing for x in start])

    def _get_density(self):

        # Determine the dtype of the file based on the mode
        mode = self.header['mode']
        if mode == 0:
            dtype = 'i1'
        elif mode == 1:
            dtype = 'i2'
        elif mode == 2:
            dtype = 'f4'

        density = np.fromfile(self.fhandle, dtype=self._endian + dtype).reshape(self.shape)
        if self.order == (1, 3, 2):
            self.density = np.swapaxes(self.density, 0, 1)
        elif self.order == (2, 1, 3):
            self.density = np.swapaxes(self.density, 1, 2)
        elif self.order == (2, 3, 1):
            self.density = np.swapaxes(self.density, 2, 0)
            self.density = np.swapaxes(self.density, 0, 1)
        elif self.order == (3, 1, 2):
            self.density = np.swapaxes(self.density, 2, 1)
            self.density = np.swapaxes(self.density, 0, 2)
        elif self.order == (3, 2, 1):
            self.density = np.swapaxes(self.density, 0, 2)

        # Upgrade precision to double if float, and to int32 if int16
        if mode == 1:
            density = density.astype(np.int32)
        elif mode == 2:
            density = density.astype(np.float64)
        self.density =density

    def _get_order(self):
        self.order = tuple(self.header[axis] for axis in ('mapc', 'mapr',
            'maps'))


class MRCParser(CCP4Parser):

    def _get_origin(self):
        origin_fields = 'xstart ystart zstart'.split()
        origin = [self.header[field] for field in origin_fields]
        return origin


def to_mrc(fid, volume, labels=[], fmt=None):

    if fmt is None:
        fmt = os.path.splitext(fid)[-1][1:]

    if fmt not in ('ccp4', 'mrc', 'map'):
        raise ValueError('Format is not recognized. Use ccp4, mrc, or map.')

    voxelspacing = volume.voxelspacing
    nz, ny, nx = volume.shape
    dtype = volume.array.dtype.name
    if dtype == 'int8':
        mode = 0
    elif dtype in ('int16', 'int32'):
        mode = 1
    elif dtype in ('float32', 'float64'):
        mode = 2
    else:
        raise TypeError("Data type ({:})is not supported.".format(dtype))
    if fmt in ('ccp4', 'map'):
        nxstart, nystart, nzstart = [int(round(x)) for x in volume.start]
    else:
        nxstart, nystart, nzstart = [0, 0, 0]
    xl, yl, zl = volume.dimensions
    alpha = beta = gamma = 90.0
    mapc, mapr, maps = [1, 2, 3]
    ispg = 1
    nsymbt = 0
    lskflg = 0
    skwmat = [0.0]*9
    skwtrn = [0.0]*3
    fut_use = [0.0]*12
    if fmt == 'mrc':
        origin = volume.origin
    else:
        origin = [0, 0, 0]
    str_map = b'MAP '
    if _BYTEORDER == 'little':
        machst = b'\x44\x41\x00\x00'
    elif _BYTEORDER == 'big':
        machst = b'\x44\x41\x00\x00'
    else:
        raise ValueError("Byteorder {:} is not recognized".format(_BYTEORDER))
    labels = b''.join([b' '] * 800)
    nlabels = 0
    min_density = volume.array.min()
    max_density = volume.array.max()
    mean_density = volume.array.mean()
    std_density = volume.array.std()

    with open(fid, 'wb') as out:
        out.write(_pack('i', nx))
        out.write(_pack('i', ny))
        out.write(_pack('i', nz))
        out.write(_pack('i', mode))
        out.write(_pack('i', nxstart))
        out.write(_pack('i', nystart))
        out.write(_pack('i', nzstart))
        out.write(_pack('i', nx))
        out.write(_pack('i', ny))
        out.write(_pack('i', nz))
        out.write(_pack('f', xl))
        out.write(_pack('f', yl))
        out.write(_pack('f', zl))
        out.write(_pack('f', alpha))
        out.write(_pack('f', beta))
        out.write(_pack('f', gamma))
        out.write(_pack('i', mapc))
        out.write(_pack('i', mapr))
        out.write(_pack('i', maps))
        out.write(_pack('f', min_density))
        out.write(_pack('f', max_density))
        out.write(_pack('f', mean_density))
        out.write(_pack('i', ispg))
        out.write(_pack('i', nsymbt))
        out.write(_pack('i', lskflg))
        for f in skwmat:
            out.write(_pack('f', f))
        for f in skwtrn:
            out.write(_pack('f', f))
        for f in fut_use:
            out.write(_pack('f', f))
        for f in origin:
            out.write(_pack('f', f))
        out.write(str_map)
        out.write(machst)
        out.write(_pack('f', std_density))
        # max 10 labels
        # nlabels = min(len(labels), 10)
        # TODO labels not handled correctly
        #for label in labels:
        #     list_label = [c for c in label]
        #     llabel = len(list_label)
        #     if llabel < 80:
        #
        #     # max 80 characters
        #     label = min(len(label), 80)
        out.write(_pack('i', nlabels))
        out.write(labels)
        # write density
        modes = [np.int8, np.int16, np.float32]
        volume.array.astype(modes[mode]).tofile(out)


class XPLORParser(object):
    """
    Class for reading XPLOR volume files created by NIH-XPLOR or CNS.
    """

    def __init__(self, fid):

        if isinstance(fid, BufferedReader):
            fname = fid.name
        elif isinstance(fid, str):
            fname = fid
            fid = open(fid)
        else:
            raise TypeError('Input should either be a file or filename')

        self.source = fname
        self._get_header()

    def _get_header(self):

        header = {}
        with open(self.source) as volume:
            # first line is blank
            volume.readline()

            line = volume.readline()
            nlabels = int(line.split()[0])

            label = [volume.readline() for n in range(nlabels)]
            header['label'] = label

            line = volume.readline()
            header['nx']      = int(line[0:8])
            header['nxstart'] = int(line[8:16])
            header['nxend']   = int(line[16:24])
            header['ny']      = int(line[24:32])
            header['nystart'] = int(line[32:40])
            header['nyend']   = int(line[40:48])
            header['nz']      = int(line[48:56])
            header['nzstart'] = int(line[56:64])
            header['nzend']   = int(line[64:72])

            line = volume.readline()
            header['xlength'] = float(line[0:12])
            header['ylength'] = float(line[12:24])
            header['zlength'] = float(line[24:36])
            header['alpha'] = float(line[36:48])
            header['beta'] = float(line[48:60])
            header['gamma'] = float(line[60:72])

            header['order'] = volume.readline()[0:3]

            self.header = header

    @property
    def voxelspacing(self):
        return self.header['xlength']/float(self.header['nx'])

    @property
    def origin(self):
        return [self.voxelspacing * x for x in
                [self.header['nxstart'], self.header['nystart'], self.header['nzstart']]]

    @property
    def density(self):
        with open(self.source) as volumefile:
            for n in range(2 + len(self.header['label']) + 3):
                volumefile.readline()
            nx = self.header['nx']
            ny = self.header['ny']
            nz = self.header['nz']

            array = np.zeros((nz, ny, nx), dtype=np.float64)

            xextend = self.header['nxend'] - self.header['nxstart'] + 1
            yextend = self.header['nyend'] - self.header['nystart'] + 1
            zextend = self.header['nzend'] - self.header['nzstart'] + 1

            nslicelines = int(np.ceil(xextend*yextend/6.0))
            for i in range(zextend):
                values = []
                nslice = int(volumefile.readline()[0:8])
                for m in range(nslicelines):
                    line = volumefile.readline()
                    for n in range(len(line)//12):
                        value = float(line[n*12: (n+1)*12])
                        values.append(value)
                array[i, :yextend, :xextend] = np.float64(values).reshape(yextend, xextend)

        return array


def to_xplor(outfile, volume, label=[]):

    nz, ny, nx = volume.shape
    voxelspacing = volume.voxelspacing
    xstart, ystart, zstart = [int(round(x)) for x in volume.start]
    xlength, ylength, zlength = volume.dimensions
    alpha = beta = gamma = 90.0

    nlabel = len(label)
    with open(outfile,'w') as out:
        out.write('\n')
        out.write('{:>8d} !NTITLE\n'.format(nlabel+1))
        # CNS requires at least one REMARK line
        out.write('REMARK\n')
        for n in range(nlabel):
            out.write(''.join(['REMARK ', label[n], '\n']))

        out.write(('{:>8d}'*9 + '\n').format(nx, xstart, xstart + nx - 1,
                                             ny, ystart, ystart + ny - 1,
                                             nz, zstart, zstart + nz - 1))
        out.write( ('{:12.5E}'*6 + '\n').format(xlength, ylength, zlength,
                                                alpha, beta, gamma))
        out.write('ZYX\n')
        #FIXME very inefficient way of writing out the volume ...
        for z in range(nz):
            out.write('{:>8d}\n'.format(z))
            n = 0
            for y in range(ny):
                for x in range(nx):
                    out.write('%12.5E'%volume.array[z,y,x])
                    n += 1
                    if (n % 6) == 0:
                        out.write('\n')
            if (nx*ny)%6 > 0:
                out.write('\n')
        out.write('{:>8d}\n'.format(-9999))
        out.write('{:12.4E} {:12.4E} '.format(volume.array.mean(), volume.array.std()))
