from __future__ import division, absolute_import, print_function
import sys
import os.path
from struct import pack, unpack
from collections import OrderedDict

import numpy as np


def parse_mrc(fid):
    mrcfile = MRCFile(fid)

    array = mrcfile.density
    voxelspacing = mrcfile.voxelspacing
    origin = mrcfile.origin

    return array, voxelspacing, origin

class MRCFile(object):
    """
    Class for MRC-type files
    """

    def __init__(self, fid, fmt=None):


        if isinstance(fid, file):
            pass
        elif isinstance(fid, str):
            fid = open(fid, 'rb')
        else:
            raise TypeError('Input should either be a file of filename')

        if fmt is None:
            fmt = os.path.splitext(fid.name)[1].strip('.')

        self._fid = fid
        self.fmt = fmt
        self._header = {}

        self._fid.seek(212)
        machst = hex(ord(self._fid.read(1)))
        if (machst == '0x44'):
            endian = '<'
        elif (machst == '0x11'):
            endian = '>'
        else:
            raise IOError("Could not determine the endianness of the file")
        self._endian = endian

        self._fid.seek(0)
        headertype = 'i'*10 + 'f'*6 + 'i'*3 + 'f'*3 + 'i'*3\
               + 'f'*27 + 'c'*8 + 'f'*1 + 'i'*1 + 'c'*(80*10)
        raw_header = unpack(endian + headertype, self._fid.read(1024))
        self._header = OrderedDict()
        self._header['nc'] = raw_header[0]
        self._header['nr'] = raw_header[1]
        self._header['ns'] = raw_header[2]
        self._header['mode'] = raw_header[3]
        self._header['ncstart'] = raw_header[4]
        self._header['nrstart'] = raw_header[5]
        self._header['nsstart'] = raw_header[6]
        self._header['nx'] = raw_header[7]
        self._header['ny'] = raw_header[8]
        self._header['nz'] = raw_header[9]
        self._header['xlength'] = raw_header[10]
        self._header['ylength'] = raw_header[11]
        self._header['zlength'] = raw_header[12]
        self._header['alpha'] = raw_header[13]
        self._header['beta'] = raw_header[14]
        self._header['gamma'] = raw_header[15]
        self._header['mapc'] = raw_header[16]
        self._header['mapr'] = raw_header[17]
        self._header['maps'] = raw_header[18]
        self._header['amin'] = raw_header[19]
        self._header['amax'] = raw_header[20]
        self._header['amean'] = raw_header[21]
        self._header['ispg'] = raw_header[22]
        self._header['nsymbt'] = raw_header[23]
        self._header['lskflg'] = raw_header[24]
        self._header['skwmat'] = raw_header[25:34]
        self._header['skwtrn'] = raw_header[34:37]
        self._header['extra'] = raw_header[37:49]
        self._header['xstart'] = raw_header[49]
        self._header['ystart'] = raw_header[50]
        self._header['zstart'] = raw_header[51]
        self._header['map'] = "".join(raw_header[52:56])
        self._header['machst'] = " ".join(map(hex,map(ord,raw_header[56:60])))
        self._header['rms'] = raw_header[60]
        self._header['nlabel'] = raw_header[61]
        self._header['label'] = "".join(raw_header[62:862]).strip()

    @property
    def fid(self):
        return self._fid

    @property
    def fmt(self):
        return self._fmt

    @fmt.setter
    def fmt(self, fmt):
        if fmt not in ('ccp4', 'mrc', 'map'):
            raise ValueError('Format should either be ccp4 or mrc.')
        self._fmt = fmt

    @property
    def header(self):
        return self._header

    @property
    def voxelspacing(self):
        return self.header['xlength']/self.header['nx']

    @property
    def origin(self):

        order = (self.header['mapc'],
                self.header['mapr'], self.header['maps'])
        if order == (1, 2, 3):
            start = (self.header['ncstart'],
                    self.header['nrstart'], self.header['nsstart'])
        elif order == (1,3,2):
            start = (self.header['ncstart'],
                    self.header['nsstart'], self.header['nrstart'])
        elif order == (2,1,3):
            start = (self.header['nrstart'],
                    self.header['ncstart'], self.header['nsstart'])
        elif order == (2,3,1):
            start = (self.header['nrstart'],
                    self.header['nsstart'], self.header['ncstart'])
        elif order == (3,1,2):
            start = (self.header['nsstart'],
                    self.header['ncstart'], self.header['nrstart'])
        elif order == (3,2,1):
            start = (self.header['nsstart'],
                    self.header['nrstart'], self.header['ncstart'])

        origin = [x * self.voxelspacing for x in start]

        if self.fmt == 'mrc':
            origin = (self.header['xstart'] + origin[0],
                    self.header['ystart'] + origin[1],
		    self.header['zstart'] + origin[2])

        return origin


    @property
    def density(self):

        # determine nc, nr, ns and mode/datatype of density
        mode = self.header['mode']
        if mode == 0:
            datatype = 'i1'
        elif mode == 1:
            datatype = 'i2'
        elif mode == 2:
            datatype = 'f4'
	else:
	    raise IOError("Datatype of density is not recoginized")

        # read the density and reshape it
        nc = self.header['nc']
        nr = self.header['nr']
        ns = self.header['ns']

        self.fid.seek(1024)
        density = np.fromfile(self.fid, dtype=self._endian + datatype).reshape((ns,nr,nc))

        order = (self.header['mapc'], self.header['mapr'], self.header['maps'])
        if order == (1,3,2):
            density = np.swapaxes(density, 0, 1)
        elif order == (2,1,3):
            density = np.swapaxes(density, 1, 2)
        elif order == (2,3,1):
            density = np.swapaxes(density, 2, 0)
            density = np.swapaxes(density, 0, 1)
        elif order == (3,1,2):
            density = np.swapaxes(density, 2, 1)
            density = np.swapaxes(density, 0, 2)
        elif order == (3,2,1):
            density = np.swapaxes(density, 0, 2)

        return density


def to_mrc(fid, volume, labels=[], fmt=None):

    if fmt is None:
        fmt = os.path.splitext(fid)[-1][1:]

    if fmt not in ('ccp4', 'mrc', 'map'):
        raise IOError('Format is not recognized. Use ccp4, mrc, or map.')

    voxelspacing = volume.voxelspacing
    with open(fid, 'wb') as out:

        nz, ny, nx = volume.shape
        out.write(pack('i', nx))
        out.write(pack('i', ny))
        out.write(pack('i', nz))

        dtype = volume.array.dtype.name
        if dtype == 'int8':
            mode = 0
        elif dtype in ('int16', 'int32'):
            mode = 1
        elif dtype in ('float32', 'float64'):
            mode = 2
        else:
            raise TypeError("Data type ({:})is not supported.".format(dtype))
        out.write(pack('i', mode))

        if fmt in ('ccp4', 'map'):
            nxstart, nystart, nzstart = [int(round(x)) for x in volume.start]
        else:
            nxstart, nystart, nzstart = [0, 0, 0]

        out.write(pack('i', nxstart))
        out.write(pack('i', nystart))
        out.write(pack('i', nzstart))

        out.write(pack('i', nx))
        out.write(pack('i', ny))
        out.write(pack('i', nz))

        xl, yl, zl = volume.dimensions
        out.write(pack('f', xl))
        out.write(pack('f', yl))
        out.write(pack('f', zl))

        alpha = beta = gamma = 90.0
        out.write(pack('f', alpha))
        out.write(pack('f', beta))
        out.write(pack('f', gamma))

        mapc, mapr, maps = [1, 2, 3]
        out.write(pack('i', mapc))
        out.write(pack('i', mapr))
        out.write(pack('i', maps))

        out.write(pack('f', volume.array.min()))
        out.write(pack('f', volume.array.max()))
        out.write(pack('f', volume.array.mean()))

        ispg = 1
        out.write(pack('i', ispg))
        nsymbt = 0
        out.write(pack('i', nsymbt))

        lskflg = 0
        out.write(pack('i', lskflg))
        skwmat = [0.0]*9
        for f in skwmat:
            out.write(pack('f', f))
        skwtrn = [0.0]*3
        for f in skwtrn:
            out.write(pack('f', f))

        fut_use = [0.0]*12
        for f in fut_use:
            out.write(pack('f', f))

        if fmt == 'mrc':
            origin = volume.origin
        else:
            origin = [0, 0, 0]

        for f in origin:
            out.write(pack('f', f))

        str_map = ['M', 'A', 'P', ' ']
        for c in str_map:
            out.write(pack('c', c))

        if sys.byteorder == 'little':
            machst = ['\x44', '\x41' ,'\x00', '\x00']
        elif sys.byteorder == 'big':
            machst = ['\x44', '\x41' ,'\x00', '\x00']
        else:
            raise ValueError("Byteorder {:} is not recognized".format(sys.byteorder))

        for c in machst:
            out.write(pack('c', c))

        out.write(pack('f', volume.array.std()))

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

        nlabels = 0
        out.write(pack('i', nlabels))

        labels = [' '] * 800
        for c in labels:
            out.write(pack('c', c))

        # write density
        if mode == 0:
            volume.array.tofile(out)
        if mode == 1:
            volume.array.astype(np.int16).tofile(out)
        if mode == 2:
            volume.array.astype(np.float32).tofile(out)
