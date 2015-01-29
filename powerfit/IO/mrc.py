import sys
from struct import pack
import numpy as np

import sys, struct
import numpy as np
from collections import OrderedDict

def parse_mrc(fid):
    mrcfile = MRCFile(fid)

    array = mrcfile.get_density()
    voxelspacing = mrcfile.get_voxelspacing()
    origin = mrcfile.get_nstart()

    return array, voxelspacing, origin

class MRCFile(object):
    """
    Class for MRC-type files
    """

    def __init__(self, source):
    
        self.source = source
	self.endian = self._get_endianness()
        self.header = self._get_header()

    def _get_endianness(self):
        """Determines the endianness of the mrc-file. 
        This is determined by the 212 to 216th byte of the file.
        It returns '<' for little-endianness and '>' for big-endianness.
        """

        with open(self.source, 'rb') as mapfile:
            byte = mapfile.read(216)[212]
            machst = hex(ord(byte))

            if machst not in ('0x44', '0x11'):
                raise ValueError, 'Machine stamp value is not recognized'

            if (machst == '0x44'):
                endian = '<'
            elif (machst == '0x11'):
                endian = '>'
	    else:
		raise IOError("Could not determine the endianness of the file")

        return endian

    def _get_header(self):
        """Parses a MRC-file and returns the header as an OrderedDict.

        See: ami.scripps.edu/software/mrctools/mrc_specification.php
        of resulting numbers.
        """

        mrc_file = self.source

        # first determine endianness of file
        endian = self._get_endianness()

        # read the header part
        headertype = 'i'*10 + 'f'*6 + 'i'*3 + 'f'*3 + 'i'*3\
               + 'f'*27 + 'c'*8 + 'f'*1 + 'i'*1 + 'c'*(80*10)

        with open(mrc_file,'rb') as mapfile:
            raw_header = struct.unpack(endian + headertype, mapfile.read(1024) )

        header = OrderedDict()
        header['nc'] = raw_header[0]
        header['nr'] = raw_header[1]
        header['ns'] = raw_header[2]
        header['mode'] = raw_header[3]
        header['ncstart'] = raw_header[4]
        header['nrstart'] = raw_header[5]
        header['nsstart'] = raw_header[6]
        header['nx'] = raw_header[7]
        header['ny'] = raw_header[8]
        header['nz'] = raw_header[9]
        header['xlength'] = raw_header[10]
        header['ylength'] = raw_header[11]
        header['zlength'] = raw_header[12]
        header['alpha'] = raw_header[13]
        header['beta'] = raw_header[14]
        header['gamma'] = raw_header[15]
        header['mapc'] = raw_header[16]
        header['mapr'] = raw_header[17]
        header['maps'] = raw_header[18]
        header['amin'] = raw_header[19]
        header['amax'] = raw_header[20]
        header['amean'] = raw_header[21]
        header['ispg'] = raw_header[22]
        header['nsymbt'] = raw_header[23]
        header['lskflg'] = raw_header[24]
        header['skwmat'] = raw_header[25:34]
        header['skwtrn'] = raw_header[34:37]
        header['extra'] = raw_header[37:49]
	header['xstart'] = raw_header[49]
	header['ystart'] = raw_header[50]
	header['zstart'] = raw_header[51]
        header['map'] = "".join(raw_header[52:56])
        header['machst'] = " ".join(map(hex,map(ord,raw_header[56:60])))
        header['rms'] = raw_header[60]
        header['nlabel'] = raw_header[61]
        header['label'] = "".join(raw_header[62:862]).strip()
        return header

    def get_voxelspacing(self):
        return self.header['xlength']/float(self.header['nx'])

    def get_nstart(self):
	return self.header['xstart'], self.header['ystart'], self.header['zstart']

    def get_density(self):
        """
        Parses a CCP4-file and returns the density.
        """

        mrc_file = self.source
        endian = self.endian
        header= self.header

        # determine nc, nr, ns and mode/datatype of density
        mode = header['mode']
        if mode == 0:
            datatype = 'i1'
        elif mode == 1:
            datatype = 'i2'
        elif mode == 2:
            datatype = 'f4'
	else:
	    raise IOError("Datatype of density is not recoginized")

        # read the density and reshape it
        nc = header['nc']
        nr = header['nr']
        ns = header['ns']

        with open(mrc_file,'rb') as mrc:
            # TODO do it properly
            if mode == 0:
                rewindfactor = 4
            if mode == 1:
                rewindfactor = 2
            if mode == 2:
                rewindfactor = 1
            density = np.fromfile(mrc, dtype = endian + datatype)[256*rewindfactor:].reshape((ns,nr,nc))
            density = density.astype(datatype)

        order = (header['mapc'], header['mapr'], header['maps'])
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


def to_mrc(fid, volume, labels=[]):

    voxelspacing = volume.voxelspacing
    with open(fid, 'wb') as out:

        nx, ny, nz = volume.shape
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

        nxstart, nystart, nzstart = [int(round(x)) for x in volume.start]
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

        for f in volume.origin:
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
