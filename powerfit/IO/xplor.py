import numpy as np
from math import ceil

def parse_xplor(fid):

    xplorfile = XPLORFile(fid)

    array = xplorfile.density
    voxelspacing = xplorfile.voxelspacing
    origin = xplorfile.origin

    return array, voxelspacing, origin

class XPLORFile(object):
    """
    Class for reading XPLOR volume files created by NIH-XPLOR or CNS.
    """

    def __init__(self, fid):

        if isinstance(fid, file):
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
            header['ylength']   = float(line[12:24])
            header['zlength'] = float(line[24:36])
            header['alpha']    = float(line[36:48])
            header['beta'] = float(line[48:60])
            header['gamma']   = float(line[60:72])

            header['order'] = volume.readline()[0:3]

            self.header = header


    @property
    def voxelspacing(self):
        return self.header['xlength']/float(self.header['nx'])
   

    @property
    def origin(self):
        return [self.voxelspacing * x for x in [self.header['nxstart'], self.header['nystart'], self.header['nzstart']]]


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

            nslicelines = int(ceil(xextend*yextend/6.0))
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
                    if (n)%6 is 0:
                        out.write('\n')
            if (nx*ny)%6 > 0:
                out.write('\n')
        out.write('{:>8d}\n'.format(-9999))
        out.write('{:12.4E} {:12.4E} '.format(volume.array.mean(), volume.array.std()))
