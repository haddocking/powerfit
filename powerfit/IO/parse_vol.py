import os.path
from powerfit.IO.mrc import parse_mrc, to_mrc
from powerfit.IO.xplor import parse_xplor, to_xplor

def parse_vol(fid, fmt=None):

    if isinstance(fid, file):
        fname = fid.name
    elif isinstance(fid, str):
        fname = fid
    else:
        raise TypeError('Input should either be a file of filename')

    if fmt is None:
        fmt = os.path.splitext(fname)[1].strip('.')

    if fmt in ('xplor', 'cns'):
        array, voxelspacing, origin = parse_xplor(fname)
    elif fmt in ('ccp4', 'mrc', 'map'):
        array, voxelspacing, origin = parse_mrc(fname)
    else:
        raise IOError("Format of volume file is not supported.")

    return array, voxelspacing, origin


def to_vol(fid, volume, fmt=None):

    if fmt is None:
        fmt = os.path.splitext(fid)[1].strip('.')

    if fmt in ('xplor', 'cns'):
        to_xplor(fid, volume)
    elif fmt in ('ccp4', 'mrc', 'map'):
        to_mrc(fid, volume)
    else:
        raise IOError("Format of volume file is not supported.")
