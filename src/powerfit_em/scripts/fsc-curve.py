import sys

from numpy.fft import fftn

from powerfit_em import Volume
from powerfit_em._powerfit import fsc_curve

def main():
    vol1 = Volume.fromfile(sys.argv[1])
    vol2 = Volume.fromfile(sys.argv[2])

    ft_vol1 = fftn(vol1.array)
    ft_vol2 = fftn(vol2.array)

    fsc = fsc_curve(ft_vol1, ft_vol2)
    res = [n / vol1.dimensions[0] for n in range(fsc.size)]
    inv_res = [1 / r for r in res[1:]]

    print(fsc)
    print()
    print(res)
    print()
    print(inv_res)
