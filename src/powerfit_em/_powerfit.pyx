import cython
import numpy as np
cimport numpy as np
from libc.math cimport ceil, exp, floor, sqrt, round

@cython.boundscheck(False)
def fsc_curve(np.ndarray[np.complex128_t, ndim=3] f1,
              np.ndarray[np.complex128_t, ndim=3] f2):

    cdef:
        int z, y, x, r, rmax
        double zmax, ymax, xmax
        np.ndarray[np.float64_t] f_1_2, f_1_1, f_2_2

    rmax = <int> np.max([f1.shape[n] for n in range(3)]) // 2
    f_1_2 = <np.ndarray[np.float64_t]> np.zeros(rmax)
    f_1_1 = <np.ndarray[np.float64_t]> np.zeros(rmax)
    f_2_2 = <np.ndarray[np.float64_t]> np.zeros(rmax)

    zmax = f1.shape[0] / 2.0
    ymax = f1.shape[1] / 2.0
    xmax = f1.shape[2] / 2.0

    for z in range(f1.shape[0]):
        if z > zmax:
            z -= f1.shape[0]
        for y in range(f1.shape[1]):
            if y > ymax:
                y -= f1.shape[1]
            for x in range(f1.shape[2]):
                if x > xmax:
                    x -= f1.shape[2]
                r = <int> round(sqrt(x * x + y * y + z * z))
                if r >= rmax:
                    continue

                f_1_2[r] += (f1[z, y, x] * f2[z, y, x].conjugate()).real
                f_1_1[r] += (f1[z, y, x] * f1[z, y, x].conjugate()).real
                f_2_2[r] += (f2[z, y, x] * f2[z, y, x].conjugate()).real

    fsc = f_1_2 / np.sqrt(f_1_1 * f_2_2)
    return fsc


@cython.boundscheck(False)
@cython.wraparound(False)
def conj_multiply(np.ndarray[np.complex128_t] in1,
        np.ndarray[np.complex128_t] in2,
        np.ndarray[np.complex128_t] out,
        ):
    cdef unsigned int n
    for n in range(out.shape[0]):
        out[n] = in1[n].conjugate() * in2[n]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calc_lcc(np.ndarray[np.float64_t] gcc,
        np.ndarray[np.float64_t] ave,
        np.ndarray[np.float64_t] ave2,
        np.ndarray[np.uint8_t] mask,
        np.ndarray[np.float64_t] lcc):

    cdef:
        unsigned int n
        np.float64_t var

    for n in range(gcc.shape[0]):
        if mask[n]:
            var = (ave2[n] - ave[n] ** 2)
            lcc[n] = gcc[n] / sqrt(var)


@cython.boundscheck(False)
@cython.cdivision(True)
def blur_points(np.ndarray[np.float64_t, ndim=2] points,
                np.ndarray[np.float64_t, ndim=1] weights,
                double sigma,
                np.ndarray[np.float64_t, ndim=3] out,
                wraparound=False,
                ):
    """Convolutes points with a Gaussian kernel

    Parameters
    ----------

    Returns
    -------
    None

    """

    cdef:
        unsigned int n
        int x, y, z, xmin, ymin, zmin, xmax, ymax, zmax
        double extend, extend2, dsigma2
        double z2, y2z2, x2y2z2

    extend = 4.0 * sigma
    extend2 = extend * extend
    dsigma2 = 2.0 * sigma * sigma

    if wraparound:
        xmin_limit = -out.shape[2] + 1
        ymin_limit = -out.shape[1] + 1
        zmin_limit = -out.shape[0] + 1
    else:
        xmin_limit = 0
        ymin_limit = 0
        zmin_limit = 0

    xmax_limit = out.shape[2] - 1
    ymax_limit = out.shape[1] - 1
    zmax_limit = out.shape[0] - 1

    for n in range(points.shape[1]):

        xmin = <int> ceil(points[0, n] - extend)
        ymin = <int> ceil(points[1, n] - extend)
        zmin = <int> ceil(points[2, n] - extend)
        xmin = max(xmin, xmin_limit)
        ymin = max(ymin, ymin_limit)
        zmin = max(zmin, zmin_limit)

        xmax = <int> floor(points[0, n] + extend)
        ymax = <int> floor(points[1, n] + extend)
        zmax = <int> floor(points[2, n] + extend)
        xmax = min(xmax, xmax_limit)
        ymax = min(ymax, ymax_limit)
        zmax = min(zmax, zmax_limit)

        for z in range(zmin, zmax+1):
            z2 = (z - points[2, n])**2
            for y in range(ymin, ymax+1):
                y2z2 = (y - points[1, n])**2 + z2
                for x in range(xmin, xmax+1):
                    x2y2z2 = (x - points[0, n])**2 + y2z2
                    if x2y2z2 <= extend2:
                        out[z,y,x] += weights[n] * exp(-x2y2z2 / dsigma2)


@cython.boundscheck(False)
@cython.cdivision(True)
def dilate_points(np.ndarray[np.float64_t, ndim=2] points,
                np.ndarray[np.float64_t] radii,
                np.ndarray[np.float64_t, ndim=3] out,
                wraparound=False,
                ):
    """Dilate points into balls on a grid

    Parameters
    ----------

    Returns
    -------
    None

    """

    cdef:
        unsigned int n
        int x, y, z, xmin, ymin, zmin, xmax, ymax, zmax
        double radius, radius2
        double z2, y2z2, x2y2z2

    if wraparound:
        xmin_limit = -out.shape[2] + 1
        ymin_limit = -out.shape[1] + 1
        zmin_limit = -out.shape[0] + 1
    else:
        xmin_limit = 0
        ymin_limit = 0
        zmin_limit = 0

    xmax_limit = out.shape[2] - 1
    ymax_limit = out.shape[1] - 1
    zmax_limit = out.shape[0] - 1

    for n in range(points.shape[1]):

        radius = radii[n]
        radius2 = radius ** 2

        xmin = <int> ceil(points[0, n] - radius)
        ymin = <int> ceil(points[1, n] - radius)
        zmin = <int> ceil(points[2, n] - radius)
        xmin = max(xmin, xmin_limit)
        ymin = max(ymin, ymin_limit)
        zmin = max(zmin, zmin_limit)

        xmax = <int> floor(points[0, n] + radius)
        ymax = <int> floor(points[1, n] + radius)
        zmax = <int> floor(points[2, n] + radius)
        xmax = min(xmax, xmax_limit) + 1
        ymax = min(ymax, ymax_limit) + 1
        zmax = min(zmax, zmax_limit) + 1

        for z in range(zmin, zmax):
            z2 = (z - points[2, n])**2
            for y in range(ymin, ymax):
                y2z2 = (y - points[1, n])**2 + z2
                for x in range(xmin, xmax):
                    x2y2z2 = (x - points[0, n])**2 + y2z2
                    if x2y2z2 <= radius2:
                        out[z,y,x] = 1
