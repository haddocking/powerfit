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

    rmax = <int> np.max([f1.shape[n] for n in range(3)]) / 2
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


@cython.boundscheck(False)
def rotate_image3d(np.ndarray[np.float64_t, ndim=3] image,
                   np.ndarray[np.float64_t, ndim=2] rotmat,
                   np.ndarray[np.float64_t, ndim=1] center,
                   int vlength,
                   np.ndarray[np.float64_t, ndim=3] out,
                   nearest=False):
    """Rotate an array around a center using trilinear interpolation

    Parameters
    ----------
    image : ndarray

    vlenght : unsigned integer
        Vertice length

    rotmat : ndarray
        Rotation matrix.

    out : ndarray
        Output array

    Notes
    -----
    The algorithm actually rotates the output array, thus the rotation matrix
    is inverted in the code, i.e. the transpose is taken during the rotation
    calculations.
    """

    cdef:
    # looping
        int x, y, z, z2, z2y2, z2y2x2, r2
    # rotation
        double xcoor_z, ycoor_z, zcoor_z
        double xcoor_yz, ycoor_yz, zcoor_yz
        double xcoor_xyz, ycoor_xyz, zcoor_xyz
    # interpolation
        int x0, y0, z0, x1, y1, z1
        double dx, dy, dz, dx1, dy1, dz1
        double c00, c01, c10, c11
        double c0, c1, c
        unsigned int near

    near = int(nearest)
    r2 = vlength ** 2

    for z in range(-vlength, vlength+1):
        xcoor_z = rotmat[2, 0]*z
        ycoor_z = rotmat[2, 1]*z
        zcoor_z = rotmat[2, 2]*z
        z2 = z * z

        for y in range(-vlength, vlength+1):
            xcoor_yz = rotmat[1, 0]*y + xcoor_z
            ycoor_yz = rotmat[1, 1]*y + ycoor_z
            zcoor_yz = rotmat[1, 2]*y + zcoor_z
            z2y2 = z2 + y * y

            for x in range(-vlength, vlength+1):
                z2y2x2 = z2y2 + x * x
                if z2y2x2 >= r2:
                    continue
                xcoor_xyz = rotmat[0, 0]*x + xcoor_yz + center[0]
                ycoor_xyz = rotmat[0, 1]*x + ycoor_yz + center[1]
                zcoor_xyz = rotmat[0, 2]*x + zcoor_yz + center[2]

                if near:
                    x0 = <int> round(xcoor_xyz)
                    y0 = <int> round(ycoor_xyz)
                    z0 = <int> round(zcoor_xyz)
                    c = image[z0, y0, x0]
                else:
                    # trilinear interpolation
                    x0 = <int> floor(xcoor_xyz)
                    y0 = <int> floor(ycoor_xyz)
                    z0 = <int> floor(zcoor_xyz)

                    x1 = x0 + 1
                    y1 = y0 + 1
                    z1 = z0 + 1

                    dx = xcoor_xyz - <double> x0
                    dy = ycoor_xyz - <double> y0
                    dz = zcoor_xyz - <double> z0
                    dx1 = 1 - dx
                    dy1 = 1 - dy
                    dz1 = 1 - dz

                    c00 = image[z0, y0, x0]*dx1 + image[z0, y0, x1]*dx
                    c10 = image[z0, y1, x0]*dx1 + image[z0, y1, x1]*dx
                    c01 = image[z1, y0, x0]*dx1 + image[z1, y0, x1]*dx
                    c11 = image[z1, y1, x0]*dx1 + image[z1, y1, x1]*dx

                    c0 = c00*dy1 + c10*dy
                    c1 = c01*dy1 + c11*dy

                    c = c0*dz1 + c1*dz

                out[z, y, x] = c


#@cython.boundscheck(False)
def rotate_grid(
        np.ndarray[np.float64_t, ndim=3] grid,
        np.ndarray[np.float64_t, ndim=2] rotmat,
        np.ndarray[np.float64_t] center, float radius,
        np.ndarray[np.float64_t, ndim=3] out, nearest=False
        ):
    """Rotate an array around a center using trilinear interpolation

    Notes
    -----
    The algorithm actually rotates the output array, thus the rotation matrix
    is inverted in the code, i.e. the transpose is taken during the rotation
    calculations.
    """

    cdef:
    # looping
        int x, y, z
    # rotation
        double xcoor, ycoor, zcoor
        double xcoor_z, ycoor_z, zcoor_z
        double xcoor_yz, ycoor_yz, zcoor_yz
        double xcoor_xyz, ycoor_xyz, zcoor_xyz
    # interpolation
        int x0, y0, z0, x1, y1, z1
        double dx, dy, dz, dx1, dy1, dz1
        double c00, c01, c10, c11
        double c0, c1, c

        unsigned int near
        int zmin, ymin, xmin, zmax, ymax, xmax
        double radius2, z_f, y_f, x_f, z2, z2y2, z2y2x2


    near = int(nearest)
    radius2 = radius ** 2
    # Determine box in which the rotation takes place
    zmin = <int> ceil(center[2] - radius)
    ymin = <int> ceil(center[1] - radius)
    xmin = <int> ceil(center[0] - radius)
    # By taking the ceiling, there is no need to add 1 to it to include it in
    # the loop later on
    zmax = <int> min(ceil(center[2] + radius), out.shape[0])
    ymax = <int> min(ceil(center[1] + radius), out.shape[1])
    xmax = <int> min(ceil(center[0] + radius), out.shape[2])

    xcoor = center[0]
    ycoor = center[1]
    zcoor = center[2]
    for z in range(zmin, zmax):
        z_f = z - center[2]
        xcoor_z = rotmat[2, 0] * z_f + xcoor
        ycoor_z = rotmat[2, 1] * z_f + ycoor
        zcoor_z = rotmat[2, 2] * z_f + zcoor
        z2 = z_f ** 2

        for y in range(ymin, ymax):
            y_f = y - center[1]
            xcoor_yz = rotmat[1, 0] * y_f + xcoor_z
            ycoor_yz = rotmat[1, 1] * y_f + ycoor_z
            zcoor_yz = rotmat[1, 2] * y_f + zcoor_z
            z2y2 = z2 + y_f ** 2

            for x in range(xmin, xmax):
                x_f = x - center[0]
                z2y2x2 = z2y2 + x_f ** 2
                if z2y2x2 >= radius2:
                    continue
                xcoor_xyz = rotmat[0, 0] * x_f + xcoor_yz
                ycoor_xyz = rotmat[0, 1] * x_f + ycoor_yz
                zcoor_xyz = rotmat[0, 2] * x_f + zcoor_yz

                if near:
                    x0 = <int> round(xcoor_xyz)
                    y0 = <int> round(ycoor_xyz)
                    z0 = <int> round(zcoor_xyz)
                    c = grid[z0, y0, x0]
                else:
                    # trilinear interpolation
                    x0 = <int> floor(xcoor_xyz)
                    y0 = <int> floor(ycoor_xyz)
                    z0 = <int> floor(zcoor_xyz)

                    x1 = x0 + 1
                    y1 = y0 + 1
                    z1 = z0 + 1

                    dx = xcoor_xyz - <double> x0
                    dy = ycoor_xyz - <double> y0
                    dz = zcoor_xyz - <double> z0
                    dx1 = 1 - dx
                    dy1 = 1 - dy
                    dz1 = 1 - dz

                    c00 = grid[z0, y0, x0]*dx1 + grid[z0, y0, x1]*dx
                    c10 = grid[z0, y1, x0]*dx1 + grid[z0, y1, x1]*dx
                    c01 = grid[z1, y0, x0]*dx1 + grid[z1, y0, x1]*dx
                    c11 = grid[z1, y1, x0]*dx1 + grid[z1, y1, x1]*dx

                    c0 = c00*dy1 + c10*dy
                    c1 = c01*dy1 + c11*dy

                    c = c0*dz1 + c1*dz

                out[z, y, x] = c


