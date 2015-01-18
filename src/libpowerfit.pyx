import cython
import numpy as np
cimport numpy as np
from libc.math cimport ceil, exp

def blur_points(np.ndarray[np.float64_t, ndim=2] points,
                np.ndarray[np.float64_t, ndim=1] weights,
                double sigma,
                np.ndarray[np.float64_t, ndim=3] out,
                ):
    """Convolutes points with a Gaussian kernel

    Parameters
    ----------

    Returns
    -------
    None

    """
    cdef unsigned int n
    cdef int x, y, z, xmin, ymin, zmin, xmax, ymax, zmax
    cdef double extend, extend2, dsigma2
    cdef double dx, x2, dy, x2y2, dz, distance2

    extend = 4.0 * sigma
    extend2 = extend**2
    dsigma2 = 2.0 * sigma**2

    for n in range(points.shape[0]):

        xmin = <int> ceil(points[n, 0] - extend)
        ymin = <int> ceil(points[n, 1] - extend)
        zmin = <int> ceil(points[n, 2] - extend)

        xmax = <int> (points[n, 0] + extend)
        ymax = <int> (points[n, 1] + extend)
        zmax = <int> (points[n, 2] + extend)

        for x in range(xmin, xmax+1):
            dx = x - points[n, 0]
            x2 = dx**2
            for y in range(ymin, ymax+1):
                dy = y - points[n, 1]
                x2y2 = x2 + dy**2
                for z in range(zmin, zmax+1):
                    dz = z - points[n, 2]
                    distance2 = x2y2 + dz**2
                    if distance2 <= extend2:
                        out[z,y,x] += weights[n, 3] * exp(-distance2/dsigma2)

def dilate_points(np.ndarray[np.float64_t, ndim=2] points,
                  double radius,
                  np.ndarray[np.float64_t, ndim=3] out,
                  ):
    """Creates a mask from the points into the volume

    Parameters
    ----------
    points : a (2,4)-dimensional numpy array
	The points have coordinates and a weight. The coordinates are
	in angstrom. 

    radius : float
	The radius of each point to where to mask in angstrom.

    out : 3D-numpy array

    Returns
    -------
    None
    """
    cdef unsigned int n
    cdef int x, y, z, xmin, ymin, zmin, xmax, ymax, zmax
    cdef double radius2
    cdef double dx, x2, dy, x2y2, dz, distance2

    radius2 = radius**2

    for n in range(points.shape[0]):

        xmin = <int> ceil(points[n, 0] - radius)
        ymin = <int> ceil(points[n, 1] - radius)
        zmin = <int> ceil(points[n, 2] - radius)

        xmax = <int> (points[n, 0] + radius)
        ymax = <int> (points[n, 1] + radius)
        zmax = <int> (points[n, 2] + radius)

        for x in range(xmin, xmax+1):
            dx = x - points[n, 0]
            x2 = dx**2
            for y in range(ymin, ymax+1):
                dy = y - points[n, 1]
                x2y2 = x2 + dy**2
                for z in range(zmin, zmax+1):
                    dz = z - points[n, 2]
                    distance2 = x2y2 + dz**2
                    if distance2 <= radius2:
                        out[z,y,x] = 1.0
