

import numpy as np
import numpy.linalg as la

class ShapeDescriptor(object):

    """Class that calculates rotation-invariant shape descriptors based on the
    radius of gyration tensor.
    """

    def __init__(self, coor):
        coor = np.asmatrix(coor)
        # Transpose coordinates if the shape is N, 3
        if coor.shape[1] == 3:
            coor = coor.T
        self.coor = coor - coor.mean(axis=1).reshape(-1, 1)
        self.gyration_tensor = self.coor * self.coor.T
        eig_values, eig_vectors = la.eigh(self.gyration_tensor)
        # Sort eigenvalues so that l_x <= l_y <= l_z.
        sort_ind = np.argsort(eig_values)
        self.principal_moments = eig_values[sort_ind]
        self.lx, self.ly, self.lz = self.principal_moments
        self.principal_axes = eig_vectors[:, sort_ind]

    @property
    def Rg2(self):
        """Return the radius of gyration squared (Rg^2)"""
        return self.principal_moments.sum()

    @property
    def asphericity(self):
        return (3 * self.lz - self.Rg2) / 2

    @property
    def acylindricity(self):
        return self.ly - self.lx

    @property
    def relative_shape_anisotropy(self):
        """Returns the relatve shape anisotropy in the interval [0, 1].

        0 for spherical conformation
        1 for for linear chains.

        See Arkin and Janke. Journal of Chemical Physics 138, 2013.
        """
        lx, ly, lz = self.principal_moments
        return 1 - 3 * (lx * ly + ly * lz + lz * lx) / self.Rg2 ** 2

    @property
    def shape_anisotropy(self):
        ave_moment = self.principal_moments.mean()
        return np.prod(self.principal_moments - ave_moment) / ave_moment ** 3

