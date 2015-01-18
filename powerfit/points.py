from __future__ import division
import numpy as np
import powerfit.libpowerfit as lpf

def rotate(points, rotmat, center=None):
    if center is None:
        return (np.mat(rotmat)*np.mat(points).T).T
    else:
        return (np.mat(rotmat) * np.mat(points - \
                np.asarray(center)).T).T + np.asarray(center)

def blur_points(points, sigma, weights, volume):

    gridsigma = sigma/volume.voxelspacing

    lpf.blur_points(gridcoor(points, volume), gridsigma, weights, volume)

def dilate_points(points, radius, volume):

    gridradius = radius/volume.voxelspacing

    lpf.dilate_points(gridcoor(points, volume), gridradius, volume)

def gridcoor(points, volume):
    return (points - volume.origin)/volume.voxelspacing
