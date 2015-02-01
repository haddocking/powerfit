from __future__ import division
import numpy as np
import powerfit.libpowerfit as lpf

def rotate(points, rotmat, center=None):
    if center is None:
        return (np.mat(rotmat)*np.mat(points).T).T
    else:
        return (np.mat(rotmat) * np.mat(points - \
                np.asarray(center)).T).T + np.asarray(center)

def blur_points(points, sigma, weights, volume=None):

    gridsigma = sigma/volume.voxelspacing

    lpf.blur_points(gridcoor(points, volume.voxelspacing).astype(np.float64), 
            gridsigma, np.asarray(weights, dtype=np.float64), volume.array.astype(np.float64))

    return volume

def dilate_points(points, radius, volume):

    gridradius = radius/volume.voxelspacing

    lpf.dilate_points(gridcoor(points, volume.voxelspacing), gridradius, volume)

def gridcoor(points, voxelspacing):
    return (points - points.mean(axis=0))/voxelspacing
