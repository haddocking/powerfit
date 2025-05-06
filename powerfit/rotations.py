
import os.path
from math import sqrt
from random import random

import numpy as np


def euler(angle, axis):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    if axis == 'x':
        matrix = [[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]]
    elif axis == 'y':
        matrix = [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]]
    elif axis == 'z':
        matrix = [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]
    else:
        raise ValueError('Axis is not recognized.')
    return np.asmatrix(matrix, dtype=np.float64)


def euler_to_rotmat(angles, order='zyz'):
    rotmat = np.asmatrix(np.identity(3, dtype=np.float64))
    for angle, axis in zip(angles, order):
        rotmat *= euler(angle, axis)
    return rotmat


def quat_to_rotmat(quaternions):

    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]

    Nq = w**2 + x**2 + y**2 + z**2
    s = np.zeros(Nq.shape, dtype=np.float64)
    s[Nq >  0.0] = 2.0/Nq[Nq > 0.0]
    s[Nq <= 0.0] = 0

    X = x*s
    Y = y*s
    Z = z*s

    rotmat = np.zeros((quaternions.shape[0],3,3), dtype=np.float64)
    rotmat[:,0,0] = 1.0 - (y*Y + z*Z)
    rotmat[:,0,1] = x*Y - w*Z
    rotmat[:,0,2] = x*Z + w*Y

    rotmat[:,1,0] = x*Y + w*Z
    rotmat[:,1,1] = 1.0 - (x*X + z*Z)
    rotmat[:,1,2] = y*Z - w*X

    rotmat[:,2,0] = x*Z - w*Y
    rotmat[:,2,1] = y*Z + w*X
    rotmat[:,2,2] = 1.0 - (x*X + y*Y)

    np.around(rotmat, decimals=8, out=rotmat)

    return rotmat


def random_rotmat():
    """Return a random rotation matrix"""

    s1 = 1
    while s1 >= 1.0:
        e1 = random() * 2 - 1
        e2 = random() * 2 - 1
        s1 = e1**2 + e2**2
            
    s2 = 1
    while s2 >= 1.0:
        e3 = random() * 2 - 1
        e4 = random() * 2 - 1
        s2 = e3**2 + e4**2
    
    q0 = e1
    q1 = e2
    q2 = e3 * sqrt((1 - s1)/s2 )
    q3 = e4 * sqrt((1 - s1)/s2 )

    quat = [[q0, q1, q2, q3]]
    return quat_to_rotmat(np.asarray(quat))[0]


def proportional_orientations(angle):
    
    # orientation sets available: name of file: (Norientations, degree)
    rot_sets = {'E.npy': (1, 360.0),
                'c48u1.npy': (24, 62.8),
                'c600v.npy': (60, 44.48),
                'c48n9.npy': (216, 36.47),
                'c600vc.npy': (360, 27.78),
                'c48u27.npy': (648, 20.83),
                'c48u83.npy': (1992, 16.29),
                'c48u181.npy': (4344, 12.29),
                'c48n309.npy': (7416, 9.72),
                'c48n527.npy': (12648, 8.17),
                'c48u815.npy': (19560, 7.4),
                'c48u1153.npy': (27672, 6.6),
                'c48u1201.npy': (28824, 6.48),
                'c48u1641.npy': (39384, 5.75),
                'c48u2219.npy': (53256, 5.27),
                'c48u2947.npy': (70728, 4.71),
                'c48u3733.npy': (89592, 4.37),
                'c48u4749.npy': (113976, 4.0),
                'c48u5879.npy': (141096, 3.74),
                'c48u7111.npy': (170664, 3.53),
                'c48u8649.npy': (207576, 3.26),
                }

    # determine the apropiate set to use
    smallestdiff = None
    for s, n in rot_sets.items():
        alpha = n[1]
        diff = abs(angle - alpha)
        if smallestdiff is None or diff < smallestdiff:
            smallestdiff = diff
            fname = s

    # read file
    infile = os.path.join(os.path.dirname(__file__), 'data', fname)
    quat_weights = np.load(infile)

    quat = quat_weights[:, :4]
    weights = quat_weights[:, -1]
    alpha = rot_sets[fname][1]

    return quat, weights, alpha
