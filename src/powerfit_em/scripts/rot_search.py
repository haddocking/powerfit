
from argparse import ArgumentParser, FileType
from time import time
import os

import numpy as np
from scipy.ndimage import laplace

from powerfit_em import Volume, Structure, quat_to_rotmat, proportional_orientations, determine_core_indices
from powerfit_em._powerfit import rotate_grid
from powerfit_em.volume import zeros_like, res_to_sigma, structure_to_shape


def parse_args():
    p = ArgumentParser(description='Perform a full rotational search of a structure in a density.')

    p.add_argument('target', type=FileType('rb'), help="Target density map.")
    p.add_argument('resolution', type=float, help="Resolution of target map")
    p.add_argument('template', type=FileType('r'), help="PDB file of template")

    p.add_argument('-a', '--angle', type=float, default=360, help="Rotational sampling interval in degree")
    p.add_argument('-l', '--laplace', dest='laplace', action='store_true', help="Use Laplacian pre-filter.")
    p.add_argument('-cw', '--core-weighted', dest='core_weighted', action='store_true', help='Use core-weighted correlation function')
    p.add_argument('-n', '--nsolutions', dest='nsolutions', type=int, default=2, help="Number of solutions written to file.")
    p.add_argument('-d', '--directory', dest='directory', type=str, default='.', help="Directory where results are stored.")
    p.add_argument('-o', '--output', dest='outfile', type=str, default='solutions.out', help="Filename holding the results.")

    args = p.parse_args()
    return args


def calc_lcc(target, template, mask, N):
    ind = mask != 0
    tmp = target * mask
    gcc = (tmp[ind] * template[ind]).sum()
    std = tmp[ind].std()
    lcc = gcc / std / N
    return lcc


def main():

    args = parse_args()
    try:
        os.makedirs(args.directory)
    except OSError:
        pass

    target = Volume.fromfile(args.target)
    structure = Structure.fromfile(args.template)
    center = structure.coor.mean(axis=1)
    radius = np.linalg.norm((structure.coor - center.reshape(-1, 1)), axis=0).max() + 0.5 * args.resolution

    template = zeros_like(target)
    rottemplate = zeros_like(target)
    mask = zeros_like(target)
    rotmask = zeros_like(target)
    structure_to_shape(structure.coor, args.resolution, out=template, shape='vol', weights=structure.atomnumber)
    structure_to_shape(structure.coor, args.resolution, out=mask, shape='mask')

    if args.laplace:
        target.array = laplace(target.array, mode='constant')
        template.array = laplace(template.array, mode='constant')
    if args.core_weighted:
        mask.array = determine_core_indices(mask.array)

    # Normalize the template density
    ind = mask.array != 0
    N = ind.sum()
    template.array *= mask.array
    template.array[ind] -= template.array[ind].mean()
    template.array[ind] /= template.array[ind].std()

    rotmat = quat_to_rotmat(proportional_orientations(args.angle)[0])

    lcc_list = []
    center -= target.origin
    center /= template.voxelspacing
    radius /= template.voxelspacing
    time0 = time()
    for n, rot in enumerate(rotmat):
        rotate_grid(template.array, rot, center, radius, rottemplate.array)
        rotate_grid(mask.array, rot, center, radius, rotmask.array, nearest=True)
        lcc = calc_lcc(target.array, rottemplate.array, rotmask.array, N)
        lcc_list.append(lcc)
        print('{:d}              \r'.format(n), end=' ')

    print('Searching took: {:.0f}m {:.0f}s'.format(*divmod(time() - time0, 60)))
    ind = np.argsort(lcc_list)[::-1]
    with open(os.path.join(args.directory, args.outfile), 'w') as f:
        line = ' '.join(['{:.4f}'] + ['{:7.4f}'] * 9) + '\n'
        for n in range(min(args.nsolutions, len(lcc_list))):
            f.write(line.format(lcc_list[ind[n]], *rotmat[ind[n]].ravel()))



if __name__ == "__main__":
    main()
