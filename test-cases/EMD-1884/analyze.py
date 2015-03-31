from __future__ import print_function
from powerfit import PDB
from os.path import join
from glob import glob

ref = PDB.fromfile('2ykr.pdb').select('chain', 'W')
ROOT = 'results'
subdirs = ['lcc', 'cw-lcc', 'l-lcc', 'l-cw-lcc']

for subdir in subdirs:

    fits = [join(ROOT, subdir, 'fit_{:d}.pdb'.format(n)) for n in range(1, 11)]

    rmsds = []
    for fit in fits:
        mob = PDB.fromfile(fit)
        rmsd = ref.rmsd(mob)
        rmsds.append(rmsd)

    min_rmsd = min(rmsds)
    rank = rmsds.index(min_rmsd) + 1
    print(subdir, min_rmsd, rank)
