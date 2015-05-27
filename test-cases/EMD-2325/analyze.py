from __future__ import print_function, division
from powerfit import PDB
from os.path import join
from glob import glob

base = PDB.fromfile('3zpz.pdb')
chains = ['O', 'P', 'Q', 'R', 'S', 'T', 'U']

ROOT = 'results-O'
subdirs = ['lcc', 'cw-lcc', 'l-lcc', 'l-cw-lcc']

for subdir in subdirs:

    for chain in chains:

        ref = base.select('chain', chain)

        fits = [join(ROOT, subdir, 'fit_{:d}.pdb'.format(n)) for n in range(1, 11)]

        min_rmsds = []
        rmsds = []
        for fit in fits:
            mob = PDB.fromfile(fit)
            rmsd = ref.rmsd(mob)
            rmsds.append(rmsd)

        min_rmsd = min(rmsds)
        rank = rmsds.index(min_rmsd) + 1
        print(subdir, chain, min_rmsd, rank)

        min_rmsds.append(min_rmsd)
    
    print('Average RMSD: ', sum(min_rmsds)/len(min_rmsds))


    print()