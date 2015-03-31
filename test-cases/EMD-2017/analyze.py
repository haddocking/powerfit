from __future__ import print_function
from powerfit import PDB
from os.path import join
from glob import glob

ref = PDB.fromfile('4adv_V.pdb')
ROOT = 'results'
subdirs = ['lcc', 'cw-lcc', 'l-lcc', 'l-cw-lcc']

for subdir in subdirs:

    fits = glob(join(ROOT, subdir, 'fit') + '*.pdb')

    for fit in fits:
        mob = PDB.fromfile(fit)
        rmsd = ref.rmsd(mob)

        if rmsd < 8:
            print(subdir, fit, rmsd)
