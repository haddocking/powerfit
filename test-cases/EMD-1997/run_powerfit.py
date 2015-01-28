from __future__ import print_function
from time import time
from powerfit import PDB, Volume, PowerFitter
from powerfit.rotations import proportional_orientations, quat_to_rotmat

q, w, a = proportional_orientations(90)
pf = PowerFitter()
pf.map = Volume.fromfile('1997.mrc')
pf.model = PDB.fromfile('1oel.pdb')
pf.rotations = quat_to_rotmat(q)
pf.resolution = 7
pf.core_weighted = True
pf.laplace = True

time0 = time()
sol = pf.search()


print('Time for search: ', time() - time0)

sol.write_local_solutions('solutions.out')
sol.best_lcc.tofile('best_lcc.mrc')
sol.write_pdb(pf.model, num=2)
