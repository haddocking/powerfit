from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.ndimage import label, maximum_position
from powerfit.volume import Volume

class Solutions(object):

    @classmethod
    def load(cls, lcc_fname, rotmat_fname, rotmat_ind_fname):
        return cls(Volume.fromfile(lcc_fname), 
                np.load(rotmat_fname), np.load(rotmat_ind_fname))

    def __init__(self, best_lcc, rotmat, rotmat_ind):

        self.best_lcc = best_lcc
        self.rotmat_ind = rotmat_ind
        self.rotmat = rotmat
        self._local_solutions = []

    def generate_local_solutions(self, steps=20):
        
        max_lcc = self.best_lcc.array.max()
        min_lcc = 0.2 * max_lcc
        
        stepsize = (max_lcc - min_lcc)/steps

        # get positions of high lcc-values using watershed algorithm
        cutoff = max_lcc
        positions = []
        for n in range(steps):
            cutoff -= stepsize
            lcc = self.best_lcc.array.copy()
            lcc[lcc < cutoff] = 0
            labels, nfeatures = label(lcc)
            positions += list(maximum_position(lcc, labels, range(1, nfeatures + 1)))
        positions = set(positions)

        # a solutions consists of the LCC-value, the xyz-coordinate 
        # in space and its corresponding rotation
        local_solutions = []
        for p in positions:
            rotmat = self.rotmat[self.rotmat_ind[p]]
            xyzcoor = [(i + ni)*self.best_lcc.voxelspacing\
                    for i, ni in zip(p, self.best_lcc.start[::-1])]

            local_solutions.append((self.best_lcc.array[p], xyzcoor, rotmat))
        local_solutions = sorted(local_solutions, key=lambda lcc: lcc[0], reverse = True)

        self._local_solutions = local_solutions

    def get_models(self, model, num=10):
        if not self._local_solutions:
            self.generate_local_solutions()
        
        num = min(num, len(self._local_solutions))
        # if num is smaller than 0, take all solutions
        if num < 0:
            num = len(self._local_solutions)

        models = []
        for n in range(0, num):
            lcc, xyzcoor, rotmat = self._local_solutions[n]
            outmodel = model.duplicate()
            outmodel.coor -= model.center
            outmodel.rotate(rotmat)
            outmodel.coor += np.asarray(xyzcoor, dtype=np.float64)[::-1]
            models.append(outmodel)
        return models

    def save(self, lcc_fname='lcc.mrc', rotmat_fname='rotmat.npy', 
            rotmat_ind_fname='rotmat_ind.npy'):

        self.best_lcc.tofile(lcc_fname)
        np.save(rotmat_fname, self.rotmat)
        np.save(rotmat_ind_fname, self.rotmat_ind)


    def write_pdb(self, model, num=10, fbase='fit'):
        if not self._local_solutions:
            self.generate_local_solutions()
        
        num = min(num, len(self._local_solutions))
        # if num is smaller than 0, take all solutions
        if num < 0:
            num = len(self._local_solutions)

        for n in range(0, num):
            lcc, xyzcoor, rotmat = self._local_solutions[n]
            outmodel = model.duplicate()
            outmodel.coor -= model.center
            outmodel.rotate(rotmat)
            outmodel.coor += np.asarray(xyzcoor, dtype=np.float64)[::-1]
            outmodel.tofile(fbase + '_{:d}.pdb'.format(n+1))


    def write_local_solutions(self, fname='solutions.out'):
        
        if not self._local_solutions:
            self.generate_local_solutions()

        with open(fname, 'w') as out:
            for sol in self._local_solutions:
                lcc, xyzcoor, rotmat = sol
                line = '{:5.3f}' + ' {:8.3f}'*3 + ' {:8.5f}'*9 + '\n'
                out.write(line.format(lcc, xyzcoor[0], xyzcoor[1], xyzcoor[2], 
                        rotmat[0][0], rotmat[0][1], rotmat[0][2],
                        rotmat[1][0], rotmat[1][1], rotmat[1][2],
                        rotmat[2][0], rotmat[2][1], rotmat[2][2],
                        ))
