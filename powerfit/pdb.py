from __future__ import absolute_import
from .IO.pdb import parse_pdb, write_pdb
import operator
import numpy as np
from numpy.linalg import norm as _norm

class PDB(object):

    @classmethod
    def fromfile(cls, pdbfile):
        return cls(parse_pdb(pdbfile))

    def __init__(self, pdbdata):
        self.data = pdbdata
        self.nmodels = np.unique(self.data['model']).size
	self.nchains = np.unique(self.data['chain']).size
        self.nresidues = np.unique(self.data['resi']).size
        self.natoms = self.data['atom_id'].shape[0]

    @property
    def coor(self):
        return np.asarray([self.data['x'], self.data['y'], self.data['z']]).T

    @coor.setter
    def coor(self, coor_array):
        self.data['x'], self.data['y'], self.data['z'] = coor_array.T
        
    @property
    def center(self):
        return self.coor.mean(axis=0)

    @property
    def chain_list(self):
        return np.unique(self.data['chain'])

    def combine(self, pdb):
        return PDB(np.hstack((self.data, pdb.data)))

    @property
    def distance_to_center(self):
        return _norm(self.coor - self.center, axis=1)

    def duplicate(self):
        return PDB(self.data.copy())

    def rmsd(self, pdb):
        return np.sqrt(((self.coor - pdb.coor)**2).mean()*3)

    def rotate(self, rotmat, center=None):
        if center is None:
            self.data['x'], self.data['y'], self.data['z'] =\
                 np.mat(rotmat) * np.mat(self.coor).T
        else:
            self.data['x'], self.data['y'], self.data['z'] =\
                 np.mat(rotmat) * np.mat(self.coor - np.asarray(center)).T + np.asarray(center)

    def select(self, identifier, value, loperator='=='):
        """A simple and probably pretty inefficient way of selection atoms"""
        if loperator == '==':
            oper = operator.eq
        elif loperator == '<':
            oper = operator.lt
        elif loperator == '>':
            oper = operator.gt
        elif loperator == '>=':
            oper = operator.ge
        elif loperator == '<=':
            oper = operator.le
        selection = np.where(oper(self.data[identifier], value))

        return PDB(self.data[selection])

    @property
    def sequence(self):
        resids, indices = np.unique(self.data['resi'], return_index=True)
        return self.data['resn'][indices]

    def tofile(self, fid):
        write_pdb(fid, self.data)
