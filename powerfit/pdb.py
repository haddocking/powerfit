from __future__ import absolute_import, print_function, division
import os.path
import operator
import numpy as np
from powerfit.IO.pdb import parse_pdb, write_pdb
from powerfit.IO.mmcif import parse_cif
import powerfit.atompar

class PDB(object):

    @classmethod
    def fromfile(cls, pdbfile):
        extension = os.path.splitext(pdbfile)[1]
     
        if extension == '.cif':
            return cls(parse_cif(pdbfile))
        elif extension in ('.pdb', '.ent'):
            return cls(parse_pdb(pdbfile))
        else:
            raise ValueError("Format of file is not recognized")

    def __init__(self, pdbdata):
        self.data = pdbdata

    @property
    def atomnumber(self):
        elements, ind = np.unique(self.data['element'], return_inverse=True)
        atomnumbers = np.asarray([powerfit.atompar.parameters[e]['Z'] for e in elements], dtype=np.float64)
        return atomnumbers[ind]

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

    @property
    def sequence(self):
        resids, indices = np.unique(self.data['resi'], return_index=True)
        return self.data['resn'][indices]

    def combine(self, pdb):
        return PDB(np.hstack((self.data, pdb.data)))

    def duplicate(self):
        return PDB(self.data.copy())

    def rmsd(self, pdb):
        return np.sqrt(((self.coor - pdb.coor)**2).mean()*3)

    def rotate(self, rotmat):
        self.data['x'], self.data['y'], self.data['z'] =\
             np.mat(rotmat) * np.mat(self.coor).T

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

    def tofile(self, fid):
        write_pdb(fid, self.data)
