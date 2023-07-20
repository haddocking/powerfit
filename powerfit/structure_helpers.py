# Original repsositry by haddock labs,
# licensed under the Apache License, Version 2.0.

# Modified by Luc Elliott, 24/04/2023, with the following modifications: 
#   Updated the code to be compatible with Python 3.7.
#   Using strutcure package Struvolpy to parse pdb files.
#   Structure class now inherits from Struvolpy Structure class.
#   Adds the properties to the Structure class and per original code:

# For more information about the original code, please see https://github.com/haddocking/powerfit. 

# Your modified code follows...

from __future__ import absolute_import
from string import capwords
import numpy as np
from .elements import ELEMENTS
from struvolpy import Structure as svpStructure
import TEMPy


MODEL = 'MODEL '
ATOM = 'ATOM  '
HETATM = 'HETATM'
TER = 'TER   '

MODEL_LINE = 'MODEL ' + ' ' * 4 + '{:>4d}\n'
ENDMDL_LINE = 'ENDMDL\n'
TER_LINE = 'TER   ' + '{:>5d}' + ' ' * 6 + '{:3s}' + ' ' + '{:1s}' + \
        '{:>4d}' + '{:1s}' + ' ' * 53 + '\n'
ATOM_LINE = '{:6s}' + '{:>5d}' + ' ' + '{:4s}' + '{:1s}' + '{:3s}' + ' ' + \
        '{:1s}' + '{:>4d}' + '{:1s}' + ' ' * 3 + '{:8.3f}' * 3 + '{:6.2f}' * 2 + \
        ' ' * 10 + '{:<2s}' + '{:2s}\n'
END_LINE = 'END   \n'

ATOM_DATA = ('record id name alt resn chain resi i x y z q b ' \
        'e charge').split()
TER_DATA = 'id resn chain resi i'.split()



class Structure(svpStructure):
    def __init__(self, filename, gemmi_structure):
        super().__init__(filename, gemmi_structure)
    

    """Private method to get a property of the atoms in the structure"""
    def __get_property(self, ptype):
        elements, ind = np.unique(
            [atom.element.name for atom in self.__get_atoms()], return_inverse=True
            )
        return np.asarray([getattr(ELEMENTS[capwords(e)], ptype) 
            for e in elements], dtype=np.float64)[ind]
    

    @property
    def atomnumber(self):
        """Return array of atom numbers"""
        return self.__get_property('number')


    @property
    def mass(self):
        return self.__get_property('mass')



    @property
    def rvdw(self):
        return self.__get_property('vdwrad')







